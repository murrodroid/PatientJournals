from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import mimetypes
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from google.genai import types
from google.cloud import storage
from tqdm import tqdm

from batch_client import get_batch_client, resolve_service_account_path
from config import config
from generation_spec import build_batch_generation_config, prompt_text
from models import resolve_model_spec
from tools import create_subfolder, get_run_logger
from upload import upload_missing_images, upload_missing_pdfs


_ALLOWED_FP_MODES = {"all", "only_fp", "exclude_fp"}
_ALLOWED_UPLOAD_SOURCES = {"pdf", "images", "auto"}
_ANTHROPIC_CUSTOM_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SBID_TOKEN_PATTERN = re.compile(r"(\d{5,})")


def _anthropic_custom_id_for_key(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:40]
    return f"gcs_{digest}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit batch job(s) from GCS image inputs."
    )
    parser.add_argument(
        "-n",
        "--num-batches",
        dest="num_batches",
        type=int,
        help=(
            "Split inputs into N smaller batch jobs. Overrides config.batch_num_chunks."
        ),
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help=(
            "Only resubmit chunks that are not successful from the latest "
            "submit run (or --run-dir)."
        ),
    )
    parser.add_argument(
        "--run-dir",
        dest="run_dir",
        help=(
            "Submit run directory to resume. "
            "Used with --rerun to choose which run to resume."
        ),
    )
    parser.add_argument(
        "--downscale",
        dest="downscale",
        type=float,
        help=(
            "Randomly sample this fraction of GCS inputs before batching. "
            "Must be > 0 and <= 1. Example: --downscale 0.1"
        ),
    )
    return parser.parse_args()


def _resolve_num_batches(args: argparse.Namespace) -> int:
    value = args.num_batches
    if value is None:
        value = int(config.batch_num_chunks or 1)
    if value <= 0:
        raise ValueError(f"num_batches must be >= 1 (received {value}).")
    return value


def _validate_batch_model_support() -> str:
    spec = resolve_model_spec(config.model)
    if not spec.supports_batch:
        raise ValueError(
            "Configured model does not support batch jobs in this pipeline. "
            f"Resolved model='{config.model}' provider='{spec.provider}' "
            f"supports_batch={spec.supports_batch}."
        )
    if spec.provider not in {"gemini", "anthropic"}:
        raise ValueError(
            "batch_submit.py currently supports provider-specific batch paths "
            f"for Gemini and Anthropic only (resolved provider='{spec.provider}')."
        )
    return spec.provider


def _get_anthropic_client():
    try:
        import anthropic
    except Exception as exc:
        raise RuntimeError(
            "Anthropic provider requested but the 'anthropic' package is unavailable."
        ) from exc

    api_key = config.api_key_for_provider("anthropic")
    return anthropic.Anthropic(api_key=api_key)


def _resolve_downscale(args: argparse.Namespace) -> float | None:
    value = args.downscale
    if value is None:
        return None
    if value <= 0.0 or value > 1.0:
        raise ValueError(f"downscale must be > 0 and <= 1 (received {value}).")
    return float(value)


def _warn_if_confidence_scores_unsupported(*, provider: str, log) -> None:
    if not bool(config.include_confidence_scores):
        return
    if provider == "anthropic":
        log(
            "include_confidence_scores=True requested, but Anthropic Messages API "
            "does not expose token logprobs in responses. "
            "Field-level confidence output will be empty."
        )


def _downscale_blobs_randomly(
    blobs: list[storage.Blob],
    *,
    downscale: float,
) -> list[storage.Blob]:
    if not blobs or downscale >= 1.0:
        return blobs

    total = len(blobs)
    target = int(round(total * downscale))
    target = max(1, min(total, target))
    if target >= total:
        return blobs

    sampled = random.sample(blobs, k=target)
    return sorted(sampled, key=lambda item: item.name)


def _split_blobs_evenly(
    blobs: list[storage.Blob],
    num_batches: int,
) -> list[list[storage.Blob]]:
    if not blobs:
        return []

    if num_batches <= 1:
        return [blobs]

    target_batches = min(num_batches, len(blobs))
    base_size, remainder = divmod(len(blobs), target_batches)
    chunks: list[list[storage.Blob]] = []
    start = 0
    for index in range(target_batches):
        size = base_size + (1 if index < remainder else 0)
        end = start + size
        if size > 0:
            chunks.append(blobs[start:end])
        start = end
    return chunks


def _chunk_requests_file_name(
    base_name: str,
    *,
    chunk_index: int,
    total_chunks: int,
) -> str:
    base = Path(base_name)
    suffix = base.suffix or ".jsonl"
    stem = base.stem
    return f"{stem}.part{chunk_index:03d}-of-{total_chunks:03d}{suffix}"


def _chunk_label(*, chunk_index: int, total_chunks: int) -> str:
    return f"chunk_{chunk_index:03d}_of_{total_chunks:03d}"


def _pages_prefix() -> str:
    return _normalize_prefix(config.gcs_pages_prefix or "")


def _resolved_upload_source() -> str:
    upload_source = str(config.upload_source or "pdf").strip().lower()
    if upload_source not in _ALLOWED_UPLOAD_SOURCES:
        raise ValueError(
            f"Unsupported upload_source: {config.upload_source}. "
            f"Expected one of: {sorted(_ALLOWED_UPLOAD_SOURCES)}"
        )
    return upload_source


def _configured_year_filter_tokens() -> list[int]:
    raw = getattr(config, "batch_year_filter", ())
    if raw is None:
        return []

    if isinstance(raw, (int, str)):
        values: list[object] = [raw]
    elif isinstance(raw, (tuple, list, set)):
        values = list(raw)
    else:
        raise ValueError(
            "config.batch_year_filter must be an int/str or a list/tuple/set "
            f"of year values (received {type(raw).__name__})."
        )

    tokens: set[int] = set()
    for value in values:
        if isinstance(value, int):
            token = int(value)
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            if not re.fullmatch(r"\d{1,4}", text):
                raise ValueError(
                    f"Invalid year token in config.batch_year_filter: {value!r}. "
                    "Use integers like 91, 94, 1891."
                )
            token = int(text)
        else:
            raise ValueError(
                "config.batch_year_filter contains unsupported value type "
                f"{type(value).__name__}: {value!r}"
            )

        if token < 0 or token > 9999:
            raise ValueError(f"Year token out of supported range [0, 9999]: {token}")
        tokens.add(token)

    return sorted(tokens)


def _load_date_mapping_year_by_sbid(path: Path) -> tuple[dict[str, int], set[int]]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"date mapping file not found: {path}. "
            "Set config.batch_date_mapping_file to a valid CSV file."
        )

    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        except csv.Error:
            dialect = csv.excel
            dialect.delimiter = ";"

        reader = csv.DictReader(handle, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError(f"date mapping file has no header row: {path}")

        columns = {
            str(name).strip().lower(): str(name)
            for name in reader.fieldnames
            if isinstance(name, str) and name.strip()
        }
        sbid_col = columns.get("sbid")
        year_col = columns.get("year")
        if not sbid_col or not year_col:
            available = ", ".join(reader.fieldnames)
            raise ValueError(
                f"date mapping file {path} must contain 'sbid' and 'year' columns. "
                f"Found: {available}"
            )

        mapping: dict[str, int] = {}
        years: set[int] = set()
        for row in reader:
            sbid_raw = str(row.get(sbid_col) or "").strip()
            year_raw = str(row.get(year_col) or "").strip()
            if not sbid_raw or not year_raw:
                continue

            sbid_match = _SBID_TOKEN_PATTERN.search(sbid_raw)
            if not sbid_match:
                continue
            sbid = sbid_match.group(1)

            try:
                year = int(year_raw)
            except ValueError:
                continue

            mapping[sbid] = year
            years.add(year)

    if not mapping:
        raise ValueError(
            f"date mapping file {path} did not produce any sbid->year rows."
        )
    return mapping, years


def _resolve_year_filter_targets(
    tokens: list[int],
    *,
    available_years: set[int],
) -> set[int]:
    if not tokens:
        return set()

    targets: set[int] = set()
    sorted_available = sorted(available_years)
    for token in tokens:
        if token >= 1000:
            if token not in available_years:
                raise ValueError(
                    f"Year {token} was requested in config.batch_year_filter but "
                    "is not present in date_mapping.csv. "
                    f"Available years: {sorted_available}"
                )
            targets.add(token)
            continue

        suffix = token % 100
        matches = sorted(year for year in available_years if year % 100 == suffix)
        if not matches:
            raise ValueError(
                f"Year token '{token}' in config.batch_year_filter did not match "
                "any year suffix in date_mapping.csv. "
                f"Available years: {sorted_available}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Year token '{token}' is ambiguous in date_mapping.csv. "
                f"Matches: {matches}. Use full year (e.g. {matches[0]})."
            )
        targets.add(matches[0])

    return targets


def _extract_sbid_from_blob_name(blob_name: str, *, pages_prefix: str) -> str | None:
    parts = list(Path(blob_name).parts)
    if len(parts) < 2:
        return None

    folder_parts = parts[:-1]
    normalized_prefix = pages_prefix.rstrip("/")
    if normalized_prefix:
        prefix_parts = list(Path(normalized_prefix).parts)
        if prefix_parts and folder_parts[: len(prefix_parts)] == prefix_parts:
            folder_parts = folder_parts[len(prefix_parts) :]

    for folder in reversed(folder_parts):
        matches = _SBID_TOKEN_PATTERN.findall(folder)
        if matches:
            return matches[-1]

    filename_stem = Path(parts[-1]).stem
    stem_matches = _SBID_TOKEN_PATTERN.findall(filename_stem)
    if stem_matches:
        return stem_matches[0]
    return None


def _apply_year_filter_to_blobs(
    blobs: list[storage.Blob],
    *,
    pages_prefix: str,
    log,
) -> list[storage.Blob]:
    tokens = _configured_year_filter_tokens()
    if not tokens:
        return sorted(blobs, key=lambda item: item.name)

    mapping_value = str(getattr(config, "batch_date_mapping_file", "") or "").strip()
    if not mapping_value:
        raise ValueError(
            "config.batch_year_filter is set but config.batch_date_mapping_file is empty."
        )
    mapping_path = Path(mapping_value).expanduser()
    sbid_to_year, years = _load_date_mapping_year_by_sbid(mapping_path)
    target_years = _resolve_year_filter_targets(tokens, available_years=years)

    filtered: list[storage.Blob] = []
    unmatched_sbid = 0
    missing_mapping = 0
    for blob in blobs:
        sbid = _extract_sbid_from_blob_name(blob.name, pages_prefix=pages_prefix)
        if not sbid:
            unmatched_sbid += 1
            continue
        year = sbid_to_year.get(sbid)
        if year is None:
            missing_mapping += 1
            continue
        if year in target_years:
            filtered.append(blob)

    if not filtered:
        raise FileNotFoundError(
            "Year filter removed all input blobs. "
            f"Requested years={sorted(target_years)} from {mapping_path}."
        )

    log(
        "Applied batch_year_filter: "
        f"requested_tokens={tokens} resolved_years={sorted(target_years)} "
        f"selected={len(filtered)}/{len(blobs)} "
        f"(unparsed_sbid={unmatched_sbid}, missing_mapping={missing_mapping})."
    )
    return sorted(filtered, key=lambda item: item.name)


def _list_input_blobs(bucket: storage.Bucket, *, log) -> list[storage.Blob]:
    _assert_gcs_input_source(config.batch_input_source)
    upload_source = _resolved_upload_source()
    override_prefix = _normalize_prefix(config.batch_input_prefix or "")
    pages_prefix = _pages_prefix()
    allowed = _allowed_extensions()

    if override_prefix:
        blobs = _list_image_blobs_for_prefix(bucket, override_prefix, allowed)
        return _apply_year_filter_to_blobs(blobs, pages_prefix=pages_prefix, log=log)

    prefer_pdf_folders = bool(config.batch_use_local_pdf_folders) and upload_source in {
        "pdf",
        "auto",
    }
    if not prefer_pdf_folders:
        all_blobs = _list_image_blobs_for_prefix(bucket, pages_prefix, allowed)
        selected = _apply_fp_mode_to_blobs(
            all_blobs,
            pages_prefix=pages_prefix,
            fp_mode=str(config.fp_mode or "all"),
            fp_suffix=str(config.fp_suffix or "_fp"),
        )
        return _apply_year_filter_to_blobs(
            selected,
            pages_prefix=pages_prefix,
            log=log,
        )

    local_pdf_paths = _list_local_pdf_paths(config.target_folder)
    if not local_pdf_paths:
        all_blobs = _list_image_blobs_for_prefix(bucket, pages_prefix, allowed)
        selected = _apply_fp_mode_to_blobs(
            all_blobs,
            pages_prefix=pages_prefix,
            fp_mode=str(config.fp_mode or "all"),
            fp_suffix=str(config.fp_suffix or "_fp"),
        )
        return _apply_year_filter_to_blobs(
            selected,
            pages_prefix=pages_prefix,
            log=log,
        )

    collected: list[storage.Blob] = []
    missing: list[str] = []
    for pdf_path in local_pdf_paths:
        folder_prefix = f"{pages_prefix}{pdf_path.name}/"
        blobs = _list_image_blobs_for_prefix(bucket, folder_prefix, allowed)
        if blobs:
            collected.extend(blobs)
        else:
            missing.append(folder_prefix)

    if not collected:
        missing_text = ", ".join(missing) if missing else "(none)"
        raise FileNotFoundError(
            f"No uploaded page images found for local PDFs in bucket "
            f"{config.gcs_bucket_name}. Missing prefixes: {missing_text}"
        )

    return _apply_year_filter_to_blobs(
        collected,
        pages_prefix=pages_prefix,
        log=log,
    )


def _assert_gcs_input_source(source: str) -> None:
    if (source or "").strip().lower() != "gcs":
        raise ValueError(
            "Only GCS input source is supported. Set config.batch_input_source = 'gcs'."
        )


def _normalize_prefix(prefix: str) -> str:
    value = prefix.strip()
    if not value:
        return ""
    value = value.strip("/")
    return f"{value}/"


def _is_fp_pdf_path(path: Path, root: Path, fp_suffix: str) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    parent_parts = rel.parts[:-1]
    return any(part.endswith(fp_suffix) for part in parent_parts) or path.stem.endswith(
        fp_suffix
    )


def _apply_fp_mode_to_pdf_paths(
    paths: list[Path],
    *,
    root: Path,
    fp_mode: str,
    fp_suffix: str,
) -> list[Path]:
    mode = fp_mode.lower()
    if mode not in _ALLOWED_FP_MODES:
        raise ValueError(
            f"Unsupported fp_mode: {fp_mode}. "
            f"Expected one of: {sorted(_ALLOWED_FP_MODES)}"
        )
    if mode == "only_fp":
        return [p for p in paths if _is_fp_pdf_path(p, root, fp_suffix)]
    if mode == "exclude_fp":
        return [p for p in paths if not _is_fp_pdf_path(p, root, fp_suffix)]
    return paths


def _ensure_unique_pdf_names(paths: list[Path]) -> None:
    by_name: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in paths:
        existing = by_name.get(path.name)
        if existing is None:
            by_name[path.name] = path
            continue
        duplicates.setdefault(path.name, [existing]).append(path)

    if not duplicates:
        return

    examples = []
    for name, duplicate_paths in sorted(duplicates.items()):
        shown = ", ".join(str(p) for p in duplicate_paths[:3])
        suffix = "..." if len(duplicate_paths) > 3 else ""
        examples.append(f"{name}: {shown}{suffix}")
    raise ValueError(
        "Duplicate PDF file names detected. "
        "Batch submit maps input folders by PDF file name, so names must be unique. "
        f"Conflicts: {'; '.join(examples)}"
    )


def _is_fp_blob_name(blob_name: str, pages_prefix: str, fp_suffix: str) -> bool:
    relative = blob_name
    if pages_prefix and blob_name.startswith(pages_prefix):
        relative = blob_name[len(pages_prefix) :]
    parts = Path(relative).parts
    if len(parts) < 2:
        return False
    folder_parts = parts[:-1]
    return any(
        part.endswith(fp_suffix) or Path(part).stem.endswith(fp_suffix)
        for part in folder_parts
    )


def _apply_fp_mode_to_blobs(
    blobs: list[storage.Blob],
    *,
    pages_prefix: str,
    fp_mode: str,
    fp_suffix: str,
) -> list[storage.Blob]:
    mode = fp_mode.lower()
    if mode not in _ALLOWED_FP_MODES:
        raise ValueError(
            f"Unsupported fp_mode: {fp_mode}. "
            f"Expected one of: {sorted(_ALLOWED_FP_MODES)}"
        )
    if mode == "all":
        return sorted(blobs, key=lambda item: item.name)

    if mode == "only_fp":
        filtered = [
            blob
            for blob in blobs
            if _is_fp_blob_name(blob.name, pages_prefix, fp_suffix)
        ]
    else:
        filtered = [
            blob
            for blob in blobs
            if not _is_fp_blob_name(blob.name, pages_prefix, fp_suffix)
        ]
    return sorted(filtered, key=lambda item: item.name)


def _list_local_pdf_paths(target_folder: str | None) -> list[Path]:
    if not target_folder:
        return []
    folder = Path(target_folder).expanduser()
    if not folder.exists() or not folder.is_dir():
        return []
    recursive = bool(config.recursive)
    fp_mode = str(config.fp_mode or "all")
    fp_suffix = str(config.fp_suffix or "_fp")
    candidates = folder.rglob("*") if recursive else folder.glob("*")
    pdfs = sorted(
        path for path in candidates if path.is_file() and path.suffix.lower() == ".pdf"
    )
    selected = _apply_fp_mode_to_pdf_paths(
        pdfs,
        root=folder,
        fp_mode=fp_mode,
        fp_suffix=fp_suffix,
    )
    _ensure_unique_pdf_names(selected)
    return selected


def _allowed_extensions() -> set[str]:
    allowed = {ext.lower().lstrip(".") for ext in config.batch_input_extensions}
    output_format = (
        str((config.image_settings or {}).get("output_format", "PNG")).strip().lower()
    )
    if output_format in {"jpeg", "jpg"}:
        allowed.add("jpg")
    elif output_format in {"tif", "tiff"}:
        allowed.add("tiff")
    elif output_format:
        allowed.add(output_format)
    return {ext for ext in allowed if ext}


def _ensure_uploaded_sources(bucket: storage.Bucket, log) -> list[str]:
    if not config.batch_auto_upload_missing:
        return []

    upload_source = _resolved_upload_source()
    prefer_pdfs = upload_source in {"pdf", "auto"}
    prefer_images = upload_source in {"images", "auto"}

    if prefer_pdfs and config.batch_use_local_pdf_folders:
        local_pdf_paths = _list_local_pdf_paths(config.target_folder)
        if local_pdf_paths:
            uploaded = upload_missing_pdfs(pdf_paths=local_pdf_paths, bucket=bucket)
            if uploaded:
                log(
                    f"Uploaded missing page images for PDF folders: {', '.join(uploaded)}"
                )
            return uploaded

    if not prefer_images:
        return []

    try:
        uploaded_images = upload_missing_images(bucket=bucket)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        log(f"Skipped local image auto-upload: {exc}")
        return []

    if uploaded_images:
        log(f"Uploaded {len(uploaded_images)} missing local image file(s).")
    return uploaded_images


def _list_image_blobs_for_prefix(
    bucket: storage.Bucket,
    prefix: str,
    allowed_extensions: set[str],
) -> list[storage.Blob]:
    blobs = list(bucket.list_blobs(prefix=prefix or None))
    filtered: list[storage.Blob] = []
    for blob in blobs:
        name = blob.name
        if name.endswith("/"):
            continue
        suffix = Path(name).suffix.lower().lstrip(".")
        if suffix in allowed_extensions:
            filtered.append(blob)
    return filtered


def _guess_mime_type(blob: storage.Blob) -> str:
    if blob.content_type:
        return blob.content_type
    guess, _ = mimetypes.guess_type(blob.name)
    return guess or "application/octet-stream"


def _schema_has_refs(schema: object) -> bool:
    if isinstance(schema, dict):
        if "$ref" in schema or "$defs" in schema:
            return True
        return any(_schema_has_refs(value) for value in schema.values())
    if isinstance(schema, list):
        return any(_schema_has_refs(item) for item in schema)
    return False


def _resolve_json_pointer(root: object, pointer: str) -> object | None:
    if not isinstance(pointer, str) or not pointer.startswith("#/"):
        return None
    current = root
    for token in pointer[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or token not in current:
            return None
        current = current[token]
    return current


def _inline_json_refs(
    node: object, root: object, stack: set[str] | None = None
) -> object:
    if stack is None:
        stack = set()

    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            if ref in stack:
                return {}
            target = _resolve_json_pointer(root, ref)
            if target is None:
                return {}
            resolved = _inline_json_refs(target, root, stack | {ref})
            extras = {
                key: _inline_json_refs(value, root, stack)
                for key, value in node.items()
                if key != "$ref"
            }
            if extras and isinstance(resolved, dict):
                merged = dict(resolved)
                merged.update(extras)
                return merged
            if extras:
                return extras
            return resolved

        output: dict[str, object] = {}
        for key, value in node.items():
            if key == "$defs":
                continue
            if key.startswith("$"):
                continue
            output[key] = _inline_json_refs(value, root, stack)
        return output

    if isinstance(node, list):
        return [_inline_json_refs(item, root, stack) for item in node]

    return node


def _vertex_compatible_schema(raw_schema: object) -> object:
    return _inline_json_refs(raw_schema, raw_schema)


def _build_request_config(*, for_vertex: bool) -> dict:
    schema_payload: object = config.output_schema
    if for_vertex:
        schema_payload = _vertex_compatible_schema(config.output_schema)

    return build_batch_generation_config(
        for_vertex=for_vertex,
        include_schema=bool(config.batch_include_response_schema),
        include_temperature=True,
        include_thinking_level=True,
        schema_payload=schema_payload,
    )


def _build_request_line(
    blob: storage.Blob,
    bucket_name: str,
    *,
    for_vertex: bool,
) -> dict:
    mime_type = _guess_mime_type(blob)
    media_part = {
        "fileData": {
            "fileUri": f"gs://{bucket_name}/{blob.name}",
            "mimeType": mime_type,
        }
    }

    parts = [media_part]
    prompt = prompt_text()
    if prompt:
        parts.append({"text": prompt})

    return {
        "key": blob.name,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
            "generationConfig": _build_request_config(for_vertex=for_vertex),
        },
    }


def _build_anthropic_manifest_line(blob: storage.Blob) -> dict[str, str]:
    key = blob.name
    return {
        "key": key,
        "mime_type": _guess_mime_type(blob),
        "custom_id": _anthropic_custom_id_for_key(key),
    }


def _iter_anthropic_manifest_entries(path: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in anthropic manifest {path} at line {line_number}."
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Invalid anthropic manifest row at line {line_number}: expected object."
                )
            key = payload.get("key")
            if not isinstance(key, str) or not key.strip():
                raise ValueError(
                    f"Missing 'key' in anthropic manifest at line {line_number}."
                )
            key = key.strip()
            mime_type = payload.get("mime_type")
            if not isinstance(mime_type, str) or not mime_type.strip():
                mime_type = "application/octet-stream"
            custom_id_raw = payload.get("custom_id")
            if isinstance(custom_id_raw, str) and custom_id_raw.strip():
                custom_id = custom_id_raw.strip()
            else:
                custom_id = _anthropic_custom_id_for_key(key)
            if not _ANTHROPIC_CUSTOM_ID_PATTERN.fullmatch(custom_id):
                raise ValueError(
                    f"Invalid custom_id in anthropic manifest at line {line_number}: "
                    f"{custom_id!r}"
                )
            entries.append(
                {
                    "key": key,
                    "mime_type": mime_type.strip(),
                    "custom_id": custom_id,
                }
            )
    return entries


def _anthropic_signed_url_expiration() -> timedelta:
    ttl_hours = max(1, int(config.anthropic_signed_url_ttl_hours or 48))
    return timedelta(hours=ttl_hours)


def _anthropic_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise TypeError("Expected output schema to be a dictionary.")

    normalized = copy.deepcopy(schema)

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            node_type = node.get("type")
            if node_type == "object" or (
                isinstance(node_type, list) and "object" in node_type
            ):
                node["additionalProperties"] = False
            for value in node.values():
                _walk(value)
            return
        if isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(normalized)
    return normalized


def _build_anthropic_batch_requests(
    *,
    bucket: storage.Bucket,
    requests_path: Path,
) -> list[dict[str, Any]]:
    manifest = _iter_anthropic_manifest_entries(requests_path)
    if not manifest:
        raise ValueError(f"Anthropic manifest is empty: {requests_path}")

    prompt = prompt_text()
    include_schema = bool(config.batch_include_response_schema)
    max_tokens = max(1, int(config.model_max_output_tokens))
    expiration = _anthropic_signed_url_expiration()
    schema_payload = (
        _anthropic_strict_json_schema(config.output_schema) if include_schema else None
    )

    requests: list[dict[str, Any]] = []
    seen_custom_ids: set[str] = set()
    for row in manifest:
        key = row["key"]
        custom_id = row["custom_id"]
        if custom_id in seen_custom_ids:
            raise ValueError(
                "Duplicate custom_id generated for Anthropic batch requests: "
                f"{custom_id}"
            )
        seen_custom_ids.add(custom_id)
        signed_url = bucket.blob(key).generate_signed_url(
            version="v4",
            method="GET",
            expiration=expiration,
        )
        content: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": signed_url,
                },
            }
        ]
        if prompt:
            content.append({"type": "text", "text": prompt})

        params: dict[str, Any] = {
            "model": config.model,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }
        if include_schema:
            params["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": schema_payload,
                }
            }
        requests.append(
            {
                "custom_id": custom_id,
                "params": params,
            }
        )
    return requests


def _upload_requests_to_gcs(
    bucket: storage.Bucket,
    run_dir_name: str,
    local_requests_path: Path,
) -> str:
    prefix = _normalize_prefix(config.batch_requests_gcs_prefix or "")
    object_name = f"{prefix}{run_dir_name}/{local_requests_path.name}"
    blob = bucket.blob(object_name)
    blob.upload_from_filename(
        str(local_requests_path),
        content_type="application/jsonl",
    )
    return f"gs://{bucket.name}/{object_name}"


def _output_dest_gcs_uri(
    bucket_name: str,
    run_dir_name: str,
    *,
    chunk_label: str | None = None,
) -> str:
    prefix = _normalize_prefix(config.batch_outputs_gcs_prefix or "")
    base = f"{prefix}{run_dir_name}".rstrip("/")
    if chunk_label:
        base = f"{base}/{chunk_label.strip('/')}"
    return f"gs://{bucket_name}/{base}"


def _write_requests_file(
    blobs: list[storage.Blob],
    bucket_name: str,
    output_path: Path,
    log,
    *,
    for_vertex: bool,
    provider: str,
) -> tuple[int, int]:
    max_bytes = int(config.batch_input_max_bytes or 0)
    total_bytes = 0
    count = 0

    with open(output_path, "w", encoding="utf-8") as handle:
        for blob in tqdm(blobs, desc="Building batch JSONL", unit="img"):
            if provider == "anthropic":
                line_obj = _build_anthropic_manifest_line(blob)
            else:
                line_obj = _build_request_line(
                    blob,
                    bucket_name,
                    for_vertex=for_vertex,
                )
            line = json.dumps(line_obj, ensure_ascii=False)
            handle.write(line)
            handle.write("\n")
            count += 1

            total_bytes += len(line.encode("utf-8")) + 1
            if max_bytes and total_bytes > max_bytes:
                raise ValueError(
                    "Batch request file exceeded batch_input_max_bytes. "
                    "Reduce inputs or split into multiple batch jobs."
                )

    log(
        f"Wrote {count} requests to {output_path.name} "
        f"({total_bytes} bytes, source=gcs)."
    )
    if provider == "anthropic":
        log(
            "Anthropic request file stores key/mime manifest only; "
            "signed image URLs are generated at submit time."
        )
    if provider == "gemini" and for_vertex and not config.batch_include_response_schema:
        log(
            "Vertex batch input omits response schema (config.batch_include_response_schema=False)."
        )
    if (
        provider == "gemini"
        and for_vertex
        and config.batch_include_response_schema
        and _schema_has_refs(config.output_schema)
    ):
        log(
            "Vertex batch input inlines schema refs and strips $-keys for compatibility."
        )
    return count, total_bytes


def _count_requests_file(path: Path) -> tuple[int, int]:
    total_bytes = path.stat().st_size if path.exists() else 0
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count, total_bytes


def _read_batch_job_payload(path: Path) -> dict:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"batch job metadata file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid batch metadata in {path}: expected object")
    return payload


def _read_batch_job_payload_if_exists(path: Path) -> dict | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _parse_chunk_file_name(file_name: str, *, base_name: str) -> tuple[int, int] | None:
    base = Path(base_name)
    suffix = base.suffix or ".jsonl"
    stem = base.stem
    pattern = re.compile(
        rf"^{re.escape(stem)}\.part(?P<index>\d+)-of-(?P<total>\d+){re.escape(suffix)}$"
    )
    match = pattern.fullmatch(file_name)
    if not match:
        return None
    try:
        chunk_index = int(match.group("index"))
        total_chunks = int(match.group("total"))
    except Exception:
        return None
    if chunk_index <= 0 or total_chunks <= 0:
        return None
    return chunk_index, total_chunks


def _discover_request_files_in_run_dir(run_dir: Path) -> tuple[dict[int, str], int]:
    by_index: dict[int, str] = {}
    inferred_total = 0
    base_name = config.batch_requests_file_name

    candidate = run_dir / base_name
    if candidate.exists() and candidate.is_file():
        by_index[1] = base_name
        inferred_total = max(inferred_total, 1)

    for path in sorted(run_dir.glob("*.jsonl")):
        parsed = _parse_chunk_file_name(path.name, base_name=base_name)
        if not parsed:
            continue
        chunk_index, total_chunks = parsed
        by_index[chunk_index] = path.name
        inferred_total = max(inferred_total, total_chunks)

    if by_index:
        inferred_total = max(inferred_total, max(by_index))
    return by_index, inferred_total


def _submitted_batch_ids_by_chunk_from_run_log(
    run_dir: Path,
) -> tuple[dict[int, str], int]:
    path = run_dir / "run.log"
    if not path.exists() or not path.is_file():
        return {}, 0

    by_index: dict[int, str] = {}
    inferred_total = 0
    pattern = re.compile(
        r"\[(chunk_(?P<index>\d+)_of_(?P<total>\d+))\].*batch_id=(?P<batch_id>\S+)"
    )
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line)
            if not match:
                continue
            try:
                chunk_index = int(match.group("index"))
                total_chunks = int(match.group("total"))
            except Exception:
                continue
            batch_id = str(match.group("batch_id") or "").strip()
            if chunk_index <= 0 or total_chunks <= 0 or not batch_id:
                continue
            by_index[chunk_index] = batch_id
            inferred_total = max(inferred_total, total_chunks)
    return by_index, inferred_total


def _latest_submit_run_dir(output_root: str) -> Path | None:
    root = Path(output_root).expanduser()
    if not root.exists() or not root.is_dir():
        return None
    run_dirs = sorted(
        (
            item
            for item in root.iterdir()
            if item.is_dir() and item.name.startswith("submit_")
        ),
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


def _resolve_rerun_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser()
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(
                f"--run-dir not found or not a directory: {run_dir}"
            )
        return run_dir

    latest = _latest_submit_run_dir(config.output_root)
    if latest is None:
        raise FileNotFoundError(
            "No previous submit run found to rerun. "
            "Run batch_submit.py first (without --rerun)."
        )
    return latest


def _normalize_job_entries(payload: dict) -> list[dict]:
    entries: list[dict] = []
    raw_jobs = payload.get("batch_jobs")
    if isinstance(raw_jobs, list):
        for index, item in enumerate(raw_jobs, start=1):
            if not isinstance(item, dict):
                continue
            entry = dict(item)
            if "chunk_index" not in entry:
                entry["chunk_index"] = index
            entries.append(entry)

    if entries:
        return entries

    single_name = payload.get("batch_job_name")
    if isinstance(single_name, str) and single_name.strip():
        requests_file = payload.get("requests_file")
        if not isinstance(requests_file, str) or not requests_file.strip():
            requests_file = config.batch_requests_file_name
        return [
            {
                "chunk_index": 1,
                "total_chunks": 1,
                "chunk_label": "chunk_001_of_001",
                "requests_file": requests_file,
                "batch_job_name": single_name.strip(),
                "request_count": int(payload.get("request_count") or 0),
                "request_bytes": int(payload.get("request_bytes") or 0),
                "input_file": payload.get("input_file"),
                "input_source": payload.get("input_source"),
                "output_destination": payload.get("output_destination"),
                "provider": payload.get("provider"),
            }
        ]
    return []


def _infer_rerun_total_chunks(
    *,
    payload: dict | None,
    payload_entries: list[dict],
    request_files_total: int,
    run_log_total: int,
) -> int:
    totals: list[int] = []
    if payload is not None:
        try:
            value = int(payload.get("num_batches_requested") or 0)
        except Exception:
            value = 0
        if value > 0:
            totals.append(value)

    if payload_entries:
        for entry in payload_entries:
            try:
                value = int(entry.get("total_chunks") or 0)
            except Exception:
                value = 0
            if value > 0:
                totals.append(value)

    if request_files_total > 0:
        totals.append(int(request_files_total))
    if run_log_total > 0:
        totals.append(int(run_log_total))

    return max(totals) if totals else 0


def _extract_downscale_from_run_log(run_dir: Path) -> float | None:
    path = run_dir / "run.log"
    if not path.exists() or not path.is_file():
        return None
    pattern = re.compile(r"Downscaled input set with fraction=(?P<value>[0-9eE+\-.]+)")
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line)
            if not match:
                continue
            try:
                value = float(match.group("value"))
            except Exception:
                continue
            if value > 0.0 and value <= 1.0:
                return value
    return None


def _ensure_requests_files_for_rerun(
    *,
    run_dir: Path,
    total_chunks: int,
    existing_files_by_index: dict[int, str],
    bucket: storage.Bucket,
    provider: str,
    client,
    log,
) -> dict[int, str]:
    if total_chunks <= 0:
        return dict(existing_files_by_index)

    expected_missing = [
        index
        for index in range(1, total_chunks + 1)
        if index not in existing_files_by_index
    ]
    if not expected_missing:
        return dict(existing_files_by_index)

    downscale = _extract_downscale_from_run_log(run_dir)
    if downscale is not None and downscale < 1.0:
        raise ValueError(
            "Cannot safely regenerate missing chunk request files for a downscaled run. "
            "Please rerun submit with the same --downscale value and stable input set, "
            "or manually recover from existing request files."
        )

    blobs = _list_input_blobs(bucket, log=log)
    if not blobs:
        raise FileNotFoundError(
            f"No input images found in bucket {config.gcs_bucket_name} "
            f"with prefix '{config.batch_input_prefix}'."
        )

    chunks = _split_blobs_evenly(blobs, total_chunks)
    if len(chunks) != total_chunks:
        raise RuntimeError(
            "Unable to regenerate missing request files for rerun because chunk "
            "reconstruction did not match expected total chunks."
        )

    files_by_index = dict(existing_files_by_index)
    for chunk_index in expected_missing:
        requests_file_name = (
            config.batch_requests_file_name
            if total_chunks == 1
            else _chunk_requests_file_name(
                config.batch_requests_file_name,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
            )
        )
        requests_path = run_dir / requests_file_name
        _write_requests_file(
            blobs=chunks[chunk_index - 1],
            bucket_name=config.gcs_bucket_name,
            output_path=requests_path,
            log=log,
            for_vertex=bool(getattr(client, "vertexai", False)),
            provider=provider,
        )
        files_by_index[chunk_index] = requests_file_name
        log(
            f"Regenerated missing request file for rerun: "
            f"chunk={chunk_index}/{total_chunks} file={requests_file_name}"
        )

    return files_by_index


def _build_rerun_entries(
    *,
    run_dir: Path,
    provider: str,
    payload_entries: list[dict],
    total_chunks: int,
    files_by_index: dict[int, str],
    submitted_by_chunk: dict[int, str],
) -> list[dict]:
    if total_chunks <= 0:
        return []

    by_chunk_payload: dict[int, dict] = {}
    for fallback_index, entry in enumerate(payload_entries, start=1):
        try:
            index = int(entry.get("chunk_index") or fallback_index)
        except Exception:
            index = fallback_index
        if index <= 0:
            continue
        by_chunk_payload[index] = dict(entry)

    entries: list[dict] = []
    for chunk_index in range(1, total_chunks + 1):
        entry = by_chunk_payload.get(chunk_index, {})

        requests_file = files_by_index.get(chunk_index)
        if not requests_file:
            requests_file = (
                config.batch_requests_file_name
                if total_chunks == 1
                else _chunk_requests_file_name(
                    config.batch_requests_file_name,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                )
            )

        requests_path = run_dir / requests_file
        if not requests_path.exists() or not requests_path.is_file():
            raise FileNotFoundError(
                f"Missing request file while preparing rerun metadata: {requests_path}"
            )

        request_count, request_bytes = _count_requests_file(requests_path)
        previous_name = str(entry.get("batch_job_name") or "").strip()
        if not previous_name:
            previous_name = submitted_by_chunk.get(chunk_index, "")

        input_source = str(entry.get("input_source") or "").strip()
        if not input_source:
            input_source = "anthropic_manifest" if provider == "anthropic" else "gcs"

        input_file = str(entry.get("input_file") or "").strip()
        if not input_file:
            input_file = requests_file if provider == "anthropic" else ""

        output_destination = entry.get("output_destination")
        if isinstance(output_destination, str):
            output_destination = output_destination.strip() or None

        rebuilt = _build_chunk_entry(
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            requests_file=requests_file,
            request_count=request_count,
            request_bytes=request_bytes,
            batch_job_name=previous_name,
            input_file=input_file,
            input_source=input_source,
            output_destination=output_destination,
            provider=provider,
        )
        history = entry.get("rerun_history")
        if isinstance(history, list):
            rebuilt["rerun_history"] = history

        entries.append(rebuilt)

    return entries


def _entries_with_replacement(
    entries: list[dict],
    *,
    chunk_index: int,
    replacement: dict,
) -> list[dict]:
    updated: list[dict] = []
    replaced = False
    for item in entries:
        try:
            current_index = int(item.get("chunk_index") or 0)
        except Exception:
            current_index = 0
        if current_index == chunk_index:
            updated.append(dict(replacement))
            replaced = True
        else:
            updated.append(dict(item))

    if not replaced:
        updated.append(dict(replacement))

    return sorted(updated, key=lambda item: int(item.get("chunk_index") or 0))


def _build_chunk_entry(
    *,
    chunk_index: int,
    total_chunks: int,
    requests_file: str,
    request_count: int,
    request_bytes: int,
    batch_job_name: str,
    input_file: str,
    input_source: str,
    output_destination: str | None,
    provider: str,
) -> dict:
    return {
        "chunk_index": int(chunk_index),
        "total_chunks": int(total_chunks),
        "chunk_label": _chunk_label(chunk_index=chunk_index, total_chunks=total_chunks),
        "requests_file": requests_file,
        "request_count": int(request_count),
        "request_bytes": int(request_bytes),
        "batch_job_name": batch_job_name,
        "input_file": input_file,
        "input_source": input_source,
        "output_destination": output_destination,
        "provider": provider,
    }


def _write_batch_job_meta(
    *,
    run_dir: Path,
    jobs: list[dict],
    num_batches_requested: int,
    client_backend: str,
    vertex_location: str | None,
    provider: str,
) -> None:
    if not jobs:
        raise ValueError("Cannot write batch metadata without jobs.")

    total_request_count = sum(int(item.get("request_count") or 0) for item in jobs)
    total_request_bytes = sum(int(item.get("request_bytes") or 0) for item in jobs)

    first = jobs[0]
    meta = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "batch_job_name": first.get("batch_job_name"),
        "batch_job_names": [
            item.get("batch_job_name") for item in jobs if item.get("batch_job_name")
        ],
        "batch_jobs": jobs,
        "request_count": total_request_count,
        "request_bytes": total_request_bytes,
        "input_file": first.get("input_file"),
        "input_source": first.get("input_source"),
        "output_destination": first.get("output_destination"),
        "model": config.model,
        "provider": provider,
        "client_backend": client_backend,
        "vertex_location": vertex_location,
        "num_batches_requested": int(num_batches_requested),
        "num_batches_submitted": len(jobs),
    }
    (run_dir / "batch_job.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _submit_chunk_job(
    *,
    client,
    provider: str,
    bucket: storage.Bucket,
    run_dir_name: str,
    requests_path: Path,
    chunk_index: int,
    total_chunks: int,
    attempt_tag: str | None,
    log,
) -> tuple[str, str, str, str | None]:
    input_source = "gcs"
    input_ref = ""
    dest_ref = None
    label = _chunk_label(chunk_index=chunk_index, total_chunks=total_chunks)
    destination_label = label
    if attempt_tag:
        destination_label = f"{label}/{attempt_tag.strip('/')}"

    display_name = (
        f"{config.batch_job_display_name}-{label}"
        if total_chunks > 1
        else config.batch_job_display_name
    )

    if provider == "anthropic":
        requests = _build_anthropic_batch_requests(
            bucket=bucket,
            requests_path=requests_path,
        )
        batch_job = client.messages.batches.create(requests=requests)
        input_source = "anthropic_manifest"
        input_ref = requests_path.name
        log(
            f"[{label}] Submitted Anthropic batch with {len(requests)} request(s). "
            f"signed_url_ttl_hours={max(1, int(config.anthropic_signed_url_ttl_hours or 48))} "
            f"batch_id={batch_job.id}"
        )
        return batch_job.id, input_source, input_ref, None

    if getattr(client, "vertexai", False):
        input_ref = _upload_requests_to_gcs(
            bucket=bucket,
            run_dir_name=run_dir_name,
            local_requests_path=requests_path,
        )
        dest_ref = _output_dest_gcs_uri(
            bucket_name=config.gcs_bucket_name,
            run_dir_name=run_dir_name,
            chunk_label=destination_label,
        )
        batch_job = client.batches.create(
            model=config.model,
            src=input_ref,
            config=types.CreateBatchJobConfig(
                display_name=display_name,
                dest=dest_ref,
            ),
        )
        log(
            f"[{label}] Uploaded request file to {input_ref} "
            f"with output destination {dest_ref}."
        )
        return batch_job.name, input_source, input_ref, dest_ref

    uploaded_file = client.files.upload(
        file=str(requests_path),
        config=types.UploadFileConfig(
            display_name=f"{display_name}-requests",
            mime_type="jsonl",
        ),
    )
    input_ref = uploaded_file.name
    input_source = "gemini_files"
    batch_job = client.batches.create(
        model=config.model,
        src=uploaded_file.name,
        config=types.CreateBatchJobConfig(display_name=display_name),
    )
    return batch_job.name, input_source, input_ref, None


def _batch_state_and_success(
    *,
    client,
    provider: str,
    batch_job_name: str,
) -> tuple[str, bool]:
    if provider == "anthropic":
        batch_job = client.messages.batches.retrieve(batch_job_name)
        status = (
            str(getattr(batch_job, "processing_status", "") or "unknown")
            .strip()
            .lower()
        )
        counts = getattr(batch_job, "request_counts", None)
        succeeded = int(getattr(counts, "succeeded", 0) or 0)
        errored = int(getattr(counts, "errored", 0) or 0)
        canceled = int(getattr(counts, "canceled", 0) or 0)
        expired = int(getattr(counts, "expired", 0) or 0)
        processing = int(getattr(counts, "processing", 0) or 0)
        state_text = (
            f"{status}(succeeded={succeeded},errored={errored},"
            f"canceled={canceled},expired={expired},processing={processing})"
        )
        successful = (
            status == "ended"
            and errored == 0
            and canceled == 0
            and expired == 0
            and processing == 0
        )
        return state_text, successful

    batch_job = client.batches.get(name=batch_job_name)
    state = str(getattr(batch_job, "state", None) or "UNKNOWN")
    return state, state == "JOB_STATE_SUCCEEDED"


def submit_batch() -> None:
    args = _parse_args()
    provider = _validate_batch_model_support()
    if not (config.service_account_file or "").strip():
        raise ValueError(
            "config.service_account_file is empty. "
            "Set it to your GCP service account JSON path."
        )
    if not (config.gcs_bucket_name or "").strip():
        raise ValueError(
            "config.gcs_bucket_name is empty. "
            "Set the GCS bucket used for batch request/input/output files."
        )

    service_account_path = resolve_service_account_path(config.service_account_file)
    storage_client = storage.Client.from_service_account_json(str(service_account_path))
    bucket = storage_client.bucket(config.gcs_bucket_name)

    vertex_location = (config.vertex_model_location or "").strip() or (
        config.gcp_location or ""
    ).strip()
    if provider == "anthropic":
        client = _get_anthropic_client()
        backend_name = "anthropic"
    else:
        client = get_batch_client(location=vertex_location)
        backend_name = "vertex" if getattr(client, "vertexai", False) else "mldev"

    if args.rerun:
        if args.num_batches is not None:
            print("--num-batches is ignored when --rerun is set.")
        if args.downscale is not None:
            print("--downscale is ignored when --rerun is set.")
        run_dir = _resolve_rerun_run_dir(args)
        log = get_run_logger(run_dir)
        _warn_if_confidence_scores_unsupported(provider=provider, log=log)
        payload = _read_batch_job_payload_if_exists(run_dir / "batch_job.json")
        payload_provider = str((payload or {}).get("provider") or "").strip().lower()
        if payload_provider in {"gemini", "anthropic"} and payload_provider != provider:
            provider = payload_provider
            if provider == "anthropic":
                client = _get_anthropic_client()
                backend_name = "anthropic"
            else:
                client = get_batch_client(location=vertex_location)
                backend_name = (
                    "vertex" if getattr(client, "vertexai", False) else "mldev"
                )
            log(
                f"Rerun provider override from metadata: provider={provider} "
                f"(model in config is '{config.model}')."
            )
            _warn_if_confidence_scores_unsupported(provider=provider, log=log)

        payload_entries = _normalize_job_entries(payload or {})
        files_by_index, files_total = _discover_request_files_in_run_dir(run_dir)
        submitted_by_chunk, run_log_total = _submitted_batch_ids_by_chunk_from_run_log(
            run_dir
        )
        total_chunks = _infer_rerun_total_chunks(
            payload=payload,
            payload_entries=payload_entries,
            request_files_total=files_total,
            run_log_total=run_log_total,
        )
        if total_chunks <= 0:
            raise ValueError(
                "Could not infer rerun chunk layout. Expected batch_job.json, "
                "chunked request files, or chunk submission lines in run.log."
            )

        files_by_index = _ensure_requests_files_for_rerun(
            run_dir=run_dir,
            total_chunks=total_chunks,
            existing_files_by_index=files_by_index,
            bucket=bucket,
            provider=provider,
            client=client,
            log=log,
        )
        entries = _build_rerun_entries(
            run_dir=run_dir,
            provider=provider,
            payload_entries=payload_entries,
            total_chunks=total_chunks,
            files_by_index=files_by_index,
            submitted_by_chunk=submitted_by_chunk,
        )
        if not entries:
            raise ValueError(f"No batch job entries found or derived in {run_dir}.")

        log(
            f"Rerun preparation complete for {run_dir.name}: "
            f"total_chunks={total_chunks}, discovered_request_files={len(files_by_index)}, "
            f"known_submitted_chunks={len(submitted_by_chunk)}."
        )
        print(
            f"Rerun prepared: {total_chunks} chunk(s), "
            f"{len(submitted_by_chunk)} with known batch IDs."
        )

        _write_batch_job_meta(
            run_dir=run_dir,
            jobs=entries,
            num_batches_requested=int(
                (payload or {}).get("num_batches_requested") or total_chunks
            ),
            client_backend=backend_name,
            vertex_location=vertex_location
            if getattr(client, "vertexai", False)
            else None,
            provider=provider,
        )

        rerun_attempt_tag = datetime.now().strftime("rerun_%Y%m%d_%H%M%S")
        current_entries = sorted(
            [dict(item) for item in entries],
            key=lambda item: int(item.get("chunk_index") or 0),
        )
        rerun_count = 0
        for fallback_index, entry in enumerate(current_entries, start=1):
            try:
                chunk_index = int(entry.get("chunk_index") or fallback_index)
            except Exception:
                chunk_index = fallback_index
            try:
                total_chunks = int(entry.get("total_chunks") or len(entries))
            except Exception:
                total_chunks = len(entries)
            label = _chunk_label(chunk_index=chunk_index, total_chunks=total_chunks)
            print(f"Checking {label} ({fallback_index}/{len(current_entries)})...")

            requests_file = entry.get("requests_file")
            if not isinstance(requests_file, str) or not requests_file.strip():
                if len(entries) == 1:
                    requests_file = config.batch_requests_file_name
                else:
                    requests_file = _chunk_requests_file_name(
                        config.batch_requests_file_name,
                        chunk_index=chunk_index,
                        total_chunks=total_chunks,
                    )
            requests_path = run_dir / requests_file
            if not requests_path.exists() or not requests_path.is_file():
                raise FileNotFoundError(
                    f"Missing request file for {label}: {requests_path}"
                )

            previous_name = str(entry.get("batch_job_name") or "").strip()
            state_text = "UNKNOWN"
            was_successful = False
            if previous_name:
                try:
                    state_text, was_successful = _batch_state_and_success(
                        client=client,
                        provider=provider,
                        batch_job_name=previous_name,
                    )
                except Exception as exc:
                    log(
                        f"[{label}] Could not resolve previous job {previous_name}; "
                        f"will resubmit chunk.",
                        exc=exc,
                    )

            if was_successful:
                log(
                    f"[{label}] Already successful. Keeping existing batch id {previous_name}."
                )
                _write_batch_job_meta(
                    run_dir=run_dir,
                    jobs=current_entries,
                    num_batches_requested=int(
                        (payload or {}).get("num_batches_requested") or total_chunks
                    ),
                    client_backend=backend_name,
                    vertex_location=vertex_location
                    if getattr(client, "vertexai", False)
                    else None,
                    provider=provider,
                )
                continue
            if previous_name:
                log(
                    f"[{label}] Previous job not successful: {previous_name} state={state_text}"
                )
            else:
                log(f"[{label}] No previous batch id recorded; submitting chunk.")

            print(f"Submitting {label}...")

            request_count, request_bytes = _count_requests_file(requests_path)
            batch_job_name, input_source, input_ref, dest_ref = _submit_chunk_job(
                client=client,
                provider=provider,
                bucket=bucket,
                run_dir_name=run_dir.name,
                requests_path=requests_path,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                attempt_tag=rerun_attempt_tag,
                log=log,
            )
            rerun_count += 1
            rebuilt = _build_chunk_entry(
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                requests_file=requests_file,
                request_count=request_count,
                request_bytes=request_bytes,
                batch_job_name=batch_job_name,
                input_file=input_ref,
                input_source=input_source,
                output_destination=dest_ref,
                provider=provider,
            )
            if previous_name and previous_name != batch_job_name:
                history = entry.get("rerun_history")
                if not isinstance(history, list):
                    history = []
                rebuilt["rerun_history"] = [*history, previous_name]
            current_entries = _entries_with_replacement(
                current_entries,
                chunk_index=chunk_index,
                replacement=rebuilt,
            )

            _write_batch_job_meta(
                run_dir=run_dir,
                jobs=current_entries,
                num_batches_requested=int(
                    (payload or {}).get("num_batches_requested") or total_chunks
                ),
                client_backend=backend_name,
                vertex_location=vertex_location
                if getattr(client, "vertexai", False)
                else None,
                provider=provider,
            )

        _write_batch_job_meta(
            run_dir=run_dir,
            jobs=current_entries,
            num_batches_requested=int(
                (payload or {}).get("num_batches_requested") or total_chunks
            ),
            client_backend=backend_name,
            vertex_location=vertex_location
            if getattr(client, "vertexai", False)
            else None,
            provider=provider,
        )
        if rerun_count == 0:
            log("Rerun requested, but all chunk jobs were already successful.")
            print("No chunks needed rerun; all jobs already succeeded.")
            return

        job_names = ", ".join(
            str(item.get("batch_job_name"))
            for item in current_entries
            if item.get("batch_job_name")
        )
        log(f"Rerun submitted {rerun_count} chunk job(s). Active jobs: {job_names}")
        print(f"Resubmitted {rerun_count} chunk job(s).")
        return

    run_dir = create_subfolder(config.output_root, prefix="submit_")
    log = get_run_logger(run_dir)
    _warn_if_confidence_scores_unsupported(provider=provider, log=log)
    _ensure_uploaded_sources(bucket, log)

    blobs = _list_input_blobs(bucket, log=log)
    if not blobs:
        raise FileNotFoundError(
            f"No input images found in bucket {config.gcs_bucket_name} "
            f"with prefix '{config.batch_input_prefix}'."
        )
    downscale = _resolve_downscale(args)
    if downscale is not None:
        original_count = len(blobs)
        blobs = _downscale_blobs_randomly(blobs, downscale=downscale)
        sampled_count = len(blobs)
        log(
            f"Downscaled input set with fraction={downscale:.6g}. "
            f"Selected {sampled_count}/{original_count} input blob(s)."
        )
        print(
            f"Downscale applied ({downscale:.6g}): "
            f"{sampled_count}/{original_count} blob(s) selected."
        )

    num_batches_requested = _resolve_num_batches(args)
    chunks = _split_blobs_evenly(blobs, num_batches_requested)
    if not chunks:
        raise RuntimeError("No request chunks could be prepared for submission.")
    if len(chunks) < num_batches_requested:
        log(
            f"Requested {num_batches_requested} chunks, "
            f"but only {len(chunks)} non-empty chunks are possible for {len(blobs)} inputs."
        )

    chunk_jobs: list[dict] = []
    total_chunks = len(chunks)
    for chunk_index, chunk_blobs in enumerate(chunks, start=1):
        if total_chunks == 1:
            requests_file_name = config.batch_requests_file_name
        else:
            requests_file_name = _chunk_requests_file_name(
                config.batch_requests_file_name,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
            )
        requests_path = run_dir / requests_file_name
        request_count, request_bytes = _write_requests_file(
            blobs=chunk_blobs,
            bucket_name=config.gcs_bucket_name,
            output_path=requests_path,
            log=log,
            for_vertex=bool(getattr(client, "vertexai", False)),
            provider=provider,
        )
        batch_job_name, input_source, input_ref, dest_ref = _submit_chunk_job(
            client=client,
            provider=provider,
            bucket=bucket,
            run_dir_name=run_dir.name,
            requests_path=requests_path,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            attempt_tag=None,
            log=log,
        )
        chunk_jobs.append(
            _build_chunk_entry(
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                requests_file=requests_file_name,
                request_count=request_count,
                request_bytes=request_bytes,
                batch_job_name=batch_job_name,
                input_file=input_ref,
                input_source=input_source,
                output_destination=dest_ref,
                provider=provider,
            )
        )

        # Persist progress immediately so batch IDs are not lost on mid-run failure.
        _write_batch_job_meta(
            run_dir=run_dir,
            jobs=chunk_jobs,
            num_batches_requested=num_batches_requested,
            client_backend=backend_name,
            vertex_location=vertex_location
            if getattr(client, "vertexai", False)
            else None,
            provider=provider,
        )

    _write_batch_job_meta(
        run_dir=run_dir,
        jobs=chunk_jobs,
        num_batches_requested=num_batches_requested,
        client_backend=backend_name,
        vertex_location=vertex_location if getattr(client, "vertexai", False) else None,
        provider=provider,
    )

    job_names = ", ".join(
        str(item.get("batch_job_name"))
        for item in chunk_jobs
        if item.get("batch_job_name")
    )
    log(f"Submitted {len(chunk_jobs)} batch job(s): {job_names}")
    print(
        f"Batch jobs submitted ({len(chunk_jobs)}): "
        + ", ".join(
            str(item.get("batch_job_name"))
            for item in chunk_jobs
            if item.get("batch_job_name")
        )
    )


if __name__ == "__main__":
    submit_batch()
