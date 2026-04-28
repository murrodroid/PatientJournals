from __future__ import annotations

import csv
import re
from pathlib import Path

from google.cloud import storage

from patientjournals.batch.upload import upload_missing_images, upload_missing_pdfs
from patientjournals.config import config


_ALLOWED_FP_MODES = {"all", "only_fp", "exclude_fp"}
_ALLOWED_UPLOAD_SOURCES = {"pdf", "images", "auto"}
_SBID_TOKEN_PATTERN = re.compile(r"(\d{5,})")


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

