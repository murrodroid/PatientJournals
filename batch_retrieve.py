from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import time
from pathlib import Path
from typing import Any

from google import genai
from google.cloud import storage
from google.genai import types
from tqdm import tqdm

from batch_client import get_batch_client, resolve_service_account_path
from config import config
from generation_spec import build_live_generation_config
from models import resolve_model_spec
from output_handler import data_to_rows
from response_parsing import extract_response_metadata
from tools import create_subfolder, flush_rows, get_run_logger


_GEMINI_TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
}
_ANTHROPIC_TERMINAL_STATES = {"ended", "succeeded", "errored", "canceled", "expired"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve batch outputs and build a dataset."
    )
    parser.add_argument(
        "--batch-name",
        dest="batch_name",
        help="Batch job name (overrides config.batch_job_name).",
    )
    parser.add_argument(
        "--run-dir",
        dest="run_dir",
        help="Run directory containing batch_job.json (overrides auto-discovery).",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Poll until the batch job completes.",
    )
    parser.add_argument(
        "--allow-partial",
        "--ack-incomplete",
        dest="allow_partial",
        action="store_true",
        help=(
            "Retrieve only completed (succeeded) batch jobs and skip chunks "
            "that are still running/failed/cancelled."
        ),
    )
    return parser.parse_args()


def _normalize_key(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def _normalize_job_state(value: object) -> str:
    if value is None:
        return "UNKNOWN"
    if isinstance(value, str):
        text = value.strip()
    else:
        name = getattr(value, "name", None)
        if isinstance(name, str) and name.strip():
            text = name.strip()
        else:
            text = str(value).strip()
    if not text:
        return "UNKNOWN"
    if "." in text:
        text = text.split(".")[-1]
    return text


def _terminal_states(provider: str) -> set[str]:
    if provider == "anthropic":
        return _ANTHROPIC_TERMINAL_STATES
    return _GEMINI_TERMINAL_STATES


def _get_anthropic_client():
    try:
        import anthropic
    except Exception as exc:
        raise RuntimeError(
            "Anthropic provider requested but the 'anthropic' package is unavailable."
        ) from exc
    api_key = config.api_key_for_provider("anthropic")
    return anthropic.Anthropic(api_key=api_key)


def _provider_from_batch_names(
    batch_names: list[str],
    *,
    submit_run_dir: Path | None,
) -> str:
    if submit_run_dir is not None:
        payload = _read_batch_job_payload(submit_run_dir / "batch_job.json")
        if payload:
            provider = payload.get("provider")
            if isinstance(provider, str) and provider.strip():
                value = provider.strip().lower()
                if value in {"gemini", "anthropic"}:
                    return value

    if batch_names and all(name.startswith("msgbatch_") for name in batch_names):
        return "anthropic"
    if batch_names and all(
        name.startswith("projects/")
        or "/locations/" in name
        or name.startswith("batches/")
        for name in batch_names
    ):
        return "gemini"

    model_provider = resolve_model_spec(config.model).provider
    if model_provider in {"gemini", "anthropic"}:
        return model_provider
    raise ValueError(
        f"Unsupported provider for batch retrieval: {model_provider}. "
        "Supported providers: gemini, anthropic."
    )


def _get_batch_job(client, batch_name: str, provider: str):
    if provider == "anthropic":
        return client.messages.batches.retrieve(batch_name)
    return client.batches.get(name=batch_name)


def _batch_job_state(batch_job: object, provider: str) -> str:
    if provider == "anthropic":
        return _normalize_job_state(getattr(batch_job, "processing_status", None))
    return _normalize_job_state(getattr(batch_job, "state", None))


def _batch_job_successful(batch_job: object, provider: str) -> bool:
    if provider == "anthropic":
        status = _batch_job_state(batch_job, provider)
        # Anthropic exposes per-request success/error counts for ended batches,
        # so retrieval should proceed once processing has ended.
        return status == "ended"

    return _batch_job_state(batch_job, provider) == "JOB_STATE_SUCCEEDED"


def _warn_if_confidence_scores_unsupported(*, provider: str, log) -> None:
    if not bool(config.include_confidence_scores):
        return
    if provider == "anthropic":
        log(
            "include_confidence_scores=True requested, but Anthropic Messages API "
            "does not expose token logprobs in responses. "
            "Field-level confidence output will be empty."
        )


def _anthropic_custom_id_for_key(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:40]
    return f"gcs_{digest}"


def _record_failure(
    failures: dict[str, str],
    *,
    key: str | None,
    line_number: int,
    reason: str,
) -> None:
    failure_key = key or f"<line:{line_number}>"
    if failure_key not in failures:
        failures[failure_key] = reason


def _sample_keys(values: set[str], limit: int) -> list[str]:
    return sorted(values)[: max(1, limit)]


def _is_parse_failure_reason(reason: str | None) -> bool:
    if not reason:
        return False
    parse_related_prefixes = (
        "schema_validation_failed",
        "empty_response_text",
        "missing_response",
        "invalid_jsonl_line",
        "invalid_record_type",
    )
    return reason.startswith(parse_related_prefixes)


def _expected_success_keys(
    *,
    expected_keys: set[str],
    observed_output_keys: set[str],
) -> set[str]:
    return expected_keys if expected_keys else observed_output_keys


def _build_api_key_generation_config() -> dict:
    return build_live_generation_config(
        include_schema=bool(config.batch_include_response_schema),
        include_temperature=True,
        include_thinking_level=True,
    )


def _guess_blob_mime_type(blob: storage.Blob, key: str) -> str:
    if blob.content_type:
        return blob.content_type
    guess, _ = mimetypes.guess_type(key)
    return guess or "application/octet-stream"


def _resolve_recovery_api_key() -> str:
    configured = (config.api_key or "").strip()
    if configured:
        return configured

    try:
        from api_keys import gemini_maarten as fallback_api_key
    except Exception:
        fallback_api_key = ""

    value = str(fallback_api_key).strip() if fallback_api_key else ""
    if value:
        return value

    raise ValueError(
        "API key recovery requires config.api_key or api_keys.gemini_maarten."
    )


def _recover_missing_pages_via_api_key(
    *,
    missing_keys: set[str],
    successful_keys: set[str],
    observed_output_keys: set[str],
    failures: dict[str, str],
    rows_to_flush: list[dict],
    log,
    force: bool = False,
) -> int:
    if not missing_keys:
        return 0

    if not config.api_recovery_enabled and not force:
        return 0

    if not force:
        max_missing = max(0, int(config.api_recovery_max_missing_pages or 0))
        if max_missing <= 0:
            raise RuntimeError(
                "API key recovery is enabled but api_recovery_max_missing_pages is <= 0."
            )

        if len(missing_keys) > max_missing:
            raise RuntimeError(
                "API key recovery aborted: "
                f"{len(missing_keys)} page(s) missing successful output, "
                f"which exceeds api_recovery_max_missing_pages={max_missing}."
            )

    bucket_name = _require_bucket_name()
    service_account_file = _require_service_account_file()
    service_account_path = resolve_service_account_path(service_account_file)
    storage_client = storage.Client.from_service_account_json(str(service_account_path))
    bucket = storage_client.bucket(bucket_name)

    recovery_model = (config.api_recovery_model or "").strip() or config.model
    recovery_client = genai.Client(api_key=_resolve_recovery_api_key())
    generation_config = _build_api_key_generation_config()

    recovered = 0
    base_line_number = 1_000_000

    for offset, key in enumerate(
        tqdm(sorted(missing_keys), desc="Recovering pages via API key", unit="page"),
        start=1,
    ):
        blob = bucket.blob(key)
        if not blob.exists():
            _record_failure(
                failures,
                key=key,
                line_number=base_line_number + offset,
                reason="recovery_blob_not_found",
            )
            log(f"Recovery skipped for missing GCS object: {key}")
            continue

        try:
            image_bytes = blob.download_as_bytes()
            mime_type = _guess_blob_mime_type(blob, key)
            response = recovery_client.models.generate_content(
                model=recovery_model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    config.input_prompt,
                ],
                config=generation_config,
            )

            metadata = extract_response_metadata(response)
            text_payload = metadata.get("text")
            if not isinstance(text_payload, str) or not text_payload.strip():
                raise ValueError("Empty response text from API key recovery.")

            parsed_model = config.output_model.model_validate_json(text_payload)
        except Exception as exc:
            _record_failure(
                failures,
                key=key,
                line_number=base_line_number + offset,
                reason=f"api_key_recovery_failed:{type(exc).__name__}",
            )
            log(f"API key recovery failed for key={key}", exc=exc)
            continue

        recovered += 1
        successful_keys.add(key)
        observed_output_keys.add(key)
        failures.pop(key, None)

        rows = data_to_rows(
            parsed_model,
            file_name=key,
            field_confidence_by_pointer=metadata.get("field_confidence_by_pointer"),
        )
        for row in rows:
            row["thoughts"] = metadata.get("thoughts") or None
        rows_to_flush.extend(rows)

    if recovered:
        log(
            f"Recovered {recovered}/{len(missing_keys)} missing page(s) "
            f"via API key using model {recovery_model}."
        )
    return recovered


def _read_batch_job_payload(path: Path) -> dict | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _read_request_keys_from_file(path: Path) -> set[str]:
    keys: set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = _normalize_key(payload.get("key"))
            if key:
                keys.add(key)
    return keys


def _extract_batch_names_from_payload(payload: dict) -> list[str]:
    names: list[str] = []
    jobs = payload.get("batch_jobs")
    if isinstance(jobs, list):
        for item in jobs:
            if not isinstance(item, dict):
                continue
            value = _normalize_key(item.get("batch_job_name"))
            if value:
                names.append(value)

    direct_names = payload.get("batch_job_names")
    if isinstance(direct_names, list):
        for value in direct_names:
            normalized = _normalize_key(value)
            if normalized:
                names.append(normalized)

    single = _normalize_key(payload.get("batch_job_name"))
    if single:
        names.append(single)

    ordered: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def _request_files_from_payload(
    payload: dict,
    *,
    selected_batch_names: list[str] | None = None,
) -> list[str]:
    selected = {
        name.strip()
        for name in (selected_batch_names or [])
        if isinstance(name, str) and name.strip()
    }
    files: list[str] = []
    jobs = payload.get("batch_jobs")
    if isinstance(jobs, list):
        for item in jobs:
            if not isinstance(item, dict):
                continue
            if selected:
                job_name = _normalize_key(item.get("batch_job_name"))
                if not job_name or job_name not in selected:
                    continue
            value = item.get("requests_file")
            if isinstance(value, str) and value.strip():
                files.append(value.strip())

    if files:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in files:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered

    if selected:
        payload_names = set(_extract_batch_names_from_payload(payload))
        if payload_names and not payload_names.intersection(selected):
            return []

    direct = payload.get("requests_file")
    if isinstance(direct, str) and direct.strip():
        return [direct.strip()]

    return [config.batch_requests_file_name]


def _find_submit_run_dir(batch_names: list[str]) -> Path | None:
    root = Path(config.output_root).expanduser()
    if not root.exists() or not root.is_dir():
        return None

    target = {name for name in batch_names if isinstance(name, str) and name.strip()}
    run_dirs = sorted(
        (
            item
            for item in root.iterdir()
            if item.is_dir() and item.name.startswith("submit_")
        ),
        reverse=True,
    )
    for run_dir in run_dirs:
        job_payload = _read_batch_job_payload(run_dir / "batch_job.json")
        if not job_payload:
            continue
        payload_names = set(_extract_batch_names_from_payload(job_payload))
        if target and not payload_names.intersection(target):
            continue
        request_files = _request_files_from_payload(job_payload)
        if any(
            (run_dir / req).exists() and (run_dir / req).is_file()
            for req in request_files
        ):
            return run_dir
    return None


def _resolve_expected_request_keys(
    *,
    submit_run_dir: Path | None,
    batch_names: list[str],
    selected_batch_names: list[str] | None = None,
    log,
) -> set[str]:
    if submit_run_dir is None:
        submit_run_dir = _find_submit_run_dir(batch_names)

    if submit_run_dir is None:
        log(
            "Could not locate submit run with batch requests file; "
            "skipping expected-page coverage validation."
        )
        return set()

    payload = _read_batch_job_payload(submit_run_dir / "batch_job.json")
    if not payload:
        log(
            f"Submit run {submit_run_dir} has no readable batch_job.json; "
            "skipping expected-page coverage validation."
        )
        return set()

    request_files = _request_files_from_payload(
        payload,
        selected_batch_names=selected_batch_names,
    )
    if not request_files and selected_batch_names:
        log(
            "No request files matched selected batch names; "
            "falling back to all request files in submit metadata."
        )
        request_files = _request_files_from_payload(payload)
    expected_keys: set[str] = set()
    loaded_files: list[Path] = []
    for request_file in request_files:
        path = submit_run_dir / request_file
        if not path.exists() or not path.is_file():
            continue
        expected_keys.update(_read_request_keys_from_file(path))
        loaded_files.append(path)

    if not loaded_files:
        log(
            f"No request files found in submit run {submit_run_dir}; "
            "skipping expected-page coverage validation."
        )
        return set()

    log(
        f"Loaded {len(expected_keys)} expected page key(s) from "
        f"{len(loaded_files)} request file(s) in {submit_run_dir}."
    )
    return expected_keys


def _resolve_anthropic_custom_id_to_key(
    *,
    submit_run_dir: Path | None,
    batch_names: list[str],
    selected_batch_names: list[str] | None = None,
    log,
) -> dict[str, str]:
    if submit_run_dir is None:
        submit_run_dir = _find_submit_run_dir(batch_names)

    if submit_run_dir is None:
        log(
            "Could not locate submit run with batch requests file; "
            "Anthropic custom_id mapping unavailable."
        )
        return {}

    payload = _read_batch_job_payload(submit_run_dir / "batch_job.json")
    if not payload:
        log(
            f"Submit run {submit_run_dir} has no readable batch_job.json; "
            "Anthropic custom_id mapping unavailable."
        )
        return {}

    request_files = _request_files_from_payload(
        payload,
        selected_batch_names=selected_batch_names,
    )
    if not request_files and selected_batch_names:
        request_files = _request_files_from_payload(payload)

    mapping: dict[str, str] = {}
    loaded_files = 0
    for request_file in request_files:
        path = submit_run_dir / request_file
        if not path.exists() or not path.is_file():
            continue
        loaded_files += 1
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                key = _normalize_key(item.get("key"))
                if not key:
                    continue
                custom_id = _normalize_key(item.get("custom_id")) or _anthropic_custom_id_for_key(key)
                existing = mapping.get(custom_id)
                if existing and existing != key:
                    raise RuntimeError(
                        "Conflicting Anthropic custom_id mapping in request files: "
                        f"{custom_id} -> {existing} vs {key}"
                    )
                mapping[custom_id] = key

    if loaded_files == 0:
        log(
            f"No request files found in submit run {submit_run_dir}; "
            "Anthropic custom_id mapping unavailable."
        )
        return {}

    log(
        f"Loaded {len(mapping)} anthropic custom_id mapping(s) from "
        f"{loaded_files} request file(s) in {submit_run_dir}."
    )
    return mapping


def _validate_page_completeness(
    *,
    expected_keys: set[str],
    observed_output_keys: set[str],
    successful_keys: set[str],
    failures: dict[str, str],
    log,
) -> None:
    sample_size = max(1, int(config.page_validation_sample_size or 1))

    if expected_keys and config.require_all_expected_pages:
        missing_from_output = expected_keys - observed_output_keys
        if missing_from_output:
            samples = _sample_keys(missing_from_output, sample_size)
            for sample in samples:
                log(f"Missing output key: {sample}")
            raise RuntimeError(
                "Batch output coverage failed: "
                f"{len(missing_from_output)} expected page(s) missing from output. "
                "Disable with config.require_all_expected_pages=False if needed."
            )

    if not config.require_all_pages_successful:
        return

    expected_success = expected_keys if expected_keys else observed_output_keys
    missing_success = expected_success - successful_keys
    if not missing_success:
        return

    samples = _sample_keys(missing_success, sample_size)
    for sample in samples:
        reason = failures.get(sample, "unknown")
        log(f"Unsuccessful page key: {sample} (reason={reason})")

    raise RuntimeError(
        "Batch success coverage failed: "
        f"{len(missing_success)} page(s) missing successful parsed output. "
        "Disable with config.require_all_pages_successful=False if needed."
    )


def _print_validation_summary(
    *,
    expected_keys: set[str],
    observed_output_keys: set[str],
    successful_keys: set[str],
    log,
) -> None:
    total_requests = len(expected_keys) if expected_keys else len(observed_output_keys)
    successful_requests = len(successful_keys)
    failed_requests = max(0, total_requests - successful_requests)
    ratio = (
        100.0 if total_requests == 0 else (successful_requests / total_requests) * 100.0
    )

    summary = (
        "Validation summary: "
        f"{successful_requests}/{total_requests} requests valid "
        f"({ratio:.2f}%), failed={failed_requests}."
    )
    print(summary)
    log(summary)


def _parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    cleaned = gcs_uri.strip()
    if not cleaned.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    remainder = cleaned[5:]
    if "/" not in remainder:
        return remainder, ""
    bucket_name, object_path = remainder.split("/", 1)
    return bucket_name, object_path


def _normalize_prefix(prefix: str) -> str:
    value = prefix.strip()
    if not value:
        return ""
    return f"{value.strip('/')}/"


def _require_bucket_name() -> str:
    bucket_name = (config.gcs_bucket_name or "").strip()
    if not bucket_name:
        raise ValueError(
            "config.gcs_bucket_name is empty. Set the GCS bucket used for batch files."
        )
    return bucket_name


def _require_service_account_file() -> str:
    service_account_file = (config.service_account_file or "").strip()
    if not service_account_file:
        raise ValueError(
            "config.service_account_file is empty. "
            "Set it to your GCP service account JSON path."
        )
    return service_account_file


def _dataset_content_type(dataset_path: Path) -> str:
    suffix = dataset_path.suffix.lower().lstrip(".")
    if suffix == "jsonl":
        return "application/jsonl"
    if suffix == "csv":
        return "text/csv"
    return "application/octet-stream"


def _upload_dataset_to_gcs(
    dataset_path: Path,
    run_dir_name: str,
    log,
) -> str | None:
    if not config.upload_dataset_to_gcs:
        return None
    if not dataset_path.exists():
        log(f"Skipped upload for missing file: {dataset_path.name}")
        return None

    bucket_name = _require_bucket_name()
    service_account_file = _require_service_account_file()
    service_account_path = resolve_service_account_path(service_account_file)
    storage_client = storage.Client.from_service_account_json(str(service_account_path))
    bucket = storage_client.bucket(bucket_name)

    prefix = _normalize_prefix(config.datasets_gcs_prefix or "")
    object_name = f"{prefix}{run_dir_name}/{dataset_path.name}"
    blob = bucket.blob(object_name)
    blob.upload_from_filename(
        str(dataset_path),
        content_type=_dataset_content_type(dataset_path),
    )
    uri = f"gs://{bucket_name}/{object_name}"
    log(f"Uploaded dataset to {uri}.")
    return uri


def _download_from_vertex_gcs_output(
    dest_gcs_uri: str,
    output_path: Path,
    log,
) -> Path:
    bucket_name, object_prefix = _parse_gcs_uri(dest_gcs_uri)
    prefix = object_prefix.rstrip("/")
    if prefix:
        prefix = f"{prefix}/"

    service_account_file = _require_service_account_file()
    service_account_path = resolve_service_account_path(service_account_file)
    storage_client = storage.Client.from_service_account_json(str(service_account_path))
    bucket = storage_client.bucket(bucket_name)
    blobs = sorted(bucket.list_blobs(prefix=prefix), key=lambda item: item.name)

    jsonl_blobs = [
        blob
        for blob in blobs
        if blob.name.endswith(".jsonl") and not blob.name.endswith("/")
    ]
    if not jsonl_blobs:
        raise RuntimeError(f"No JSONL outputs found at {dest_gcs_uri}.")

    with open(output_path, "wb") as handle:
        for blob in jsonl_blobs:
            data = blob.download_as_bytes()
            handle.write(data)
            if data and not data.endswith(b"\n"):
                handle.write(b"\n")

    log(
        f"Downloaded {len(jsonl_blobs)} output file(s) from "
        f"{dest_gcs_uri} into {output_path.name}."
    )
    return output_path


def _download_from_mldev_output(
    client: genai.Client,
    file_name: str,
    output_path: Path,
) -> Path:
    downloaded = None
    try:
        downloaded = client.files.download(
            file=file_name,
            destination=str(output_path),
        )
    except TypeError:
        downloaded = client.files.download(file=file_name)

    if downloaded is not None:
        if hasattr(downloaded, "read"):
            output_path.write_bytes(downloaded.read())
        else:
            output_path.write_bytes(downloaded)
    return output_path


def _sdk_obj_to_dict(value: object) -> object:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            pass
    return value


def _download_from_anthropic_output(
    client,
    batch_name: str,
    output_path: Path,
    log,
) -> Path:
    count = 0
    with open(output_path, "w", encoding="utf-8") as handle:
        for item in client.messages.batches.results(batch_name):
            payload = _sdk_obj_to_dict(item)
            handle.write(json.dumps(payload, ensure_ascii=False, default=str))
            handle.write("\n")
            count += 1
    if count == 0:
        raise RuntimeError(f"No result rows found for Anthropic batch {batch_name}.")
    log(f"Downloaded {count} Anthropic result row(s) into {output_path.name}.")
    return output_path


def _extract_anthropic_response_metadata(response: object) -> dict[str, object]:
    message = response
    if hasattr(message, "model_dump"):
        message = message.model_dump(mode="json")
    elif hasattr(message, "to_dict"):
        message = message.to_dict()

    content = None
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    if not isinstance(content, list):
        return {"text": None, "thoughts": None, "field_confidence_by_pointer": {}}

    text_chunks: list[str] = []
    thought_chunks: list[str] = []
    for block in content:
        if isinstance(block, dict):
            block_type = str(block.get("type") or "").strip().lower()
            if block_type == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    text_chunks.append(text)
            elif block_type == "thinking":
                thinking = block.get("thinking")
                if isinstance(thinking, str) and thinking.strip():
                    thought_chunks.append(thinking)
        else:
            block_type = str(getattr(block, "type", "") or "").strip().lower()
            if block_type == "text":
                text = getattr(block, "text", None)
                if isinstance(text, str) and text.strip():
                    text_chunks.append(text)
            elif block_type == "thinking":
                thinking = getattr(block, "thinking", None)
                if isinstance(thinking, str) and thinking.strip():
                    thought_chunks.append(thinking)

    text = "".join(text_chunks).strip() if text_chunks else None
    thoughts = "\n\n".join(thought_chunks).strip() if thought_chunks else None
    return {
        "text": text,
        "thoughts": thoughts,
        "field_confidence_by_pointer": {},
    }


def _await_completion(
    client,
    batch_name: str,
    provider: str,
    log,
) -> object:
    poll_interval = max(1, int(config.batch_poll_interval_seconds))
    terminal_states = _terminal_states(provider)
    while True:
        batch_job = _get_batch_job(client, batch_name, provider)
        state = _batch_job_state(batch_job, provider)
        if state in terminal_states:
            return batch_job
        log(f"Batch {batch_name} state={state}. Waiting {poll_interval}s.")
        time.sleep(poll_interval)


def _read_batch_names_from_job_file(path: Path) -> list[str]:
    payload = _read_batch_job_payload(path)
    if not payload:
        return []
    return _extract_batch_names_from_payload(payload)


def _latest_batch_job_file(output_root: str) -> Path | None:
    root = Path(output_root).expanduser()
    if not root.exists() or not root.is_dir():
        return None
    run_dirs = sorted((item for item in root.iterdir() if item.is_dir()), reverse=True)
    for run_dir in run_dirs:
        candidate = run_dir / "batch_job.json"
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _resolve_batch_targets(args: argparse.Namespace) -> tuple[list[str], Path | None]:
    if args.batch_name:
        submit_run_dir = Path(args.run_dir).expanduser() if args.run_dir else None
        return [args.batch_name], submit_run_dir

    if args.run_dir:
        candidate = Path(args.run_dir).expanduser() / "batch_job.json"
        batch_names = _read_batch_names_from_job_file(candidate)
        if batch_names:
            return batch_names, candidate.parent
        raise ValueError(f"No batch job names found in {candidate}.")

    latest_job_file = _latest_batch_job_file(config.output_root)
    if latest_job_file is not None:
        batch_names = _read_batch_names_from_job_file(latest_job_file)
        if batch_names:
            return batch_names, latest_job_file.parent

    if config.batch_job_name:
        return [config.batch_job_name], None

    raise ValueError(
        "Batch job name not found. Use --batch-name, --run-dir, or set config.batch_job_name."
    )


def _extract_location_from_batch_name(batch_name: str) -> str | None:
    parts = [part for part in batch_name.split("/") if part]
    for index, part in enumerate(parts):
        if part == "locations" and index + 1 < len(parts):
            return parts[index + 1]
    return None


def _flush_rows(
    *,
    rows_to_flush: list[dict],
    out_path: Path,
    output_dataset_format: str,
    header_written: bool,
) -> tuple[bool, int]:
    if not rows_to_flush:
        return header_written, 0

    header_written = flush_rows(
        rows=rows_to_flush,
        out_path=str(out_path),
        header_written=header_written,
        output_format=output_dataset_format,
        sep=config.csv_sep,
    )
    count = len(rows_to_flush)
    rows_to_flush.clear()
    return header_written, count


def retrieve_batch() -> Path:
    args = _parse_args()
    batch_names, submit_run_dir = _resolve_batch_targets(args)
    if not batch_names:
        raise ValueError("No batch jobs resolved for retrieval.")

    run_dir = create_subfolder(config.output_root, prefix="retrieve_")
    log = get_run_logger(run_dir)

    provider = _provider_from_batch_names(
        batch_names,
        submit_run_dir=submit_run_dir,
    )
    _warn_if_confidence_scores_unsupported(provider=provider, log=log)
    if provider == "anthropic":
        client = _get_anthropic_client()
    else:
        client_location = (
            _extract_location_from_batch_name(batch_names[0])
            or (config.vertex_model_location or "").strip()
            or (config.gcp_location or "").strip()
            or None
        )
        client = get_batch_client(location=client_location)

    batch_jobs: list[tuple[str, object]] = []
    incomplete_batches: list[tuple[str, str]] = []
    for batch_name in batch_names:
        batch_job = _get_batch_job(client, batch_name, provider)
        state = _batch_job_state(batch_job, provider)
        if not _batch_job_successful(batch_job, provider):
            if args.wait:
                batch_job = _await_completion(
                    client=client,
                    batch_name=batch_name,
                    provider=provider,
                    log=log,
                )
                state = _batch_job_state(batch_job, provider)

        if not _batch_job_successful(batch_job, provider):
            incomplete_batches.append((batch_name, state))
            continue
        batch_jobs.append((batch_name, batch_job))

    if incomplete_batches and not args.allow_partial:
        examples = ", ".join(
            f"{name} ({state})" for name, state in incomplete_batches[:5]
        )
        suffix = "..." if len(incomplete_batches) > 5 else ""
        raise RuntimeError(
            "Not all batch jobs are complete/succeeded. "
            f"Incomplete jobs={len(incomplete_batches)} [{examples}{suffix}]. "
            "Re-run with --wait, or use --allow-partial to retrieve only finished chunks."
        )

    if not batch_jobs:
        raise RuntimeError(
            "No succeeded batch jobs available to retrieve. "
            "Use --wait to block until completion."
        )

    if incomplete_batches:
        skipped = ", ".join(
            f"{name} ({state})" for name, state in incomplete_batches[:5]
        )
        suffix = "..." if len(incomplete_batches) > 5 else ""
        log(
            f"Partial retrieval enabled: skipping {len(incomplete_batches)} "
            f"non-succeeded job(s): {skipped}{suffix}"
        )
        print(
            f"Partial retrieval: using {len(batch_jobs)}/{len(batch_names)} "
            "succeeded batch job(s)."
        )

    raw_outputs: list[tuple[str, Path]] = []
    for index, (batch_name, batch_job) in enumerate(batch_jobs, start=1):
        raw_path = run_dir / f"batch_output_{index:03d}.jsonl"

        if provider == "anthropic":
            raw_path = _download_from_anthropic_output(
                client=client,
                batch_name=batch_name,
                output_path=raw_path,
                log=log,
            )
        else:
            dest = getattr(batch_job, "dest", None)
            file_name = getattr(dest, "file_name", None) if dest else None
            dest_gcs_uri = getattr(dest, "gcs_uri", None) if dest else None

            if dest_gcs_uri:
                raw_path = _download_from_vertex_gcs_output(
                    dest_gcs_uri=dest_gcs_uri,
                    output_path=raw_path,
                    log=log,
                )
            elif file_name:
                raw_path = _download_from_mldev_output(
                    client=client,
                    file_name=file_name,
                    output_path=raw_path,
                )
            else:
                raise RuntimeError(f"Batch {batch_name} missing output destination.")
        raw_outputs.append((batch_name, raw_path))

    anthropic_custom_id_to_key: dict[str, str] = {}
    if provider == "anthropic":
        anthropic_custom_id_to_key = _resolve_anthropic_custom_id_to_key(
            submit_run_dir=submit_run_dir,
            batch_names=batch_names,
            selected_batch_names=[name for name, _ in batch_jobs],
            log=log,
        )

    out_name = config.dataset_file_name
    output_dataset_format = config.output_format
    out_path = (
        run_dir / f"{run_dir.name}_{out_name}.{output_dataset_format.lstrip('.')}"
    )

    flush_every = max(1, int(config.flush_every or config.batch_size))
    rows_to_flush: list[dict] = []
    header_written = False
    total_rows = 0
    error_rows = 0

    output_keys_seen: set[str] = set()
    successful_page_keys: set[str] = set()
    failures: dict[str, str] = {}

    global_line_number = 0
    for output_index, (batch_name, raw_path) in enumerate(raw_outputs, start=1):
        with open(raw_path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(
                tqdm(handle, desc=f"Parsing {batch_name}", unit="line"),
                start=1,
            ):
                global_line_number += 1
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    log(f"Invalid JSONL line in batch output ({batch_name}).", exc=exc)
                    error_rows += 1
                    _record_failure(
                        failures,
                        key=None,
                        line_number=global_line_number,
                        reason="invalid_jsonl_line",
                    )
                    continue

                if provider == "anthropic":
                    custom_id = (
                        _normalize_key(record.get("custom_id"))
                        if isinstance(record, dict)
                        else None
                    )
                    key = (
                        anthropic_custom_id_to_key.get(custom_id, custom_id)
                        if custom_id
                        else None
                    )
                else:
                    key = (
                        _normalize_key(record.get("key"))
                        if isinstance(record, dict)
                        else None
                    )
                if key:
                    output_keys_seen.add(key)

                if not isinstance(record, dict):
                    error_rows += 1
                    _record_failure(
                        failures,
                        key=key,
                        line_number=global_line_number,
                        reason="invalid_record_type",
                    )
                    continue

                if provider == "anthropic":
                    result = record.get("result")
                    if not isinstance(result, dict):
                        error_rows += 1
                        log(f"Missing/invalid result payload for key={key}")
                        _record_failure(
                            failures,
                            key=key,
                            line_number=global_line_number,
                            reason="missing_result",
                        )
                        continue
                    result_type = str(result.get("type") or "").strip().lower()
                    if result_type != "succeeded":
                        error_rows += 1
                        log(
                            f"Anthropic batch non-success for key={key}: type={result_type}"
                        )
                        _record_failure(
                            failures,
                            key=key,
                            line_number=global_line_number,
                            reason=f"batch_{result_type or 'unknown'}",
                        )
                        continue
                    response = result.get("message")
                    if response is None:
                        error_rows += 1
                        log(f"Missing message in Anthropic result for key={key}")
                        _record_failure(
                            failures,
                            key=key,
                            line_number=global_line_number,
                            reason="missing_response",
                        )
                        continue
                    metadata = _extract_anthropic_response_metadata(response)
                else:
                    if record.get("error"):
                        error_rows += 1
                        log(f"Batch error for key={key}: {record.get('error')}")
                        _record_failure(
                            failures,
                            key=key,
                            line_number=global_line_number,
                            reason="batch_error",
                        )
                        continue

                    response = record.get("response")
                    if response is None:
                        error_rows += 1
                        log(f"Missing response for key={key}")
                        _record_failure(
                            failures,
                            key=key,
                            line_number=global_line_number,
                            reason="missing_response",
                        )
                        continue

                    metadata = extract_response_metadata(response)

                text_payload = metadata.get("text")
                if not text_payload:
                    error_rows += 1
                    log(f"Empty response text for key={key}")
                    _record_failure(
                        failures,
                        key=key,
                        line_number=global_line_number,
                        reason="empty_response_text",
                    )
                    continue

                try:
                    parsed_model = config.output_model.model_validate_json(text_payload)
                except Exception as exc:
                    error_rows += 1
                    log(f"Schema validation failed for key={key}", exc=exc)
                    _record_failure(
                        failures,
                        key=key,
                        line_number=global_line_number,
                        reason="schema_validation_failed",
                    )
                    continue

                file_key = key or f"<batch:{output_index}-line:{line_number}>"
                if key:
                    successful_page_keys.add(key)

                rows = data_to_rows(
                    parsed_model,
                    file_name=file_key,
                    field_confidence_by_pointer=metadata.get(
                        "field_confidence_by_pointer"
                    ),
                )
                for row in rows:
                    row["thoughts"] = metadata.get("thoughts") or None
                rows_to_flush.extend(rows)

                if len(rows_to_flush) >= flush_every:
                    header_written, wrote = _flush_rows(
                        rows_to_flush=rows_to_flush,
                        out_path=out_path,
                        output_dataset_format=output_dataset_format,
                        header_written=header_written,
                    )
                    total_rows += wrote

    expected_keys = _resolve_expected_request_keys(
        submit_run_dir=submit_run_dir,
        batch_names=batch_names,
        selected_batch_names=[name for name, _ in batch_jobs],
        log=log,
    )
    expected_success = _expected_success_keys(
        expected_keys=expected_keys,
        observed_output_keys=output_keys_seen,
    )
    missing_success_keys = expected_success - successful_page_keys
    parse_failure_missing_keys = {
        key
        for key in missing_success_keys
        if _is_parse_failure_reason(failures.get(key))
    }

    forced_parse_recovery_used = False
    if missing_success_keys and provider == "gemini":
        recovery_keys: set[str] = set()
        recovery_force = False

        if config.api_recovery_enabled:
            recovery_keys = missing_success_keys
        elif parse_failure_missing_keys:
            recovery_keys = parse_failure_missing_keys
            recovery_force = True
            forced_parse_recovery_used = True
            log(
                "Detected parse-related batch failures; attempting API-key recovery "
                "for affected page(s) even though api_recovery_enabled=False."
            )

        if recovery_keys:
            recovered_count = _recover_missing_pages_via_api_key(
                missing_keys=recovery_keys,
                successful_keys=successful_page_keys,
                observed_output_keys=output_keys_seen,
                failures=failures,
                rows_to_flush=rows_to_flush,
                log=log,
                force=recovery_force,
            )
            if recovered_count:
                remaining = (
                    _expected_success_keys(
                        expected_keys=expected_keys,
                        observed_output_keys=output_keys_seen,
                    )
                    - successful_page_keys
                )
                log(f"API key recovery remaining unsuccessful pages: {len(remaining)}.")
    elif missing_success_keys and config.api_recovery_enabled:
        log(
            f"Skipping api_recovery_enabled for provider '{provider}': "
            "API key recovery is currently implemented for Gemini only."
        )

    try:
        _validate_page_completeness(
            expected_keys=expected_keys,
            observed_output_keys=output_keys_seen,
            successful_keys=successful_page_keys,
            failures=failures,
            log=log,
        )
    except RuntimeError as exc:
        if forced_parse_recovery_used:
            log(
                "Coverage validation failed after forced parse recovery. "
                "Continuing to write partial dataset.",
                exc=exc,
            )
        else:
            raise

    _print_validation_summary(
        expected_keys=expected_keys,
        observed_output_keys=output_keys_seen,
        successful_keys=successful_page_keys,
        log=log,
    )

    if rows_to_flush:
        header_written, wrote = _flush_rows(
            rows_to_flush=rows_to_flush,
            out_path=out_path,
            output_dataset_format=output_dataset_format,
            header_written=header_written,
        )
        total_rows += wrote

    log(
        f"Retrieved {total_rows} processed row(s) from {len(batch_jobs)} completed "
        f"batch job(s) into {out_path.name} (errors={error_rows})."
    )
    _upload_dataset_to_gcs(out_path, run_dir.name, log)
    return out_path


if __name__ == "__main__":
    retrieve_batch()
