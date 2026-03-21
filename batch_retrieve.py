from __future__ import annotations

import argparse
import json
import mimetypes
import time
from pathlib import Path

from google import genai
from google.cloud import storage
from google.genai import types
from tqdm import tqdm

from batch_client import get_batch_client, resolve_service_account_path
from config import config
from output_handler import data_to_rows
from tools import create_subfolder, flush_rows, get_run_logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve Gemini batch outputs and build a dataset."
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
    return parser.parse_args()


def _normalize_key(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def _extract_text(response: object) -> str | None:
    if not isinstance(response, dict):
        return None

    candidates = response.get("candidates") or []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content") or {}
        if not isinstance(content, dict):
            continue
        parts = content.get("parts") or []
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                return text

    text = response.get("text")
    if isinstance(text, str) and text.strip():
        return text

    return None


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


def _expected_success_keys(
    *,
    expected_keys: set[str],
    observed_output_keys: set[str],
) -> set[str]:
    return expected_keys if expected_keys else observed_output_keys


def _build_api_key_generation_config() -> dict:
    generation_config: dict[str, object] = {
        "response_mime_type": config.response_mime_type
    }
    if not config.batch_include_response_schema:
        return generation_config
    schema_field = config.response_schema_field
    if schema_field:
        generation_config[schema_field] = config.output_schema
    return generation_config


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
    raw_rows_to_flush: list[dict],
    log,
) -> int:
    if not missing_keys:
        return 0

    if not config.api_recovery_enabled:
        return 0

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

            text_payload = getattr(response, "text", None)
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

        rows_to_flush.extend(data_to_rows(parsed_model, file_name=key))
        raw_rows_to_flush.append(
            {
                "file_name": key,
                "source": "api_recovery",
                "response": parsed_model.model_dump(mode="python"),
            }
        )

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


def _find_submit_run_dir(batch_name: str) -> Path | None:
    root = Path(config.output_root).expanduser()
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
    for run_dir in run_dirs:
        job_payload = _read_batch_job_payload(run_dir / "batch_job.json")
        if not job_payload:
            continue
        job_name = _normalize_key(job_payload.get("batch_job_name"))
        if job_name != batch_name:
            continue
        requests_file = run_dir / config.batch_requests_file_name
        if requests_file.exists() and requests_file.is_file():
            return run_dir
    return None


def _resolve_expected_request_keys(
    args: argparse.Namespace,
    batch_name: str,
    log,
) -> set[str]:
    submit_run_dir: Path | None = None
    if args.run_dir:
        candidate = Path(args.run_dir).expanduser()
        requests_file = candidate / config.batch_requests_file_name
        if requests_file.exists() and requests_file.is_file():
            submit_run_dir = candidate

    if submit_run_dir is None:
        submit_run_dir = _find_submit_run_dir(batch_name)

    if submit_run_dir is None:
        log(
            "Could not locate submit run with batch requests file; "
            "skipping expected-page coverage validation."
        )
        return set()

    requests_file = submit_run_dir / config.batch_requests_file_name
    expected_keys = _read_request_keys_from_file(requests_file)
    log(
        f"Loaded {len(expected_keys)} expected page key(s) from "
        f"{requests_file}."
    )
    return expected_keys


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
        100.0
        if total_requests == 0
        else (successful_requests / total_requests) * 100.0
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
            "config.gcs_bucket_name is empty. "
            "Set the GCS bucket used for batch files."
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
    run_dir: Path,
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

    raw_path = run_dir / "batch_output.jsonl"
    with open(raw_path, "wb") as handle:
        for blob in jsonl_blobs:
            data = blob.download_as_bytes()
            handle.write(data)
            if data and not data.endswith(b"\n"):
                handle.write(b"\n")

    log(
        f"Downloaded {len(jsonl_blobs)} output file(s) from "
        f"{dest_gcs_uri} into {raw_path.name}."
    )
    return raw_path


def _download_from_mldev_output(
    client: genai.Client,
    file_name: str,
    run_dir: Path,
) -> Path:
    raw_path = run_dir / "batch_output.jsonl"
    downloaded = None
    try:
        downloaded = client.files.download(
            file=file_name,
            destination=str(raw_path),
        )
    except TypeError:
        downloaded = client.files.download(file=file_name)

    if downloaded is not None:
        if hasattr(downloaded, "read"):
            raw_path.write_bytes(downloaded.read())
        else:
            raw_path.write_bytes(downloaded)
    return raw_path


def _await_completion(
    client: genai.Client,
    batch_name: str,
    log,
) -> object:
    poll_interval = max(1, int(config.batch_poll_interval_seconds))
    while True:
        batch_job = client.batches.get(name=batch_name)
        state = getattr(batch_job, "state", None)
        if state in {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
        }:
            return batch_job
        log(f"Batch {batch_name} state={state}. Waiting {poll_interval}s.")
        time.sleep(poll_interval)


def _read_batch_name_from_job_file(path: Path) -> str | None:
    payload = _read_batch_job_payload(path)
    if not payload:
        return None
    return _normalize_key(payload.get("batch_job_name"))


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


def _resolve_batch_name(args: argparse.Namespace) -> str:
    if args.batch_name:
        return args.batch_name

    if args.run_dir:
        candidate = Path(args.run_dir).expanduser() / "batch_job.json"
        batch_name = _read_batch_name_from_job_file(candidate)
        if batch_name:
            return batch_name
        raise ValueError(f"No batch_job_name found in {candidate}.")

    latest_job_file = _latest_batch_job_file(config.output_root)
    if latest_job_file is not None:
        batch_name = _read_batch_name_from_job_file(latest_job_file)
        if batch_name:
            return batch_name

    if config.batch_job_name:
        return config.batch_job_name

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
    batch_name = _resolve_batch_name(args)

    run_dir = create_subfolder(config.output_root, prefix="retrieve_")
    log = get_run_logger(run_dir)

    client_location = (
        _extract_location_from_batch_name(batch_name)
        or (config.vertex_model_location or "").strip()
        or (config.gcp_location or "").strip()
        or None
    )
    client = get_batch_client(location=client_location)
    batch_job = client.batches.get(name=batch_name)
    state = getattr(batch_job, "state", None)
    if state != "JOB_STATE_SUCCEEDED":
        if args.wait:
            batch_job = _await_completion(client, batch_name, log)
            state = getattr(batch_job, "state", None)
        else:
            raise RuntimeError(f"Batch {batch_name} not complete (state={state}).")

    dest = getattr(batch_job, "dest", None)
    file_name = getattr(dest, "file_name", None) if dest else None
    dest_gcs_uri = getattr(dest, "gcs_uri", None) if dest else None

    if dest_gcs_uri:
        raw_path = _download_from_vertex_gcs_output(
            dest_gcs_uri=dest_gcs_uri,
            run_dir=run_dir,
            log=log,
        )
    elif file_name:
        raw_path = _download_from_mldev_output(
            client=client,
            file_name=file_name,
            run_dir=run_dir,
        )
    else:
        raise RuntimeError(f"Batch {batch_name} missing output destination.")

    out_name = config.dataset_file_name
    output_dataset_format = config.output_format
    out_path = run_dir / f"{run_dir.name}_{out_name}.{output_dataset_format.lstrip('.')}"
    raw_out_path = run_dir / f"{run_dir.name}_{out_name}_raw.{output_dataset_format.lstrip('.')}"

    flush_every = max(1, int(config.flush_every or config.batch_size))
    rows_to_flush: list[dict] = []
    raw_rows_to_flush: list[dict] = []
    header_written = False
    raw_header_written = False
    total_rows = 0
    raw_total_rows = 0
    error_rows = 0

    output_keys_seen: set[str] = set()
    successful_page_keys: set[str] = set()
    failures: dict[str, str] = {}

    with open(raw_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(
            tqdm(handle, desc="Parsing batch results", unit="line"),
            start=1,
        ):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                log("Invalid JSONL line in batch output.", exc=exc)
                error_rows += 1
                _record_failure(
                    failures,
                    key=None,
                    line_number=line_number,
                    reason="invalid_jsonl_line",
                )
                continue

            key = _normalize_key(record.get("key")) if isinstance(record, dict) else None
            if key:
                output_keys_seen.add(key)

            if not isinstance(record, dict):
                error_rows += 1
                _record_failure(
                    failures,
                    key=key,
                    line_number=line_number,
                    reason="invalid_record_type",
                )
                continue

            if record.get("error"):
                error_rows += 1
                log(f"Batch error for key={key}: {record.get('error')}")
                _record_failure(
                    failures,
                    key=key,
                    line_number=line_number,
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
                    line_number=line_number,
                    reason="missing_response",
                )
                continue

            text_payload = _extract_text(response)
            if not text_payload:
                error_rows += 1
                log(f"Empty response text for key={key}")
                _record_failure(
                    failures,
                    key=key,
                    line_number=line_number,
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
                    line_number=line_number,
                    reason="schema_validation_failed",
                )
                continue

            file_key = key or f"<line:{line_number}>"
            if key:
                successful_page_keys.add(key)

            rows_to_flush.extend(data_to_rows(parsed_model, file_name=file_key))
            raw_rows_to_flush.append(
                {
                    "file_name": file_key,
                    "source": "batch",
                    "response": parsed_model.model_dump(mode="python"),
                }
            )

            if len(rows_to_flush) >= flush_every:
                header_written, wrote = _flush_rows(
                    rows_to_flush=rows_to_flush,
                    out_path=out_path,
                    output_dataset_format=output_dataset_format,
                    header_written=header_written,
                )
                total_rows += wrote

            if len(raw_rows_to_flush) >= flush_every:
                raw_header_written, raw_wrote = _flush_rows(
                    rows_to_flush=raw_rows_to_flush,
                    out_path=raw_out_path,
                    output_dataset_format=output_dataset_format,
                    header_written=raw_header_written,
                )
                raw_total_rows += raw_wrote

    expected_keys = _resolve_expected_request_keys(args, batch_name, log)
    expected_success = _expected_success_keys(
        expected_keys=expected_keys,
        observed_output_keys=output_keys_seen,
    )
    missing_success_keys = expected_success - successful_page_keys

    if missing_success_keys and config.api_recovery_enabled:
        recovered_count = _recover_missing_pages_via_api_key(
            missing_keys=missing_success_keys,
            successful_keys=successful_page_keys,
            observed_output_keys=output_keys_seen,
            failures=failures,
            rows_to_flush=rows_to_flush,
            raw_rows_to_flush=raw_rows_to_flush,
            log=log,
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

    _validate_page_completeness(
        expected_keys=expected_keys,
        observed_output_keys=output_keys_seen,
        successful_keys=successful_page_keys,
        failures=failures,
        log=log,
    )
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

    if raw_rows_to_flush:
        raw_header_written, raw_wrote = _flush_rows(
            rows_to_flush=raw_rows_to_flush,
            out_path=raw_out_path,
            output_dataset_format=output_dataset_format,
            header_written=raw_header_written,
        )
        raw_total_rows += raw_wrote

    log(
        f"Retrieved {total_rows} processed row(s) into {out_path.name} "
        f"and {raw_total_rows} raw row(s) into {raw_out_path.name} "
        f"(errors={error_rows})."
    )
    _upload_dataset_to_gcs(out_path, run_dir.name, log)
    _upload_dataset_to_gcs(raw_out_path, run_dir.name, log)
    return out_path


if __name__ == "__main__":
    retrieve_batch()
