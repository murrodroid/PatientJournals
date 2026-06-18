from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import mimetypes
import time
from concurrent.futures import ThreadPoolExecutor, as_completed as futures_as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from google import genai
from google.cloud import storage
from google.genai import types
from tqdm import tqdm

from patientjournals.batch.client import get_batch_client, resolve_service_account_path
from patientjournals.batch.output_records import (
    add_response_metadata_columns,
    parse_gemini_output_record,
)
from patientjournals.batch.results import RetrieveBatchResult
from patientjournals.batch.retry import (
    _collect_failed_retry_keys,
    _submit_failed_pages_as_batch,
)
from patientjournals.config import config
from patientjournals.shared.generation_spec import (
    build_live_generation_config,
)
from patientjournals.config.models import resolve_model_spec
from patientjournals.shared.api_retry import (
    is_retryable_api_error,
    retry_delay_seconds,
)
from patientjournals.shared.output_handler import data_to_rows
from patientjournals.shared.processing_metrics import (
    MANIFEST_FILE_NAME,
    append_processing_record,
    base_image_record,
    utc_now_iso,
    write_processing_summary,
)
from patientjournals.shared.response_parsing import extract_response_metadata
from patientjournals.shared import run_layout
from patientjournals.shared.tools import create_subfolder, flush_rows, get_run_logger


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
        action="append",
        help=(
            "Batch job name. Repeat to retrieve selected chunks from one run. "
            "Overrides config.batch_job_name when supplied."
        ),
    )
    parser.add_argument(
        "--run-dir",
        dest="run_dir",
        help="Run directory containing batch_job.json (overrides auto-discovery).",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help=(
            "Directory where retrieved outputs and the dataset should be written. "
            "Defaults to a new runs/retrieves/<timestamp> folder."
        ),
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
            "Permit partial retrieval. For Gemini/Vertex, this also parses "
            "available output files from non-succeeded jobs and skips jobs "
            "without downloadable outputs."
        ),
    )
    parser.add_argument(
        "--submit-failed",
        dest="submit_failed",
        action="store_true",
        help=(
            "After retrieval, submit a separate retry batch containing keys "
            "that errored or failed JSON/schema validation."
        ),
    )
    parser.add_argument(
        "--recover-missing-with-api",
        dest="recover_missing_with_api",
        action="store_true",
        help=(
            "For Gemini retrieval, send missing expected pages through the live API "
            "after partial batch output parsing. This is parallelized with "
            "config.api_concurrent_tasks."
        ),
    )
    parser.add_argument(
        "--duplicate-strategy",
        choices=("first_successful", "provide_all"),
        default=None,
        help=(
            "How to handle duplicate successful output keys across chunks. "
            "Default comes from config.batch_duplicate_strategy."
        ),
    )
    return parser.parse_args()


def _should_submit_failed_batch(args: argparse.Namespace) -> bool:
    return bool(args.submit_failed or config.batch_submit_failed_pages)


def _normalize_key(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def _arg_batch_names(args: argparse.Namespace) -> list[str]:
    raw = getattr(args, "batch_name", None)
    values: list[object]
    if isinstance(raw, list | tuple):
        values = list(raw)
    elif raw:
        values = [raw]
    else:
        values = []

    names: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = _normalize_key(value)
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _effective_duplicate_strategy(args: argparse.Namespace) -> str:
    strategy = (
        getattr(args, "duplicate_strategy", None)
        or getattr(config, "batch_duplicate_strategy", "first_successful")
        or "first_successful"
    )
    strategy = str(strategy).strip().lower()
    if strategy not in {"first_successful", "provide_all"}:
        raise ValueError(
            "Unsupported duplicate strategy: "
            f"{strategy!r}. Expected first_successful or provide_all."
        )
    return strategy


def _parallel_worker_count(total: int) -> int:
    return max(1, min(max(1, total), int(config.api_concurrent_tasks or 1)))


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


@dataclass
class _RecoveryResult:
    key: str
    line_number: int
    parsed_model: Any | None = None
    metadata: dict[str, Any] | None = None
    failure_reason: str | None = None
    exception: BaseException | None = None
    attempts: int = 0
    generation_seconds: float | None = None
    total_seconds: float | None = None
    downloaded_bytes: int | None = None
    mime_type: str | None = None


async def _generate_recovery_response(
    *,
    recovery_client,
    recovery_model: str,
    image_bytes: bytes,
    mime_type: str,
    generation_config: dict,
) -> object:
    aio_client = getattr(recovery_client, "aio", None)
    aio_models = getattr(aio_client, "models", None)
    async_generate = getattr(aio_models, "generate_content", None)
    if callable(async_generate):
        return await async_generate(
            model=recovery_model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                config.input_prompt,
            ],
            config=generation_config,
        )

    return await asyncio.to_thread(
        recovery_client.models.generate_content,
        model=recovery_model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            config.input_prompt,
        ],
        config=generation_config,
    )


async def _recover_one_missing_page_via_api_key(
    *,
    key: str,
    line_number: int,
    bucket,
    recovery_client,
    recovery_model: str,
    generation_config: dict,
    log,
) -> _RecoveryResult:
    blob = bucket.blob(key)
    max_attempts = max(1, int(config.api_max_attempts))
    total_started = time.perf_counter()

    for attempt in range(1, max_attempts + 1):
        try:
            exists = await asyncio.to_thread(blob.exists)
            if not exists:
                return _RecoveryResult(
                    key=key,
                    line_number=line_number,
                    failure_reason="recovery_blob_not_found",
                    attempts=attempt,
                    total_seconds=time.perf_counter() - total_started,
                )

            image_bytes = await asyncio.to_thread(blob.download_as_bytes)
            mime_type = _guess_blob_mime_type(blob, key)
            generation_started = time.perf_counter()
            response = await _generate_recovery_response(
                recovery_client=recovery_client,
                recovery_model=recovery_model,
                image_bytes=image_bytes,
                mime_type=mime_type,
                generation_config=generation_config,
            )
            generation_seconds = time.perf_counter() - generation_started

            metadata = extract_response_metadata(response)
            text_payload = metadata.get("text")
            if not isinstance(text_payload, str) or not text_payload.strip():
                raise ValueError("Empty response text from API key recovery.")

            parsed_model = config.output_model.model_validate_json(text_payload)
            return _RecoveryResult(
                key=key,
                line_number=line_number,
                parsed_model=parsed_model,
                metadata=metadata,
                attempts=attempt,
                generation_seconds=generation_seconds,
                total_seconds=time.perf_counter() - total_started,
                downloaded_bytes=len(image_bytes),
                mime_type=mime_type,
            )
        except Exception as exc:
            retryable = is_retryable_api_error(exc)
            if retryable and attempt < max_attempts:
                delay = retry_delay_seconds(attempt)
                log(
                    f"Transient API key recovery error for key={key} "
                    f"(attempt {attempt}/{max_attempts}). "
                    f"Retrying in {delay:.1f}s. Error: {exc}"
                )
                await asyncio.sleep(delay)
                continue

            return _RecoveryResult(
                key=key,
                line_number=line_number,
                failure_reason=f"api_key_recovery_failed:{type(exc).__name__}",
                exception=exc,
                attempts=attempt,
                total_seconds=time.perf_counter() - total_started,
            )

    return _RecoveryResult(
        key=key,
        line_number=line_number,
        failure_reason="api_key_recovery_failed:unknown",
        attempts=max_attempts,
        total_seconds=time.perf_counter() - total_started,
    )


async def _recover_missing_pages_via_api_key_async(
    *,
    missing_keys: set[str],
    bucket,
    recovery_client,
    recovery_model: str,
    generation_config: dict,
    log,
) -> list[_RecoveryResult]:
    concurrency = max(1, int(config.api_concurrent_tasks or 1))
    semaphore = asyncio.Semaphore(concurrency)
    base_line_number = 1_000_000

    async def run_one(offset: int, key: str) -> _RecoveryResult:
        async with semaphore:
            return await _recover_one_missing_page_via_api_key(
                key=key,
                line_number=base_line_number + offset,
                bucket=bucket,
                recovery_client=recovery_client,
                recovery_model=recovery_model,
                generation_config=generation_config,
                log=log,
            )

    tasks = [
        asyncio.create_task(run_one(offset, key))
        for offset, key in enumerate(sorted(missing_keys), start=1)
    ]
    results: list[_RecoveryResult] = []
    for task in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Recovering pages via API key",
        unit="page",
    ):
        results.append(await task)
    return sorted(results, key=lambda result: result.line_number)


def _recover_missing_pages_via_api_key(
    *,
    missing_keys: set[str],
    successful_keys: set[str],
    observed_output_keys: set[str],
    failures: dict[str, str],
    rows_to_flush: list[dict],
    log,
    force: bool = False,
    manifest_path: Path | None = None,
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
    concurrency = max(1, int(config.api_concurrent_tasks or 1))
    max_attempts = max(1, int(config.api_max_attempts))

    recovered = 0
    log(
        f"Starting API key recovery for {len(missing_keys)} page(s) "
        f"using model {recovery_model} "
        f"(concurrency={concurrency}, max_attempts={max_attempts})."
    )
    results = asyncio.run(
        _recover_missing_pages_via_api_key_async(
            missing_keys=missing_keys,
            bucket=bucket,
            recovery_client=recovery_client,
            recovery_model=recovery_model,
            generation_config=generation_config,
            log=log,
        )
    )

    for result in results:
        key = result.key
        if result.parsed_model is None:
            reason = result.failure_reason or "api_key_recovery_failed:unknown"
            if manifest_path is not None:
                append_processing_record(
                    manifest_path,
                    base_image_record(
                        image_reference=key,
                        source="api_recovery",
                        status="failed",
                        model=recovery_model,
                        provider="gemini",
                        attempts=result.attempts,
                        max_attempts=max_attempts,
                        total_seconds=result.total_seconds,
                        rows_written=0,
                        failure_reason=reason,
                        error_type=type(result.exception).__name__
                        if result.exception
                        else None,
                        error_message=str(result.exception)
                        if result.exception
                        else None,
                        extra={
                            "mime_type": result.mime_type,
                            "downloaded_bytes": result.downloaded_bytes,
                        },
                    ),
                )
            _record_failure(
                failures,
                key=key,
                line_number=result.line_number,
                reason=reason,
            )
            if reason == "recovery_blob_not_found":
                log(f"Recovery skipped for missing GCS object: {key}")
            else:
                log(
                    f"API key recovery failed for key={key}",
                    exc=result.exception,
                )
            continue

        recovered += 1
        successful_keys.add(key)
        observed_output_keys.add(key)
        failures.pop(key, None)

        metadata = result.metadata or {}
        rows = data_to_rows(
            result.parsed_model,
            file_name=key,
            field_confidence_by_pointer=metadata.get("field_confidence_by_pointer"),
        )
        add_response_metadata_columns(rows, metadata)
        rows_to_flush.extend(rows)
        if manifest_path is not None:
            append_processing_record(
                manifest_path,
                base_image_record(
                    image_reference=key,
                    source="api_recovery",
                    status="success",
                    model=recovery_model,
                    provider="gemini",
                    attempts=result.attempts,
                    max_attempts=max_attempts,
                    generation_seconds=result.generation_seconds,
                    total_seconds=result.total_seconds,
                    rows_written=len(rows),
                    extra={
                        "mime_type": result.mime_type,
                        "downloaded_bytes": result.downloaded_bytes,
                    },
                ),
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
    target = {name for name in batch_names if isinstance(name, str) and name.strip()}
    run_dirs = run_layout.iter_run_dirs(config.output_root, "submit")
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


def _output_destinations_from_submit_run(
    submit_run_dir: Path | None,
) -> dict[str, str]:
    if submit_run_dir is None:
        return {}
    payload = _read_batch_job_payload(submit_run_dir / "batch_job.json")
    if not payload:
        return {}

    destinations: dict[str, str] = {}
    jobs = payload.get("batch_jobs")
    if not isinstance(jobs, list):
        return destinations

    for item in jobs:
        if not isinstance(item, dict):
            continue
        batch_name = _normalize_key(item.get("batch_job_name"))
        destination = _normalize_key(item.get("output_destination"))
        if batch_name and destination:
            destinations[batch_name] = destination
    return destinations


def _gemini_output_reference(
    batch_job: object,
    *,
    metadata_destination: str | None = None,
) -> tuple[str, str] | None:
    dest = getattr(batch_job, "dest", None)
    file_name = getattr(dest, "file_name", None) if dest else None
    dest_gcs_uri = getattr(dest, "gcs_uri", None) if dest else None

    if isinstance(dest_gcs_uri, str) and dest_gcs_uri.strip():
        return "gcs", dest_gcs_uri.strip()
    if isinstance(file_name, str) and file_name.strip():
        return "file", file_name.strip()
    if metadata_destination:
        return "gcs", metadata_destination
    return None


def _validate_page_completeness(
    *,
    expected_keys: set[str],
    observed_output_keys: set[str],
    successful_keys: set[str],
    failures: dict[str, str],
    require_all_expected_pages: bool | None = None,
    require_all_pages_successful: bool | None = None,
    log,
) -> None:
    sample_size = max(1, int(config.page_validation_sample_size or 1))
    require_expected = (
        config.require_all_expected_pages
        if require_all_expected_pages is None
        else require_all_expected_pages
    )
    require_success = (
        config.require_all_pages_successful
        if require_all_pages_successful is None
        else require_all_pages_successful
    )

    if expected_keys and require_expected:
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

    if not require_success:
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

    try:
        bucket_name = _require_bucket_name()
        service_account_file = _require_service_account_file()
        service_account_path = resolve_service_account_path(service_account_file)
        storage_client = storage.Client.from_service_account_json(
            str(service_account_path)
        )
        bucket = storage_client.bucket(bucket_name)

        prefix = _normalize_prefix(config.datasets_gcs_prefix or "")
        object_name = f"{prefix}{run_dir_name}/{dataset_path.name}"
        blob = bucket.blob(object_name)
        blob.upload_from_filename(
            str(dataset_path),
            content_type=_dataset_content_type(dataset_path),
        )
    except Exception as exc:  # noqa: BLE001
        log(f"Dataset upload skipped or failed for {dataset_path.name}.", exc=exc)
        return None

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
    for run_dir in run_layout.iter_run_dirs(output_root, "submit"):
        candidate = run_dir / "batch_job.json"
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _resolve_batch_targets(args: argparse.Namespace) -> tuple[list[str], Path | None]:
    cli_batch_names = _arg_batch_names(args)
    if cli_batch_names:
        submit_run_dir = Path(args.run_dir).expanduser() if args.run_dir else None
        return cli_batch_names, submit_run_dir

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


def _resolve_retrieve_run_dir(
    args: argparse.Namespace,
    *,
    submit_run_dir: Path | None,
) -> Path:
    output_dir = getattr(args, "output_dir", None)
    if output_dir:
        run_dir = Path(output_dir).expanduser()
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    return create_subfolder(config.output_root, category="retrieve")


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


def retrieve_batch(args: argparse.Namespace | None = None) -> RetrieveBatchResult:
    args = args or _parse_args()
    submit_failed_requested = _should_submit_failed_batch(args)
    batch_names, submit_run_dir = _resolve_batch_targets(args)
    if not batch_names:
        raise ValueError("No batch jobs resolved for retrieval.")

    run_dir = _resolve_retrieve_run_dir(args, submit_run_dir=submit_run_dir)
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

    duplicate_strategy = _effective_duplicate_strategy(args)
    recover_missing_with_api = bool(getattr(args, "recover_missing_with_api", False))
    manifest_path = run_dir / MANIFEST_FILE_NAME

    output_destinations_by_name = _output_destinations_from_submit_run(submit_run_dir)
    batch_jobs: list[tuple[int, str, object, bool, tuple[str, str] | None]] = []
    incomplete_batches: list[tuple[str, str]] = []
    partial_output_batches: list[tuple[str, str]] = []

    def resolve_one(index: int, batch_name: str):
        batch_job = _get_batch_job(client, batch_name, provider)
        state = _batch_job_state(batch_job, provider)
        if not _batch_job_successful(batch_job, provider):
            if bool(getattr(args, "wait", False)):
                batch_job = _await_completion(
                    client=client,
                    batch_name=batch_name,
                    provider=provider,
                    log=log,
                )
                state = _batch_job_state(batch_job, provider)

        is_successful = _batch_job_successful(batch_job, provider)
        output_ref: tuple[str, str] | None = None
        if provider != "anthropic":
            output_ref = _gemini_output_reference(
                batch_job,
                metadata_destination=output_destinations_by_name.get(batch_name),
            )

        if not is_successful:
            if provider == "gemini" and output_ref is not None:
                return index, batch_name, batch_job, state, True, output_ref, True
            return index, batch_name, batch_job, state, False, output_ref, False
        return index, batch_name, batch_job, state, False, output_ref, True

    workers = _parallel_worker_count(len(batch_names))
    log(
        f"Resolving {len(batch_names)} batch chunk state(s) in parallel "
        f"(workers={workers})."
    )
    resolved_jobs = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(resolve_one, index, batch_name)
            for index, batch_name in enumerate(batch_names, start=1)
        ]
        for future in tqdm(
            futures_as_completed(futures),
            total=len(futures),
            desc="Resolving batch chunks",
            unit="chunk",
        ):
            resolved_jobs.append(future.result())

    for (
        index,
        batch_name,
        batch_job,
        state,
        is_partial_output,
        output_ref,
        downloadable,
    ) in sorted(resolved_jobs, key=lambda item: int(item[0])):
        if not downloadable:
            incomplete_batches.append((batch_name, state))
            continue
        if is_partial_output:
            partial_output_batches.append((batch_name, state))
            if not bool(getattr(args, "allow_partial", False)):
                incomplete_batches.append((batch_name, state))
                continue
        batch_jobs.append((index, batch_name, batch_job, is_partial_output, output_ref))

    if incomplete_batches and not bool(getattr(args, "allow_partial", False)):
        examples = ", ".join(
            f"{name} ({state})" for name, state in incomplete_batches[:5]
        )
        suffix = "..." if len(incomplete_batches) > 5 else ""
        raise RuntimeError(
            "Not all batch jobs are complete/succeeded. "
            f"Incomplete jobs={len(incomplete_batches)} [{examples}{suffix}]. "
            "Re-run with --wait, or use --allow-partial to retrieve available "
            "output rows from non-succeeded chunks."
        )

    if not batch_jobs and not (recover_missing_with_api and provider == "gemini"):
        raise RuntimeError(
            "No batch jobs with downloadable outputs are available to retrieve. "
            "Use --wait to block until completion, or use "
            "--allow-partial --recover-missing-with-api for Gemini API recovery."
        )

    if partial_output_batches:
        examples = ", ".join(
            f"{name} ({state})" for name, state in partial_output_batches[:5]
        )
        suffix = "..." if len(partial_output_batches) > 5 else ""
        log(
            f"Partial retrieval enabled: including available output files from "
            f"{len(partial_output_batches)} non-succeeded Gemini job(s): "
            f"{examples}{suffix}"
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
            "batch job(s) with downloadable outputs."
        )

    raw_outputs: list[tuple[str, Path]] = []
    skipped_output_batches: list[tuple[str, str]] = []
    def download_one(
        output_index: int,
        batch_name: str,
        batch_job: object,
        is_partial_output: bool,
        output_ref: tuple[str, str] | None,
    ) -> tuple[int, str, Path | None, str | None]:
        raw_path = run_dir / f"batch_output_{output_index:03d}.jsonl"
        if provider == "anthropic":
            raw_path = _download_from_anthropic_output(
                client=client,
                batch_name=batch_name,
                output_path=raw_path,
                log=log,
            )
            return output_index, batch_name, raw_path, None

        if output_ref is None:
            raise RuntimeError(f"Batch {batch_name} missing output destination.")
        ref_kind, ref_value = output_ref
        try:
            if ref_kind == "gcs":
                raw_path = _download_from_vertex_gcs_output(
                    dest_gcs_uri=ref_value,
                    output_path=raw_path,
                    log=log,
                )
            elif ref_kind == "file":
                raw_path = _download_from_mldev_output(
                    client=client,
                    file_name=ref_value,
                    output_path=raw_path,
                )
            else:
                raise RuntimeError(
                    f"Unsupported output destination type '{ref_kind}' "
                    f"for batch {batch_name}."
                )
        except RuntimeError as exc:
            if is_partial_output and bool(getattr(args, "allow_partial", False)):
                return output_index, batch_name, None, str(exc)
            raise
        return output_index, batch_name, raw_path, None

    if batch_jobs:
        download_workers = _parallel_worker_count(len(batch_jobs))
        log(
            f"Downloading {len(batch_jobs)} batch output chunk(s) in parallel "
            f"(workers={download_workers})."
        )
        downloaded = []
        with ThreadPoolExecutor(max_workers=download_workers) as executor:
            futures = [
                executor.submit(
                    download_one,
                    output_index,
                    batch_name,
                    batch_job,
                    is_partial_output,
                    output_ref,
                )
                for output_index, (
                    _source_index,
                    batch_name,
                    batch_job,
                    is_partial_output,
                    output_ref,
                ) in enumerate(batch_jobs, start=1)
            ]
            for future in tqdm(
                futures_as_completed(futures),
                total=len(futures),
                desc="Downloading batch outputs",
                unit="chunk",
            ):
                downloaded.append(future.result())

        for _index, batch_name, raw_path, skip_reason in sorted(
            downloaded,
            key=lambda item: int(item[0]),
        ):
            if raw_path is None:
                skipped_output_batches.append((batch_name, skip_reason or "unknown"))
                log(
                    f"Skipping partial-output batch with no downloadable JSONL: "
                    f"{batch_name} ({skip_reason or 'unknown'})"
                )
                continue
            raw_outputs.append((batch_name, raw_path))

    if skipped_output_batches:
        print(
            "Skipped partial-output batch job(s) without JSONL files: "
            f"{len(skipped_output_batches)}"
        )

    if not raw_outputs and not (recover_missing_with_api and provider == "gemini"):
        raise RuntimeError("No output JSONL files were downloaded.")

    anthropic_custom_id_to_key: dict[str, str] = {}
    if provider == "anthropic":
        anthropic_custom_id_to_key = _resolve_anthropic_custom_id_to_key(
            submit_run_dir=submit_run_dir,
            batch_names=batch_names,
            selected_batch_names=[name for name, _ in raw_outputs],
            log=log,
        )

    out_name = config.dataset_file_name
    output_dataset_format = config.output_format
    final_out_path = (
        run_dir / f"{run_dir.name}_{out_name}.{output_dataset_format.lstrip('.')}"
    )
    out_path = final_out_path
    if final_out_path.exists():
        out_path = run_dir / f".{final_out_path.name}.tmp"
        out_path.unlink(missing_ok=True)

    flush_every = max(1, int(config.flush_every or config.batch_size))
    rows_to_flush: list[dict] = []
    header_written = False
    total_rows = 0
    error_rows = 0
    duplicate_rows_skipped = 0
    recovered_pages = 0

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
                    append_processing_record(
                        manifest_path,
                        base_image_record(
                            image_reference=None,
                            source="batch_retrieve",
                            status="failed",
                            model=config.model,
                            provider=provider,
                            attempts=1,
                            max_attempts=1,
                            rows_written=0,
                            failure_reason="invalid_jsonl_line",
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                            extra={
                                "batch_name": batch_name,
                                "raw_output_file": raw_path.name,
                                "line_number": line_number,
                            },
                        ),
                    )
                    _record_failure(
                        failures,
                        key=None,
                        line_number=global_line_number,
                        reason="invalid_jsonl_line",
                    )
                    continue

                key: str | None = None
                metadata: dict[str, object] = {}
                parsed_model = None
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
                    if key:
                        output_keys_seen.add(key)

                    if not isinstance(record, dict):
                        error_rows += 1
                        append_processing_record(
                            manifest_path,
                            base_image_record(
                                image_reference=key,
                                source="batch_retrieve",
                                status="failed",
                                model=config.model,
                                provider=provider,
                                attempts=1,
                                max_attempts=1,
                                rows_written=0,
                                failure_reason="invalid_record_type",
                                extra={
                                    "batch_name": batch_name,
                                    "raw_output_file": raw_path.name,
                                    "line_number": line_number,
                                },
                            ),
                        )
                        _record_failure(
                            failures,
                            key=key,
                            line_number=global_line_number,
                            reason="invalid_record_type",
                        )
                        continue

                    result = record.get("result")
                    if not isinstance(result, dict):
                        error_rows += 1
                        log(f"Missing/invalid result payload for key={key}")
                        append_processing_record(
                            manifest_path,
                            base_image_record(
                                image_reference=key,
                                source="batch_retrieve",
                                status="failed",
                                model=config.model,
                                provider=provider,
                                attempts=1,
                                max_attempts=1,
                                rows_written=0,
                                failure_reason="missing_result",
                                extra={
                                    "batch_name": batch_name,
                                    "raw_output_file": raw_path.name,
                                    "line_number": line_number,
                                },
                            ),
                        )
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
                        append_processing_record(
                            manifest_path,
                            base_image_record(
                                image_reference=key,
                                source="batch_retrieve",
                                status="failed",
                                model=config.model,
                                provider=provider,
                                attempts=1,
                                max_attempts=1,
                                rows_written=0,
                                failure_reason=f"batch_{result_type or 'unknown'}",
                                extra={
                                    "batch_name": batch_name,
                                    "raw_output_file": raw_path.name,
                                    "line_number": line_number,
                                },
                            ),
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
                        append_processing_record(
                            manifest_path,
                            base_image_record(
                                image_reference=key,
                                source="batch_retrieve",
                                status="failed",
                                model=config.model,
                                provider=provider,
                                attempts=1,
                                max_attempts=1,
                                rows_written=0,
                                failure_reason="missing_response",
                                extra={
                                    "batch_name": batch_name,
                                    "raw_output_file": raw_path.name,
                                    "line_number": line_number,
                                },
                            ),
                        )
                        _record_failure(
                            failures,
                            key=key,
                            line_number=global_line_number,
                            reason="missing_response",
                        )
                        continue
                    metadata = _extract_anthropic_response_metadata(response)

                    text_payload = metadata.get("text")
                    if not text_payload:
                        error_rows += 1
                        log(f"Empty response text for key={key}")
                        append_processing_record(
                            manifest_path,
                            base_image_record(
                                image_reference=key,
                                source="batch_retrieve",
                                status="failed",
                                model=config.model,
                                provider=provider,
                                attempts=1,
                                max_attempts=1,
                                rows_written=0,
                                failure_reason="empty_response_text",
                                extra={
                                    "batch_name": batch_name,
                                    "raw_output_file": raw_path.name,
                                    "line_number": line_number,
                                },
                            ),
                        )
                        _record_failure(
                            failures,
                            key=key,
                            line_number=global_line_number,
                            reason="empty_response_text",
                        )
                        continue

                    try:
                        parsed_model = config.output_model.model_validate_json(
                            text_payload
                        )
                    except Exception as exc:
                        error_rows += 1
                        log(f"Schema validation failed for key={key}", exc=exc)
                        append_processing_record(
                            manifest_path,
                            base_image_record(
                                image_reference=key,
                                source="batch_retrieve",
                                status="failed",
                                model=config.model,
                                provider=provider,
                                attempts=1,
                                max_attempts=1,
                                rows_written=0,
                                failure_reason="schema_validation_failed",
                                error_type=type(exc).__name__,
                                error_message=str(exc),
                                extra={
                                    "batch_name": batch_name,
                                    "raw_output_file": raw_path.name,
                                    "line_number": line_number,
                                },
                            ),
                        )
                        _record_failure(
                            failures,
                            key=key,
                            line_number=global_line_number,
                            reason="schema_validation_failed",
                        )
                        continue
                else:
                    parse_result = parse_gemini_output_record(
                        record,
                        source=batch_name,
                        line_number=line_number,
                    )
                    key = parse_result.key
                    if key:
                        output_keys_seen.add(key)
                    if not parse_result.is_valid:
                        reason = parse_result.reason or "unknown"
                        error_rows += 1
                        if parse_result.detail:
                            log(
                                f"Gemini output rejected for key={key}: "
                                f"{reason} ({parse_result.detail})"
                            )
                        else:
                            log(f"Gemini output rejected for key={key}: {reason}")
                        _record_failure(
                            failures,
                            key=key,
                            line_number=global_line_number,
                            reason=reason,
                        )
                        append_processing_record(
                            manifest_path,
                            base_image_record(
                                image_reference=key,
                                source="batch_retrieve",
                                status="failed",
                                model=config.model,
                                provider=provider,
                                attempts=1,
                                max_attempts=1,
                                rows_written=0,
                                failure_reason=reason,
                                error_message=parse_result.detail,
                                extra={
                                    "batch_name": batch_name,
                                    "raw_output_file": raw_path.name,
                                    "line_number": line_number,
                                },
                            ),
                        )
                        continue
                    metadata = parse_result.metadata
                    parsed_model = parse_result.parsed_model

                if parsed_model is None:
                    error_rows += 1
                    append_processing_record(
                        manifest_path,
                        base_image_record(
                            image_reference=key,
                            source="batch_retrieve",
                            status="failed",
                            model=config.model,
                            provider=provider,
                            attempts=1,
                            max_attempts=1,
                            rows_written=0,
                            failure_reason="missing_parsed_model",
                            extra={
                                "batch_name": batch_name,
                                "raw_output_file": raw_path.name,
                                "line_number": line_number,
                            },
                        ),
                    )
                    _record_failure(
                        failures,
                        key=key,
                        line_number=global_line_number,
                        reason="missing_parsed_model",
                    )
                    continue

                file_key = key or f"<batch:{output_index}-line:{line_number}>"
                if (
                    key
                    and duplicate_strategy == "first_successful"
                    and key in successful_page_keys
                ):
                    duplicate_rows_skipped += 1
                    append_processing_record(
                        manifest_path,
                        base_image_record(
                            image_reference=key,
                            source="batch_retrieve",
                            status="duplicate_skipped",
                            model=config.model,
                            provider=provider,
                            attempts=1,
                            max_attempts=1,
                            rows_written=0,
                            extra={
                                "batch_name": batch_name,
                                "raw_output_file": raw_path.name,
                                "line_number": line_number,
                                "duplicate_strategy": duplicate_strategy,
                                "duplicate_action": "kept_first_successful",
                            },
                        ),
                    )
                    continue
                if key:
                    successful_page_keys.add(key)

                rows = data_to_rows(
                    parsed_model,
                    file_name=file_key,
                    field_confidence_by_pointer=metadata.get(
                        "field_confidence_by_pointer"
                    ),
                )
                add_response_metadata_columns(rows, metadata)
                rows_to_flush.extend(rows)
                append_processing_record(
                    manifest_path,
                    base_image_record(
                        image_reference=file_key,
                        source="batch_retrieve",
                        status="success",
                        model=config.model,
                        provider=provider,
                        attempts=1,
                        max_attempts=1,
                        rows_written=len(rows),
                        extra={
                            "batch_name": batch_name,
                            "raw_output_file": raw_path.name,
                            "line_number": line_number,
                            "duplicate_strategy": duplicate_strategy,
                            "duplicate_action": "provided"
                            if duplicate_strategy == "provide_all"
                            else "kept",
                        },
                    ),
                )

                if len(rows_to_flush) >= flush_every:
                    header_written, wrote = _flush_rows(
                        rows_to_flush=rows_to_flush,
                        out_path=out_path,
                        output_dataset_format=output_dataset_format,
                        header_written=header_written,
                    )
                    total_rows += wrote

    expected_batch_names = (
        batch_names
        if recover_missing_with_api
        else [name for name, _ in raw_outputs]
    )
    expected_keys = _resolve_expected_request_keys(
        submit_run_dir=submit_run_dir,
        batch_names=batch_names,
        selected_batch_names=expected_batch_names,
        log=log,
    )
    expected_success = _expected_success_keys(
        expected_keys=expected_keys,
        observed_output_keys=output_keys_seen,
    )
    missing_success_keys = expected_success - successful_page_keys
    if missing_success_keys and provider == "gemini":
        recovery_keys: set[str] = set()
        recovery_force = False

        if recover_missing_with_api:
            recovery_keys = missing_success_keys
            recovery_force = True
            log(
                "Recover-missing API mode enabled: attempting live API recovery "
                f"for {len(recovery_keys)} missing expected page(s)."
            )
        elif config.api_recovery_enabled:
            recovery_keys = missing_success_keys

        if recovery_keys:
            recovered_count = _recover_missing_pages_via_api_key(
                missing_keys=recovery_keys,
                successful_keys=successful_page_keys,
                observed_output_keys=output_keys_seen,
                failures=failures,
                rows_to_flush=rows_to_flush,
                log=log,
                force=recovery_force,
                manifest_path=manifest_path,
            )
            recovered_pages += recovered_count
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

    final_expected_success = _expected_success_keys(
        expected_keys=expected_keys,
        observed_output_keys=output_keys_seen,
    )
    failed_retry_keys, failed_retry_reasons = _collect_failed_retry_keys(
        expected_success_keys=final_expected_success,
        successful_page_keys=successful_page_keys,
        failures=failures,
    )
    if submit_failed_requested:
        if incomplete_batches:
            log(
                "Skipping failed-page retry submission because retrieval is partial "
                "and not all chunk jobs are complete."
            )
            print(
                "Skipped failed-page retry submission: retrieval used partial chunks."
            )
        elif failed_retry_keys:
            retry_submission = _submit_failed_pages_as_batch(
                failed_keys=failed_retry_keys,
                failure_reasons=failed_retry_reasons,
                provider=provider,
                client=client,
                batch_names=batch_names,
                submit_run_dir=submit_run_dir,
                log=log,
            )
            if retry_submission is not None:
                retry_run_dir, retry_batch_name, retry_count = retry_submission
                print(
                    f"Submitted failed-page retry batch ({retry_count} key(s)): "
                    f"{retry_batch_name} [{retry_run_dir}]"
                )
        else:
            log(
                "Failed-page retry submission requested, but no failed keys were detected."
            )
            print("No failed keys detected; retry batch was not submitted.")

    try:
        if bool(getattr(args, "allow_partial", False)):
            log(
                "Partial retrieval enabled: expected-page and successful-page "
                "coverage checks will be reported without failing the run."
            )
        _validate_page_completeness(
            expected_keys=expected_keys,
            observed_output_keys=output_keys_seen,
            successful_keys=successful_page_keys,
            failures=failures,
            require_all_expected_pages=(
                bool(config.require_all_expected_pages)
                and not bool(getattr(args, "allow_partial", False))
            ),
            require_all_pages_successful=(
                bool(config.require_all_pages_successful)
                and not bool(getattr(args, "allow_partial", False))
            ),
            log=log,
        )
    except RuntimeError:
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

    if out_path != final_out_path and out_path.exists():
        out_path.replace(final_out_path)
        out_path = final_out_path
    elif out_path != final_out_path:
        out_path = final_out_path

    log(
        f"Retrieved {total_rows} processed row(s) from {len(batch_jobs)} completed "
        f"batch job(s) into {out_path.name} "
        f"(errors={error_rows}, duplicates_skipped={duplicate_rows_skipped}, "
        f"recovered={recovered_pages})."
    )
    summary_path = write_processing_summary(run_dir)
    log(f"Wrote image processing manifest: {manifest_path.name}")
    log(f"Wrote image processing summary: {summary_path.name}")
    dataset_gcs_uri = _upload_dataset_to_gcs(out_path, run_dir.name, log) or ""
    return RetrieveBatchResult(
        dataset_path=out_path,
        run_dir=run_dir,
        provider=provider,
        batch_count=len(batch_jobs),
        output_file_count=len(raw_outputs),
        rows_written=total_rows,
        error_rows=error_rows,
        expected_pages=len(expected_keys),
        observed_pages=len(output_keys_seen),
        successful_pages=len(successful_page_keys),
        duplicate_rows_skipped=duplicate_rows_skipped,
        recovered_pages=recovered_pages,
        manifest_path=manifest_path,
        dataset_gcs_uri=dataset_gcs_uri,
    )


if __name__ == "__main__":
    retrieve_batch()
