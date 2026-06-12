from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from patientjournals.batch.client import get_batch_client, resolve_service_account_path
from patientjournals.config import config
from patientjournals.config.models import resolve_model_spec
from patientjournals.shared import run_layout


_GEMINI_TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
}
_ANTHROPIC_TERMINAL_STATES = {"ended"}


@dataclass(frozen=True)
class ModelProgress:
    processed: int | None
    total: int | None
    detail: str = ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check status of batch job(s)."
    )
    parser.add_argument(
        "--batch-name",
        dest="batch_name",
        help="Batch job name/id (overrides config.batch_job_name).",
    )
    parser.add_argument(
        "--run-dir",
        dest="run_dir",
        help="Run directory containing batch_job.json (overrides auto-discovery).",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Poll until the job reaches a terminal state.",
    )
    parser.add_argument(
        "--cancel",
        action="store_true",
        help="Cancel the batch job and exit.",
    )
    parser.add_argument(
        "--simple",
        "--summary",
        dest="simple",
        action="store_true",
        help=(
            "Show aggregate counts only (finished/pending and per-state), "
            "instead of detailed per-job output."
        ),
    )
    return parser.parse_args()


def _extract_batch_names_from_payload(payload: dict) -> list[str]:
    names: list[str] = []
    jobs = payload.get("batch_jobs")
    if isinstance(jobs, list):
        for item in jobs:
            if not isinstance(item, dict):
                continue
            value = item.get("batch_job_name")
            if isinstance(value, str) and value.strip():
                names.append(value.strip())

    direct_names = payload.get("batch_job_names")
    if isinstance(direct_names, list):
        for value in direct_names:
            if isinstance(value, str) and value.strip():
                names.append(value.strip())

    single = payload.get("batch_job_name")
    if isinstance(single, str) and single.strip():
        names.append(single.strip())

    ordered: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


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


def _read_batch_names_from_job_file(path: Path) -> list[str]:
    payload = _read_batch_job_payload(path)
    if not payload:
        return []
    return _extract_batch_names_from_payload(payload)


def _request_count_from_payload(payload: dict | None) -> int | None:
    if not payload:
        return None
    value = payload.get("request_count")
    if isinstance(value, int):
        return value
    jobs = payload.get("batch_jobs")
    if not isinstance(jobs, list):
        return None
    total = 0
    found = False
    for item in jobs:
        if not isinstance(item, dict):
            continue
        count = item.get("request_count")
        if isinstance(count, int):
            total += count
            found = True
    return total if found else None


def _latest_batch_job_file(output_root: str) -> Path | None:
    for run_dir in run_layout.iter_run_dirs(output_root, "submit"):
        candidate = run_dir / "batch_job.json"
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _resolve_batch_names(args: argparse.Namespace) -> tuple[list[str], Path | None]:
    if args.batch_name:
        run_dir = Path(args.run_dir).expanduser() if args.run_dir else None
        return [args.batch_name], run_dir

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser()
        candidate = run_dir / "batch_job.json"
        batch_names = _read_batch_names_from_job_file(candidate)
        if batch_names:
            return batch_names, run_dir
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


def _normalize_state(value: object) -> str:
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


def _aggregate_state_lines(states: list[str], provider: str) -> tuple[list[str], tuple]:
    terminal_states = _terminal_states(provider)
    counts: dict[str, int] = {}
    for state in states:
        counts[state] = counts.get(state, 0) + 1

    total = len(states)
    finished = sum(counts.get(state, 0) for state in terminal_states)
    pending = max(0, total - finished)

    ordered_state_pairs = sorted(counts.items(), key=lambda kv: kv[0])
    state_text = ", ".join(f"{name}={count}" for name, count in ordered_state_pairs)
    lines = [
        f"Batches: {total}",
        f"Finished: {finished}",
        f"Pending: {pending}",
    ]
    if provider == "anthropic":
        lines.append(f"Ended: {counts.get('ended', 0)}")
    else:
        lines.append(f"Succeeded: {counts.get('JOB_STATE_SUCCEEDED', 0)}")
        lines.append(f"Failed: {counts.get('JOB_STATE_FAILED', 0)}")
        lines.append(f"Cancelled: {counts.get('JOB_STATE_CANCELLED', 0)}")
    if state_text:
        lines.append(f"States: {state_text}")

    digest = (
        total,
        finished,
        pending,
        tuple(ordered_state_pairs),
    )
    return lines, digest


def _parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    cleaned = gcs_uri.strip()
    if not cleaned.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    remainder = cleaned[5:]
    if "/" not in remainder:
        return remainder, ""
    bucket_name, object_path = remainder.split("/", 1)
    return bucket_name, object_path


def _normalize_gcs_prefix(prefix: str) -> str:
    value = prefix.strip()
    if not value:
        return ""
    return f"{value.strip('/')}/"


def _output_destinations_from_submit_run(run_dir: Path | None) -> dict[str, str]:
    if run_dir is None:
        return {}
    payload = _read_batch_job_payload(run_dir / "batch_job.json")
    if not payload:
        return {}
    jobs = payload.get("batch_jobs")
    if not isinstance(jobs, list):
        return {}

    destinations: dict[str, str] = {}
    for item in jobs:
        if not isinstance(item, dict):
            continue
        batch_name = item.get("batch_job_name")
        destination = item.get("output_destination")
        if isinstance(batch_name, str) and isinstance(destination, str):
            if batch_name.strip() and destination.strip():
                destinations[batch_name.strip()] = destination.strip()
    return destinations


def _gemini_output_gcs_uri(
    batch_job: object,
    *,
    metadata_destination: str | None = None,
) -> str | None:
    dest = getattr(batch_job, "dest", None)
    dest_gcs_uri = getattr(dest, "gcs_uri", None) if dest else None
    if isinstance(dest_gcs_uri, str) and dest_gcs_uri.strip():
        return dest_gcs_uri.strip()
    if isinstance(metadata_destination, str) and metadata_destination.strip():
        return metadata_destination.strip()
    return None


def _count_jsonl_blob_lines(blob: object) -> int:
    count = 0
    with blob.open("rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _count_gemini_prediction_rows(gcs_uris: list[str]) -> tuple[int, int]:
    if not gcs_uris:
        return 0, 0

    from google.cloud import storage

    service_account_path = resolve_service_account_path(config.service_account_file)
    storage_client = storage.Client.from_service_account_json(str(service_account_path))
    processed = 0
    file_count = 0
    seen: set[tuple[str, str]] = set()
    for gcs_uri in gcs_uris:
        bucket_name, object_prefix = _parse_gcs_uri(gcs_uri)
        prefix = _normalize_gcs_prefix(object_prefix)
        bucket = storage_client.bucket(bucket_name)
        blobs = sorted(bucket.list_blobs(prefix=prefix), key=lambda item: item.name)
        for blob in blobs:
            if blob.name.endswith("/") or not blob.name.endswith(".jsonl"):
                continue
            key = (bucket_name, blob.name)
            if key in seen:
                continue
            seen.add(key)
            processed += _count_jsonl_blob_lines(blob)
            file_count += 1
    return processed, file_count


def _anthropic_model_progress(batch_jobs: list[object], total: int | None) -> ModelProgress:
    processed = 0
    found_counts = False
    for batch_job in batch_jobs:
        counts = getattr(batch_job, "request_counts", None)
        if counts is None:
            continue
        found_counts = True
        succeeded = int(getattr(counts, "succeeded", 0) or 0)
        errored = int(getattr(counts, "errored", 0) or 0)
        canceled = int(getattr(counts, "canceled", 0) or 0)
        expired = int(getattr(counts, "expired", 0) or 0)
        processed += succeeded + errored + canceled + expired
    if not found_counts:
        return ModelProgress(None, total, "request counts unavailable")
    return ModelProgress(processed, total, "Anthropic request counts")


def _gemini_model_progress(
    batch_jobs_by_name: dict[str, object],
    *,
    run_dir: Path | None,
    total: int | None,
) -> ModelProgress:
    metadata_destinations = _output_destinations_from_submit_run(run_dir)
    gcs_uris = [
        uri
        for batch_name, batch_job in batch_jobs_by_name.items()
        if (
            uri := _gemini_output_gcs_uri(
                batch_job,
                metadata_destination=metadata_destinations.get(batch_name),
            )
        )
    ]
    if not gcs_uris:
        return ModelProgress(None, total, "prediction output destination unavailable")
    try:
        processed, file_count = _count_gemini_prediction_rows(gcs_uris)
    except Exception as exc:
        return ModelProgress(None, total, f"prediction count unavailable: {type(exc).__name__}")
    return ModelProgress(processed, total, f"{file_count} prediction file(s)")


def _model_progress_line(progress: ModelProgress) -> str:
    if progress.processed is None:
        suffix = f" ({progress.detail})" if progress.detail else ""
        return f"Model outputs: unavailable{suffix}"
    if progress.total and progress.total > 0:
        ratio = (progress.processed / progress.total) * 100.0
        suffix = f" ({progress.detail})" if progress.detail else ""
        return (
            f"Model outputs: {progress.processed}/{progress.total} "
            f"({ratio:.2f}%){suffix}"
        )
    suffix = f" ({progress.detail})" if progress.detail else ""
    return f"Model outputs: {progress.processed}{suffix}"


def _fmt_time(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _batch_summary(batch_job: object, provider: str) -> str:
    if provider == "anthropic":
        batch_id = getattr(batch_job, "id", None)
        state = _normalize_state(getattr(batch_job, "processing_status", None))
        created_at = getattr(batch_job, "created_at", None)
        ended_at = getattr(batch_job, "ended_at", None)
        expires_at = getattr(batch_job, "expires_at", None)
        counts = getattr(batch_job, "request_counts", None)
        processing = int(getattr(counts, "processing", 0) or 0)
        succeeded = int(getattr(counts, "succeeded", 0) or 0)
        errored = int(getattr(counts, "errored", 0) or 0)
        canceled = int(getattr(counts, "canceled", 0) or 0)
        expired = int(getattr(counts, "expired", 0) or 0)
        lines = [
            f"Batch: {batch_id}",
            f"State: {state}",
            f"Created: {_fmt_time(created_at)}",
            f"Ended: {_fmt_time(ended_at)}",
            f"Expires: {_fmt_time(expires_at)}",
            (
                "Counts: "
                f"processing={processing}, succeeded={succeeded}, errored={errored}, "
                f"canceled={canceled}, expired={expired}"
            ),
        ]
        return "\n".join(lines)

    name = getattr(batch_job, "name", None)
    state = _normalize_state(getattr(batch_job, "state", None))
    model = getattr(batch_job, "model", None)
    create_time = getattr(batch_job, "create_time", None)
    update_time = getattr(batch_job, "update_time", None)

    dest = getattr(batch_job, "dest", None)
    dest_file_name = getattr(dest, "file_name", None) if dest else None
    dest_gcs_uri = getattr(dest, "gcs_uri", None) if dest else None
    error = getattr(batch_job, "error", None)

    lines = [
        f"Batch: {name}",
        f"State: {state}",
        f"Model: {model}",
        f"Created: {_fmt_time(create_time)}",
        f"Updated: {_fmt_time(update_time)}",
    ]
    if error:
        lines.append(f"Error: {error}")
    if dest_file_name:
        lines.append(f"Output file: {dest_file_name}")
    if dest_gcs_uri:
        lines.append(f"Output GCS: {dest_gcs_uri}")
    return "\n".join(lines)


def _provider_from_batch_names(
    batch_names: list[str],
    *,
    run_dir: Path | None,
) -> str:
    if run_dir is not None:
        payload = _read_batch_job_payload(run_dir / "batch_job.json")
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
        f"Unsupported provider for batch status: {model_provider}. "
        "Supported providers: gemini, anthropic."
    )


def _get_anthropic_client():
    try:
        import anthropic
    except Exception as exc:
        raise RuntimeError(
            "Anthropic provider requested but the 'anthropic' package is unavailable."
        ) from exc
    api_key = config.api_key_for_provider("anthropic")
    return anthropic.Anthropic(api_key=api_key)


def _get_client(provider: str, batch_names: list[str]):
    if provider == "anthropic":
        return _get_anthropic_client()

    client_location = (
        _extract_location_from_batch_name(batch_names[0])
        or (config.vertex_model_location or "").strip()
        or (config.gcp_location or "").strip()
        or None
    )
    return get_batch_client(location=client_location)


def _get_batch_job(client, batch_name: str, provider: str):
    if provider == "anthropic":
        return client.messages.batches.retrieve(batch_name)
    return client.batches.get(name=batch_name)


def _cancel_batch_job(client, batch_name: str, provider: str):
    if provider == "anthropic":
        return client.messages.batches.cancel(batch_name)
    return client.batches.cancel(name=batch_name)


def _batch_state(batch_job: object, provider: str) -> str:
    if provider == "anthropic":
        return _normalize_state(getattr(batch_job, "processing_status", None))
    return _normalize_state(getattr(batch_job, "state", None))


def main() -> None:
    args = _parse_args()
    batch_names, run_dir = _resolve_batch_names(args)
    if not batch_names:
        raise ValueError("No batch jobs resolved.")

    if args.cancel and args.watch:
        raise ValueError("--cancel and --watch cannot be used together.")

    provider = _provider_from_batch_names(batch_names, run_dir=run_dir)
    client = _get_client(provider, batch_names)
    terminal_states = _terminal_states(provider)
    submit_payload = _read_batch_job_payload(run_dir / "batch_job.json") if run_dir else None
    request_count = _request_count_from_payload(submit_payload)

    if args.cancel:
        cancelled = 0
        for batch_name in batch_names:
            batch_job = _get_batch_job(client, batch_name, provider)
            state = _batch_state(batch_job, provider)
            if state in terminal_states:
                print(f"Batch {batch_name} already terminal (state={state}); nothing to cancel.")
                continue
            _cancel_batch_job(client, batch_name, provider)
            print(f"Cancel requested for batch {batch_name}.")
            cancelled += 1
        if cancelled == 0:
            print("No non-terminal jobs to cancel.")
        return

    poll_interval = max(1, int(config.batch_poll_interval_seconds))

    if args.simple:
        last_digest: tuple | None = None
        while True:
            states: list[str] = []
            batch_jobs_by_name: dict[str, object] = {}
            all_terminal = True
            for batch_name in batch_names:
                batch_job = _get_batch_job(client, batch_name, provider)
                batch_jobs_by_name[batch_name] = batch_job
                state = _batch_state(batch_job, provider)
                states.append(state)
                if state not in terminal_states:
                    all_terminal = False

            lines, digest = _aggregate_state_lines(states, provider)
            if provider == "anthropic":
                progress = _anthropic_model_progress(
                    list(batch_jobs_by_name.values()),
                    request_count,
                )
            else:
                progress = _gemini_model_progress(
                    batch_jobs_by_name,
                    run_dir=run_dir,
                    total=request_count,
                )
            lines.append(_model_progress_line(progress))
            digest = (
                *digest,
                progress.processed,
                progress.total,
                progress.detail,
            )
            if digest != last_digest:
                print("\n".join(lines))
                print("")
                last_digest = digest

            if not args.watch or all_terminal:
                break
            time.sleep(poll_interval)
        return

    last_state_by_name: dict[str, str | None] = {}
    while True:
        all_terminal = True
        for batch_name in batch_names:
            batch_job = _get_batch_job(client, batch_name, provider)
            state = _batch_state(batch_job, provider)
            if state != last_state_by_name.get(batch_name):
                print(_batch_summary(batch_job, provider))
                print("")
                last_state_by_name[batch_name] = state
            if state not in terminal_states:
                all_terminal = False
        if not args.watch or all_terminal:
            break
        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
