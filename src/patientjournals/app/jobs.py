from __future__ import annotations

import argparse
import json
import os
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from patientjournals.app.models import (
    AppSettings,
    BatchChunkSummary,
    CommandSpec,
    DuplicateStrategy,
    JobSummary,
    SubmitJobDraft,
    app_settings_path,
)
from patientjournals.app.settings_store import (
    command_override_payload,
    write_command_overrides,
)
from patientjournals.config import config
from patientjournals.config.schemas import resolve_output_schema
from patientjournals.local.service import LocalRunRequest, LocalRunResult, run_local_job


@dataclass(frozen=True)
class RegisteredJob:
    job_id: str
    created_at: str
    kind: str
    status: str
    command: str
    config_path: str
    pid: int | None = None
    run_dir: str = ""
    detail: str = ""

    def summary(self) -> JobSummary:
        return JobSummary(
            job_id=self.job_id,
            source="app",
            kind=self.kind,
            status=self.status,
            created_at=self.created_at,
            run_dir=self.run_dir,
            command=self.command,
            detail=self.detail,
        )


class JobRegistry:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = (
            Path(path).expanduser()
            if path is not None
            else app_settings_path().parent / "jobs.json"
        )

    def list(self) -> list[RegisteredJob]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid app job registry: {self.path}") from exc
        if not isinstance(payload, list):
            raise ValueError(f"Invalid app job registry payload: {self.path}")
        jobs: list[RegisteredJob] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                jobs.append(RegisteredJob(**item))
            except TypeError:
                continue
        return jobs

    def save(self, jobs: Iterable[RegisteredJob]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([asdict(job) for job in jobs], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def add(self, job: RegisteredJob) -> None:
        jobs = self.list()
        jobs.insert(0, job)
        self.save(jobs)


def build_submit_command(
    draft: SubmitJobDraft,
    settings: AppSettings,
) -> CommandSpec:
    overrides = command_override_payload(
        settings,
        model_name=draft.model_name,
        schema_name=draft.schema_name,
        output_format=draft.output_format,
        local_path=draft.local_path,
        cloud_prefix=draft.cloud_prefix,
        cloud_prefixes=draft.cloud_prefixes,
    )

    if draft.run_mode == "local_api":
        args: list[str] = []
        if draft.local_path:
            args.extend(["--data-folder", draft.local_path])
        if draft.continue_dataset:
            args.extend(["--continue-dataset", draft.continue_dataset])
        return CommandSpec(
            module="patientjournals.local.cli",
            args=tuple(args),
            config_overrides=overrides,
        )

    args = []
    if draft.num_batches:
        args.extend(["--num-batches", str(draft.num_batches)])
    if draft.continue_dataset:
        args.extend(["--continue-dataset", draft.continue_dataset])
    return CommandSpec(
        module="patientjournals.batch.submit",
        args=tuple(args),
        config_overrides=overrides,
    )


def _apply_runtime_overrides(payload: dict[str, object]) -> dict[str, object]:
    keys = {
        "batch_backend",
        "gcp_auth_mode",
        "service_account_file",
        "gcp_project_id",
        "gcp_location",
        "vertex_model_location",
        "gcs_bucket_name",
        "gcs_pages_prefix",
        "batch_requests_gcs_prefix",
        "batch_outputs_gcs_prefix",
        "batch_input_prefix",
        "batch_input_prefixes",
        "target_folder",
        "upload_images_folder",
        "model",
        "output_format",
        "output_root",
        "output_model",
        "api_key",
        "provider_api_keys",
        "batch_duplicate_strategy",
        "batch_restrict_image_names",
    }
    previous = {key: getattr(config, key) for key in keys if hasattr(config, key)}
    aliases = {
        "auth_mode": "gcp_auth_mode",
        "local_runs_root": "output_root",
    }
    for key, value in payload.items():
        target_key = aliases.get(key, key)
        if target_key == "schema_name":
            if isinstance(value, str) and value.strip():
                config.output_model = resolve_output_schema(value)
            continue
        if hasattr(config, target_key) and value is not None:
            setattr(config, target_key, value)
    config.__post_init__()
    return previous


def _restore_runtime_overrides(previous: dict[str, object]) -> None:
    for key, value in previous.items():
        setattr(config, key, value)
    config.__post_init__()


async def run_local_draft_direct(
    draft: SubmitJobDraft,
    settings: AppSettings,
    *,
    progress_callback=None,
) -> LocalRunResult:
    if draft.run_mode != "local_api":
        raise ValueError("Direct app execution is currently supported for local_api only.")
    overrides = command_override_payload(
        settings,
        model_name=draft.model_name,
        schema_name=draft.schema_name,
        output_format=draft.output_format,
        local_path=draft.local_path,
        cloud_prefix=draft.cloud_prefix,
        cloud_prefixes=draft.cloud_prefixes,
    )
    previous = _apply_runtime_overrides(overrides)
    try:
        return await run_local_job(
            LocalRunRequest(
                data_folder=draft.local_path,
                continue_dataset=draft.continue_dataset,
            ),
            progress_callback=progress_callback,
        )
    finally:
        _restore_runtime_overrides(previous)


@dataclass(frozen=True)
class BatchSubmitOutcome:
    run_dir: str
    model: str
    chunk_count: int
    request_count: int
    batch_job_names: tuple[str, ...]
    status: str


def _summarize_batch_run(run_dir: Path) -> BatchSubmitOutcome:
    payload = _read_json_file(run_dir / "batch_job.json")
    chunks = _batch_chunk_summaries_from_payload(payload)
    request_count = sum(chunk.request_count for chunk in chunks) or int(
        payload.get("request_count") or 0
    )
    names = tuple(
        str(item.get("batch_job_name"))
        for item in (payload.get("batch_jobs") or [])
        if isinstance(item, dict) and item.get("batch_job_name")
    )
    return BatchSubmitOutcome(
        run_dir=str(run_dir),
        model=str(payload.get("model") or config.model or ""),
        chunk_count=len(chunks) or len(names),
        request_count=request_count,
        batch_job_names=names,
        status=_run_dir_status(run_dir),
    )


def _batch_submit_namespace(
    *,
    num_batches: int | None = None,
    continue_dataset: str = "",
    rerun: bool = False,
    run_dir: str = "",
) -> argparse.Namespace:
    return argparse.Namespace(
        num_batches=num_batches,
        rerun=rerun,
        run_dir=run_dir or None,
        continue_dataset=continue_dataset or None,
        downscale=None,
    )


def local_image_names(folder: str | Path, *, recursive: bool = True) -> set[str]:
    """Image file basenames in a local folder, matching batch input extensions.

    These are the exact images a local-folder batch submission should be scoped
    to, so it cannot fan out to the whole bucket prefix.
    """
    root = Path(folder).expanduser()
    if not folder or not root.exists() or not root.is_dir():
        return set()
    extensions = {
        f".{str(ext).lower().lstrip('.')}"
        for ext in (config.batch_input_extensions or ())
        if str(ext).strip()
    }
    paths = root.rglob("*") if recursive else root.glob("*")
    return {
        path.name
        for path in paths
        if path.is_file()
        and path.suffix.lower() in extensions
        and not path.name.startswith("._")
    }


def run_batch_draft_direct(
    draft: SubmitJobDraft,
    settings: AppSettings,
) -> BatchSubmitOutcome | None:
    """Submit a cloud batch in-process so the run directory is captured immediately."""
    if draft.run_mode != "cloud_batch":
        raise ValueError("run_batch_draft_direct only supports cloud_batch drafts.")
    from patientjournals.batch.submit import submit_batch

    overrides = command_override_payload(
        settings,
        model_name=draft.model_name,
        schema_name=draft.schema_name,
        output_format=draft.output_format,
        local_path=draft.local_path,
        cloud_prefix=draft.cloud_prefix,
        cloud_prefixes=draft.cloud_prefixes,
    )
    # Scope a local-folder batch to exactly the images in that folder so it can
    # never expand to every object under the bucket prefix.
    if draft.dataset_source == "local" and draft.local_path:
        names = local_image_names(draft.local_path, recursive=True)
        if not names:
            raise FileNotFoundError(
                f"No batch input images found in local folder '{draft.local_path}'. "
                "Select a folder containing image files for a local batch submission."
            )
        overrides["batch_restrict_image_names"] = tuple(sorted(names))
    previous = _apply_runtime_overrides(overrides)
    try:
        run_dir = submit_batch(
            args=_batch_submit_namespace(
                num_batches=draft.num_batches,
                continue_dataset=draft.continue_dataset,
            )
        )
        if run_dir is None:
            return None
        return _summarize_batch_run(Path(run_dir))
    finally:
        _restore_runtime_overrides(previous)


def run_batch_rerun_direct(
    run_dir: str | Path,
    settings: AppSettings,
) -> BatchSubmitOutcome:
    """Resubmit the chunks of an existing run that did not complete successfully."""
    from patientjournals.batch.submit import submit_batch

    overrides = command_override_payload(settings)
    previous = _apply_runtime_overrides(overrides)
    try:
        result_dir = submit_batch(
            args=_batch_submit_namespace(rerun=True, run_dir=str(run_dir))
        )
        return _summarize_batch_run(Path(result_dir or run_dir))
    finally:
        _restore_runtime_overrides(previous)


BATCH_RESULTS_FILE = "batch_results.json"


def run_retrieve_direct(
    run_dir: str | Path,
    settings: AppSettings,
    *,
    allow_partial: bool = False,
    recover_missing_with_api: bool = False,
    submit_failed: bool = False,
    duplicate_strategy: str = "",
    record_results: bool = True,
) -> dict[str, object]:
    """Retrieve a submitted batch in-process and record the result on the submit run.

    When ``record_results`` is true, writes ``batch_results.json`` into the submit
    run directory so the job row can show success/failed counts. Returns the
    results payload.
    """
    from patientjournals.batch.service import BatchResultService, BatchRetrieveRequest

    effective_strategy = (
        duplicate_strategy
        or settings.batch_duplicate_strategy
        or "first_successful"
    )
    overrides = command_override_payload(
        settings,
        duplicate_strategy=str(effective_strategy),
    )
    previous = _apply_runtime_overrides(overrides)
    try:
        result = BatchResultService().retrieve(
            BatchRetrieveRequest(
                run_dir=str(run_dir),
                allow_partial=allow_partial,
                recover_missing_with_api=recover_missing_with_api,
                submit_failed=submit_failed,
                duplicate_strategy=str(effective_strategy),
            )
        )
    finally:
        _restore_runtime_overrides(previous)

    payload = {
        "retrieved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_path": str(result.dataset_path),
        "provider": result.provider,
        "batch_count": result.batch_count,
        "rows_written": result.rows_written,
        "error_rows": result.error_rows,
        "expected_pages": result.expected_pages,
        "observed_pages": result.observed_pages,
        "successful_pages": result.successful_pages,
        "recovered_pages": result.recovered_pages,
        "missing_pages": max(0, result.expected_pages - result.successful_pages),
        "submit_failed": bool(submit_failed),
    }
    if record_results:
        results_path = Path(run_dir).expanduser() / BATCH_RESULTS_FILE
        results_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return payload


def resubmit_failed_requests(
    run_dir: str | Path,
    settings: AppSettings,
) -> dict[str, object]:
    """Resubmit the requests that did not succeed as a fresh batch.

    Clears the recorded results so the job returns to a pending state until the
    resubmitted work is retrieved again.
    """
    payload = run_retrieve_direct(
        run_dir,
        settings,
        allow_partial=True,
        submit_failed=True,
        record_results=False,
    )
    results_path = Path(run_dir).expanduser() / BATCH_RESULTS_FILE
    results_path.unlink(missing_ok=True)
    return payload


def cancel_batch_run(run_dir: str | Path, settings: AppSettings) -> int:
    """Cancel every non-terminal batch job belonging to a submit run.

    Returns the number of jobs for which cancellation was requested.
    """
    from patientjournals.batch import status as status_module

    chunks = list_batch_chunks(run_dir)
    names = [chunk.batch_job_name for chunk in chunks if chunk.batch_job_name]
    if not names:
        return 0
    overrides = command_override_payload(settings)
    previous = _apply_runtime_overrides(overrides)
    cancelled = 0
    try:
        run_path = Path(run_dir).expanduser()
        provider = status_module._provider_from_batch_names(names, run_dir=run_path)
        client = status_module._get_client(provider, names)
        terminal = status_module._terminal_states(provider)
        for name in names:
            try:
                batch_job = status_module._get_batch_job(client, name, provider)
                if status_module._batch_state(batch_job, provider) in terminal:
                    continue
                status_module._cancel_batch_job(client, name, provider)
                cancelled += 1
            except Exception:  # noqa: BLE001
                continue
    finally:
        _restore_runtime_overrides(previous)
    return cancelled


def read_run_error(run_dir: str | Path) -> str:
    """Return the text of any locally written error file for a run, if present."""
    run_path = Path(run_dir).expanduser()
    for error_file in sorted(run_path.glob("error_*.txt")):
        try:
            text = error_file.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text:
            return text
    return ""


def read_dataset_preview(
    dataset_path: str | Path,
    *,
    limit: int = 20,
) -> tuple[list[str], list[dict[str, object]]]:
    """Read up to ``limit`` rows from a dataset for a quick on-screen preview.

    Returns (column_names, rows). Supports JSONL and CSV datasets.
    """
    path = Path(dataset_path).expanduser()
    if not path.exists() or not path.is_file():
        return [], []

    rows: list[dict[str, object]] = []
    columns: list[str] = []
    suffix = path.suffix.lower()
    if suffix == ".csv":
        import csv

        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            for index, row in enumerate(reader):
                if index >= limit:
                    break
                rows.append(dict(row))
    else:
        with open(path, "r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    rows.append(record)
                    for key in record:
                        if key not in columns:
                            columns.append(key)
    return columns, rows


def build_retrieve_command(
    settings: AppSettings,
    *,
    run_dir: str = "",
    batch_name: str = "",
    batch_names: tuple[str, ...] = (),
    output_format: str = "",
    allow_partial: bool = False,
    wait: bool = False,
    recover_missing_with_api: bool = False,
    submit_failed: bool = False,
    duplicate_strategy: DuplicateStrategy | str = "",
) -> CommandSpec:
    effective_strategy = (
        duplicate_strategy
        or settings.batch_duplicate_strategy
        or "first_successful"
    )
    overrides = command_override_payload(
        settings,
        output_format=output_format,
        duplicate_strategy=str(effective_strategy),
    )
    args: list[str] = []
    if run_dir:
        args.extend(["--run-dir", run_dir])
    selected_batch_names = [name for name in (*batch_names, batch_name) if name]
    for name in selected_batch_names:
        args.extend(["--batch-name", name])
    if allow_partial:
        args.append("--allow-partial")
    if wait:
        args.append("--wait")
    if recover_missing_with_api:
        args.append("--recover-missing-with-api")
    if submit_failed:
        args.append("--submit-failed")
    if effective_strategy:
        args.extend(["--duplicate-strategy", str(effective_strategy)])
    return CommandSpec(
        module="patientjournals.batch.retrieve",
        args=tuple(args),
        config_overrides=overrides,
    )


def build_validation_command(
    settings: AppSettings,
    *,
    images: str,
    results: str,
    username: str,
    corrections: bool = False,
) -> CommandSpec:
    overrides = command_override_payload(settings)
    args = ["--user", username, "--images", images, "--results", results]
    if corrections:
        args.append("--corrections")
    return CommandSpec(
        module="patientjournals.validation.cli",
        args=tuple(args),
        config_overrides=overrides,
    )


def start_command(
    command: CommandSpec,
    *,
    registry: JobRegistry | None = None,
    kind: str = "command",
) -> RegisteredJob:
    job_id = uuid.uuid4().hex[:12]
    config_path = write_command_overrides(
        command.config_overrides,
        stem=f"job_{job_id}",
    )
    env = os.environ.copy()
    env["PATIENTJOURNALS_CONFIG_JSON"] = str(config_path)
    process = subprocess.Popen(command.argv(), env=env)  # noqa: S603
    job = RegisteredJob(
        job_id=job_id,
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        kind=kind,
        status="running",
        command=command.display(),
        config_path=str(config_path),
        pid=process.pid,
    )
    (registry or JobRegistry()).add(job)
    return job


def _read_json_file(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _batch_chunk_summaries_from_payload(payload: dict) -> list[BatchChunkSummary]:
    jobs = payload.get("batch_jobs")
    if not isinstance(jobs, list):
        return []

    summaries: list[BatchChunkSummary] = []
    for item in jobs:
        if not isinstance(item, dict):
            continue
        batch_job_name = str(item.get("batch_job_name") or "")
        if not batch_job_name:
            continue
        chunk_index = int(item.get("chunk_index") or len(summaries) + 1)
        total_chunks = int(item.get("total_chunks") or len(jobs) or 1)
        summaries.append(
            BatchChunkSummary(
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                chunk_label=str(
                    item.get("chunk_label")
                    or f"chunk_{chunk_index:03d}_of_{total_chunks:03d}"
                ),
                batch_job_name=batch_job_name,
                request_count=int(item.get("request_count") or 0),
                output_destination=str(item.get("output_destination") or ""),
                requests_file=str(item.get("requests_file") or ""),
                provider=str(item.get("provider") or payload.get("provider") or ""),
            )
        )
    return sorted(summaries, key=lambda item: item.chunk_index)


def list_batch_chunks(run_dir: str | Path) -> list[BatchChunkSummary]:
    payload = _read_json_file(Path(run_dir).expanduser() / "batch_job.json")
    return _batch_chunk_summaries_from_payload(payload)


def list_batch_chunks_with_state(run_dir: str | Path) -> list[BatchChunkSummary]:
    chunks = list_batch_chunks(run_dir)
    if not chunks:
        return []

    from concurrent.futures import ThreadPoolExecutor, as_completed

    from patientjournals.batch import status as status_module

    batch_names = [chunk.batch_job_name for chunk in chunks]
    run_path = Path(run_dir).expanduser()
    provider = status_module._provider_from_batch_names(batch_names, run_dir=run_path)
    client = status_module._get_client(provider, batch_names)
    workers = max(1, min(len(batch_names), int(config.api_concurrent_tasks or 1)))

    state_by_name: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(status_module._get_batch_job, client, name, provider): name
            for name in batch_names
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                batch_job = future.result()
                state_by_name[name] = status_module._batch_state(batch_job, provider)
            except Exception as exc:  # noqa: BLE001
                state_by_name[name] = f"error:{type(exc).__name__}"

    return [
        BatchChunkSummary(
            chunk_index=chunk.chunk_index,
            total_chunks=chunk.total_chunks,
            chunk_label=chunk.chunk_label,
            batch_job_name=chunk.batch_job_name,
            request_count=chunk.request_count,
            status=state_by_name.get(chunk.batch_job_name, chunk.status),
            output_destination=chunk.output_destination,
            requests_file=chunk.requests_file,
            provider=chunk.provider,
        )
        for chunk in chunks
    ]


def _is_success_state(state: str) -> bool:
    return "SUCCEEDED" in state or state == "ENDED"


def _is_failure_state(state: str) -> bool:
    return any(
        token in state for token in ("FAILED", "CANCELLED", "CANCELED", "EXPIRED")
    )


def aggregate_batch_state(chunks: list[BatchChunkSummary]) -> str:
    """Reduce per-chunk live states into a single job-level status.

    Returns "succeeded", "failed", or "running"; "" when the state cannot be
    determined (no chunks, or any chunk lookup errored).
    """
    states = [chunk.status.upper() for chunk in chunks]
    if not states or any(state.startswith("ERROR:") for state in states):
        return ""
    if all(_is_success_state(state) or _is_failure_state(state) for state in states):
        return "failed" if any(_is_failure_state(state) for state in states) else "succeeded"
    return "running"


def resolve_batch_run_state(run_dir: str | Path) -> str:
    """Query the batch API once and aggregate chunk states into a job-level status.

    Returns "" when the state cannot be determined (no credentials, network
    error) so callers can fall back to the on-disk status.
    """
    try:
        chunks = list_batch_chunks_with_state(run_dir)
    except Exception:  # noqa: BLE001
        return ""
    return aggregate_batch_state(chunks)


def poll_local_batch_states(
    settings: AppSettings,
    run_root: str | Path | None = None,
) -> dict[str, str]:
    """One-shot API poll mapping each unfinished local batch run_dir to a live status.

    Applies the app settings as runtime config so the batch client authenticates
    the same way submission does. Only runs that look unfinished on disk are polled.
    """
    candidates = [
        job
        for job in list_submit_jobs(run_root or settings.local_runs_root)
        if job.run_dir
        and not job.retrieved
        and job.status in {"submitted", "unknown", "running"}
    ]
    if not candidates:
        return {}
    overrides = command_override_payload(settings)
    previous = _apply_runtime_overrides(overrides)
    states: dict[str, str] = {}
    try:
        for job in candidates:
            state = resolve_batch_run_state(job.run_dir)
            if state:
                states[job.run_dir] = state
    finally:
        _restore_runtime_overrides(previous)
    return states


def _run_dir_status(run_dir: Path) -> str:
    if any(run_dir.glob("error_*.txt")):
        return "failed"
    if (run_dir / "batch_job.json").exists():
        return "submitted"
    if any(run_dir.glob("*_dataset.jsonl")) or any(run_dir.glob("*_dataset.csv")):
        return "finished"
    return "unknown"


# Operational run directories that are steps performed on a job, not jobs
# themselves; they only clutter the Jobs list.
_OPERATIONAL_RUN_PREFIXES = ("retrieve_", "collect_outputs_")


def _has_dataset(run_dir: Path) -> bool:
    return any(run_dir.glob("*_dataset.jsonl")) or any(run_dir.glob("*_dataset.csv"))


def _describe_input_location(run_dir: Path, batch_meta: dict) -> str:
    """A human-readable description of where a submit job's inputs came from."""
    metadata = _read_json_file(run_dir / "metadata.json")
    cfg = ((metadata.get("config_values") or {}).get("config")) or {}
    bucket = str(cfg.get("gcs_bucket_name") or batch_meta.get("gcs_bucket_name") or "")
    pages = str(cfg.get("gcs_pages_prefix") or "").strip("/")
    restrict = cfg.get("batch_restrict_image_names") or []
    prefixes = [
        str(prefix).strip("/")
        for prefix in (cfg.get("batch_input_prefixes") or [])
        if str(prefix).strip()
    ]
    if not prefixes and cfg.get("batch_input_prefix"):
        prefixes = [str(cfg.get("batch_input_prefix")).strip("/")]
    target = str(cfg.get("target_folder") or "")

    def gs(prefix: str) -> str:
        if bucket:
            return f"gs://{bucket}/{prefix}".rstrip("/")
        return prefix or "(bucket prefix)"

    if restrict:
        base = Path(target).name or target or "local folder"
        return f"{base} — {len(restrict)} scoped image(s) in {gs(pages)}"
    if prefixes:
        return ", ".join(gs(prefix) for prefix in prefixes)
    return gs(pages)


def list_submit_jobs(run_root: str | Path | None = None) -> list[JobSummary]:
    """One row per batch submission. Retrieval is folded into the same row.

    Only directories that actually submitted a batch (have ``batch_job.json``)
    are returned — operational and incomplete runs are ignored, so a job appears
    exactly once across its whole submit → retrieve → rerun lifecycle.
    """
    root = Path(run_root or config.output_root).expanduser()
    if not root.exists() or not root.is_dir():
        return []
    summaries: list[JobSummary] = []
    for run_dir in sorted((item for item in root.iterdir() if item.is_dir()), reverse=True):
        if run_dir.name.startswith(_OPERATIONAL_RUN_PREFIXES):
            continue
        batch_meta = _read_json_file(run_dir / "batch_job.json")
        if not batch_meta:
            continue

        metadata = _read_json_file(run_dir / "metadata.json")
        created_at = str(metadata.get("created_at") or batch_meta.get("created_at") or "")
        model = str(batch_meta.get("model") or "")
        chunks = _batch_chunk_summaries_from_payload(batch_meta)
        chunk_count = len(chunks) or len(batch_meta.get("batch_job_names") or [])
        images = sum(chunk.request_count for chunk in chunks) or int(
            batch_meta.get("request_count") or 0
        )
        location = _describe_input_location(run_dir, batch_meta)

        results = _read_json_file(run_dir / BATCH_RESULTS_FILE)
        retrieved = bool(results)
        succeeded: int | None = None
        failed: int | None = None
        status = _run_dir_status(run_dir)
        if retrieved:
            succeeded = int(results.get("successful_pages") or 0)
            expected = int(results.get("expected_pages") or images or 0)
            failed = max(0, expected - succeeded)
            status = "retrieved"

        detail = f"{images} image(s)"
        if chunk_count:
            detail = f"{detail}, {chunk_count} chunk(s)"

        summaries.append(
            JobSummary(
                job_id=run_dir.name,
                source="local",
                kind="batch",
                status=status,
                created_at=created_at,
                model=model,
                run_dir=str(run_dir),
                detail=detail,
                input_location=location,
                image_count=images,
                chunk_count=chunk_count,
                retrieved=retrieved,
                succeeded=succeeded,
                failed=failed,
            )
        )
    return summaries


# Submits are now tracked authoritatively by their run directory (see
# list_local_run_jobs), so the legacy subprocess registry rows for these kinds
# are duplicates that otherwise linger as perpetual "running" entries.
_RUN_DIR_TRACKED_KINDS = {"cloud_batch", "local_api", "batch", "submit"}


def list_app_registry_jobs(registry: JobRegistry | None = None) -> list[JobSummary]:
    summaries: list[JobSummary] = []
    for job in (registry or JobRegistry()).list():
        if job.kind in _RUN_DIR_TRACKED_KINDS and not job.run_dir:
            continue
        summaries.append(job.summary())
    return summaries


def list_cloud_batch_jobs(*, limit: int = 100) -> list[JobSummary]:
    from patientjournals.batch.client import get_batch_client
    from patientjournals.batch.status import _normalize_state

    client = get_batch_client()
    batches = getattr(client, "batches", None)
    list_fn = getattr(batches, "list", None)
    if not callable(list_fn):
        raise RuntimeError("The configured Google client does not expose batches.list().")

    jobs: list[JobSummary] = []
    for index, batch in enumerate(list_fn(), start=1):
        if index > limit:
            break
        name = str(getattr(batch, "name", "") or f"cloud-{index}")
        jobs.append(
            JobSummary(
                job_id=name,
                source="cloud",
                kind="batch",
                status=_normalize_state(getattr(batch, "state", None)),
                created_at=str(getattr(batch, "create_time", "") or ""),
                model=str(getattr(batch, "model", "") or ""),
                detail=str(getattr(batch, "display_name", "") or ""),
            )
        )
    return jobs
