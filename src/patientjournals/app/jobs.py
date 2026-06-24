from __future__ import annotations

import argparse
import json
import os
import subprocess
import uuid
from collections import Counter
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
from patientjournals.config.models import resolve_model_spec
from patientjournals.config.schemas import resolve_output_schema
from patientjournals.local.service import LocalRunRequest, LocalRunResult, run_local_job
from patientjournals.shared.dataset_coverage import load_dataset_image_coverage
from patientjournals.shared.identity import image_name_from_reference
from patientjournals.shared.processing_metrics import (
    MANIFEST_FILE_NAME,
    read_processing_records,
)
from patientjournals.shared import run_layout


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
        "datasets_gcs_prefix",
        "upload_dataset_to_gcs",
        "validations_gcs_prefix",
        "upload_validation_to_gcs",
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
        "api_recovery_enabled",
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


@dataclass(frozen=True)
class BatchRunReadiness:
    state: str
    detail: str = ""
    output_rows: int | None = None
    expected_rows: int | None = None


def _summarize_batch_run(run_dir: Path) -> BatchSubmitOutcome:
    payload = _read_json_file(run_dir / "batch_job.json")
    chunks = _batch_chunk_summaries_from_payload(payload)
    request_count = _primary_request_count_from_payload(payload)
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


def _api_recovery_error_rows(failures: dict[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key, reason in sorted(failures.items()):
        image_name = image_name_from_reference(key) or (Path(key).name if key else "")
        rows.append(
            {
                "image_name": image_name or key,
                "key": key,
                "failure_reason": str(reason or "api_key_recovery_failed:unknown"),
            }
        )
    return rows


def _api_recovery_error_summary(error_rows: list[dict[str, str]]) -> str:
    if not error_rows:
        return ""
    counts = Counter(row.get("failure_reason") or "unknown" for row in error_rows)
    summary = ", ".join(f"{count} {reason}" for reason, count in counts.most_common())
    sample = "; ".join(
        f"{row.get('image_name') or row.get('key')}: {row.get('failure_reason')}"
        for row in error_rows[:5]
    )
    if len(error_rows) > 5:
        sample = f"{sample}; +{len(error_rows) - 5} more"
    return f"API recovery failed for {len(error_rows)} page(s): {summary}. {sample}"


def run_retrieve_direct(
    run_dir: str | Path,
    settings: AppSettings,
    *,
    allow_partial: bool = False,
    recover_missing_with_api: bool = False,
    submit_failed: bool = False,
    failed_retry_num_batches: int | None = None,
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
    overrides["api_recovery_enabled"] = bool(recover_missing_with_api)
    previous = _apply_runtime_overrides(overrides)
    try:
        result = BatchResultService().retrieve(
            BatchRetrieveRequest(
                run_dir=str(run_dir),
                output_dir=str(run_dir),
                allow_partial=allow_partial,
                recover_missing_with_api=recover_missing_with_api,
                submit_failed=submit_failed,
                failed_retry_num_batches=failed_retry_num_batches,
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
        "dataset_gcs_uri": result.dataset_gcs_uri,
        "submit_failed": bool(submit_failed),
    }
    if recover_missing_with_api:
        payload["api_recovery_attempted"] = True
        payload["api_recovery_completed"] = True
        payload["api_recovered_row_count"] = int(result.recovered_pages or 0)
        payload["api_recovered_rows"] = []
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
    *,
    num_batches: int = 1,
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
        failed_retry_num_batches=num_batches,
        record_results=False,
    )
    results_path = Path(run_dir).expanduser() / BATCH_RESULTS_FILE
    results_path.unlink(missing_ok=True)
    return payload


def batch_run_provider(run_dir: str | Path) -> str:
    """Return the provider ("gemini"/"anthropic") recorded for a submit run."""
    meta = _read_json_file(Path(run_dir).expanduser() / "batch_job.json")
    provider = str(meta.get("provider") or "").strip().lower()
    if provider in {"gemini", "anthropic"}:
        return provider

    model = str(meta.get("model") or "").strip()
    if model:
        try:
            model_provider = resolve_model_spec(model).provider
        except ValueError:
            lowered = model.lower()
            if "gemini" in lowered:
                return "gemini"
            if "claude" in lowered or "anthropic" in lowered:
                return "anthropic"
        else:
            if model_provider in {"gemini", "anthropic"}:
                return model_provider

    names = [
        str(item.get("batch_job_name") or "")
        for item in (meta.get("batch_jobs") or [])
        if isinstance(item, dict) and item.get("batch_job_name")
    ]
    if not names and meta.get("batch_job_name"):
        names = [str(meta.get("batch_job_name"))]
    if names and all(name.startswith("msgbatch_") for name in names):
        return "anthropic"
    if names and all(
        name.startswith("projects/")
        or "/locations/" in name
        or name.startswith("batches/")
        for name in names
    ):
        return "gemini"
    return ""


def recover_failed_via_api(
    run_dir: str | Path,
    settings: AppSettings,
) -> dict[str, object]:
    """Fill in failed/missing pages with synchronous API calls and record results.

    This completes the dataset in one pass without submitting a new batch. It is
    appropriate for a small number of failures; large counts should use a rerun
    batch instead. Gemini-only.
    """
    return run_retrieve_direct(
        run_dir,
        settings,
        allow_partial=True,
        recover_missing_with_api=True,
    )


def recover_dataset_gaps(
    run_dir: str | Path,
    settings: AppSettings,
) -> dict[str, object]:
    """Recover only the pages genuinely missing from the existing dataset via API.

    Unlike a full retrieve (which re-derives every batch failure from scratch and
    redoes work already recovered), this targets exactly the expected pages that
    are absent from the current dataset, recovers them with concurrent API calls,
    and appends the new rows. Falls back to a full recover-retrieve if the job has
    not been retrieved yet or expected keys cannot be resolved.
    """
    from patientjournals.batch import retrieve as retrieve_module
    from patientjournals.shared.identity import image_name_from_reference
    from patientjournals.shared.tools import (
        flush_rows,
        get_run_logger,
        load_existing_dataset,
    )

    recorded = read_recorded_results(run_dir)
    dataset_path = find_dataset_near(recorded.get("dataset_path") or "")
    if not dataset_path:
        return recover_failed_via_api(run_dir, settings)

    overrides = command_override_payload(settings)
    previous = _apply_runtime_overrides(overrides)
    try:
        run_path = Path(run_dir).expanduser()
        log = get_run_logger(run_path)
        batch_meta = _read_json_file(run_path / "batch_job.json")
        batch_names = [
            str(item.get("batch_job_name"))
            for item in (batch_meta.get("batch_jobs") or [])
            if item.get("batch_job_name")
        ]
        output_format, covered_names, existing_count = load_existing_dataset(dataset_path)
        expected_keys = retrieve_module._resolve_expected_request_keys(
            submit_run_dir=run_path,
            batch_names=batch_names,
            selected_batch_names=batch_names,
            log=log,
        )
        if not expected_keys:
            # Cannot determine the page universe; fall back to a full retrieve.
            return recover_failed_via_api(run_dir, settings)

        def basename(key: str) -> str:
            return image_name_from_reference(key) or key

        missing_keys = {
            key for key in expected_keys if basename(key) not in covered_names
        }
        log(
            f"Incremental API recovery: {len(missing_keys)} of {len(expected_keys)} "
            f"expected page(s) missing from existing dataset."
        )

        recovered = 0
        recovered_rows: list[dict] = []
        api_failures: dict[str, str] = {}
        if missing_keys:
            rows_to_flush: list[dict] = []
            recovered = retrieve_module._recover_missing_pages_via_api_key(
                missing_keys=missing_keys,
                successful_keys=set(),
                observed_output_keys=set(),
                failures=api_failures,
                rows_to_flush=rows_to_flush,
                log=log,
                force=True,
                manifest_path=None,
            )
            recovered_rows = [dict(row) for row in rows_to_flush]
            if rows_to_flush:
                flush_rows(
                    rows=rows_to_flush,
                    out_path=str(dataset_path),
                    header_written=True,
                    output_format=output_format,
                )

        recovered_at = datetime.now().astimezone().isoformat(timespec="seconds")
        previous_recovered = int(recorded.get("recovered_pages") or 0)
        recovered_total = previous_recovered + recovered
        recovery_history = list(recorded.get("recovery_history") or [])
        if recovered:
            recovery_history.append(
                {
                    "recovered_at": recovered_at,
                    "recovered_pages": recovered,
                    "method": "api",
                }
            )
        dataset_gcs_uri = retrieve_module._upload_dataset_to_gcs(
            Path(dataset_path),
            run_path.name,
            log,
        ) or str(recorded.get("dataset_gcs_uri") or "")

        successful = len(covered_names) + recovered
        missing_after_recovery = max(0, len(missing_keys) - recovered)
        api_error_rows = _api_recovery_error_rows(api_failures)
        if missing_after_recovery and not api_error_rows:
            recovered_names = {
                str(row.get("image_name") or "")
                for row in recovered_rows
                if isinstance(row, dict)
            }
            unresolved = [
                key
                for key in sorted(missing_keys)
                if basename(key) not in recovered_names
            ]
            api_error_rows = _api_recovery_error_rows(
                {
                    key: "api_key_recovery_failed:no_recovered_row"
                    for key in unresolved[:missing_after_recovery]
                }
            )
        api_failed = bool(missing_after_recovery)
        payload = {
            "retrieved_at": recovered_at,
            "dataset_path": str(dataset_path),
            "dataset_gcs_uri": dataset_gcs_uri,
            "provider": str(batch_meta.get("provider") or ""),
            "batch_count": len(batch_names),
            "rows_written": successful,
            "error_rows": max(0, len(missing_keys) - recovered),
            "expected_pages": len(expected_keys),
            "observed_pages": len(expected_keys),
            "successful_pages": successful,
            "recovered_pages": recovered_total,
            "missing_pages": missing_after_recovery,
            "submit_failed": False,
            "recovery_history": recovery_history,
            "api_recovery_attempted": True,
            "api_recovery_completed": True,
            "api_recovery_failed": api_failed,
            "api_recovery_errors": api_error_rows,
            "api_recovery_error_summary": _api_recovery_error_summary(api_error_rows),
            "api_recovered_rows": recovered_rows,
            "api_recovered_row_count": len(recovered_rows),
        }
        (run_path / BATCH_RESULTS_FILE).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return payload
    finally:
        _restore_runtime_overrides(previous)


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
    diagnostics = run_failure_diagnostics(run_path)
    if diagnostics:
        return diagnostics
    return ""


def _dataset_files_in_run_dir(run_dir: Path) -> list[Path]:
    return sorted(
        [
            *run_dir.glob("*_dataset.jsonl"),
            *run_dir.glob("*_dataset.csv"),
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _read_request_keys_from_run_dir(run_dir: Path, batch_meta: dict) -> set[str]:
    request_files: list[str] = []
    jobs = batch_meta.get("batch_jobs")
    if isinstance(jobs, list):
        for item in jobs:
            if not isinstance(item, dict):
                continue
            value = item.get("requests_file")
            if isinstance(value, str) and value.strip():
                request_files.append(value.strip())
    direct = batch_meta.get("requests_file")
    if isinstance(direct, str) and direct.strip():
        request_files.append(direct.strip())
    if not request_files:
        request_files.append(config.batch_requests_file_name)

    keys: set[str] = set()
    for file_name in dict.fromkeys(request_files):
        path = run_dir / file_name
        if not path.exists() or not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                key = payload.get("key")
                if isinstance(key, str) and key.strip():
                    keys.add(key.strip())
    return keys


def _count_output_rows(run_dir: Path) -> int:
    total = 0
    for path in sorted(run_dir.glob("batch_output_*.jsonl")):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                total += sum(1 for line in handle if line.strip())
        except OSError:
            continue
    return total


def _derived_results_from_artifacts(run_dir: str | Path) -> dict:
    run_path = Path(run_dir).expanduser()
    batch_meta = _read_json_file(run_path / "batch_job.json")
    if not batch_meta:
        return {}
    dataset_files = _dataset_files_in_run_dir(run_path)
    if not dataset_files:
        return {}
    dataset_path = dataset_files[0]
    try:
        _fmt, covered_names, row_count = load_dataset_image_coverage(
            dataset_path,
            csv_sep=config.csv_sep,
            bucket_name=str(batch_meta.get("gcs_bucket_name") or config.gcs_bucket_name),
        )
    except Exception:  # noqa: BLE001
        return {}

    request_keys = _read_request_keys_from_run_dir(run_path, batch_meta)
    expected_names = {
        image_name_from_reference(key)
        for key in request_keys
        if image_name_from_reference(key)
    }
    request_count = int(batch_meta.get("request_count") or 0)
    expected_pages = len(expected_names) or request_count or len(covered_names)
    successful_pages = (
        len(covered_names & expected_names)
        if expected_names
        else len(covered_names)
    )
    observed_pages = _count_output_rows(run_path) or expected_pages
    missing_pages = max(0, expected_pages - successful_pages)
    chunks = _batch_chunk_summaries_from_payload(batch_meta)
    return {
        "retrieved_at": datetime.fromtimestamp(
            dataset_path.stat().st_mtime
        ).astimezone().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path),
        "provider": str(batch_meta.get("provider") or ""),
        "batch_count": len(chunks) or len(batch_meta.get("batch_job_names") or []),
        "rows_written": row_count,
        "error_rows": max(0, observed_pages - successful_pages),
        "expected_pages": expected_pages,
        "observed_pages": observed_pages,
        "successful_pages": successful_pages,
        "recovered_pages": 0,
        "missing_pages": missing_pages,
        "submit_failed": False,
        "result_inferred": True,
        "inference_note": (
            "Derived from dataset, request file, and downloaded batch output "
            "because batch_results.json was missing."
        ),
    }


def run_failure_diagnostics(run_dir: str | Path, *, limit: int = 12) -> str:
    run_path = Path(run_dir).expanduser()
    manifest_path = run_path / MANIFEST_FILE_NAME
    records = read_processing_records(manifest_path)
    if not records:
        return ""

    seen: set[tuple[str, str]] = set()
    reason_counts: Counter[str] = Counter()
    examples: list[str] = []
    for record in records:
        if str(record.get("status") or "") != "failed":
            continue
        reason = str(record.get("failure_reason") or "unknown")
        image_name = str(record.get("image_name") or record.get("image_reference") or "")
        key = (image_name, reason)
        if key in seen:
            continue
        seen.add(key)
        reason_counts[reason] += 1
        if len(examples) < limit:
            examples.append(f"{image_name or '(unknown image)'}: {reason}")

    if not reason_counts:
        return ""

    counts = ", ".join(
        f"{reason}={count}" for reason, count in reason_counts.most_common()
    )
    lines = [f"Processing failure codes: {counts}"]
    if examples:
        lines.append("Examples:")
        lines.extend(f"- {item}" for item in examples)
    return "\n".join(lines)


def read_recorded_results(run_dir: str | Path) -> dict:
    """Return the recorded retrieval results for a run, if it has been retrieved."""
    run_path = Path(run_dir).expanduser()
    return _read_json_file(run_path / BATCH_RESULTS_FILE) or _derived_results_from_artifacts(
        run_path
    )


def repair_recorded_results(run_dir: str | Path) -> dict:
    """Persist derived results when a dataset exists but batch_results.json is absent."""
    run_path = Path(run_dir).expanduser()
    existing = _read_json_file(run_path / BATCH_RESULTS_FILE)
    if existing:
        return existing
    derived = _derived_results_from_artifacts(run_path)
    if derived:
        (run_path / BATCH_RESULTS_FILE).write_text(
            json.dumps(derived, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return derived


def repair_all_missing_recorded_results(
    run_root: str | Path | None = None,
) -> int:
    root = Path(run_root or config.output_root).expanduser()
    repaired = 0
    for run_dir in run_layout.iter_run_dirs(root, "submit"):
        if (run_dir / BATCH_RESULTS_FILE).exists():
            continue
        if repair_recorded_results(run_dir):
            repaired += 1
    return repaired


def find_dataset_near(reference: str | Path) -> str:
    """Locate a dataset file at ``reference`` or, failing that, in its directory.

    Returns the resolved path as a string, or "" if none is found.
    """
    if not reference:
        return ""
    path = Path(reference).expanduser()
    if path.is_file():
        return str(path)
    search_dir = path if path.is_dir() else path.parent
    if not search_dir.is_dir():
        return ""
    for pattern in ("*_dataset.jsonl", "*_dataset.csv", "*.jsonl", "*.csv"):
        matches = sorted(search_dir.glob(pattern))
        if matches:
            return str(matches[0])
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


def _write_json_file(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _stored_batch_status(payload: dict, item: dict) -> str:
    status = str(item.get("status") or "").strip()
    if status and status.lower() != "unknown":
        return status
    payload_status = str(payload.get("status") or "").strip()
    if payload_status and payload_status.lower() != "unknown":
        return payload_status
    return "submitted"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _is_retry_batch_meta(payload: dict) -> bool:
    role = str(payload.get("job_group_role") or "").strip().lower()
    return bool(
        role == "retry"
        or payload.get("retry_source_run")
        or payload.get("retry_source_run_id")
    )


def _submit_root_for_run_dir(run_dir: Path) -> Path:
    return run_dir.parent.parent if run_dir.parent.name == "submits" else run_dir.parent


def _same_run_dir(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left == right


def _resolve_retry_source_run_dir(
    root: Path,
    retry_run_dir: Path,
    retry_meta: dict,
) -> Path | None:
    raw = str(retry_meta.get("retry_source_run") or "").strip()
    candidates: list[Path] = []
    if raw:
        raw_path = Path(raw).expanduser()
        candidates.append(raw_path)
        if not raw_path.is_absolute():
            candidates.extend(
                [
                    root / raw_path,
                    run_layout.category_root(root, "submit") / raw_path,
                    retry_run_dir.parent / raw_path,
                ]
            )
        candidates.append(run_layout.category_root(root, "submit") / raw_path.name)
        candidates.append(root / raw_path.name)

    source_id = str(retry_meta.get("retry_source_run_id") or "").strip()
    if source_id:
        candidates.extend(
            [
                run_layout.category_root(root, "submit") / source_id,
                root / source_id,
            ]
        )

    for candidate in candidates:
        if candidate == retry_run_dir:
            continue
        if (candidate / "batch_job.json").is_file():
            return candidate
    return None


def _retry_child_run_dirs(source_run_dir: Path) -> list[Path]:
    root = _submit_root_for_run_dir(source_run_dir)
    children: list[Path] = []
    for candidate in run_layout.iter_run_dirs(root, "submit"):
        if _same_run_dir(candidate, source_run_dir):
            continue
        meta = _read_json_file(candidate / "batch_job.json")
        if not meta or not _is_retry_batch_meta(meta):
            continue
        source = _resolve_retry_source_run_dir(root, candidate, meta)
        if source is not None and _same_run_dir(source, source_run_dir):
            children.append(candidate)
    return children


def _retry_attempt_label(parent_meta: dict) -> str:
    retry_runs = parent_meta.get("retry_runs")
    count = len(retry_runs) if isinstance(retry_runs, list) else 0
    return f"retry_failed_{count + 1:03d}"


def _patch_retry_child_metadata(
    *,
    source_run_dir: Path,
    retry_run_dir: Path,
    retry_meta: dict,
) -> bool:
    group_id = str(retry_meta.get("job_group_id") or source_run_dir.name)
    changed = False
    if retry_meta.get("job_group_id") != group_id:
        retry_meta["job_group_id"] = group_id
        changed = True
    if retry_meta.get("job_group_role") != "retry":
        retry_meta["job_group_role"] = "retry"
        changed = True
    if retry_meta.get("retry_source_run_id") != source_run_dir.name:
        retry_meta["retry_source_run_id"] = source_run_dir.name
        changed = True
    if changed:
        _write_json_file(retry_run_dir / "batch_job.json", retry_meta)
    return changed


def _append_retry_child_to_source_metadata(
    *,
    source_run_dir: Path,
    retry_run_dir: Path,
) -> bool:
    source_path = source_run_dir / "batch_job.json"
    retry_path = retry_run_dir / "batch_job.json"
    source_meta = _read_json_file(source_path)
    retry_meta = _read_json_file(retry_path)
    if not source_meta or not retry_meta or not _is_retry_batch_meta(retry_meta):
        return False

    changed = False
    group_id = str(source_meta.get("job_group_id") or source_run_dir.name)
    if source_meta.get("job_group_id") != group_id:
        source_meta["job_group_id"] = group_id
        changed = True
    if source_meta.get("job_group_role") != "root":
        source_meta["job_group_role"] = "root"
        changed = True

    retry_jobs = [
        item
        for item in (retry_meta.get("batch_jobs") or [])
        if isinstance(item, dict) and item.get("batch_job_name")
    ]
    if not retry_jobs:
        return changed

    jobs = source_meta.get("batch_jobs")
    if not isinstance(jobs, list):
        jobs = []
        source_meta["batch_jobs"] = jobs
        changed = True
    existing_names = {
        item.get("batch_job_name")
        for item in jobs
        if isinstance(item, dict) and item.get("batch_job_name")
    }
    chunk_indices = [
        int(item.get("chunk_index") or 0)
        for item in jobs
        if isinstance(item, dict)
    ]
    next_chunk_index = max(chunk_indices or [0]) + 1
    retry_attempt_label = _retry_attempt_label(source_meta)
    total_retry_jobs = len(retry_jobs)
    appended_names: list[object] = []
    for retry_index, retry_job in enumerate(retry_jobs, start=1):
        batch_name = retry_job.get("batch_job_name")
        if batch_name in existing_names:
            continue
        retry_entry = dict(retry_job)
        retry_label = (
            retry_attempt_label
            if total_retry_jobs == 1
            else f"{retry_attempt_label}_{retry_index:03d}_of_{total_retry_jobs:03d}"
        )
        retry_entry.update(
            {
                "chunk_index": next_chunk_index,
                "total_chunks": next_chunk_index,
                "chunk_label": retry_label,
                "status": _stored_batch_status(retry_meta, retry_job),
                "is_retry": True,
                "retry_run_dir": str(retry_run_dir),
                "retry_run_id": retry_run_dir.name,
                "retry_source_run": str(source_run_dir),
                "retry_source_run_id": source_run_dir.name,
                "job_group_id": group_id,
                "retry_failed_keys_file": retry_meta.get("retry_failed_keys_file"),
            }
        )
        jobs.append(retry_entry)
        existing_names.add(batch_name)
        appended_names.append(batch_name)
        next_chunk_index += 1
        changed = True

    names = [
        item.get("batch_job_name")
        for item in jobs
        if isinstance(item, dict) and item.get("batch_job_name")
    ]
    deduped_names = list(dict.fromkeys(names))
    if source_meta.get("batch_job_names") != deduped_names:
        source_meta["batch_job_names"] = deduped_names
        changed = True

    retry_runs = source_meta.get("retry_runs")
    if not isinstance(retry_runs, list):
        retry_runs = []
        source_meta["retry_runs"] = retry_runs
        changed = True
    if not any(
        isinstance(item, dict) and item.get("run_id") == retry_run_dir.name
        for item in retry_runs
    ):
        retry_runs.append(
            {
                "run_dir": str(retry_run_dir),
                "run_id": retry_run_dir.name,
                "batch_job_name": retry_jobs[0].get("batch_job_name"),
                "batch_job_names": appended_names
                or [item.get("batch_job_name") for item in retry_jobs],
                "batch_count": total_retry_jobs,
                "request_count": int(retry_meta.get("request_count") or 0),
                "created_at": str(retry_meta.get("created_at") or ""),
            }
        )
        changed = True

    child_changed = _patch_retry_child_metadata(
        source_run_dir=source_run_dir,
        retry_run_dir=retry_run_dir,
        retry_meta=retry_meta,
    )
    if changed:
        _write_json_file(source_path, source_meta)
    return changed or child_changed


def repair_retry_metadata_links(run_root: str | Path | None = None) -> int:
    root = Path(run_root or config.output_root).expanduser()
    repaired = 0
    for retry_run_dir in run_layout.iter_run_dirs(root, "submit"):
        retry_meta = _read_json_file(retry_run_dir / "batch_job.json")
        if not retry_meta or not _is_retry_batch_meta(retry_meta):
            continue
        source_run_dir = _resolve_retry_source_run_dir(root, retry_run_dir, retry_meta)
        if source_run_dir is None:
            continue
        if _append_retry_child_to_source_metadata(
            source_run_dir=source_run_dir,
            retry_run_dir=retry_run_dir,
        ):
            repaired += 1
    return repaired


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
                status=_stored_batch_status(payload, item),
                output_destination=str(item.get("output_destination") or ""),
                requests_file=str(item.get("requests_file") or ""),
                provider=str(item.get("provider") or payload.get("provider") or ""),
            )
        )
    return sorted(summaries, key=lambda item: item.chunk_index)


def _primary_request_count_from_payload(payload: dict) -> int:
    jobs = payload.get("batch_jobs")
    if isinstance(jobs, list):
        count = sum(
            int(item.get("request_count") or 0)
            for item in jobs
            if isinstance(item, dict) and not bool(item.get("is_retry"))
        )
        if count:
            return count
    return int(payload.get("request_count") or 0)


def _linked_batch_chunk_summaries(run_dir: Path) -> list[BatchChunkSummary]:
    payload = _read_json_file(run_dir / "batch_job.json")
    chunks = _batch_chunk_summaries_from_payload(payload)
    seen_names = {chunk.batch_job_name for chunk in chunks}
    next_index = max((chunk.chunk_index for chunk in chunks), default=0) + 1
    for retry_run_dir in _retry_child_run_dirs(run_dir):
        retry_payload = _read_json_file(retry_run_dir / "batch_job.json")
        for chunk in _batch_chunk_summaries_from_payload(retry_payload):
            if chunk.batch_job_name in seen_names:
                continue
            chunks.append(
                BatchChunkSummary(
                    chunk_index=next_index,
                    total_chunks=next_index,
                    chunk_label=f"retry_failed_{next_index:03d}",
                    batch_job_name=chunk.batch_job_name,
                    request_count=chunk.request_count,
                    status=chunk.status,
                    output_destination=chunk.output_destination,
                    requests_file=chunk.requests_file,
                    provider=chunk.provider,
                )
            )
            seen_names.add(chunk.batch_job_name)
            next_index += 1
    return sorted(chunks, key=lambda item: item.chunk_index)


def _record_chunk_statuses_in_payload(
    payload: dict,
    state_by_name: dict[str, str],
    checked_at: str,
) -> bool:
    jobs = payload.get("batch_jobs")
    if not isinstance(jobs, list):
        return False

    changed = False
    for item in jobs:
        if not isinstance(item, dict):
            continue
        batch_name = str(item.get("batch_job_name") or "")
        state = state_by_name.get(batch_name)
        if not state:
            continue
        if item.get("status") != state:
            item["status"] = state
            changed = True
        if item.get("status_checked_at") != checked_at:
            item["status_checked_at"] = checked_at
            changed = True
    return changed


def record_batch_chunk_statuses(
    run_dir: str | Path,
    state_by_name: dict[str, str],
) -> None:
    """Persist the last checked provider state for parent and retry batch chunks."""
    if not state_by_name:
        return
    checked_at = _now_iso()
    run_path = Path(run_dir).expanduser()
    parent_path = run_path / "batch_job.json"
    parent_payload = _read_json_file(parent_path)
    if parent_payload and _record_chunk_statuses_in_payload(
        parent_payload,
        state_by_name,
        checked_at,
    ):
        _write_json_file(parent_path, parent_payload)

    for retry_run_dir in _retry_child_run_dirs(run_path):
        retry_path = retry_run_dir / "batch_job.json"
        retry_payload = _read_json_file(retry_path)
        if retry_payload and _record_chunk_statuses_in_payload(
            retry_payload,
            state_by_name,
            checked_at,
        ):
            _write_json_file(retry_path, retry_payload)


def list_batch_chunks(run_dir: str | Path) -> list[BatchChunkSummary]:
    return _linked_batch_chunk_summaries(Path(run_dir).expanduser())


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

    try:
        record_batch_chunk_statuses(run_path, state_by_name)
    except Exception:  # noqa: BLE001
        pass

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
    return state.strip().upper() in {"JOB_STATE_SUCCEEDED", "SUCCEEDED", "ENDED"}


def _is_failure_state(state: str) -> bool:
    return state.strip().upper() in {
        "JOB_STATE_FAILED",
        "FAILED",
        "ERRORED",
        "JOB_STATE_PARTIALLY_SUCCEEDED",
        "PARTIALLY_SUCCEEDED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_CANCELED",
        "CANCELLED",
        "CANCELED",
        "EXPIRED",
        "JOB_STATE_EXPIRED",
    }


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


def _batch_model_progress(run_dir: str | Path):
    from patientjournals.batch import status as status_module

    run_path = Path(run_dir).expanduser()
    payload = _read_json_file(run_path / "batch_job.json")
    chunks = _batch_chunk_summaries_from_payload(payload)
    batch_names = [chunk.batch_job_name for chunk in chunks if chunk.batch_job_name]
    if not batch_names:
        return None

    provider = status_module._provider_from_batch_names(batch_names, run_dir=run_path)
    client = status_module._get_client(provider, batch_names)
    total = status_module._request_count_from_payload(payload)
    if total is None:
        total = sum(chunk.request_count for chunk in chunks) or None

    if provider == "anthropic":
        batch_jobs = [
            status_module._get_batch_job(client, name, provider)
            for name in batch_names
        ]
        return status_module._anthropic_model_progress(batch_jobs, total)

    batch_jobs_by_name = {
        name: status_module._get_batch_job(client, name, provider)
        for name in batch_names
    }
    return status_module._gemini_model_progress(
        batch_jobs_by_name,
        run_dir=run_path,
        total=total,
    )


def resolve_batch_run_readiness(
    run_dir: str | Path,
    *,
    chunks: list[BatchChunkSummary] | None = None,
) -> BatchRunReadiness:
    """Return the app-facing batch state, including output-file readiness.

    Some Gemini jobs can report a succeeded job state before prediction JSONL
    files are fully visible in GCS. The app should not offer retrieval until
    those rows are present, otherwise a user can accidentally record a zero-row
    partial retrieval as if every page failed.
    """
    current_chunks = chunks if chunks is not None else list_batch_chunks_with_state(run_dir)
    state = aggregate_batch_state(current_chunks)
    if state != "succeeded":
        return BatchRunReadiness(state=state)

    try:
        progress = _batch_model_progress(run_dir)
    except Exception as exc:  # noqa: BLE001
        return BatchRunReadiness(
            state="succeeded",
            detail=f"output readiness unavailable: {type(exc).__name__}",
        )
    if progress is None:
        return BatchRunReadiness(state="succeeded")

    if (
        progress.processed is not None
        and progress.total is not None
        and progress.total > 0
        and progress.processed < progress.total
    ):
        return BatchRunReadiness(
            state="finalizing",
            detail=(
                f"model outputs {progress.processed}/{progress.total}; "
                "waiting for prediction files"
            ),
            output_rows=progress.processed,
            expected_rows=progress.total,
        )

    return BatchRunReadiness(
        state="succeeded",
        detail=progress.detail,
        output_rows=progress.processed,
        expected_rows=progress.total,
    )


def resolve_batch_run_state(run_dir: str | Path) -> str:
    """Query the batch API once and aggregate chunk states into a job-level status.

    Returns "" when the state cannot be determined (no credentials, network
    error) so callers can fall back to the on-disk status.
    """
    try:
        return resolve_batch_run_readiness(run_dir).state
    except Exception:  # noqa: BLE001
        return ""


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
        and (
            not job.retrieved
            or job.status in {"retry_submitted", "submitted", "running"}
        )
        and job.status in {"submitted", "unknown", "running", "retry_submitted"}
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
    repair_retry_metadata_links(root)
    summaries: list[JobSummary] = []
    for run_dir in run_layout.iter_run_dirs(root, "submit"):
        batch_meta = _read_json_file(run_dir / "batch_job.json")
        if not batch_meta:
            continue
        if _is_retry_batch_meta(batch_meta):
            source_run_dir = _resolve_retry_source_run_dir(root, run_dir, batch_meta)
            if source_run_dir is not None:
                continue

        metadata = _read_json_file(run_dir / "metadata.json")
        created_at = str(metadata.get("created_at") or batch_meta.get("created_at") or "")
        model = str(batch_meta.get("model") or "")
        chunks = list_batch_chunks(run_dir)
        chunk_count = len(chunks) or len(batch_meta.get("batch_job_names") or [])
        images = _primary_request_count_from_payload(batch_meta)
        location = _describe_input_location(run_dir, batch_meta)
        retry_runs = batch_meta.get("retry_runs")
        retry_count = len(retry_runs) if isinstance(retry_runs, list) else 0

        results = read_recorded_results(run_dir)
        retrieved = bool(results)
        succeeded: int | None = None
        failed: int | None = None
        recovered = 0
        status = _run_dir_status(run_dir)
        if retrieved:
            succeeded = int(results.get("successful_pages") or 0)
            expected = int(results.get("expected_pages") or images or 0)
            failed = max(0, expected - succeeded)
            recovered = int(results.get("recovered_pages") or 0)
            status = "retrieved"
        if retry_count and failed:
            status = "retry_submitted"

        detail = f"{images} image(s)"
        if chunk_count:
            detail = f"{detail}, {chunk_count} chunk(s)"
        if retry_count:
            label = "retry batch" if retry_count == 1 else "retry batches"
            detail = f"{detail}, {retry_count} {label}"

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
                recovered=recovered,
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
