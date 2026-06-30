from __future__ import annotations

import asyncio
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any

from patientjournals.app.access import run_access_checks
from patientjournals.app.dashboard import analyze_dataset_file, summarize_dashboard
from patientjournals.app.datasets import (
    combine_dataset_files,
    download_cloud_dataset,
    list_cloud_dataset_choices,
    list_cloud_dataset_library,
    list_local_dataset_library,
    prepare_dataset_sources,
)
from patientjournals.app.job_store import JobStore, utc_now_iso
from patientjournals.app.jobs import (
    _apply_runtime_overrides,
    _restore_runtime_overrides,
    build_validation_command,
    cancel_batch_run,
    finalize_dataset_with_failed_rows,
    list_batch_chunks,
    list_batch_chunks_with_state,
    list_submit_jobs,
    read_dataset_preview,
    recover_dataset_gaps,
    resubmit_failed_requests,
    resolve_batch_run_readiness,
    run_batch_draft_direct,
    run_local_draft_direct,
    run_retrieve_direct,
    start_command,
)
from patientjournals.app.models import AppSettings, SubmitJobDraft
from patientjournals.app.settings_store import (
    command_override_payload,
    load_app_settings,
    save_app_settings,
)
from patientjournals.config import config
from patientjournals.validation.browser import BrowserValidationManager


def serializable(value: Any) -> Any:
    if is_dataclass(value):
        return serializable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serializable(item) for item in value]
    return value


class WorkflowService:
    """App-facing workflow API.

    Tk, web handlers, and tests should call this layer rather than coordinating
    retrieval, retry, dashboard, and dataset helpers directly.
    """

    def __init__(
        self,
        settings: AppSettings | None = None,
        *,
        settings_path: str | Path | None = None,
    ) -> None:
        self.settings_path = settings_path
        self.settings = settings or load_app_settings(settings_path)
        self.store = JobStore(self.settings.local_runs_root)
        self.validation_manager = BrowserValidationManager()

    def list_jobs(self) -> list[dict[str, Any]]:
        return serializable(list_submit_jobs(self.settings.local_runs_root))

    def cloud_settings(self) -> dict[str, Any]:
        return serializable(self.settings)

    def save_cloud_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "auth_mode",
            "service_account_file",
            "gcp_project_id",
            "gcp_location",
            "vertex_model_location",
            "gcs_bucket_name",
            "gcs_pages_prefix",
            "batch_requests_gcs_prefix",
            "batch_outputs_gcs_prefix",
            "datasets_gcs_prefix",
            "validations_gcs_prefix",
            "upload_validation_to_gcs",
        }
        updates = {key: payload[key] for key in allowed if key in payload}
        self.settings = replace(self.settings, **updates)
        save_app_settings(self.settings, self.settings_path)
        self.store = JobStore(self.settings.local_runs_root)
        return self.cloud_settings()

    def cloud_access_report(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if payload:
            self.save_cloud_settings(payload)
        report = run_access_checks(self.settings)
        return {
            "ready": report.ready,
            "failed": report.failed,
            "warnings": report.warnings,
            "passed": report.passed,
            "results": serializable(report.results),
        }

    def start_cloud_browser_login(
        self,
        *,
        mode: str = "adc",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if payload:
            self.save_cloud_settings({**payload, "auth_mode": "adc" if mode == "adc" else payload.get("auth_mode", "adc")})
        commands = {
            "adc": ("gcloud", "auth", "application-default", "login"),
            "gcloud": ("gcloud", "auth", "login"),
        }
        command = commands.get(mode)
        if command is None:
            raise ValueError(f"Unknown cloud login mode: {mode}")
        process = subprocess.Popen(command)  # noqa: S603
        return {
            "status": "started",
            "mode": mode,
            "pid": process.pid,
            "command": " ".join(command),
        }

    def submit_batch(self, draft: SubmitJobDraft) -> dict[str, Any]:
        if draft.run_mode == "local_api":
            result = asyncio.run(run_local_draft_direct(draft, self.settings))
            self._record_local_result(draft, result)
            return serializable(result)
        outcome = run_batch_draft_direct(draft, self.settings)
        return serializable(outcome or {"status": "covered"})

    def _record_local_result(self, draft: SubmitJobDraft, result: Any) -> None:
        run_dir = Path(result.run_dir).expanduser()
        dataset_path = Path(result.dataset_path).expanduser()
        job_id = self.store.job_id_for_run_dir(run_dir)
        now = utc_now_iso()
        self.store.write(
            job_id,
            {
                "schema_version": 2,
                "job_id": job_id,
                "kind": "local",
                "status": str(result.status),
                "created_at": now,
                "model": draft.model_name,
                "provider": "",
                "legacy": {"submit_run_dir": str(run_dir)},
                "input": {
                    "location": draft.local_path or str(config.target_folder or ""),
                    "image_count": int(result.total_images or 0),
                },
                "batches": {"source_run_dir": str(run_dir), "chunk_count": 0},
                "dataset": {
                    "current_path": str(dataset_path) if dataset_path.exists() else "",
                    "current_gcs_uri": "",
                    "versions": [
                        {
                            "created_at": now,
                            "operation": "local_run",
                            "path": str(dataset_path),
                            "source_path": str(dataset_path),
                            "rows_written": int(result.rows_written or 0),
                            "successful_pages": int(result.covered_after or 0),
                            "missing_pages": max(
                                0,
                                int(result.total_images or 0)
                                - int(result.covered_after or 0),
                            ),
                        }
                    ]
                    if dataset_path.exists()
                    else [],
                },
                "retrieval": {},
                "metrics": {
                    "expected_pages": int(result.total_images or 0),
                    "successful_pages": int(result.covered_after or 0),
                    "missing_pages": max(
                        0,
                        int(result.total_images or 0) - int(result.covered_after or 0),
                    ),
                    "rows_written": int(result.rows_written or 0),
                    "skipped_images": int(result.skipped_images or 0),
                    "operation": "local_run",
                },
                "operations": [],
            },
        )
        self.store.append_event(job_id, "local_run", serializable(result))

    def retrieve_results(
        self,
        run_dir: str,
        *,
        ignore_failed: bool = False,
        duplicate_strategy: str = "",
        force: bool = False,
    ) -> dict[str, Any]:
        payload = run_retrieve_direct(
            run_dir,
            self.settings,
            allow_partial=True,
            ignore_failed=ignore_failed,
            duplicate_strategy=duplicate_strategy,
            force=force,
        )
        dataset_path = str(payload.get("dataset_path") or "")
        columns, rows = read_dataset_preview(dataset_path, limit=50)
        return {
            "result": serializable(payload),
            "preview": {"columns": columns, "rows": serializable(rows)},
        }

    def retrieve_many(
        self,
        run_dirs: list[str] | tuple[str, ...],
        *,
        ignore_failed: bool = False,
        duplicate_strategy: str = "",
        force: bool = False,
    ) -> dict[str, Any]:
        selected = [item for item in dict.fromkeys(str(path) for path in run_dirs) if item]
        if not selected:
            raise ValueError("Select at least one job to retrieve.")
        max_workers = min(max(1, len(selected)), 4)
        results: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    self.retrieve_results,
                    run_dir,
                    ignore_failed=ignore_failed,
                    duplicate_strategy=duplicate_strategy,
                    force=force,
                ): run_dir
                for run_dir in selected
            }
            for future in as_completed(futures):
                run_dir = futures[future]
                try:
                    results.append({"run_dir": run_dir, "payload": future.result()})
                except Exception as exc:  # noqa: BLE001
                    failures.append({"run_dir": run_dir, "error": str(exc)})
        return {
            "requested": len(selected),
            "succeeded": len(results),
            "failed": len(failures),
            "results": serializable(results),
            "failures": failures,
        }

    def finalize_failed_rows(self, run_dir: str) -> dict[str, Any]:
        return serializable(finalize_dataset_with_failed_rows(run_dir, self.settings))

    def recover_missing_with_api(self, run_dir: str) -> dict[str, Any]:
        return serializable(recover_dataset_gaps(run_dir, self.settings))

    def resubmit_failed(self, run_dir: str, *, num_batches: int = 1) -> dict[str, Any]:
        return serializable(
            resubmit_failed_requests(run_dir, self.settings, num_batches=num_batches)
        )

    def cancel_batch(self, run_dir: str) -> dict[str, int]:
        return {"cancelled": cancel_batch_run(run_dir, self.settings)}

    def job_chunks(self, run_dir: str, *, live: bool = False) -> list[dict[str, Any]]:
        chunks = list_batch_chunks_with_state(run_dir) if live else list_batch_chunks(run_dir)
        return serializable(chunks)

    def readiness(self, run_dir: str) -> dict[str, Any]:
        return serializable(resolve_batch_run_readiness(run_dir))

    def list_datasets(self, *, include_cloud: bool = False) -> dict[str, Any]:
        local_items = list_local_dataset_library(self.settings.local_runs_root)
        cloud_items = []
        if include_cloud:
            previous = _apply_runtime_overrides(command_override_payload(self.settings))
            try:
                cloud_items = list_cloud_dataset_library(
                    bucket_name=self.settings.gcs_bucket_name,
                    datasets_prefix=self.settings.datasets_gcs_prefix,
                )
            finally:
                _restore_runtime_overrides(previous)
        return {"local": serializable(local_items), "cloud": serializable(cloud_items)}

    def cloud_input_choices(self) -> list[dict[str, Any]]:
        previous = _apply_runtime_overrides(command_override_payload(self.settings))
        try:
            return serializable(
                list_cloud_dataset_choices(
                    bucket_name=self.settings.gcs_bucket_name,
                    pages_prefix=self.settings.gcs_pages_prefix,
                )
            )
        finally:
            _restore_runtime_overrides(previous)

    def combine_datasets(
        self,
        dataset_items: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        *,
        output_name: str,
        duplicate_strategy: str = "first_successful",
    ) -> dict[str, Any]:
        if not output_name.strip():
            raise ValueError("Enter a name for the combined dataset.")
        previous = _apply_runtime_overrides(command_override_payload(self.settings))
        try:
            sources = prepare_dataset_sources(
                dataset_items,
                download_root=(
                    Path(self.settings.local_runs_root)
                    / "datasets"
                    / "_cloud_cache"
                ),
            )
            return serializable(
                combine_dataset_files(
                    sources,
                    output_name=output_name,
                    output_root=self.settings.local_runs_root,
                    duplicate_strategy=duplicate_strategy,
                    upload_to_cloud=bool(
                        self.settings.gcs_bucket_name
                        and getattr(config, "upload_dataset_to_gcs", True)
                    ),
                    bucket_name=self.settings.gcs_bucket_name,
                    datasets_prefix=self.settings.datasets_gcs_prefix,
                )
            )
        finally:
            _restore_runtime_overrides(previous)

    def local_input_choices(self, *, limit: int = 200) -> list[dict[str, Any]]:
        root_value = (
            self.settings.validation_images_root
            or getattr(config, "upload_images_folder", "")
            or getattr(config, "target_folder", "")
        )
        root = Path(str(root_value or "")).expanduser()
        if not root.exists() or not root.is_dir():
            return []

        children = []
        try:
            children = [path for path in root.iterdir() if path.is_dir()]
        except OSError:
            children = []
        candidates = [root, *sorted(children, key=lambda path: _mtime(path), reverse=True)]
        items: list[dict[str, Any]] = []
        for path in candidates[: max(1, limit)]:
            count = _count_images(path)
            if count <= 0:
                continue
            items.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "image_count": count,
                    "updated_at": _updated_label(path),
                }
            )
        return items

    def dashboard(self, *, include_cloud_validations: bool = True) -> dict[str, Any]:
        return serializable(
            summarize_dashboard(
                run_root=self.settings.local_runs_root,
                validations_root="validations",
                cloud_validations_bucket=(
                    self.settings.gcs_bucket_name if include_cloud_validations else ""
                ),
                cloud_validations_prefix=self.settings.validations_gcs_prefix,
            )
        )

    def analyze_dataset(self, dataset_path: str) -> dict[str, Any]:
        return serializable(analyze_dataset_file(dataset_path))

    def _resolve_validation_dataset(self, results: str) -> tuple[Path, str]:
        value = str(results or "").strip()
        if not value:
            raise ValueError("Select a dataset first.")
        if value.startswith("gs://"):
            previous = _apply_runtime_overrides(command_override_payload(self.settings))
            try:
                path = download_cloud_dataset(
                    value,
                    destination_root=(
                        Path(self.settings.local_runs_root)
                        / "validations"
                        / "_dataset_cache"
                    ),
                )
            finally:
                _restore_runtime_overrides(previous)
            return path, value
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {results}")
        return path, path.name

    def start_browser_validation(
        self,
        *,
        results: str,
        image_source: str = "cloud",
        images: str = "",
        cloud_prefixes: list[str] | tuple[str, ...] = (),
        username: str = "researcher",
        corrections: bool = True,
        sampling_mode: str = "balanced_ucb",
    ) -> dict[str, Any]:
        dataset_path, dataset_label = self._resolve_validation_dataset(results)
        previous = _apply_runtime_overrides(command_override_payload(self.settings))
        try:
            return serializable(
                self.validation_manager.start_session(
                    dataset_path=dataset_path,
                    dataset_label=dataset_label,
                    username=username.strip() or "researcher",
                    allow_corrections=corrections,
                    sampling_mode=sampling_mode,
                    image_source=image_source,
                    image_root=images,
                    cloud_prefixes=tuple(str(item) for item in cloud_prefixes if item),
                    bucket_name=self.settings.gcs_bucket_name,
                )
            )
        finally:
            _restore_runtime_overrides(previous)

    def browser_validation_current(self, session_id: str) -> dict[str, Any]:
        return serializable(self.validation_manager.current(session_id))

    def mark_browser_validation(
        self,
        *,
        session_id: str,
        label: str,
        corrected_text: str = "",
    ) -> dict[str, Any]:
        return serializable(
            self.validation_manager.mark(
                session_id,
                label=label,
                corrected_text=corrected_text,
            )
        )

    def finish_browser_validation(self, session_id: str) -> dict[str, Any]:
        previous = _apply_runtime_overrides(command_override_payload(self.settings))
        try:
            return serializable(self.validation_manager.finish(session_id))
        finally:
            _restore_runtime_overrides(previous)

    def browser_validation_image(self, session_id: str) -> tuple[bytes, str]:
        return self.validation_manager.local_image_bytes(session_id)

    def start_validation(
        self,
        *,
        results: str,
        images: str = "",
        username: str = "researcher",
        corrections: bool = True,
        sampling_mode: str = "random",
    ) -> dict[str, Any]:
        dataset = Path(results).expanduser()
        if not dataset.is_file():
            raise FileNotFoundError(f"Dataset not found: {results}")
        image_root = images or self.settings.validation_images_root or str(config.target_folder)
        if not Path(image_root).expanduser().is_dir():
            raise FileNotFoundError(f"Image folder not found: {image_root}")
        command = build_validation_command(
            self.settings,
            images=image_root,
            results=str(dataset),
            username=username.strip() or "researcher",
            corrections=corrections,
            sampling_mode=sampling_mode,
        )
        return serializable(start_command(command, kind="validation"))


def _image_extensions() -> set[str]:
    return {
        f".{str(ext).lower().lstrip('.')}"
        for ext in getattr(config, "batch_input_extensions", ())
        if str(ext).strip()
    } or {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _updated_label(path: Path) -> str:
    try:
        from datetime import datetime

        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return ""


def _count_images(path: Path, *, cap: int = 10000) -> int:
    extensions = _image_extensions()
    count = 0
    try:
        iterator = path.rglob("*") if getattr(config, "recursive", True) else path.glob("*")
        for item in iterator:
            if item.is_file() and item.suffix.lower() in extensions and not item.name.startswith("._"):
                count += 1
                if count >= cap:
                    return count
    except OSError:
        return count
    return count
