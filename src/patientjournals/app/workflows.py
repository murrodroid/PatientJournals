from __future__ import annotations

import asyncio
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Any

from patientjournals.app.access import resolve_validator_identity, run_access_checks
from patientjournals.app.dashboard import analyze_dataset_file, summarize_dashboard
from patientjournals.app.datasets import (
    combine_dataset_files,
    download_cloud_dataset,
    list_cloud_dataset_choices,
    list_cloud_dataset_library,
    list_local_dataset_library,
    prepare_dataset_sources,
    read_dataset_page,
)
from patientjournals.app.job_store import JobStore, utc_now_iso
from patientjournals.app.image_access import ImageAccessService
from patientjournals.app.jobs import (
    _apply_runtime_overrides,
    _restore_runtime_overrides,
    build_validation_command,
    cancel_batch_run,
    command_overrides_for_run,
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
from patientjournals.app.schemas import SchemaService
from patientjournals.app.settings_store import (
    command_override_payload,
    load_app_settings,
    save_app_settings,
)
from patientjournals.config import config
from patientjournals.config.models import resolve_model_spec
from patientjournals.validation.browser import BrowserValidationManager


_RUNTIME_CONFIG_LOCK = RLock()


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
        self.schema_service = SchemaService(
            self.store,
            bucket_name=self.settings.gcs_bucket_name,
            schemas_prefix=self.settings.schemas_gcs_prefix,
        )
        self.validation_manager = BrowserValidationManager()
        self.image_access = ImageAccessService(self.settings)

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
            "schemas_gcs_prefix",
            "upload_validation_to_gcs",
            "ocr_enabled",
            "subagents",
            "model_validation_enabled",
            "verification_model",
            "verification_thinking_level",
            "verification_scope",
            "verification_control_sample_percent",
            "verification_max_output_tokens",
            "verification_num_chunks",
        }
        updates = {key: payload[key] for key in allowed if key in payload}
        if "verification_thinking_level" in updates and updates[
            "verification_thinking_level"
        ] not in {"low", "medium", "high"}:
            raise ValueError("Verification thinking must be low, medium, or high.")
        if "verification_scope" in updates and updates["verification_scope"] not in {
            "all",
            "flagged",
        }:
            raise ValueError("Verification scope must be all or flagged.")
        # Automatic acceptance is a pipeline invariant, not a user preference.
        updates["verification_apply_mode"] = "apply_patches"
        for key in ("verification_max_output_tokens", "verification_num_chunks"):
            if key in updates:
                updates[key] = max(1, int(updates[key]))
        if "verification_control_sample_percent" in updates:
            updates["verification_control_sample_percent"] = max(
                0.0,
                min(100.0, float(updates["verification_control_sample_percent"])),
            )
        next_settings = replace(self.settings, **updates)
        if (
            next_settings.model_validation_enabled
            and next_settings.verification_scope == "flagged"
            and next_settings.verification_control_sample_percent <= 0.0
        ):
            raise ValueError(
                "Risk-routed verification requires a positive routine-page "
                "control sample. Choose all-page scope to use 0%."
            )
        self.settings = next_settings
        save_app_settings(self.settings, self.settings_path)
        self.store = JobStore(self.settings.local_runs_root)
        self.schema_service = SchemaService(
            self.store,
            bucket_name=self.settings.gcs_bucket_name,
            schemas_prefix=self.settings.schemas_gcs_prefix,
        )
        self.image_access.update_settings(self.settings)
        return self.cloud_settings()

    def list_schemas(self, *, sync_cloud: bool = True) -> dict[str, Any]:
        previous = _apply_runtime_overrides(command_override_payload(self.settings))
        try:
            return serializable(
                self.schema_service.list_versions(sync_cloud=sync_cloud)
            )
        finally:
            _restore_runtime_overrides(previous)

    def create_schema_version(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw_fields = payload.get("fields") or []
        if not isinstance(raw_fields, list):
            raise ValueError("Schema fields must be a list.")
        identity = self.validation_identity()
        previous = _apply_runtime_overrides(command_override_payload(self.settings))
        try:
            return serializable(
                self.schema_service.create_version(
                    name=str(payload.get("name") or ""),
                    fields=[item for item in raw_fields if isinstance(item, dict)],
                    created_by=identity.get("account") or identity.get("username") or "unknown",
                    parent_version_id=str(payload.get("parent_version_id") or ""),
                    make_active=bool(payload.get("make_active")),
                )
            )
        finally:
            _restore_runtime_overrides(previous)

    def set_active_schema(self, version_id: str) -> dict[str, Any]:
        previous = _apply_runtime_overrides(command_override_payload(self.settings))
        try:
            return serializable(self.schema_service.set_active(version_id))
        finally:
            _restore_runtime_overrides(previous)

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
        if draft.model_validation_enabled:
            draft = replace(draft, verification_apply_mode="apply_patches")
            if draft.run_mode != "cloud_batch":
                raise ValueError("Second-pass model validation requires Cloud batch.")
            verifier = resolve_model_spec(draft.verification_model)
            if not verifier.supports_batch:
                raise ValueError(
                    f"Verifier model does not support batch jobs: {verifier.name}"
                )
            if draft.verification_scope not in {"all", "flagged"}:
                raise ValueError("Verification scope must be all or flagged.")
            if not 0.0 <= draft.verification_control_sample_percent <= 100.0:
                raise ValueError("Control sample percent must be between 0 and 100.")
            if (
                draft.verification_scope == "flagged"
                and draft.verification_control_sample_percent <= 0.0
            ):
                raise ValueError(
                    "Risk-routed verification requires a positive routine-page "
                    "control sample so an all-routine job still has a final-model "
                    "validation batch."
                )
            if draft.verification_thinking_level not in {"low", "medium", "high"}:
                raise ValueError("Verifier thinking must be low, medium, or high.")
            if draft.verification_max_output_tokens < 1:
                raise ValueError("Verifier max output tokens must be at least 1.")
            if draft.verification_num_chunks < 1:
                raise ValueError("Validation batch chunks must be at least 1.")
        previous = _apply_runtime_overrides(command_override_payload(self.settings))
        try:
            self.schema_service.sync_from_cloud()
        finally:
            _restore_runtime_overrides(previous)
        schema_version = self.schema_service.resolve_version(
            draft.schema_version_id or draft.schema_name
        )
        if not schema_version:
            raise ValueError(
                f"Schema version not found: {draft.schema_version_id or draft.schema_name}"
            )
        draft = replace(
            draft,
            schema_name=str(schema_version["name"]),
            schema_version_id=str(schema_version["version_id"]),
            schema_payload=dict(schema_version["schema_json"]),
        )
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
                "schema": {
                    "name": draft.schema_name,
                    "version_id": draft.schema_version_id,
                },
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
        readiness = self._resolve_run_readiness(run_dir)
        if readiness.state not in {"succeeded", "failed"}:
            detail = f" ({readiness.detail})" if readiness.detail else ""
            raise RuntimeError(
                "Batch retrieval is not ready: "
                f"state={readiness.state or 'unknown'}{detail}."
            )
        payload = run_retrieve_direct(
            run_dir,
            self.settings,
            allow_partial=readiness.state == "failed",
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

    def submit_model_validation(
        self,
        run_dir: str,
        *,
        model: str = "",
        thinking_level: str = "",
        scope: str = "",
        control_sample_percent: float | None = None,
        max_output_tokens: int | None = None,
        num_chunks: int | None = None,
    ) -> dict[str, Any]:
        """Submit the candidate-aware verifier batch for a retrieved extraction."""
        from patientjournals.batch.verify import (
            ModelValidationSubmitRequest,
            submit_model_validation,
        )

        selected_model = model or self.settings.verification_model
        selected_thinking = thinking_level or self.settings.verification_thinking_level
        selected_scope = scope or self.settings.verification_scope
        if selected_scope not in {"all", "flagged"}:
            raise ValueError("Verification scope must be all or flagged.")
        selected_control_sample = max(
            0.0,
            min(
                100.0,
                float(
                    self.settings.verification_control_sample_percent
                    if control_sample_percent is None
                    else control_sample_percent
                ),
            ),
        )
        results_path = Path(run_dir).expanduser() / "batch_results.json"
        if results_path.is_file():
            results_payload = json.loads(results_path.read_text(encoding="utf-8"))
            expected_pages = int(results_payload.get("expected_pages") or 0)
            successful_pages = int(results_payload.get("successful_pages") or 0)
            if (
                expected_pages <= 0
                or successful_pages != expected_pages
                or not str(results_payload.get("page_candidates_path") or "")
                or not str(results_payload.get("deterministic_routing_path") or "")
            ):
                raise RuntimeError(
                    "Final-model submission requires one routed deterministic "
                    "candidate for every expected page. Recover or resubmit failed "
                    "pages, then retrieve the complete routing sweep first."
                )
        selected_chunks = max(
            1, int(num_chunks or self.settings.verification_num_chunks)
        )
        selected_tokens = max(
            1,
            int(
                max_output_tokens
                or self.settings.verification_max_output_tokens
            ),
        )
        verifier = resolve_model_spec(selected_model)
        if not verifier.supports_batch:
            raise ValueError(
                f"Verifier model does not support batch jobs: {verifier.name}"
            )

        overrides = command_overrides_for_run(self.settings, run_dir)
        routed_control_sample = max(
            0.0,
            min(
                100.0,
                float(
                    overrides.get("verification_control_sample_percent")
                    if overrides.get("verification_control_sample_percent") is not None
                    else self.settings.verification_control_sample_percent
                ),
            ),
        )
        if (
            control_sample_percent is not None
            and abs(selected_control_sample - routed_control_sample) > 1e-9
        ):
            raise ValueError(
                "The routine-page control sample is fixed when extraction retrieval "
                "writes deterministic routing. Select it during job submission, or "
                "retrieve the extraction again with the desired value."
            )
        selected_control_sample = routed_control_sample
        overrides.update(
            {
                "model_validation_enabled": True,
                "verification_model": selected_model,
                "verification_thinking_level": selected_thinking,
                "verification_scope": selected_scope,
                "verification_control_sample_percent": selected_control_sample,
                "verification_apply_mode": "apply_patches",
                "verification_max_output_tokens": selected_tokens,
                "verification_num_chunks": selected_chunks,
            }
        )
        with _RUNTIME_CONFIG_LOCK:
            previous = _apply_runtime_overrides(overrides)
            try:
                result = submit_model_validation(
                    ModelValidationSubmitRequest(
                        source_run_dir=run_dir,
                        model=selected_model,
                        thinking_level=selected_thinking,
                        scope=selected_scope,
                        apply_mode="apply_patches",
                        max_output_tokens=selected_tokens,
                        num_chunks=selected_chunks,
                    )
                )
            finally:
                _restore_runtime_overrides(previous)

        payload = serializable(result)
        self.store.mark_model_validation_submitted(
            run_dir,
            verification_run_dir=str(payload.get("run_dir") or ""),
            model=selected_model,
            apply_mode="apply_patches",
            thinking_level=selected_thinking,
            scope=selected_scope,
            control_sample_percent=selected_control_sample,
            max_output_tokens=selected_tokens,
            num_chunks=selected_chunks,
        )
        return payload

    def retrieve_model_validation(
        self,
        run_dir: str,
        *,
        wait: bool = False,
        allow_partial: bool = False,
    ) -> dict[str, Any]:
        """Retrieve and record the verifier batch linked to an extraction job."""
        from patientjournals.batch.verify import (
            ModelValidationRetrieveRequest,
            retrieve_model_validation,
        )

        record = self.store.record_for_run_dir(run_dir)
        validation = (
            record.get("model_validation")
            if isinstance(record.get("model_validation"), dict)
            else {}
        )
        verification_run_dir = str(
            validation.get("verification_run_dir") or ""
        )
        if not verification_run_dir:
            raise ValueError(f"No verifier batch is linked to extraction job: {run_dir}")
        overrides = command_overrides_for_run(self.settings, run_dir)
        with _RUNTIME_CONFIG_LOCK:
            previous = _apply_runtime_overrides(overrides)
            try:
                result = retrieve_model_validation(
                    ModelValidationRetrieveRequest(
                        run_dir=verification_run_dir,
                        wait=wait,
                        allow_partial=allow_partial,
                    )
                )
            finally:
                _restore_runtime_overrides(previous)

        payload = serializable(result)
        payload["verification_run_dir"] = verification_run_dir
        payload.setdefault(
            "verification_model",
            str(payload.get("model") or validation.get("model") or ""),
        )
        payload.setdefault("model_validation_status", str(payload.get("status") or ""))
        payload.setdefault("rows_written", int(payload.get("dataset_rows") or 0))
        publish_dataset = bool(payload.get("publishable"))
        if publish_dataset and not payload.get("dataset_path"):
            raise RuntimeError(
                "Verifier marked the result publishable without a dataset_path."
            )
        payload = self.store.record_model_validation_result(
            run_dir,
            payload,
            publish_dataset=publish_dataset,
        )
        return payload

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
        if not live:
            return serializable(list_batch_chunks(run_dir))
        previous = _apply_runtime_overrides(
            command_overrides_for_run(self.settings, run_dir)
        )
        try:
            chunks = list_batch_chunks_with_state(run_dir)
        finally:
            _restore_runtime_overrides(previous)
        return serializable(chunks)

    def readiness(self, run_dir: str) -> dict[str, Any]:
        return serializable(self._resolve_run_readiness(run_dir))

    def live_batch_status(self, run_dir: str) -> dict[str, Any]:
        """Resolve live chunk states and output readiness under one run snapshot."""

        previous = _apply_runtime_overrides(
            command_overrides_for_run(self.settings, run_dir)
        )
        try:
            chunks = list_batch_chunks_with_state(run_dir)
            readiness = resolve_batch_run_readiness(run_dir, chunks=chunks)
        finally:
            _restore_runtime_overrides(previous)
        return {
            "chunks": serializable(chunks),
            "readiness": serializable(readiness),
        }

    def _resolve_run_readiness(self, run_dir: str):
        """Read provider/GCS state while the source run's config snapshot is active."""

        previous = _apply_runtime_overrides(
            command_overrides_for_run(self.settings, run_dir)
        )
        try:
            return resolve_batch_run_readiness(run_dir)
        finally:
            _restore_runtime_overrides(previous)

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

    def validation_identity(self) -> dict[str, str]:
        return resolve_validator_identity(self.settings)

    def analyze_dataset(self, dataset_path: str) -> dict[str, Any]:
        self.schema_service.sync_from_cloud()
        records = self.store.list_schema_versions()
        schema_names_by_version = {
            str(record["version_id"]): str(record.get("name") or "")
            for record in records
        }
        return serializable(
            analyze_dataset_file(
                dataset_path,
                schema_fields_by_version=(
                    self.schema_service.validation_field_paths_by_version()
                ),
                schema_names_by_version=schema_names_by_version,
            )
        )

    def inspect_dataset(
        self,
        dataset_location: str,
        *,
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        dataset_path, dataset_label = self._resolve_validation_dataset(dataset_location)
        payload = read_dataset_page(dataset_path, offset=offset, limit=limit)
        payload["dataset_location"] = dataset_location
        payload["dataset_label"] = dataset_label
        return serializable(payload)

    def dataset_image_link(
        self,
        *,
        image_name: str,
        object_hint: str = "",
    ) -> dict[str, str]:
        previous = _apply_runtime_overrides(command_override_payload(self.settings))
        try:
            return self.image_access.dataset_image_link(
                image_name=image_name,
                object_hint=object_hint,
            )
        finally:
            _restore_runtime_overrides(previous)

    def local_image_bytes(self, token: str) -> tuple[bytes, str]:
        return self.image_access.local_image_bytes(token)

    def preview_submission(self, payload: dict[str, Any]) -> dict[str, Any]:
        prefixes = payload.get("cloud_prefixes") or ()
        if isinstance(prefixes, str):
            prefixes = (prefixes,)
        previous = _apply_runtime_overrides(command_override_payload(self.settings))
        try:
            return serializable(
                self.image_access.submission_preview(
                    source=str(payload.get("dataset_source") or "local"),
                    local_path=str(payload.get("local_path") or ""),
                    cloud_prefixes=tuple(str(item) for item in prefixes if item),
                    sample_size=int(payload.get("sample_size") or 6),
                )
            )
        finally:
            _restore_runtime_overrides(previous)

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
                    use_cache=True,
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
        corrections: bool = True,
        sampling_mode: str = "balanced_ucb",
        offline: bool = False,
    ) -> dict[str, Any]:
        dataset_path, dataset_label = self._resolve_validation_dataset(results)
        identity = self.validation_identity()
        self.schema_service.sync_from_cloud()
        schema_fields_by_version = (
            self.schema_service.validation_field_paths_by_version()
        )
        previous = _apply_runtime_overrides(command_override_payload(self.settings))
        try:
            return serializable(
                self.validation_manager.start_session(
                    dataset_path=dataset_path,
                    dataset_label=dataset_label,
                    username=identity["username"],
                    validator_account=identity.get("account", ""),
                    allow_corrections=corrections,
                    sampling_mode=sampling_mode,
                    image_source=image_source,
                    image_root=images,
                    cloud_prefixes=tuple(str(item) for item in cloud_prefixes if item),
                    bucket_name=self.settings.gcs_bucket_name,
                    sync_to_cloud=not offline,
                    schema_fields_by_version=schema_fields_by_version,
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
