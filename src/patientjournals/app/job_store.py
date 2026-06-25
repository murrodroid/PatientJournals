from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


JOB_STORE_SCHEMA_VERSION = 1
JOBS_DIR_NAME = "jobs"
JOB_FILE_NAME = "job.json"
EVENTS_FILE_NAME = "events.jsonl"


def utc_now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json_file(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_job_id(value: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in {"_", "-", "."} else "_"
        for char in value.strip()
    ).strip("._")
    return cleaned or "job"


def _submit_root_for_run_dir(run_dir: Path) -> Path:
    return run_dir.parent.parent if run_dir.parent.name == "submits" else run_dir.parent


def _dataset_files(job_dir: Path) -> tuple[Path, Path]:
    dataset_dir = job_dir / "datasets"
    versions_dir = dataset_dir / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir, versions_dir


def _copy_dataset_into_job(
    *,
    job_dir: Path,
    source_path: Path,
    operation: str,
    version_count: int,
) -> Path:
    if not source_path.exists() or not source_path.is_file():
        return source_path

    dataset_dir, versions_dir = _dataset_files(job_dir)
    suffix = source_path.suffix or ".jsonl"
    current_path = dataset_dir / f"current{suffix}"
    if source_path.resolve() != current_path.resolve():
        shutil.copy2(source_path, current_path)

    version_path = versions_dir / f"v{version_count + 1:03d}_{operation}{suffix}"
    if current_path.exists():
        shutil.copy2(current_path, version_path)
    return current_path


class JobStore:
    """Canonical app-facing state for batch jobs.

    Legacy submit/retry directories remain operation artifacts. The app-level job
    entity lives under ``runs/jobs/<job_id>/`` and records the current dataset,
    retrieval signature, metrics, and links back to the operational artifacts.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        self.jobs_root = self.root / JOBS_DIR_NAME

    @classmethod
    def for_run_dir(cls, run_dir: str | Path) -> "JobStore":
        return cls(_submit_root_for_run_dir(Path(run_dir).expanduser()))

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_root / _safe_job_id(job_id)

    def job_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / JOB_FILE_NAME

    def event_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / EVENTS_FILE_NAME

    def read(self, job_id: str) -> dict:
        return read_json_file(self.job_path(job_id))

    def write(self, job_id: str, record: dict) -> None:
        record["schema_version"] = JOB_STORE_SCHEMA_VERSION
        record["job_id"] = _safe_job_id(job_id)
        record["updated_at"] = utc_now_iso()
        write_json_file(self.job_path(job_id), record)

    def append_event(self, job_id: str, event_type: str, payload: dict | None = None) -> None:
        path = self.event_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "at": utc_now_iso(),
            "type": event_type,
            "payload": payload or {},
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False))
            handle.write("\n")

    def job_id_for_run_dir(self, run_dir: str | Path) -> str:
        return _safe_job_id(Path(run_dir).expanduser().name)

    def record_for_run_dir(self, run_dir: str | Path) -> dict:
        return self.read(self.job_id_for_run_dir(run_dir))

    def sync_legacy_submit_run(
        self,
        run_dir: str | Path,
        *,
        batch_meta: dict,
        created_at: str = "",
        model: str = "",
        input_location: str = "",
        image_count: int = 0,
        chunk_count: int = 0,
        status: str = "submitted",
        results: dict | None = None,
    ) -> dict:
        run_path = Path(run_dir).expanduser()
        job_id = self.job_id_for_run_dir(run_path)
        record = self.read(job_id)
        if not record:
            record = {
                "schema_version": JOB_STORE_SCHEMA_VERSION,
                "job_id": job_id,
                "kind": "batch",
                "created_at": created_at or str(batch_meta.get("created_at") or ""),
                "legacy": {"submit_run_dir": str(run_path)},
                "operations": [],
                "dataset": {"versions": []},
                "retrieval": {},
                "metrics": {},
            }

        record["kind"] = "batch"
        canonical_status = status
        if results and status == "retrieved":
            canonical_status = (
                "retrieved_complete"
                if int(results.get("missing_pages") or 0) == 0
                else "retrieved_partial"
            )
        record["status"] = canonical_status
        record["model"] = model or str(batch_meta.get("model") or record.get("model") or "")
        record["created_at"] = created_at or str(record.get("created_at") or "")
        record["legacy"] = {**(record.get("legacy") or {}), "submit_run_dir": str(run_path)}
        record["input"] = {
            "location": input_location,
            "image_count": int(image_count or 0),
        }
        record["batches"] = {
            "source_run_dir": str(run_path),
            "batch_job_names": [
                item.get("batch_job_name")
                for item in (batch_meta.get("batch_jobs") or [])
                if isinstance(item, dict) and item.get("batch_job_name")
            ]
            or list(batch_meta.get("batch_job_names") or []),
            "chunk_count": int(chunk_count or 0),
            "attempts": list(batch_meta.get("retry_runs") or []),
        }
        if results:
            self._apply_results_payload(record, results, operation="legacy_retrieval")
        self.write(job_id, record)
        return record

    def build_retrieval_signature(
        self,
        run_dir: str | Path,
        *,
        allow_partial: bool = False,
        recover_missing_with_api: bool = False,
        duplicate_strategy: str = "",
    ) -> str:
        run_path = Path(run_dir).expanduser()
        batch_meta = read_json_file(run_path / "batch_job.json")
        batch_jobs = []
        for item in batch_meta.get("batch_jobs") or []:
            if not isinstance(item, dict):
                continue
            batch_jobs.append(
                {
                    "batch_job_name": item.get("batch_job_name"),
                    "requests_file": item.get("requests_file"),
                    "request_count": item.get("request_count"),
                    "output_destination": item.get("output_destination"),
                    "is_retry": bool(item.get("is_retry")),
                    "retry_run_id": item.get("retry_run_id"),
                }
            )
        payload = {
            "schema_version": JOB_STORE_SCHEMA_VERSION,
            "source_run_dir": str(run_path),
            "model": batch_meta.get("model"),
            "provider": batch_meta.get("provider"),
            "batch_jobs": batch_jobs,
            "allow_partial": bool(allow_partial),
            "recover_missing_with_api": bool(recover_missing_with_api),
            "duplicate_strategy": duplicate_strategy or "",
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def cached_retrieval(
        self,
        run_dir: str | Path,
        *,
        signature: str,
    ) -> dict:
        record = self.record_for_run_dir(run_dir)
        retrieval = record.get("retrieval") if isinstance(record, dict) else {}
        if not isinstance(retrieval, dict):
            return {}
        if retrieval.get("signature") != signature:
            return {}
        payload = retrieval.get("payload")
        if not isinstance(payload, dict):
            return {}
        dataset_path = Path(str(payload.get("dataset_path") or "")).expanduser()
        if not dataset_path.is_file():
            return {}
        return dict(payload)

    def current_results_for_run_dir(self, run_dir: str | Path) -> dict:
        record = self.record_for_run_dir(run_dir)
        retrieval = record.get("retrieval") if isinstance(record, dict) else {}
        if not isinstance(retrieval, dict):
            return {}
        payload = retrieval.get("payload")
        if not isinstance(payload, dict):
            return {}
        dataset_path = Path(str(payload.get("dataset_path") or "")).expanduser()
        if dataset_path and dataset_path.is_file():
            return dict(payload)
        return {}

    def record_retrieval(
        self,
        run_dir: str | Path,
        payload: dict,
        *,
        signature: str,
        operation: str = "retrieve",
    ) -> dict:
        run_path = Path(run_dir).expanduser()
        job_id = self.job_id_for_run_dir(run_path)
        record = self.read(job_id)
        if not record:
            record = {
                "schema_version": JOB_STORE_SCHEMA_VERSION,
                "job_id": job_id,
                "kind": "batch",
                "created_at": "",
                "legacy": {"submit_run_dir": str(run_path)},
                "operations": [],
                "dataset": {"versions": []},
                "retrieval": {},
                "metrics": {},
            }

        updated_payload = dict(payload)
        dataset_path = Path(str(updated_payload.get("dataset_path") or "")).expanduser()
        dataset = record.get("dataset")
        if not isinstance(dataset, dict):
            dataset = {"versions": []}
        versions = dataset.get("versions")
        if not isinstance(versions, list):
            versions = []
        if dataset_path.is_file():
            canonical_path = _copy_dataset_into_job(
                job_dir=self.job_dir(job_id),
                source_path=dataset_path,
                operation=operation,
                version_count=len(versions),
            )
            updated_payload["dataset_path"] = str(canonical_path)
            dataset["current_path"] = str(canonical_path)
            dataset["current_gcs_uri"] = str(updated_payload.get("dataset_gcs_uri") or "")
            versions.append(
                {
                    "created_at": utc_now_iso(),
                    "operation": operation,
                    "path": str(canonical_path),
                    "source_path": str(dataset_path),
                    "rows_written": int(updated_payload.get("rows_written") or 0),
                    "successful_pages": int(updated_payload.get("successful_pages") or 0),
                    "missing_pages": int(updated_payload.get("missing_pages") or 0),
                }
            )
            dataset["versions"] = versions
            record["dataset"] = dataset

        self._apply_results_payload(record, updated_payload, operation=operation)
        record["retrieval"] = {
            "signature": signature,
            "retrieved_at": str(updated_payload.get("retrieved_at") or utc_now_iso()),
            "operation": operation,
            "payload": updated_payload,
        }
        record["status"] = (
            "retrieved_complete"
            if int(updated_payload.get("missing_pages") or 0) == 0
            else "retrieved_partial"
        )
        self.write(job_id, record)
        self.append_event(job_id, operation, {"signature": signature})
        return updated_payload

    def mark_retry_submitted(self, run_dir: str | Path) -> None:
        job_id = self.job_id_for_run_dir(run_dir)
        record = self.read(job_id)
        if not record:
            return
        record["status"] = "retry_submitted"
        record["retrieval"] = {}
        self.write(job_id, record)
        self.append_event(job_id, "retry_submitted", {})

    def _apply_results_payload(
        self,
        record: dict,
        payload: dict,
        *,
        operation: str,
    ) -> None:
        record["metrics"] = {
            "expected_pages": int(payload.get("expected_pages") or 0),
            "successful_pages": int(payload.get("successful_pages") or 0),
            "missing_pages": int(payload.get("missing_pages") or 0),
            "recovered_pages": int(payload.get("recovered_pages") or 0),
            "rows_written": int(payload.get("rows_written") or 0),
            "operation": operation,
        }
