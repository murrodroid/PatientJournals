from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable


JOB_STORE_SCHEMA_VERSION = 2
JOBS_DIR_NAME = "jobs"
JOB_FILE_NAME = "job.json"
EVENTS_FILE_NAME = "events.jsonl"
DB_FILE_NAME = "app_state.sqlite3"


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


def _json_dumps(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)


def _json_loads(value: str | None) -> dict:
    if not value:
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


class JobStore:
    """SQLite-backed app state for jobs.

    Operational run folders are artifacts only. The app-facing job entity,
    retrieval payload, current dataset pointer, and events live in
    ``runs/app_state.sqlite3``.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        self.jobs_root = self.root / JOBS_DIR_NAME
        self.db_path = self.root / DB_FILE_NAME
        self._ensure_schema()

    @classmethod
    def for_run_dir(cls, run_dir: str | Path) -> "JobStore":
        return cls(_submit_root_for_run_dir(Path(run_dir).expanduser()))

    def _connect(self) -> sqlite3.Connection:
        self.root.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _ensure_schema(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL DEFAULT 'batch',
                    status TEXT NOT NULL DEFAULT 'unknown',
                    created_at TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT '',
                    run_dir TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '',
                    provider TEXT NOT NULL DEFAULT '',
                    input_location TEXT NOT NULL DEFAULT '',
                    image_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    current_dataset_path TEXT NOT NULL DEFAULT '',
                    current_dataset_gcs_uri TEXT NOT NULL DEFAULT '',
                    record_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dataset_versions (
                    version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    path TEXT NOT NULL,
                    source_path TEXT NOT NULL DEFAULT '',
                    rows_written INTEGER NOT NULL DEFAULT 0,
                    successful_pages INTEGER NOT NULL DEFAULT 0,
                    missing_pages INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT NOT NULL DEFAULT '',
                    finished_at TEXT NOT NULL DEFAULT '',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_kind_updated ON jobs(kind, updated_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_job ON job_events(job_id, event_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status, updated_at)"
            )
            conn.execute(f"PRAGMA user_version={JOB_STORE_SCHEMA_VERSION}")

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_root / _safe_job_id(job_id)

    def job_path(self, job_id: str) -> Path:
        # Deprecated JSON location. Kept only so callers can find old artifacts.
        return self.job_dir(job_id) / JOB_FILE_NAME

    def event_path(self, job_id: str) -> Path:
        # Deprecated JSONL location. Events are stored in SQLite.
        return self.job_dir(job_id) / EVENTS_FILE_NAME

    def _legacy_json_record(self, job_id: str) -> dict:
        path = self.job_path(job_id)
        record = read_json_file(path)
        if record:
            self.write(job_id, record)
        return record

    def read(self, job_id: str) -> dict:
        safe_id = _safe_job_id(job_id)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT record_json FROM jobs WHERE job_id = ?",
                (safe_id,),
            ).fetchone()
        if row is None:
            return self._legacy_json_record(safe_id)
        return _json_loads(str(row["record_json"]))

    def write(self, job_id: str, record: dict) -> None:
        safe_id = _safe_job_id(job_id)
        now = utc_now_iso()
        updated = dict(record)
        updated["schema_version"] = JOB_STORE_SCHEMA_VERSION
        updated["job_id"] = safe_id
        updated["updated_at"] = now

        legacy = updated.get("legacy") if isinstance(updated.get("legacy"), dict) else {}
        input_payload = updated.get("input") if isinstance(updated.get("input"), dict) else {}
        batches = updated.get("batches") if isinstance(updated.get("batches"), dict) else {}
        dataset = updated.get("dataset") if isinstance(updated.get("dataset"), dict) else {}

        run_dir = str(legacy.get("submit_run_dir") or batches.get("source_run_dir") or "")
        current_dataset_path = str(dataset.get("current_path") or "")
        current_dataset_gcs_uri = str(dataset.get("current_gcs_uri") or "")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, kind, status, created_at, updated_at, run_dir, model,
                    provider, input_location, image_count, chunk_count,
                    current_dataset_path, current_dataset_gcs_uri, record_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    kind = excluded.kind,
                    status = excluded.status,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    run_dir = excluded.run_dir,
                    model = excluded.model,
                    provider = excluded.provider,
                    input_location = excluded.input_location,
                    image_count = excluded.image_count,
                    chunk_count = excluded.chunk_count,
                    current_dataset_path = excluded.current_dataset_path,
                    current_dataset_gcs_uri = excluded.current_dataset_gcs_uri,
                    record_json = excluded.record_json
                """,
                (
                    safe_id,
                    str(updated.get("kind") or "batch"),
                    str(updated.get("status") or "unknown"),
                    str(updated.get("created_at") or ""),
                    now,
                    run_dir,
                    str(updated.get("model") or ""),
                    str(updated.get("provider") or ""),
                    str(input_payload.get("location") or ""),
                    int(input_payload.get("image_count") or 0),
                    int(batches.get("chunk_count") or 0),
                    current_dataset_path,
                    current_dataset_gcs_uri,
                    _json_dumps(updated),
                ),
            )

    def append_event(self, job_id: str, event_type: str, payload: dict | None = None) -> None:
        safe_id = _safe_job_id(job_id)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO job_events (job_id, created_at, event_type, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    safe_id,
                    utc_now_iso(),
                    event_type,
                    _json_dumps(payload or {}),
                ),
            )

    def events(self, job_id: str, *, limit: int = 200) -> list[dict]:
        safe_id = _safe_job_id(job_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT created_at, event_type, payload_json
                FROM job_events
                WHERE job_id = ?
                ORDER BY event_id DESC
                LIMIT ?
                """,
                (safe_id, max(1, limit)),
            ).fetchall()
        return [
            {
                "at": str(row["created_at"]),
                "type": str(row["event_type"]),
                "payload": _json_loads(str(row["payload_json"])),
            }
            for row in rows
        ]

    def list_records(self, *, kind: str | None = None) -> list[dict]:
        query = "SELECT record_json FROM jobs"
        params: tuple[object, ...] = ()
        if kind:
            query += " WHERE kind = ?"
            params = (kind,)
        query += " ORDER BY COALESCE(NULLIF(created_at, ''), updated_at) DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_json_loads(str(row["record_json"])) for row in rows]

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
        record["provider"] = str(batch_meta.get("provider") or record.get("provider") or "")
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
        ignore_failed: bool = False,
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
            "ignore_failed": bool(ignore_failed),
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
            retrieval = {}
        payload = retrieval.get("payload")
        if isinstance(payload, dict):
            dataset_path = Path(str(payload.get("dataset_path") or "")).expanduser()
            if dataset_path and dataset_path.is_file():
                return dict(payload)

        dataset = record.get("dataset") if isinstance(record, dict) else {}
        if not isinstance(dataset, dict):
            return {}
        current_path = Path(str(dataset.get("current_path") or "")).expanduser()
        if not current_path.is_file():
            return {}
        metrics = record.get("metrics") if isinstance(record, dict) else {}
        if not isinstance(metrics, dict):
            metrics = {}
        batches = record.get("batches") if isinstance(record, dict) else {}
        if not isinstance(batches, dict):
            batches = {}
        batch_names = batches.get("batch_job_names")
        batch_count = len(batch_names) if isinstance(batch_names, list) else 0
        failed_rows_included = int(metrics.get("failed_rows_included") or 0)
        operation = str(metrics.get("operation") or "")
        return {
            "retrieved_at": str(record.get("updated_at") or utc_now_iso()),
            "dataset_path": str(current_path),
            "dataset_gcs_uri": str(dataset.get("current_gcs_uri") or ""),
            "provider": str(record.get("provider") or ""),
            "batch_count": batch_count,
            "rows_written": int(metrics.get("rows_written") or 0),
            "error_rows": int(metrics.get("missing_pages") or 0),
            "expected_pages": int(metrics.get("expected_pages") or 0),
            "observed_pages": int(metrics.get("expected_pages") or 0),
            "successful_pages": int(metrics.get("successful_pages") or 0),
            "recovered_pages": int(metrics.get("recovered_pages") or 0),
            "failed_rows_included": failed_rows_included,
            "missing_pages": int(metrics.get("missing_pages") or 0),
            "submit_failed": False,
            "ignore_failed": bool(metrics.get("ignore_failed"))
            or operation == "finalize_failed"
            or failed_rows_included > 0,
            "finalized_with_failed_rows": operation == "finalize_failed",
            "api_recovery_attempted": bool(metrics.get("api_recovery_attempted")),
            "api_recovery_completed": bool(metrics.get("api_recovery_completed")),
        }

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
        if "missing_pages" not in updated_payload:
            expected_pages = int(updated_payload.get("expected_pages") or 0)
            successful_pages = int(updated_payload.get("successful_pages") or 0)
            updated_payload["missing_pages"] = max(0, expected_pages - successful_pages)
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
            version = {
                "created_at": utc_now_iso(),
                "operation": operation,
                "path": str(canonical_path),
                "source_path": str(dataset_path),
                "rows_written": int(updated_payload.get("rows_written") or 0),
                "successful_pages": int(updated_payload.get("successful_pages") or 0),
                "missing_pages": int(updated_payload.get("missing_pages") or 0),
            }
            versions.append(version)
            dataset["versions"] = versions
            record["dataset"] = dataset
            self.write(job_id, record)
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO dataset_versions (
                        job_id, created_at, operation, path, source_path,
                        rows_written, successful_pages, missing_pages
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        version["created_at"],
                        operation,
                        str(canonical_path),
                        str(dataset_path),
                        int(updated_payload.get("rows_written") or 0),
                        int(updated_payload.get("successful_pages") or 0),
                        int(updated_payload.get("missing_pages") or 0),
                    ),
                )

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
        expected_pages = int(payload.get("expected_pages") or 0)
        successful_pages = int(payload.get("successful_pages") or 0)
        missing_pages = (
            int(payload.get("missing_pages") or 0)
            if "missing_pages" in payload
            else max(0, expected_pages - successful_pages)
        )
        record["metrics"] = {
            "expected_pages": expected_pages,
            "successful_pages": successful_pages,
            "missing_pages": missing_pages,
            "recovered_pages": int(payload.get("recovered_pages") or 0),
            "failed_rows_included": int(payload.get("failed_rows_included") or 0),
            "rows_written": int(payload.get("rows_written") or 0),
            "ignore_failed": bool(payload.get("ignore_failed")),
            "api_recovery_attempted": bool(payload.get("api_recovery_attempted")),
            "api_recovery_completed": bool(payload.get("api_recovery_completed")),
            "operation": operation,
        }

    def upsert_task(
        self,
        task_id: str,
        *,
        kind: str,
        status: str,
        metadata: dict | None = None,
        result: dict | None = None,
        error: str = "",
        started_at: str = "",
        finished_at: str = "",
    ) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT created_at, started_at FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            created_at = str(existing["created_at"]) if existing else now
            persisted_started = str(existing["started_at"]) if existing else ""
            conn.execute(
                """
                INSERT INTO tasks (
                    task_id, kind, status, created_at, updated_at, started_at,
                    finished_at, result_json, error, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    kind = excluded.kind,
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    started_at = excluded.started_at,
                    finished_at = excluded.finished_at,
                    result_json = excluded.result_json,
                    error = excluded.error,
                    metadata_json = excluded.metadata_json
                """,
                (
                    task_id,
                    kind,
                    status,
                    created_at,
                    now,
                    started_at or persisted_started,
                    finished_at,
                    _json_dumps(result or {}),
                    error,
                    _json_dumps(metadata or {}),
                ),
            )

    def list_tasks(self, *, limit: int = 100) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT task_id, kind, status, created_at, updated_at, started_at,
                       finished_at, result_json, error, metadata_json
                FROM tasks
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()
        return [
            {
                "task_id": str(row["task_id"]),
                "kind": str(row["kind"]),
                "status": str(row["status"]),
                "created_at": str(row["created_at"]),
                "updated_at": str(row["updated_at"]),
                "started_at": str(row["started_at"]),
                "finished_at": str(row["finished_at"]),
                "result": _json_loads(str(row["result_json"])),
                "error": str(row["error"]),
                "metadata": _json_loads(str(row["metadata_json"])),
            }
            for row in rows
        ]

    def dataset_versions(self, job_id: str) -> list[dict]:
        safe_id = _safe_job_id(job_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT created_at, operation, path, source_path, rows_written,
                       successful_pages, missing_pages
                FROM dataset_versions
                WHERE job_id = ?
                ORDER BY version_id DESC
                """,
                (safe_id,),
            ).fetchall()
        return [
            {
                "created_at": str(row["created_at"]),
                "operation": str(row["operation"]),
                "path": str(row["path"]),
                "source_path": str(row["source_path"]),
                "rows_written": int(row["rows_written"]),
                "successful_pages": int(row["successful_pages"]),
                "missing_pages": int(row["missing_pages"]),
            }
            for row in rows
        ]


def records_by_run_dir(records: Iterable[dict]) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for record in records:
        legacy = record.get("legacy") if isinstance(record.get("legacy"), dict) else {}
        batches = record.get("batches") if isinstance(record.get("batches"), dict) else {}
        run_dir = str(legacy.get("submit_run_dir") or batches.get("source_run_dir") or "")
        if run_dir:
            output[run_dir] = record
    return output
