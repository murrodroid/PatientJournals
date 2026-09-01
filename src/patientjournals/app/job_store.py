from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

from patientjournals.validation.publication import publication_idempotency_key


JOB_STORE_SCHEMA_VERSION = 3
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
    version_number: int | None = None,
) -> tuple[Path, Path]:
    if not source_path.exists() or not source_path.is_file():
        return source_path, source_path

    dataset_dir, versions_dir = _dataset_files(job_dir)
    suffix = source_path.suffix or ".jsonl"
    current_path = dataset_dir / f"current{suffix}"
    selected_version = int(version_number or (version_count + 1))
    if selected_version <= 0:
        raise ValueError("Dataset version number must be positive.")
    version_path = versions_dir / f"v{selected_version:03d}_{operation}{suffix}"
    if version_path.exists() and version_path.read_bytes() != source_path.read_bytes():
        raise RuntimeError(
            f"Immutable dataset version already exists with different bytes: {version_path}"
        )

    def atomic_copy(target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        shutil.copy2(source_path, temporary)
        temporary.replace(target)

    # Establish the immutable version before advancing the mutable current pointer.
    # A crash at either boundary is therefore safe to replay.
    if not version_path.exists():
        atomic_copy(version_path)
    if source_path.resolve() != current_path.resolve():
        atomic_copy(current_path)
    return current_path, version_path


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


def _portable_run_id(explicit_id: object, path_value: object) -> str:
    """Return the run ID used by shared publication, independent of its root."""

    raw_id = str(explicit_id or "").strip()
    if raw_id:
        return _safe_job_id(raw_id)
    raw_path = str(path_value or "").strip()
    if not raw_path:
        return ""
    return _safe_job_id(Path(raw_path).expanduser().name)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_validation_idempotency_key(
    *,
    source_run_id: str,
    verification_run_id: str,
    candidate_hash: str,
    verification_model: str,
    verification_prompt_hash: str,
) -> str:
    identity = {
        "source_run_id": source_run_id,
        "verification_run_id": verification_run_id,
        "candidate_hash": candidate_hash,
        "verification_model": verification_model,
        "verification_prompt_hash": verification_prompt_hash,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _dataset_publication_idempotency_key(
    *,
    source_run_id: str,
    verification_run_id: str,
    candidate_hash: str,
    verification_prompt_hash: str,
    dataset_sha256: str,
    publication_provenance_sha256: str,
) -> str:
    """Use the shared publisher's portable, content-bound identity."""

    return publication_idempotency_key(
        source_run_id=source_run_id,
        verification_run_id=verification_run_id,
        candidate_hash=candidate_hash,
        verification_prompt_hash=verification_prompt_hash,
        dataset_sha256=dataset_sha256,
        publication_provenance_sha256=publication_provenance_sha256,
    )


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
                """
                CREATE TABLE IF NOT EXISTS schema_versions (
                    version_id TEXT PRIMARY KEY,
                    family_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    version_number INTEGER NOT NULL,
                    parent_version_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL DEFAULT '',
                    prompt_name TEXT NOT NULL DEFAULT 'frontpage',
                    source TEXT NOT NULL DEFAULT 'local',
                    schema_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL DEFAULT ''
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_schema_family_version "
                "ON schema_versions(family_id, version_number DESC)"
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

        legacy = (
            updated.get("legacy") if isinstance(updated.get("legacy"), dict) else {}
        )
        input_payload = (
            updated.get("input") if isinstance(updated.get("input"), dict) else {}
        )
        batches = (
            updated.get("batches") if isinstance(updated.get("batches"), dict) else {}
        )
        dataset = (
            updated.get("dataset") if isinstance(updated.get("dataset"), dict) else {}
        )

        run_dir = str(
            legacy.get("submit_run_dir") or batches.get("source_run_dir") or ""
        )
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

    def append_event(
        self, job_id: str, event_type: str, payload: dict | None = None
    ) -> None:
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
        record["model"] = model or str(
            batch_meta.get("model") or record.get("model") or ""
        )
        record["provider"] = str(
            batch_meta.get("provider") or record.get("provider") or ""
        )
        existing_schema = record.get("schema")
        if not isinstance(existing_schema, dict):
            existing_schema = {}
        record["schema"] = {
            "name": str(
                batch_meta.get("schema_name") or existing_schema.get("name") or ""
            ),
            "version_id": str(
                batch_meta.get("schema_version_id")
                or existing_schema.get("version_id")
                or ""
            ),
        }
        record["created_at"] = created_at or str(record.get("created_at") or "")
        record["legacy"] = {
            **(record.get("legacy") or {}),
            "submit_run_dir": str(run_path),
        }
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
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode(
            "utf-8"
        )
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
        version_number: int | None = None,
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
            if version_number is not None:
                occupied = [
                    item
                    for item in versions
                    if isinstance(item, dict)
                    and int(item.get("version") or 0) == int(version_number)
                ]
                if occupied:
                    raise RuntimeError(
                        f"Dataset version v{int(version_number):03d} is already recorded."
                    )
            canonical_path, version_path = _copy_dataset_into_job(
                job_dir=self.job_dir(job_id),
                source_path=dataset_path,
                operation=operation,
                version_count=len(versions),
                version_number=version_number,
            )
            updated_payload["dataset_path"] = str(canonical_path)
            dataset["current_path"] = str(canonical_path)
            dataset["current_gcs_uri"] = str(
                updated_payload.get("dataset_gcs_uri") or ""
            )
            version = {
                "version": int(version_number or (len(versions) + 1)),
                "created_at": utc_now_iso(),
                "operation": operation,
                "path": str(version_path),
                "source_path": str(dataset_path),
                "rows_written": int(updated_payload.get("rows_written") or 0),
                "successful_pages": int(updated_payload.get("successful_pages") or 0),
                "missing_pages": int(updated_payload.get("missing_pages") or 0),
            }
            if operation == "model_validation":
                version.update(
                    {
                        "version_id": str(
                            updated_payload.get("dataset_version_id") or ""
                        ),
                        "dataset_sha256": str(
                            updated_payload.get("dataset_sha256") or ""
                        ),
                        "dataset_gcs_uri": str(
                            updated_payload.get("dataset_gcs_uri") or ""
                        ),
                        "dataset_gcs_generation": str(
                            updated_payload.get("dataset_gcs_generation") or ""
                        ),
                        "dataset_version_ledger_gcs_uri": str(
                            updated_payload.get("dataset_version_ledger_gcs_uri") or ""
                        ),
                        "publication_idempotency_key": str(
                            updated_payload.get("dataset_publication_idempotency_key")
                            or ""
                        ),
                        "publication_provenance_sha256": str(
                            updated_payload.get("publication_provenance_sha256") or ""
                        ),
                        "source_run_id": str(
                            updated_payload.get("source_run_id") or ""
                        ),
                        "verification_run_id": str(
                            updated_payload.get("verification_run_id") or ""
                        ),
                        "candidate_hash": str(
                            updated_payload.get("candidate_hash") or ""
                        ),
                        "verification_prompt_hash": str(
                            updated_payload.get("verification_prompt_hash") or ""
                        ),
                    }
                )
            versions.append(version)
            dataset["versions"] = versions
            record["dataset"] = dataset
            self.write(job_id, record)
            self._ensure_dataset_version_index(job_id, version)

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

    def _ensure_dataset_version_index(self, job_id: str, version: dict) -> None:
        """Idempotently mirror a JSON dataset-version record into SQLite."""

        version_path = str(version.get("path") or "")
        if not version_path:
            raise ValueError("Dataset version index record has no path.")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                """
                SELECT operation
                FROM dataset_versions
                WHERE job_id = ? AND path = ?
                """,
                (_safe_job_id(job_id), version_path),
            ).fetchone()
            if existing is not None:
                if str(existing["operation"]) != str(version.get("operation") or ""):
                    raise RuntimeError(
                        "Dataset version index path is already bound to a different "
                        "operation."
                    )
                return
            conn.execute(
                """
                INSERT INTO dataset_versions (
                    job_id, created_at, operation, path, source_path,
                    rows_written, successful_pages, missing_pages
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _safe_job_id(job_id),
                    str(version.get("created_at") or utc_now_iso()),
                    str(version.get("operation") or ""),
                    version_path,
                    str(version.get("source_path") or ""),
                    int(version.get("rows_written") or 0),
                    int(version.get("successful_pages") or 0),
                    int(version.get("missing_pages") or 0),
                ),
            )

    def _reconcile_model_validation_version(
        self,
        *,
        job_id: str,
        record: dict,
        updated_payload: dict,
        version_number: int,
        idempotency_key: str,
    ) -> tuple[dict, dict] | None:
        """Finish app recording after a crash that already wrote the shared vNNN.

        The existing app version is reusable only when its content digest and all
        immutable cloud/publication bindings match. Missing provenance is treated
        as ambiguous and fails closed.
        """

        dataset = record.get("dataset")
        if not isinstance(dataset, dict):
            return None
        versions = dataset.get("versions")
        if not isinstance(versions, list):
            return None
        matches = [
            (index, item)
            for index, item in enumerate(versions)
            if isinstance(item, dict)
            and int(item.get("version") or 0) == int(version_number)
        ]
        if not matches:
            return None
        if len(matches) != 1:
            raise RuntimeError(
                f"Dataset version v{version_number:03d} is recorded more than once."
            )

        version_index, existing_version = matches[0]
        if str(existing_version.get("operation") or "") != "model_validation":
            raise RuntimeError(
                f"Dataset version v{version_number:03d} belongs to a different "
                "operation."
            )

        # A crash may have happened after the final retrieval payload was written
        # or one write earlier, when only the version record existed. Merge both
        # locations, preferring the version's immutable fields.
        recorded_provenance: dict = {}
        retrieval = record.get("retrieval")
        if isinstance(retrieval, dict):
            retrieval_payload = retrieval.get("payload")
            if isinstance(retrieval_payload, dict) and int(
                retrieval_payload.get("dataset_version") or 0
            ) == int(version_number):
                recorded_provenance.update(retrieval_payload)
        recorded_provenance.update(existing_version)

        comparisons = (
            ("version_id", "dataset_version_id"),
            ("dataset_sha256", "dataset_sha256"),
            ("dataset_gcs_uri", "dataset_gcs_uri"),
            ("dataset_gcs_generation", "dataset_gcs_generation"),
            (
                "dataset_version_ledger_gcs_uri",
                "dataset_version_ledger_gcs_uri",
            ),
            (
                "publication_idempotency_key",
                "dataset_publication_idempotency_key",
            ),
            (
                "publication_provenance_sha256",
                "publication_provenance_sha256",
            ),
            ("source_run_id", "source_run_id"),
            ("verification_run_id", "verification_run_id"),
            ("candidate_hash", "candidate_hash"),
            ("verification_prompt_hash", "verification_prompt_hash"),
        )
        for recorded_key, incoming_key in comparisons:
            recorded_value = str(recorded_provenance.get(recorded_key) or "")
            incoming_value = str(updated_payload.get(incoming_key) or "")
            if not recorded_value:
                raise RuntimeError(
                    f"Cannot safely reconcile v{version_number:03d}: recorded "
                    f"publication provenance is missing {recorded_key}."
                )
            if recorded_value != incoming_value:
                raise RuntimeError(
                    f"Conflicting model-validation replay for v{version_number:03d}: "
                    f"{incoming_key} changed."
                )

        source_path = Path(str(updated_payload.get("dataset_path") or "")).expanduser()
        if not source_path.is_file():
            raise FileNotFoundError(
                "Cannot reconcile the recorded dataset version because the shared "
                f"publication file is missing: {source_path}"
            )
        expected_digest = str(updated_payload.get("dataset_sha256") or "")
        if _file_sha256(source_path) != expected_digest:
            raise RuntimeError(
                "Cannot reconcile the recorded dataset version from different bytes."
            )

        current_path, version_path = _copy_dataset_into_job(
            job_dir=self.job_dir(job_id),
            source_path=source_path,
            operation="model_validation",
            version_count=len(versions),
            version_number=version_number,
        )
        reconciled_version = {
            **existing_version,
            "path": str(version_path),
            "source_path": str(source_path),
        }
        versions[version_index] = reconciled_version
        dataset["versions"] = versions
        dataset["current_path"] = str(current_path)
        dataset["current_gcs_uri"] = str(updated_payload.get("dataset_gcs_uri") or "")
        record["dataset"] = dataset
        updated_payload["dataset_path"] = str(current_path)
        updated_payload["dataset_version_path"] = str(version_path)
        updated_payload["reconciled_recorded_dataset_version"] = True
        self._apply_results_payload(
            record, updated_payload, operation="model_validation"
        )
        record["retrieval"] = {
            "signature": idempotency_key,
            "retrieved_at": str(updated_payload.get("retrieved_at") or utc_now_iso()),
            "operation": "model_validation",
            "payload": updated_payload,
        }
        record["status"] = "retrieved_complete"
        self.write(job_id, record)
        self._ensure_dataset_version_index(job_id, reconciled_version)
        self.append_event(
            job_id,
            "model_validation_version_reconciled",
            {
                "dataset_version": version_number,
                "publication_idempotency_key": str(
                    updated_payload.get("dataset_publication_idempotency_key") or ""
                ),
            },
        )
        return updated_payload, self.read(job_id)

    def record_candidate_retrieval(
        self,
        run_dir: str | Path,
        payload: dict,
        *,
        signature: str,
    ) -> dict:
        """Record extraction retrieval without publishing a dataset version.

        Model-validation-enabled jobs deliberately keep their extracted dataset
        as a pre-version candidate.  The first entry in ``dataset_versions`` is
        created only after the validator has completed and its accepted patches
        have passed the original output schema again.
        """

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
        expected_pages = int(updated_payload.get("expected_pages") or 0)
        successful_pages = int(updated_payload.get("successful_pages") or 0)
        updated_payload.setdefault(
            "missing_pages", max(0, expected_pages - successful_pages)
        )
        candidate_path = str(updated_payload.get("dataset_path") or "")
        updated_payload["candidate_dataset_path"] = candidate_path
        candidates_complete = (
            expected_pages > 0
            and successful_pages == expected_pages
            and bool(updated_payload.get("page_candidates_path"))
            and bool(updated_payload.get("deterministic_routing_path"))
        )
        candidate_status = "pending" if candidates_complete else "candidate_incomplete"
        updated_payload["model_validation_status"] = candidate_status

        self._apply_results_payload(
            record,
            updated_payload,
            operation="candidate_retrieve",
        )
        record["retrieval"] = {
            "signature": signature,
            "retrieved_at": str(updated_payload.get("retrieved_at") or utc_now_iso()),
            "operation": "candidate_retrieve",
            "payload": updated_payload,
        }
        validation = record.get("model_validation")
        if not isinstance(validation, dict):
            validation = {}
        record["model_validation"] = {
            **validation,
            "enabled": True,
            "status": candidate_status,
            "candidate_dataset_path": candidate_path,
            "page_candidates_path": str(
                updated_payload.get("page_candidates_path") or ""
            ),
            "page_candidates_gcs_uri": str(
                updated_payload.get("page_candidates_gcs_uri") or ""
            ),
            "page_candidates_sha256": str(
                updated_payload.get("page_candidates_sha256") or ""
            ),
            "page_candidates_gcs_generation": str(
                updated_payload.get("page_candidates_gcs_generation") or ""
            ),
            "deterministic_routing_path": str(
                updated_payload.get("deterministic_routing_path") or ""
            ),
            "deterministic_routing_gcs_uri": str(
                updated_payload.get("deterministic_routing_gcs_uri") or ""
            ),
            "deterministic_routing_sha256": str(
                updated_payload.get("deterministic_routing_sha256") or ""
            ),
            "deterministic_routing_gcs_generation": str(
                updated_payload.get("deterministic_routing_gcs_generation") or ""
            ),
            "subagent_combined_gcs_uri": str(
                updated_payload.get("subagent_combined_gcs_uri") or ""
            ),
            "subagent_combined_sha256": str(
                updated_payload.get("subagent_combined_sha256") or ""
            ),
            "subagent_combined_gcs_generation": str(
                updated_payload.get("subagent_combined_gcs_generation") or ""
            ),
            "subagent_failures_gcs_uri": str(
                updated_payload.get("subagent_failures_gcs_uri") or ""
            ),
            "subagent_failures_sha256": str(
                updated_payload.get("subagent_failures_sha256") or ""
            ),
            "subagent_failures_gcs_generation": str(
                updated_payload.get("subagent_failures_gcs_generation") or ""
            ),
            "deterministic_flagged_pages": int(
                updated_payload.get("deterministic_flagged_pages") or 0
            ),
            "deterministic_routine_pages": int(
                updated_payload.get("deterministic_routine_pages") or 0
            ),
        }
        record["status"] = (
            "validation_pending"
            if candidates_complete
            else "validation_candidate_incomplete"
        )
        self.write(job_id, record)
        self.append_event(job_id, "candidate_retrieve", {"signature": signature})
        return updated_payload

    def mark_model_validation_submitted(
        self,
        run_dir: str | Path,
        *,
        verification_run_dir: str | Path,
        model: str,
        apply_mode: str,
        thinking_level: str = "",
        scope: str = "flagged",
        control_sample_percent: float = 0.0,
        max_output_tokens: int | None = None,
        num_chunks: int | None = None,
    ) -> dict:
        """Link a verifier batch to its extraction job without versioning data."""

        job_id = self.job_id_for_run_dir(run_dir)
        record = self.read(job_id)
        if not record:
            raise ValueError(f"Extraction job is not registered: {run_dir}")
        validation = record.get("model_validation")
        if not isinstance(validation, dict):
            validation = {}
        completed_runs = validation.get("runs")
        if not isinstance(completed_runs, list):
            completed_runs = []
        source_run_id = _portable_run_id(validation.get("source_run_id"), run_dir)
        verification_run_id = _portable_run_id("", verification_run_dir)
        record["model_validation"] = {
            **validation,
            "enabled": True,
            "status": "submitted",
            "verification_run_dir": str(Path(verification_run_dir).expanduser()),
            "source_run_id": source_run_id,
            "verification_run_id": verification_run_id,
            "model": str(model),
            "apply_mode": str(apply_mode),
            "thinking_level": str(thinking_level),
            "scope": str(scope or "flagged"),
            "control_sample_percent": max(
                0.0, min(100.0, float(control_sample_percent))
            ),
            "max_output_tokens": int(max_output_tokens or 0),
            "num_chunks": int(num_chunks or 0),
            "attempt_number": len(completed_runs) + 1,
            "runs": completed_runs,
            "submitted_at": utc_now_iso(),
        }
        record["status"] = "validation_submitted"
        self.write(job_id, record)
        self.append_event(
            job_id,
            "model_validation_submitted",
            {
                "verification_run_dir": str(Path(verification_run_dir).expanduser()),
                "source_run_id": source_run_id,
                "verification_run_id": verification_run_id,
                "model": str(model),
                "apply_mode": str(apply_mode),
                "thinking_level": str(thinking_level),
                "scope": str(scope or "flagged"),
                "control_sample_percent": max(
                    0.0, min(100.0, float(control_sample_percent))
                ),
                "max_output_tokens": int(max_output_tokens or 0),
                "num_chunks": int(num_chunks or 0),
            },
        )
        return record

    def record_model_validation_result(
        self,
        run_dir: str | Path,
        payload: dict,
        *,
        publish_dataset: bool,
    ) -> dict:
        """Record verifier results and optionally publish a dataset version once."""

        run_path = Path(run_dir).expanduser()
        job_id = self.job_id_for_run_dir(run_path)
        record = self.read(job_id)
        if not record:
            raise ValueError(f"Extraction job is not registered: {run_dir}")

        updated_payload = dict(payload)
        validation = record.get("model_validation")
        if not isinstance(validation, dict):
            validation = {}
        completed_runs = validation.get("runs")
        if not isinstance(completed_runs, list):
            completed_runs = []
        verification_run_dir = str(
            updated_payload.get("verification_run_dir")
            or validation.get("verification_run_dir")
            or ""
        )
        source_run_id = _portable_run_id("", run_path)
        provided_source_run_id = str(
            updated_payload.get("source_run_id")
            or validation.get("source_run_id")
            or ""
        )
        if (
            provided_source_run_id
            and _portable_run_id(provided_source_run_id, "") != source_run_id
        ):
            raise ValueError(
                "Model-validation source_run_id does not match the extraction run."
            )
        verification_run_id = _portable_run_id(
            updated_payload.get("verification_run_id")
            or validation.get("verification_run_id"),
            verification_run_dir,
        )
        path_verification_run_id = _portable_run_id("", verification_run_dir)
        if (
            verification_run_id
            and path_verification_run_id
            and verification_run_id != path_verification_run_id
        ):
            raise ValueError(
                "Model-validation verification_run_id does not match its run path."
            )
        updated_payload["source_run_id"] = source_run_id
        updated_payload["verification_run_id"] = verification_run_id
        verification_model = str(
            updated_payload.get("verification_model") or validation.get("model") or ""
        )
        candidate_hash = str(updated_payload.get("candidate_hash") or "")
        verification_prompt_hash = str(
            updated_payload.get("verification_prompt_hash") or ""
        )
        claimed_dataset_sha256 = str(updated_payload.get("dataset_sha256") or "")
        claimed_publication_provenance_sha256 = str(
            updated_payload.get("publication_provenance_sha256") or ""
        )
        publication_idempotency_key = ""
        if publish_dataset:
            if updated_payload.get("publishable") is not True:
                raise ValueError(
                    "Refusing model-validation publication without publishable=True."
                )
            if not claimed_dataset_sha256:
                raise ValueError(
                    "Refusing model-validation publication without dataset_sha256."
                )
            if not claimed_publication_provenance_sha256:
                raise ValueError(
                    "Refusing model-validation publication without "
                    "publication_provenance_sha256."
                )
            publication_idempotency_key = _dataset_publication_idempotency_key(
                source_run_id=source_run_id,
                verification_run_id=verification_run_id,
                candidate_hash=candidate_hash,
                verification_prompt_hash=verification_prompt_hash,
                dataset_sha256=claimed_dataset_sha256,
                publication_provenance_sha256=(claimed_publication_provenance_sha256),
            )
            provided_publication_key = str(
                updated_payload.get("dataset_publication_idempotency_key")
                or updated_payload.get("publication_idempotency_key")
                or ""
            )
            if (
                provided_publication_key
                and provided_publication_key != publication_idempotency_key
            ):
                raise ValueError(
                    "Model-validation publication identity does not match its "
                    "run IDs, candidate, prompt, and dataset digest."
                )
            updated_payload["dataset_publication_idempotency_key"] = (
                publication_idempotency_key
            )
            idempotency_key = publication_idempotency_key
        else:
            idempotency_key = _model_validation_idempotency_key(
                source_run_id=source_run_id,
                verification_run_id=verification_run_id,
                candidate_hash=candidate_hash,
                verification_model=verification_model,
                verification_prompt_hash=verification_prompt_hash,
            )
        updated_payload["model_validation_idempotency_key"] = idempotency_key

        # Retrieval is safe to retry. Once this exact verifier run has published,
        # reuse its immutable version and original artifacts instead of appending
        # a new dataset version. Run IDs and the shared publisher's content-bound
        # identity are portable across relocated run-directory roots.
        published_run = next(
            (
                item
                for item in completed_runs
                if isinstance(item, dict)
                and bool(item.get("published_dataset"))
                and (
                    _portable_run_id(item.get("source_run_id"), run_path)
                    == source_run_id
                    and _portable_run_id(
                        item.get("verification_run_id"),
                        item.get("verification_run_dir"),
                    )
                    == verification_run_id
                )
            ),
            None,
        )
        if published_run is not None:
            recorded_payload = published_run.get("recorded_payload")
            if not isinstance(recorded_payload, dict):
                recorded_payload = {}

            def recorded_value(summary_key: str, payload_key: str) -> str:
                return str(
                    published_run.get(summary_key)
                    or recorded_payload.get(payload_key)
                    or ""
                )

            recorded_candidate_hash = recorded_value("candidate_hash", "candidate_hash")
            recorded_prompt_hash = recorded_value(
                "verification_prompt_hash", "verification_prompt_hash"
            )
            recorded_dataset_sha = recorded_value("dataset_sha256", "dataset_sha256")
            recorded_publication_provenance_sha = recorded_value(
                "publication_provenance_sha256",
                "publication_provenance_sha256",
            )
            recorded_publication_key = recorded_value(
                "publication_idempotency_key",
                "dataset_publication_idempotency_key",
            )
            if not recorded_publication_key and recorded_dataset_sha:
                recorded_publication_key = _dataset_publication_idempotency_key(
                    source_run_id=source_run_id,
                    verification_run_id=verification_run_id,
                    candidate_hash=recorded_candidate_hash,
                    verification_prompt_hash=recorded_prompt_hash,
                    dataset_sha256=recorded_dataset_sha,
                    publication_provenance_sha256=(recorded_publication_provenance_sha),
                )

            for incoming_key, recorded in (
                ("candidate_hash", recorded_candidate_hash),
                ("verification_model", recorded_value("model", "verification_model")),
                ("verification_prompt_hash", recorded_prompt_hash),
                ("dataset_sha256", recorded_dataset_sha),
                (
                    "publication_provenance_sha256",
                    recorded_publication_provenance_sha,
                ),
                (
                    "dataset_version",
                    recorded_value("dataset_version", "dataset_version"),
                ),
                (
                    "dataset_version_id",
                    recorded_value("dataset_version_id", "dataset_version_id"),
                ),
                (
                    "dataset_gcs_uri",
                    recorded_value("dataset_gcs_uri", "dataset_gcs_uri"),
                ),
                (
                    "dataset_gcs_generation",
                    recorded_value("dataset_gcs_generation", "dataset_gcs_generation"),
                ),
                (
                    "dataset_version_ledger_gcs_uri",
                    recorded_value(
                        "dataset_version_ledger_gcs_uri",
                        "dataset_version_ledger_gcs_uri",
                    ),
                ),
                (
                    "dataset_publication_idempotency_key",
                    recorded_publication_key,
                ),
            ):
                incoming_value = str(updated_payload.get(incoming_key) or "")
                if incoming_value != recorded:
                    raise ValueError(
                        "Conflicting model-validation replay for verification run "
                        f"{verification_run_id}: {incoming_key} changed."
                    )

            incoming_dataset_path = Path(
                str(updated_payload.get("dataset_path") or "")
            ).expanduser()
            if (
                incoming_dataset_path.is_file()
                and _file_sha256(incoming_dataset_path) != recorded_dataset_sha
            ):
                raise ValueError(
                    "Conflicting model-validation replay has different local "
                    "dataset bytes."
                )

            replay_payload = dict(updated_payload)
            replay_payload.update(recorded_payload)
            for artifact_key in (
                "results_path",
                "failures_path",
                "summary_path",
                "field_corrections_path",
                "field_corrections_gcs_uri",
                "field_corrections_gcs_generation",
                "field_corrections_sha256",
                "patched_candidates_path",
                "dataset_gcs_uri",
                "artifact_gcs_uris",
            ):
                if artifact_key in published_run:
                    replay_payload[artifact_key] = published_run[artifact_key]
            replay_payload.update(
                {
                    "verification_run_dir": str(
                        published_run.get("verification_run_dir") or ""
                    ),
                    "model_validation_status": str(
                        published_run.get("status") or "published"
                    ),
                    "publishable": True,
                    "dataset_version": published_run.get("dataset_version"),
                    "dataset_version_path": str(
                        published_run.get("dataset_version_path") or ""
                    ),
                    "model_validation_idempotency_key": str(
                        published_run.get("idempotency_key") or idempotency_key
                    ),
                    "dataset_publication_idempotency_key": (recorded_publication_key),
                    "source_run_id": source_run_id,
                    "verification_run_id": verification_run_id,
                    "idempotent_replay": True,
                }
            )
            if replay_payload["dataset_version_path"]:
                replay_payload["dataset_path"] = replay_payload["dataset_version_path"]
            self.append_event(
                job_id,
                "model_validation_retrieval_reused",
                {
                    "verification_run_dir": replay_payload["verification_run_dir"],
                    "dataset_version": replay_payload["dataset_version"],
                },
            )
            return replay_payload

        if publish_dataset:
            dataset_path = Path(
                str(updated_payload.get("dataset_path") or "")
            ).expanduser()
            shared_version = int(updated_payload.get("dataset_version") or 0)
            shared_version_id = str(updated_payload.get("dataset_version_id") or "")
            expected_pages = int(updated_payload.get("expected_pages") or 0)
            completed_pages = int(
                updated_payload.get("completed_pages")
                or updated_payload.get("successful_pages")
                or 0
            )
            if updated_payload.get("publishable") is not True:
                raise ValueError(
                    "Refusing model-validation publication without publishable=True."
                )
            if not dataset_path.is_file():
                raise FileNotFoundError(
                    f"Publishable model-validation dataset is missing: {dataset_path}"
                )
            actual_dataset_sha256 = _file_sha256(dataset_path)
            if actual_dataset_sha256 != claimed_dataset_sha256:
                raise ValueError(
                    "Refusing model-validation publication because dataset_sha256 "
                    "does not match the local shared-version bytes."
                )
            if shared_version <= 0 or shared_version_id != f"v{shared_version:03d}":
                raise ValueError(
                    "Refusing model-validation publication without a shared "
                    "vNNN dataset version allocated by the batch publisher."
                )
            for required_key in (
                "dataset_gcs_uri",
                "dataset_gcs_generation",
                "dataset_sha256",
                "publication_provenance_sha256",
                "dataset_version_ledger_gcs_uri",
            ):
                if not str(updated_payload.get(required_key) or "").strip():
                    raise ValueError(
                        "Refusing model-validation publication without immutable "
                        f"cloud version provenance: {required_key}."
                    )
            if (
                expected_pages <= 0
                or completed_pages != expected_pages
                or int(updated_payload.get("missing_pages") or 0) != 0
                or int(updated_payload.get("failed_pages") or 0) != 0
                or int(updated_payload.get("unverifiable_pages") or 0) != 0
            ):
                raise ValueError(
                    "Refusing model-validation publication until every expected page "
                    "is complete, verifiable, and failure-free."
                )
            reconciled = self._reconcile_model_validation_version(
                job_id=job_id,
                record=record,
                updated_payload=updated_payload,
                version_number=shared_version,
                idempotency_key=idempotency_key,
            )
            if reconciled is None:
                updated_payload = self.record_retrieval(
                    run_path,
                    updated_payload,
                    signature=idempotency_key,
                    operation="model_validation",
                    version_number=shared_version,
                )
                record = self.read(job_id)
            else:
                updated_payload, record = reconciled

        validation = record.get("model_validation")
        if not isinstance(validation, dict):
            validation = {}
        terminal_status = (
            "published"
            if publish_dataset
            else str(updated_payload.get("model_validation_status") or "report_only")
        )
        completed_runs = validation.get("runs")
        if not isinstance(completed_runs, list):
            completed_runs = []
        verification_run_dir = str(
            updated_payload.get("verification_run_dir")
            or validation.get("verification_run_dir")
            or ""
        )
        dataset = record.get("dataset")
        versions = dataset.get("versions") if isinstance(dataset, dict) else []
        if not isinstance(versions, list):
            versions = []
        dataset_version = (
            int(updated_payload.get("dataset_version") or 0)
            if publish_dataset
            else None
        )
        dataset_version_path = (
            next(
                (
                    str(item.get("path") or "")
                    for item in versions
                    if isinstance(item, dict)
                    and int(item.get("version") or 0) == int(dataset_version or 0)
                ),
                "",
            )
            if publish_dataset
            else ""
        )
        if publish_dataset:
            updated_payload["dataset_version"] = dataset_version
            updated_payload["dataset_version_path"] = dataset_version_path
        updated_payload["model_validation_idempotency_key"] = idempotency_key
        run_summary = {
            "verification_run_dir": verification_run_dir,
            "source_run_id": source_run_id,
            "verification_run_id": verification_run_id,
            "idempotency_key": idempotency_key,
            "publication_idempotency_key": (
                publication_idempotency_key if publish_dataset else ""
            ),
            "status": terminal_status,
            "model": str(
                updated_payload.get("verification_model")
                or validation.get("model")
                or ""
            ),
            "thinking_level": str(
                updated_payload.get("verification_thinking_level")
                or validation.get("thinking_level")
                or ""
            ),
            "scope": str(
                updated_payload.get("verification_scope")
                or validation.get("scope")
                or "all"
            ),
            "apply_mode": str(
                updated_payload.get("apply_mode") or validation.get("apply_mode") or ""
            ),
            "max_output_tokens": int(
                updated_payload.get("verification_max_output_tokens")
                or validation.get("max_output_tokens")
                or 0
            ),
            "num_chunks": int(
                updated_payload.get("verification_num_chunks")
                or validation.get("num_chunks")
                or 0
            ),
            "candidate_hash": str(updated_payload.get("candidate_hash") or ""),
            "verification_prompt_hash": str(
                updated_payload.get("verification_prompt_hash") or ""
            ),
            "expected_pages": int(updated_payload.get("expected_pages") or 0),
            "completed_pages": int(updated_payload.get("completed_pages") or 0),
            "confirmed_pages": int(updated_payload.get("confirmed_pages") or 0),
            "needs_correction_pages": int(
                updated_payload.get("needs_correction_pages") or 0
            ),
            "unverifiable_pages": int(updated_payload.get("unverifiable_pages") or 0),
            "published_dataset": bool(publish_dataset),
            "dataset_version": dataset_version,
            "dataset_version_id": str(updated_payload.get("dataset_version_id") or ""),
            "dataset_version_path": dataset_version_path,
            "dataset_sha256": str(updated_payload.get("dataset_sha256") or ""),
            "publication_provenance_sha256": str(
                updated_payload.get("publication_provenance_sha256") or ""
            ),
            "results_path": str(updated_payload.get("results_path") or ""),
            "failures_path": str(updated_payload.get("failures_path") or ""),
            "summary_path": str(updated_payload.get("summary_path") or ""),
            "field_corrections_path": str(
                updated_payload.get("field_corrections_path") or ""
            ),
            "field_corrections_gcs_uri": str(
                updated_payload.get("field_corrections_gcs_uri") or ""
            ),
            "field_corrections_gcs_generation": str(
                updated_payload.get("field_corrections_gcs_generation") or ""
            ),
            "field_corrections_sha256": str(
                updated_payload.get("field_corrections_sha256") or ""
            ),
            "corrected_fields": int(updated_payload.get("corrected_fields") or 0),
            "accepted_correction_fields": int(
                updated_payload.get("accepted_correction_fields") or 0
            ),
            "correction_acceptance_policy": str(
                updated_payload.get("correction_acceptance_policy") or ""
            ),
            "patched_candidates_path": str(
                updated_payload.get("patched_candidates_path") or ""
            ),
            "dataset_gcs_uri": str(updated_payload.get("dataset_gcs_uri") or ""),
            "dataset_gcs_generation": str(
                updated_payload.get("dataset_gcs_generation") or ""
            ),
            "dataset_version_ledger_gcs_uri": str(
                updated_payload.get("dataset_version_ledger_gcs_uri") or ""
            ),
            "artifact_gcs_uris": (
                dict(updated_payload.get("artifact_gcs_uris") or {})
                if isinstance(updated_payload.get("artifact_gcs_uris"), dict)
                else {}
            ),
            "recorded_payload": dict(updated_payload),
            "submitted_at": str(validation.get("submitted_at") or ""),
            "completed_at": utc_now_iso(),
        }
        previous_index = next(
            (
                index
                for index, item in enumerate(completed_runs)
                if isinstance(item, dict)
                and _portable_run_id(item.get("source_run_id"), run_path)
                == source_run_id
                and _portable_run_id(
                    item.get("verification_run_id"),
                    item.get("verification_run_dir"),
                )
                == verification_run_id
            ),
            None,
        )
        if previous_index is None:
            completed_runs.append(run_summary)
        else:
            completed_runs[previous_index] = run_summary
        record["model_validation"] = {
            **validation,
            **updated_payload,
            "enabled": True,
            "status": terminal_status,
            "runs": completed_runs,
            "completed_at": run_summary["completed_at"],
        }
        record["status"] = (
            "retrieved_complete" if publish_dataset else f"validation_{terminal_status}"
        )
        self.write(job_id, record)
        self.append_event(
            job_id,
            "model_validation_completed",
            {
                "status": terminal_status,
                "published_dataset": bool(publish_dataset),
            },
        )
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

    def upsert_schema_version(self, record: dict) -> None:
        schema_payload = record.get("schema_json")
        if not isinstance(schema_payload, dict):
            raise ValueError("Schema version is missing schema_json.")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO schema_versions (
                    version_id, family_id, name, version_number,
                    parent_version_id, created_at, created_by, prompt_name,
                    source, schema_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(version_id) DO NOTHING
                """,
                (
                    str(record.get("version_id") or ""),
                    str(record.get("family_id") or ""),
                    str(record.get("name") or ""),
                    int(record.get("version_number") or 1),
                    str(record.get("parent_version_id") or ""),
                    str(record.get("created_at") or utc_now_iso()),
                    str(record.get("created_by") or ""),
                    str(record.get("prompt_name") or "frontpage"),
                    str(record.get("source") or "local"),
                    _json_dumps(schema_payload),
                ),
            )

    def schema_version(self, version_id: str) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM schema_versions WHERE version_id = ?",
                (str(version_id),),
            ).fetchone()
        return self._schema_row(row) if row is not None else {}

    def list_schema_versions(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM schema_versions
                ORDER BY name COLLATE NOCASE, version_number DESC, created_at DESC
                """
            ).fetchall()
        return [self._schema_row(row) for row in rows]

    @staticmethod
    def _schema_row(row: sqlite3.Row) -> dict:
        return {
            "version_id": str(row["version_id"]),
            "family_id": str(row["family_id"]),
            "name": str(row["name"]),
            "version_number": int(row["version_number"]),
            "parent_version_id": str(row["parent_version_id"]),
            "created_at": str(row["created_at"]),
            "created_by": str(row["created_by"]),
            "prompt_name": str(row["prompt_name"]),
            "source": str(row["source"]),
            "schema_json": _json_loads(str(row["schema_json"])),
        }

    def set_schema_state(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO schema_state (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(key), str(value)),
            )

    def schema_state(self, key: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM schema_state WHERE key = ?",
                (str(key),),
            ).fetchone()
        return str(row["value"]) if row is not None else ""


def records_by_run_dir(records: Iterable[dict]) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for record in records:
        legacy = record.get("legacy") if isinstance(record.get("legacy"), dict) else {}
        batches = (
            record.get("batches") if isinstance(record.get("batches"), dict) else {}
        )
        run_dir = str(
            legacy.get("submit_run_dir") or batches.get("source_run_dir") or ""
        )
        if run_dir:
            output[run_dir] = record
    return output
