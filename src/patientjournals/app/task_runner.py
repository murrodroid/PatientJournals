from __future__ import annotations

import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from patientjournals.app.job_store import JobStore, utc_now_iso
from patientjournals.app.workflows import serializable


class TaskRunner:
    """Small in-process background task runner with SQLite task state."""

    def __init__(self, store: JobStore, *, max_workers: int = 4) -> None:
        self.store = store
        self.executor = ThreadPoolExecutor(max_workers=max(1, max_workers))
        self._running: set[str] = set()

    def submit(
        self,
        kind: str,
        func: Callable[[], object],
        *,
        metadata: dict | None = None,
    ) -> str:
        task_id = f"{kind}_{uuid.uuid4().hex[:12]}"
        self.store.upsert_task(
            task_id,
            kind=kind,
            status="pending",
            metadata=metadata or {},
        )

        def run() -> None:
            self._running.add(task_id)
            self.store.upsert_task(
                task_id,
                kind=kind,
                status="running",
                metadata=metadata or {},
                started_at=utc_now_iso(),
            )
            try:
                result = serializable(func())
            except Exception as exc:  # noqa: BLE001
                self.store.upsert_task(
                    task_id,
                    kind=kind,
                    status="failed",
                    metadata=metadata or {},
                    error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                    finished_at=utc_now_iso(),
                )
            else:
                payload = result if isinstance(result, dict) else {"value": result}
                self.store.upsert_task(
                    task_id,
                    kind=kind,
                    status="succeeded",
                    metadata=metadata or {},
                    result=payload,
                    finished_at=utc_now_iso(),
                )
            finally:
                self._running.discard(task_id)

        self.executor.submit(run)
        return task_id

    def list_tasks(self, *, limit: int = 100) -> list[dict]:
        return self.store.list_tasks(limit=limit)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)
