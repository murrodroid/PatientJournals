from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from patientjournals.shared.identity import image_name_from_reference


MANIFEST_FILE_NAME = "image_processing_manifest.jsonl"
SUMMARY_FILE_NAME = "image_processing_summary.json"


def utc_now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def append_processing_record(path: str | Path, record: dict[str, Any]) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str))
        handle.write("\n")


def read_processing_records(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    if not manifest_path.exists() or not manifest_path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _numeric_summary(values: Iterable[object]) -> dict[str, float | int | None]:
    numbers = [value for value in (_number(item) for item in values) if value is not None]
    if not numbers:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "mean": mean(numbers),
        "median": median(numbers),
    }


def _counter(values: Iterable[object]) -> dict[str, int]:
    counts = Counter(str(value) for value in values if value not in {None, ""})
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def summarize_processing_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "record_count": len(records),
        "images": len(
            {
                str(record.get("image_name") or "")
                for record in records
                if record.get("image_name")
            }
        ),
        "status_counts": _counter(record.get("status") for record in records),
        "source_counts": _counter(record.get("source") for record in records),
        "provider_counts": _counter(record.get("provider") for record in records),
        "model_counts": _counter(record.get("model") for record in records),
        "attempts": _numeric_summary(record.get("attempts") for record in records),
        "generation_seconds": _numeric_summary(
            record.get("generation_seconds") for record in records
        ),
        "total_seconds": _numeric_summary(record.get("total_seconds") for record in records),
        "rows_written": _numeric_summary(record.get("rows_written") for record in records),
        "failure_reasons": _counter(record.get("failure_reason") for record in records),
        "duplicate_actions": _counter(record.get("duplicate_action") for record in records),
    }


def write_processing_summary(
    run_dir: str | Path,
    records: list[dict[str, Any]] | None = None,
) -> Path:
    run_path = Path(run_dir)
    manifest_path = run_path / MANIFEST_FILE_NAME
    payload = summarize_processing_records(
        records if records is not None else read_processing_records(manifest_path)
    )
    payload["manifest_path"] = str(manifest_path)
    payload["generated_at"] = utc_now_iso()
    summary_path = run_path / SUMMARY_FILE_NAME
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return summary_path


def base_image_record(
    *,
    image_reference: str | Path | None,
    source: str,
    status: str,
    model: str = "",
    provider: str = "",
    attempts: int | None = None,
    max_attempts: int | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
    generation_seconds: float | None = None,
    total_seconds: float | None = None,
    rows_written: int | None = None,
    failure_reason: str | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    preprocessing: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference = str(image_reference or "")
    record: dict[str, Any] = {
        "recorded_at": utc_now_iso(),
        "source": source,
        "status": status,
        "image_reference": reference,
        "image_name": image_name_from_reference(reference) if reference else "",
        "model": model,
        "provider": provider,
        "attempts": attempts,
        "max_attempts": max_attempts,
        "started_at": started_at,
        "completed_at": completed_at,
        "generation_seconds": generation_seconds,
        "total_seconds": total_seconds,
        "rows_written": rows_written,
        "failure_reason": failure_reason,
        "error_type": error_type,
        "error_message": error_message,
        "preprocessing": preprocessing or {},
    }
    if extra:
        record.update(extra)
    return record
