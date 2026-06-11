from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from patientjournals.shared.processing_metrics import (
    MANIFEST_FILE_NAME,
    read_processing_records,
    summarize_processing_records,
)


@dataclass(frozen=True)
class DashboardSummary:
    dataset_count: int
    dataset_rows: int
    latest_dataset: str
    validation_count: int
    validation_label_counts: dict[str, int]
    processing_record_count: int
    processing_image_count: int
    status_counts: dict[str, int]
    source_counts: dict[str, int]
    attempts: dict[str, float | int | None]
    generation_seconds: dict[str, float | int | None]
    failure_reasons: dict[str, int]
    duplicate_actions: dict[str, int]


def _count_jsonl_rows(path: Path) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _count_csv_rows(path: Path) -> int:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="$")
        rows = sum(1 for row in reader if row)
    return max(0, rows - 1)


def count_dataset_rows(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _count_jsonl_rows(path)
    if suffix == ".csv":
        return _count_csv_rows(path)
    return 0


def find_dataset_files(run_root: str | Path = "runs") -> list[Path]:
    root = Path(run_root).expanduser()
    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        [
            *root.rglob("*_dataset.jsonl"),
            *root.rglob("*_dataset.csv"),
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def latest_dataset_path(run_root: str | Path = "runs") -> Path | None:
    files = find_dataset_files(run_root)
    return files[0] if files else None


def _iter_validation_rows(validations_root: str | Path = "validations") -> Iterable[dict[str, str]]:
    root = Path(validations_root).expanduser()
    if not root.exists():
        return []
    files = [root] if root.is_file() else sorted(root.rglob("*_validations.csv"))
    rows: list[dict[str, str]] = []
    for path in files:
        try:
            with open(path, "r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    rows.append(dict(row))
        except OSError:
            continue
    return rows


def _counter(values: Iterable[object]) -> dict[str, int]:
    counts = Counter(str(value) for value in values if value not in {None, ""})
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value) if value not in {None, ""} else None
    except (TypeError, ValueError):
        return None


def numeric_distribution(values: Iterable[object]) -> dict[str, float | int | None]:
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


def _load_processing_records(run_root: str | Path = "runs") -> list[dict[str, Any]]:
    root = Path(run_root).expanduser()
    if not root.exists() or not root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for manifest in sorted(root.rglob(MANIFEST_FILE_NAME)):
        for record in read_processing_records(manifest):
            enriched = dict(record)
            enriched["manifest_path"] = str(manifest)
            records.append(enriched)
    return records


def summarize_dashboard(
    *,
    run_root: str | Path = "runs",
    validations_root: str | Path = "validations",
) -> DashboardSummary:
    dataset_files = find_dataset_files(run_root)
    dataset_rows = sum(count_dataset_rows(path) for path in dataset_files)
    latest = str(dataset_files[0]) if dataset_files else ""

    validation_rows = list(_iter_validation_rows(validations_root))
    validation_label_counts = _counter(
        str(row.get("label") or "").lower() for row in validation_rows
    )

    processing_records = _load_processing_records(run_root)
    processing_summary = summarize_processing_records(processing_records)
    return DashboardSummary(
        dataset_count=len(dataset_files),
        dataset_rows=dataset_rows,
        latest_dataset=latest,
        validation_count=len(validation_rows),
        validation_label_counts=validation_label_counts,
        processing_record_count=int(processing_summary.get("record_count") or 0),
        processing_image_count=int(processing_summary.get("images") or 0),
        status_counts={
            str(key): int(value)
            for key, value in (processing_summary.get("status_counts") or {}).items()
        },
        source_counts={
            str(key): int(value)
            for key, value in (processing_summary.get("source_counts") or {}).items()
        },
        attempts=dict(processing_summary.get("attempts") or {}),
        generation_seconds=dict(processing_summary.get("generation_seconds") or {}),
        failure_reasons={
            str(key): int(value)
            for key, value in (processing_summary.get("failure_reasons") or {}).items()
        },
        duplicate_actions={
            str(key): int(value)
            for key, value in (processing_summary.get("duplicate_actions") or {}).items()
        },
    )


def dashboard_summary_json(summary: DashboardSummary) -> str:
    return json.dumps(summary.__dict__, indent=2, ensure_ascii=False, default=str)
