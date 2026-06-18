from __future__ import annotations

import csv
import io
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from patientjournals.shared.processing_metrics import (
    MANIFEST_FILE_NAME,
    read_processing_records,
    summarize_processing_records,
)
from patientjournals.data.bucket import build_storage_bucket, list_bucket_blobs, normalize_prefix


@dataclass(frozen=True)
class ValidationMetricSummary:
    metric: str
    decisions: int
    scored: int
    accepted: int
    somewhat_accepted: int
    rejected: int
    corrected: int
    unsure: int
    accuracy: float | None


@dataclass(frozen=True)
class ValidationRunSummary:
    run_id: str
    dataset_file: str
    validator_id: str
    validations_path: str
    decisions: int
    scored: int
    accepted: int
    somewhat_accepted: int
    rejected: int
    corrected: int
    unsure: int
    accuracy: float | None


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
    validation_runs: tuple[ValidationRunSummary, ...]
    validation_metrics: tuple[ValidationMetricSummary, ...]
    validation_sync_error: str = ""


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
                    item = dict(row)
                    item["_validation_path"] = str(path)
                    item["_validation_run"] = path.parent.name
                    try:
                        item["_validation_mtime"] = str(path.stat().st_mtime)
                    except OSError:
                        item["_validation_mtime"] = "0"
                    rows.append(item)
        except OSError:
            continue
    return rows


def _iter_cloud_validation_rows(
    *,
    bucket_name: str = "",
    validations_prefix: str = "validations",
) -> tuple[list[dict[str, str]], str]:
    if not bucket_name.strip():
        return [], ""
    try:
        bucket = build_storage_bucket(bucket_name)
        prefix = normalize_prefix(validations_prefix)
        blobs = list_bucket_blobs(bucket, prefix=prefix)
    except Exception as exc:  # noqa: BLE001
        return [], f"Shared validations unavailable: {type(exc).__name__}: {exc}"

    rows: list[dict[str, str]] = []
    resolved_bucket = str(getattr(bucket, "name", "") or bucket_name)
    for blob in blobs:
        name = str(getattr(blob, "name", "") or "")
        if not name.endswith("_validations.csv"):
            continue
        try:
            text = blob.download_as_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            return rows, f"Could not read shared validation {name}: {exc}"
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            item = dict(row)
            item["_validation_path"] = f"gs://{resolved_bucket}/{name}"
            item["_validation_run"] = Path(name).parent.name
            updated = getattr(blob, "updated", None) or getattr(blob, "time_created", None)
            item["_validation_mtime"] = (
                str(updated.timestamp()) if hasattr(updated, "timestamp") else "0"
            )
            item["_validation_source"] = "cloud"
            rows.append(item)
    return rows, ""


def _dedupe_validation_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    unique: dict[tuple[str, str, str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            str(row.get("validator_id") or ""),
            str(row.get("dataset_file") or ""),
            str(row.get("image_name") or row.get("file_name") or ""),
            str(row.get("column_name") or ""),
            str(row.get("decided_at") or ""),
            str(row.get("label") or ""),
        )
        previous = unique.get(key)
        if previous is None or previous.get("_validation_source") != "cloud":
            unique[key] = row
    return list(unique.values())


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


_ACCEPT_LABELS = {"accept", "accepted"}
_SOMEWHAT_LABELS = {"somewhat_accept", "somewhat accept", "somewhat-accept", "partial"}
_REJECT_LABELS = {"reject", "rejected"}
_CORRECTED_LABELS = {"corrected", "correction"}
_UNSURE_LABELS = {"unsure", "unknown"}


def _normalized_label(value: object) -> str:
    return str(value or "").strip().lower()


def _score_label(value: object) -> float | None:
    label = _normalized_label(value)
    if label in _ACCEPT_LABELS:
        return 1.0
    if label in _SOMEWHAT_LABELS:
        return 0.5
    if label in _REJECT_LABELS or label in _CORRECTED_LABELS:
        return 0.0
    if label in _UNSURE_LABELS:
        return None
    return None


def _validation_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    labels = [_normalized_label(row.get("label")) for row in rows]
    return {
        "accepted": sum(1 for label in labels if label in _ACCEPT_LABELS),
        "somewhat_accepted": sum(1 for label in labels if label in _SOMEWHAT_LABELS),
        "rejected": sum(1 for label in labels if label in _REJECT_LABELS),
        "corrected": sum(1 for label in labels if label in _CORRECTED_LABELS),
        "unsure": sum(1 for label in labels if label in _UNSURE_LABELS),
    }


def _validation_accuracy(rows: list[dict[str, str]]) -> tuple[int, float | None]:
    scores = [
        score
        for score in (_score_label(row.get("label")) for row in rows)
        if score is not None
    ]
    if not scores:
        return 0, None
    return len(scores), round(sum(scores) / len(scores) * 100.0, 1)


def _summarize_validation_runs(
    rows: list[dict[str, str]],
) -> tuple[ValidationRunSummary, ...]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        run_id = str(row.get("_validation_run") or "validation")
        path = str(row.get("_validation_path") or "")
        grouped.setdefault((run_id, path), []).append(row)

    summaries: list[ValidationRunSummary] = []
    for (run_id, path), group in grouped.items():
        counts = _validation_counts(group)
        scored, accuracy = _validation_accuracy(group)
        dataset_file = str(group[0].get("dataset_file") or "unknown")
        validator_id = str(group[0].get("validator_id") or "unknown")
        summaries.append(
            ValidationRunSummary(
                run_id=run_id,
                dataset_file=dataset_file,
                validator_id=validator_id,
                validations_path=path,
                decisions=len(group),
                scored=scored,
                accuracy=accuracy,
                **counts,
            )
        )
    return tuple(
        sorted(
            summaries,
            key=lambda item: (
                Path(item.validations_path).stat().st_mtime
                if item.validations_path and Path(item.validations_path).exists()
                else 0.0
            ),
            reverse=True,
        )
    )


def _summarize_validation_metrics(
    rows: list[dict[str, str]],
) -> tuple[ValidationMetricSummary, ...]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        metric = str(row.get("column_name") or "").strip() or "(unknown)"
        grouped.setdefault(metric, []).append(row)

    summaries: list[ValidationMetricSummary] = []
    for metric, group in grouped.items():
        counts = _validation_counts(group)
        scored, accuracy = _validation_accuracy(group)
        summaries.append(
            ValidationMetricSummary(
                metric=metric,
                decisions=len(group),
                scored=scored,
                accuracy=accuracy,
                **counts,
            )
        )
    return tuple(
        sorted(
            summaries,
            key=lambda item: (
                -(item.accuracy if item.accuracy is not None else -1.0),
                -item.decisions,
                item.metric,
            ),
        )
    )


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
    cloud_validations_bucket: str = "",
    cloud_validations_prefix: str = "validations",
) -> DashboardSummary:
    dataset_files = find_dataset_files(run_root)
    dataset_rows = sum(count_dataset_rows(path) for path in dataset_files)
    latest = str(dataset_files[0]) if dataset_files else ""

    local_validation_rows = list(_iter_validation_rows(validations_root))
    cloud_validation_rows, validation_sync_error = _iter_cloud_validation_rows(
        bucket_name=cloud_validations_bucket,
        validations_prefix=cloud_validations_prefix,
    )
    validation_rows = _dedupe_validation_rows(
        [*local_validation_rows, *cloud_validation_rows]
    )
    validation_label_counts = _counter(
        str(row.get("label") or "").lower() for row in validation_rows
    )
    validation_runs = _summarize_validation_runs(validation_rows)
    validation_metrics = _summarize_validation_metrics(validation_rows)

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
        validation_runs=validation_runs,
        validation_metrics=validation_metrics,
        validation_sync_error=validation_sync_error,
    )


def dashboard_summary_json(summary: DashboardSummary) -> str:
    return json.dumps(asdict(summary), indent=2, ensure_ascii=False, default=str)
