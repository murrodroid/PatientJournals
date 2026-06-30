from __future__ import annotations

import csv
import io
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, get_args, get_origin

from patientjournals.config import config
from patientjournals.shared.field_classification import is_schema_data_field
from patientjournals.config.schemas import list_output_schemas
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


@dataclass(frozen=True)
class DatasetColumnSummary:
    column: str
    populated: int
    missing: int
    completeness: float


@dataclass(frozen=True)
class DatasetAnalysis:
    dataset_path: str
    row_count: int
    column_count: int
    columns: tuple[str, ...]
    failed_rows: int
    failure_reasons: dict[str, int]
    field_completeness: tuple[DatasetColumnSummary, ...]
    schema_field_completeness: tuple[DatasetColumnSummary, ...]
    metadata_field_completeness: tuple[DatasetColumnSummary, ...]
    attempts: dict[str, float | int | None]
    avg_logprobs: dict[str, float | int | None]
    sample_rows: tuple[dict[str, object], ...]


def _count_jsonl_rows(path: Path) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _count_csv_rows(path: Path) -> int:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=config.csv_sep)
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
    files_by_path = {
        path.resolve(): path
        for path in [
            *root.rglob("*_dataset.jsonl"),
            *root.rglob("*_dataset.csv"),
            *root.rglob("jobs/*/datasets/current.jsonl"),
            *root.rglob("jobs/*/datasets/current.csv"),
        ]
        if path.is_file()
    }

    def run_id(path: Path) -> str:
        if path.parent.name == "datasets":
            return path.parent.parent.name
        return path.parent.name

    def is_current(path: Path) -> bool:
        return path.parent.name == "datasets" and path.stem == "current"

    canonical = {run_id(path) for path in files_by_path.values() if is_current(path)}
    visible = [
        path
        for path in files_by_path.values()
        if is_current(path) or run_id(path) not in canonical
    ]
    return sorted(
        visible,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _dataset_rows(path: Path, *, limit: int | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=config.csv_sep)
            for row in reader:
                rows.append(dict(row))
                if limit is not None and len(rows) >= limit:
                    break
        return rows
    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
                    if limit is not None and len(rows) >= limit:
                        break
    return rows


def _flatten_mapping(
    value: dict[str, object],
    *,
    prefix: str = "",
) -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            if item:
                flat.update(_flatten_mapping(item, prefix=path))
            else:
                flat[path] = item
        else:
            flat[path] = item
    return flat


def analyze_dataset_file(
    dataset_path: str | Path,
    *,
    sample_limit: int = 25,
) -> DatasetAnalysis:
    path = Path(dataset_path).expanduser()
    rows = _dataset_rows(path)
    flat_rows = [_flatten_mapping(row) for row in rows]
    columns: list[str] = []
    for row in flat_rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))

    row_count = len(rows)
    failed_rows = sum(1 for row in rows if _truthy(row.get("failed")))
    failure_reasons = _counter(row.get("failure_reason") for row in rows)
    schema_completeness: list[DatasetColumnSummary] = []
    metadata_completeness: list[DatasetColumnSummary] = []
    for column in columns:
        populated = sum(1 for row in flat_rows if _is_populated(row.get(column)))
        missing = max(0, row_count - populated)
        summary = DatasetColumnSummary(
            column=column,
            populated=populated,
            missing=missing,
            completeness=(populated / row_count * 100.0) if row_count else 0.0,
        )
        if _is_schema_column(column):
            schema_completeness.append(summary)
        else:
            metadata_completeness.append(summary)

    sorted_schema = tuple(
        sorted(schema_completeness, key=lambda item: (item.completeness, item.column))
    )
    sorted_metadata = tuple(
        sorted(metadata_completeness, key=lambda item: (item.completeness, item.column))
    )

    return DatasetAnalysis(
        dataset_path=str(path),
        row_count=row_count,
        column_count=len(columns),
        columns=tuple(columns),
        failed_rows=failed_rows,
        failure_reasons=failure_reasons,
        field_completeness=sorted_schema,
        schema_field_completeness=sorted_schema,
        metadata_field_completeness=sorted_metadata,
        attempts=numeric_distribution(row.get("attempts") for row in rows),
        avg_logprobs=numeric_distribution(row.get("avg_logprobs") for row in rows),
        sample_rows=tuple(flat_rows[: max(1, sample_limit)]),
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


def _unwrap_optional(field_type: object) -> object:
    origin = get_origin(field_type)
    if origin is None:
        return field_type
    if origin in {list, tuple, set}:
        return field_type
    args = [arg for arg in get_args(field_type) if arg is not type(None)]
    return args[0] if len(args) == 1 else field_type


def _schema_models() -> tuple[type, ...]:
    models: list[type] = []
    configured = getattr(config, "output_model", None)
    if isinstance(configured, type) and hasattr(configured, "model_fields"):
        models.append(configured)
    for model in list_output_schemas().values():
        if model not in models:
            models.append(model)
    return tuple(models)


def _field_type_for_model(model: type, path: str) -> object | None:
    current: object = model
    field_type: object | None = None
    for part in path.split("."):
        if not hasattr(current, "model_fields"):
            return None
        field_info = current.model_fields.get(part)
        if field_info is None:
            return None
        field_type = _unwrap_optional(field_info.annotation)
        origin = get_origin(field_type)
        if origin in {list, tuple, set}:
            args = get_args(field_type)
            field_type = args[0] if args else None
            origin = get_origin(field_type)
        if isinstance(field_type, type) and hasattr(field_type, "model_fields"):
            current = field_type
        else:
            current = None
    return field_type


def _is_metadata_column(column: str) -> bool:
    return not is_schema_data_field(column)


def _is_schema_column(column: str) -> bool:
    if _is_metadata_column(column):
        return False
    return any(_field_type_for_model(model, column) is not None for model in _schema_models())


def _is_populated(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _counter_key(value: object) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(value)
    return str(value)


def _counter(values: Iterable[object]) -> dict[str, int]:
    counts = Counter(_counter_key(value) for value in values if _is_populated(value))
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not _is_populated(value):
        return None
    if isinstance(value, (list, tuple, set, dict)):
        return None
    try:
        return float(value)
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
