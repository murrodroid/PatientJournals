from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from patientjournals.app.models import (
    CloudDatasetChoice,
    CloudResolution,
    DatasetLibraryItem,
    DatasetSummary,
)
from patientjournals.config import config
from patientjournals.data.bucket import (
    build_storage_bucket,
    list_bucket_blobs,
    normalize_prefix,
    select_bucket_image_blobs,
    summarize_bucket_data,
)
from patientjournals.data.inspection import collect_files, summarize_batch_data
from patientjournals.shared.identity import (
    build_image_name_set,
    duplicate_image_names,
    ensure_row_image_name,
    image_name_from_reference,
    row_image_name,
)


def inspect_local_dataset(
    root: str | Path,
    *,
    glob_pattern: str | None = None,
    recursive: bool = True,
) -> DatasetSummary:
    root_path, _all_files, image_files = collect_files(
        root,
        glob_pattern=glob_pattern,
        recursive=recursive,
    )
    duplicates = tuple(sorted(duplicate_image_names(str(path) for path in image_files)))
    status = "error" if duplicates else "ok"
    detail = (
        "Duplicate image names must be resolved before submitting."
        if duplicates
        else ""
    )
    return DatasetSummary(
        source="local",
        name=root_path.name,
        root=str(root_path),
        image_count=len(image_files),
        duplicate_image_names=duplicates,
        status=status,
        detail=detail,
    )


def local_summary_report(
    root: str | Path,
    *,
    glob_pattern: str | None = None,
    recursive: bool = True,
) -> dict:
    return summarize_batch_data(
        root,
        glob_pattern=glob_pattern,
        recursive=recursive,
    )


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


def count_dataset_rows(path: str | Path) -> int:
    dataset = Path(path).expanduser()
    suffix = dataset.suffix.lower()
    if suffix == ".jsonl":
        return _count_jsonl_rows(dataset)
    if suffix == ".csv":
        return _count_csv_rows(dataset)
    return 0


def _dataset_content_type(dataset_path: Path) -> str:
    suffix = dataset_path.suffix.lower().lstrip(".")
    if suffix == "jsonl":
        return "application/jsonl"
    if suffix == "json":
        return "application/json"
    if suffix == "csv":
        return "text/csv"
    return "application/octet-stream"


def _safe_dataset_slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or datetime.now().strftime("combined_%Y%m%d_%H%M%S")


def _unique_dataset_dir(root: Path, slug: str) -> tuple[Path, str]:
    candidate = root / slug
    if not candidate.exists():
        return candidate, slug
    stamped = f"{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    candidate = root / stamped
    index = 2
    while candidate.exists():
        candidate = root / f"{stamped}_{index}"
        index += 1
    return candidate, candidate.name


def _source_location(item: Mapping[str, Any] | str | Path) -> str:
    if isinstance(item, Mapping):
        for key in ("local_path", "path", "gcs_uri", "location"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
    return str(item)


def _source_name(item: Mapping[str, Any] | str | Path, location: str) -> str:
    if isinstance(item, Mapping):
        value = item.get("name") or item.get("run_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return Path(location).name if location else "dataset"


def _iter_dataset_rows(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if not isinstance(payload, dict):
                    raise ValueError(f"{path} line {line_number} is not a JSON object.")
                yield dict(payload)
        return
    if suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=config.csv_sep)
            for row in reader:
                yield dict(row)
        return
    raise ValueError(f"Unsupported dataset format: {path}")


def _row_failed(row: Mapping[str, Any]) -> bool:
    value = row.get("failed")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "failed", "error"}
    status = row.get("status")
    if isinstance(status, str):
        return status.strip().lower() in {"failed", "error", "missing"}
    return False


def prepare_dataset_sources(
    items: Iterable[Mapping[str, Any] | str | Path],
    *,
    download_root: str | Path,
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        location = _source_location(item)
        if not location:
            raise ValueError(f"Dataset selection {index + 1} is missing a location.")
        if location.startswith("gs://"):
            local_path = download_cloud_dataset(
                location,
                destination_root=Path(download_root).expanduser() / f"source_{index + 1}",
            )
        else:
            local_path = Path(location).expanduser()
        if not local_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {local_path}")
        sources.append(
            {
                "name": _source_name(item, location),
                "location": location,
                "local_path": str(local_path),
                "source": (
                    str(item.get("source") or "cloud")
                    if isinstance(item, Mapping) and location.startswith("gs://")
                    else str(item.get("source") or "local")
                    if isinstance(item, Mapping)
                    else "cloud"
                    if location.startswith("gs://")
                    else "local"
                ),
            }
        )
    if not sources:
        raise ValueError("Select at least one dataset to combine.")
    return sources


def upload_dataset_to_cloud(
    dataset_path: str | Path,
    *,
    dataset_name: str,
    bucket_name: str | None = None,
    datasets_prefix: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    path = Path(dataset_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    bucket = build_storage_bucket(bucket_name)
    resolved_bucket = str(getattr(bucket, "name", "") or bucket_name or "")
    prefix = normalize_prefix(datasets_prefix or config.datasets_gcs_prefix)
    object_name = f"{prefix}{dataset_name}/{path.name}"
    blob = bucket.blob(object_name)
    if metadata:
        blob.metadata = {
            str(key): json.dumps(value, ensure_ascii=False)
            if isinstance(value, (dict, list, tuple))
            else str(value)
            for key, value in metadata.items()
            if value is not None
        }
    blob.upload_from_filename(str(path), content_type=_dataset_content_type(path))
    return f"gs://{resolved_bucket}/{object_name}"


def combine_dataset_files(
    sources: Iterable[Mapping[str, Any]],
    *,
    output_name: str,
    output_root: str | Path = "runs",
    duplicate_strategy: str = "first_successful",
    upload_to_cloud: bool = False,
    bucket_name: str | None = None,
    datasets_prefix: str | None = None,
) -> dict[str, Any]:
    strategy = (duplicate_strategy or "first_successful").strip().lower()
    aliases = {
        "first": "first_successful",
        "ignore_duplicates": "first_successful",
        "ignore": "first_successful",
        "include_all": "provide_all",
        "all": "provide_all",
    }
    strategy = aliases.get(strategy, strategy)
    if strategy not in {"first_successful", "provide_all"}:
        raise ValueError(
            "Duplicate strategy must be first_successful or provide_all."
        )

    resolved_sources = list(sources)
    if not resolved_sources:
        raise ValueError("Select at least one dataset to combine.")

    output_slug = _safe_dataset_slug(output_name)
    library_root = Path(output_root).expanduser() / "datasets"
    output_dir, final_slug = _unique_dataset_dir(library_root, output_slug)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / f"{final_slug}_dataset.jsonl"
    manifest_path = output_dir / "dataset_manifest.json"

    output_rows: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    first_seen: dict[str, dict[str, Any]] = {}
    source_stats: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    duplicate_names: set[str] = set()
    rows_read = 0
    duplicates_skipped = 0
    duplicates_included = 0
    duplicates_replaced = 0

    for source in resolved_sources:
        path = Path(str(source.get("local_path") or source.get("path") or "")).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")
        source_rows = 0
        source_duplicate_rows = 0
        source_name = str(source.get("name") or path.name)
        source_location = str(source.get("location") or path)
        for source_row_number, raw_row in enumerate(_iter_dataset_rows(path), start=1):
            rows_read += 1
            source_rows += 1
            row = dict(raw_row)
            image_name = ensure_row_image_name(row) or row_image_name(row)
            if not image_name:
                output_rows.append(row)
                continue
            if image_name not in seen:
                seen[image_name] = len(output_rows)
                first_seen[image_name] = {
                    "source": source_name,
                    "location": source_location,
                    "row_number": source_row_number,
                    "failed": _row_failed(row),
                }
                output_rows.append(row)
                continue

            duplicate_names.add(image_name)
            source_duplicate_rows += 1
            original = first_seen.get(image_name, {})
            duplicate_record = {
                "image_name": image_name,
                "source": source_name,
                "location": source_location,
                "row_number": source_row_number,
                "first_source": original.get("source", ""),
                "first_location": original.get("location", ""),
                "first_row_number": original.get("row_number", ""),
                "action": "",
            }
            if strategy == "provide_all":
                duplicate_record["action"] = "included_duplicate"
                output_rows.append(row)
                duplicates_included += 1
            else:
                existing_index = seen[image_name]
                existing_failed = _row_failed(output_rows[existing_index])
                current_failed = _row_failed(row)
                if existing_failed and not current_failed:
                    output_rows[existing_index] = row
                    first_seen[image_name] = {
                        "source": source_name,
                        "location": source_location,
                        "row_number": source_row_number,
                        "failed": False,
                    }
                    duplicate_record["action"] = "replaced_failed_duplicate"
                    duplicates_replaced += 1
                else:
                    duplicate_record["action"] = "skipped_duplicate"
                duplicates_skipped += 1
            duplicate_rows.append(duplicate_record)
        source_stats.append(
            {
                "name": source_name,
                "source": source.get("source", ""),
                "location": source_location,
                "local_path": str(path),
                "rows_read": source_rows,
                "duplicate_rows": source_duplicate_rows,
            }
        )

    with open(dataset_path, "w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_name": output_name.strip(),
        "final_dataset_name": final_slug,
        "dataset_path": str(dataset_path),
        "duplicate_strategy": strategy,
        "source_count": len(source_stats),
        "sources": source_stats,
        "rows_read": rows_read,
        "rows_written": len(output_rows),
        "duplicates_detected": len(duplicate_rows),
        "duplicate_image_count": len(duplicate_names),
        "duplicates_skipped": duplicates_skipped,
        "duplicates_included": duplicates_included,
        "duplicates_replaced": duplicates_replaced,
        "duplicate_image_names": sorted(duplicate_names),
        "duplicate_rows": duplicate_rows,
        "cloud_uri": "",
        "manifest_cloud_uri": "",
        "cloud_sync_error": "",
    }

    cloud_uri = ""
    manifest_cloud_uri = ""
    cloud_error = ""
    if upload_to_cloud:
        try:
            cloud_uri = upload_dataset_to_cloud(
                dataset_path,
                dataset_name=final_slug,
                bucket_name=bucket_name,
                datasets_prefix=datasets_prefix,
                metadata={
                    "rows": len(output_rows),
                    "row_count": len(output_rows),
                    "source_count": len(source_stats),
                    "duplicate_strategy": strategy,
                    "duplicates_detected": len(duplicate_rows),
                    "duplicates_skipped": duplicates_skipped,
                    "duplicates_included": duplicates_included,
                    "duplicates_replaced": duplicates_replaced,
                    "combined": True,
                },
            )
        except Exception as exc:  # noqa: BLE001
            cloud_error = str(exc)
    manifest["cloud_uri"] = cloud_uri
    manifest["manifest_cloud_uri"] = manifest_cloud_uri
    manifest["cloud_sync_error"] = cloud_error
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if upload_to_cloud and cloud_uri and not cloud_error:
        try:
            manifest_cloud_uri = upload_dataset_to_cloud(
                manifest_path,
                dataset_name=final_slug,
                bucket_name=bucket_name,
                datasets_prefix=datasets_prefix,
                metadata={
                    "kind": "dataset_manifest",
                    "dataset_name": final_slug,
                    "dataset_uri": cloud_uri,
                    "rows": len(output_rows),
                    "duplicates_detected": len(duplicate_rows),
                },
            )
            manifest["manifest_cloud_uri"] = manifest_cloud_uri
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            cloud_error = f"Dataset uploaded, but manifest upload failed: {exc}"
            manifest["cloud_sync_error"] = cloud_error
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    return {
        "dataset_name": final_slug,
        "dataset_path": str(dataset_path),
        "manifest_path": str(manifest_path),
        "cloud_uri": cloud_uri,
        "manifest_cloud_uri": manifest_cloud_uri,
        "cloud_sync_error": cloud_error,
        "row_count": len(output_rows),
        "rows_read": rows_read,
        "source_count": len(source_stats),
        "duplicate_strategy": strategy,
        "duplicates_detected": len(duplicate_rows),
        "duplicate_image_count": len(duplicate_names),
        "duplicates_skipped": duplicates_skipped,
        "duplicates_included": duplicates_included,
        "duplicates_replaced": duplicates_replaced,
        "duplicate_image_names": sorted(duplicate_names)[:100],
    }


def _format_blob_updated(blob: object) -> str:
    value = getattr(blob, "updated", None) or getattr(blob, "time_created", None)
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value or "")


def list_local_dataset_library(
    run_root: str | Path = "runs",
    *,
    limit: int = 200,
) -> list[DatasetLibraryItem]:
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
    def local_run_id(path: Path) -> str:
        if path.parent.name == "datasets" and path.parent.parent.name:
            return path.parent.parent.name
        return path.parent.name

    def is_current_dataset(path: Path) -> bool:
        return path.parent.name == "datasets" and path.stem == "current"

    canonical_run_ids = {
        local_run_id(path)
        for path in files_by_path.values()
        if is_current_dataset(path)
    }
    visible_files = [
        path
        for path in files_by_path.values()
        if is_current_dataset(path) or local_run_id(path) not in canonical_run_ids
    ]
    files = sorted(
        visible_files,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    items: list[DatasetLibraryItem] = []
    for path in files[: max(1, limit)]:
        try:
            row_count = count_dataset_rows(path)
            size_bytes = path.stat().st_size
            updated_at = datetime.fromtimestamp(path.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M"
            )
        except OSError:
            continue
        run_id = local_run_id(path)
        items.append(
            DatasetLibraryItem(
                source="local",
                name=path.name,
                location=str(path),
                row_count=row_count,
                size_bytes=size_bytes,
                updated_at=updated_at,
                run_id=run_id,
                local_path=str(path),
            )
        )
    return items


def list_cloud_dataset_library(
    *,
    bucket_name: str | None = None,
    datasets_prefix: str | None = None,
    limit: int = 500,
) -> list[DatasetLibraryItem]:
    bucket = build_storage_bucket(bucket_name)
    resolved_bucket = str(getattr(bucket, "name", "") or bucket_name or "")
    prefix = normalize_prefix(datasets_prefix or config.datasets_gcs_prefix)
    blobs = list_bucket_blobs(bucket, prefix=prefix)
    items: list[DatasetLibraryItem] = []
    for blob in blobs:
        name = str(getattr(blob, "name", "") or "")
        if not name or name.endswith("/"):
            continue
        if Path(name).suffix.lower() not in {".jsonl", ".csv"}:
            continue
        metadata = getattr(blob, "metadata", None) or {}
        rows_value = metadata.get("rows") or metadata.get("row_count")
        try:
            row_count = int(rows_value) if rows_value not in {None, ""} else None
        except (TypeError, ValueError):
            row_count = None
        uri = f"gs://{resolved_bucket}/{name}"
        items.append(
            DatasetLibraryItem(
                source="cloud",
                name=Path(name).name,
                location=uri,
                row_count=row_count,
                size_bytes=getattr(blob, "size", None),
                updated_at=_format_blob_updated(blob),
                run_id=Path(name).parent.name,
                gcs_uri=uri,
            )
        )
        if len(items) >= max(1, limit):
            break
    return sorted(items, key=lambda item: item.updated_at, reverse=True)


def _split_gcs_uri(uri: str) -> tuple[str, str]:
    value = uri.strip()
    if not value.startswith("gs://"):
        raise ValueError(f"Expected a gs:// dataset URI, got: {uri}")
    bucket_and_object = value[5:]
    bucket, _, object_name = bucket_and_object.partition("/")
    if not bucket or not object_name:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return bucket, object_name


def download_cloud_dataset(
    gcs_uri: str,
    *,
    destination_root: str | Path = "datasets",
) -> Path:
    bucket_name, object_name = _split_gcs_uri(gcs_uri)
    bucket = build_storage_bucket(bucket_name)
    destination = Path(destination_root).expanduser() / Path(object_name).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    bucket.blob(object_name).download_to_filename(str(destination))
    return destination


def inspect_cloud_dataset(
    *,
    bucket_name: str | None = None,
    prefix: str | None = None,
    glob_pattern: str | None = None,
) -> DatasetSummary:
    report = summarize_bucket_data(
        bucket_name=bucket_name,
        prefix=prefix,
        glob_pattern=glob_pattern,
    )
    return DatasetSummary(
        source="cloud",
        name=report.get("prefix") or report.get("bucket") or "cloud",
        root=report.get("root", ""),
        image_count=int(report.get("image_files") or 0),
        status="ok",
    )


def list_cloud_dataset_prefixes(
    *,
    bucket_name: str | None = None,
    pages_prefix: str = "",
    limit: int = 200,
) -> list[str]:
    bucket = build_storage_bucket(bucket_name)
    prefix = normalize_prefix(pages_prefix)
    blobs = list_bucket_blobs(bucket, prefix=prefix)
    prefixes: set[str] = set()
    for blob in blobs:
        name = str(getattr(blob, "name", "") or "")
        if not name or name.endswith("/"):
            continue
        relative = name[len(prefix) :] if prefix and name.startswith(prefix) else name
        parts = Path(relative).parts
        if len(parts) > 1:
            prefixes.add(f"{prefix}{parts[0]}".strip("/"))
        else:
            prefixes.add(prefix.strip("/"))
        if len(prefixes) >= limit:
            break
    return sorted(item for item in prefixes if item)


def list_cloud_dataset_choices(
    *,
    bucket_name: str | None = None,
    pages_prefix: str = "",
    glob_pattern: str | None = None,
    limit: int = 500,
) -> list[CloudDatasetChoice]:
    bucket = build_storage_bucket(bucket_name)
    prefix = normalize_prefix(pages_prefix)
    blobs = list_bucket_blobs(bucket, prefix=prefix)
    image_blobs = select_bucket_image_blobs(blobs, glob_pattern=glob_pattern)

    object_counts: dict[str, int] = {}
    image_counts: dict[str, int] = {}
    updated_labels: dict[str, str] = {}
    updated_sort_values: dict[str, float] = {}

    def blob_updated(blob) -> tuple[float, str]:
        value = getattr(blob, "updated", None) or getattr(blob, "time_created", None)
        if isinstance(value, datetime):
            return value.timestamp(), value.strftime("%Y-%m-%d %H:%M")
        if value:
            return 0.0, str(value)
        return 0.0, ""

    for blob in image_blobs:
        name = str(getattr(blob, "name", "") or "")
        if not name or name.endswith("/"):
            continue
        relative = name[len(prefix) :] if prefix and name.startswith(prefix) else name
        parts = Path(relative).parts
        dataset_prefix = (
            f"{prefix}{parts[0]}".strip("/")
            if len(parts) > 1
            else prefix.strip("/")
        )
        if not dataset_prefix:
            continue
        object_counts[dataset_prefix] = object_counts.get(dataset_prefix, 0) + 1
        image_counts[dataset_prefix] = image_counts.get(dataset_prefix, 0) + 1
        updated_sort_value, updated_label = blob_updated(blob)
        if updated_sort_value >= updated_sort_values.get(dataset_prefix, -1.0):
            updated_sort_values[dataset_prefix] = updated_sort_value
            updated_labels[dataset_prefix] = updated_label

    dataset_prefixes = sorted(
        image_counts,
        key=lambda item: (-updated_sort_values.get(item, 0.0), item),
    )
    choices = [
        CloudDatasetChoice(
            prefix=dataset_prefix,
            image_count=image_counts.get(dataset_prefix, 0),
            object_count=object_counts.get(dataset_prefix, 0),
            updated_at=updated_labels.get(dataset_prefix, ""),
        )
        for dataset_prefix in dataset_prefixes
    ]
    return choices[: max(1, limit)]


def resolve_local_images_on_cloud(
    local_root: str | Path,
    *,
    bucket_name: str | None = None,
    cloud_prefix: str = "",
    glob_pattern: str | None = None,
    recursive: bool = True,
) -> CloudResolution:
    root_path, _all_files, image_files = collect_files(
        local_root,
        glob_pattern=glob_pattern,
        recursive=recursive,
    )
    local_refs = [str(path) for path in image_files]
    local_names = build_image_name_set(local_refs)
    local_duplicates = tuple(sorted(duplicate_image_names(local_refs)))

    bucket = build_storage_bucket(bucket_name)
    blobs = list_bucket_blobs(bucket, prefix=normalize_prefix(cloud_prefix))
    image_blobs = select_bucket_image_blobs(blobs, glob_pattern=glob_pattern)
    cloud_refs = [str(getattr(blob, "name", "") or "") for blob in image_blobs]
    cloud_names = build_image_name_set(cloud_refs)
    cloud_duplicates = tuple(sorted(duplicate_image_names(cloud_refs)))

    matched = local_names & cloud_names
    missing = tuple(sorted(local_names - cloud_names))
    return CloudResolution(
        local_root=str(root_path),
        bucket_name=str(getattr(bucket, "name", bucket_name or "")),
        prefix=normalize_prefix(cloud_prefix),
        local_image_count=len(local_names),
        cloud_image_count=len(cloud_names),
        matched_image_count=len(matched),
        missing_image_names=missing,
        duplicate_local_image_names=local_duplicates,
        duplicate_cloud_image_names=cloud_duplicates,
    )


def cloud_object_by_image_name(
    *,
    bucket_name: str | None = None,
    cloud_prefix: str = "",
    glob_pattern: str | None = None,
) -> dict[str, str]:
    bucket = build_storage_bucket(bucket_name)
    blobs = list_bucket_blobs(bucket, prefix=normalize_prefix(cloud_prefix))
    image_blobs = select_bucket_image_blobs(blobs, glob_pattern=glob_pattern)
    mapping: dict[str, str] = {}
    duplicates: set[str] = set()
    for blob in image_blobs:
        object_name = str(getattr(blob, "name", "") or "")
        image_name = image_name_from_reference(object_name)
        if not image_name:
            continue
        if image_name in mapping:
            duplicates.add(image_name)
            continue
        mapping[image_name] = object_name
    if duplicates:
        examples = ", ".join(sorted(duplicates)[:10])
        raise ValueError(
            "Cloud prefix contains duplicate image names. "
            f"Examples: {examples}"
        )
    return mapping
