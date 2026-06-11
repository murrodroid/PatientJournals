from __future__ import annotations

from datetime import datetime
from pathlib import Path

from patientjournals.app.models import CloudDatasetChoice, CloudResolution, DatasetSummary
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
    image_name_from_reference,
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
