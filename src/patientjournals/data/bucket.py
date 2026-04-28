from __future__ import annotations

import fnmatch
from collections import Counter, defaultdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from patientjournals.batch.client import resolve_service_account_path
from patientjournals.config import config
from patientjournals.data.inspection import (
    FORMAT_EXTENSIONS,
    configured_image_extensions,
    _numeric_stats,
)


def resolve_bucket_name(bucket_name: str | None = None) -> str:
    value = (bucket_name or config.gcs_bucket_name or "").strip()
    if not value:
        raise ValueError("Bucket name is empty. Pass --bucket-name or set config.gcs_bucket_name.")
    return value


def normalize_prefix(prefix: str | None) -> str:
    value = str(prefix or "").strip().strip("/")
    return f"{value}/" if value else ""


def build_storage_bucket(bucket_name: str | None = None):
    from google.cloud import storage

    resolved_name = resolve_bucket_name(bucket_name)
    service_account_file = str(config.service_account_file or "").strip()
    if service_account_file:
        candidate = Path(service_account_file).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if candidate.exists():
            service_account_path = resolve_service_account_path(service_account_file)
            client = storage.Client.from_service_account_json(str(service_account_path))
        else:
            client = storage.Client()
    else:
        client = storage.Client()
    return client.bucket(resolved_name)


def _is_folder_placeholder(blob: object) -> bool:
    name = str(getattr(blob, "name", "") or "")
    return name.endswith("/")


def _blob_extension(blob: object) -> str:
    return Path(str(getattr(blob, "name", "") or "")).suffix.lower().lstrip(".")


def _blob_size(blob: object) -> int | None:
    size = getattr(blob, "size", None)
    if isinstance(size, int):
        return size
    try:
        return int(size)
    except (TypeError, ValueError):
        return None


def _bucket_relative_name(blob_name: str, prefix: str) -> str:
    if prefix and blob_name.startswith(prefix):
        return blob_name[len(prefix) :]
    return blob_name


def _bucket_parent(blob_name: str, prefix: str) -> str:
    relative = _bucket_relative_name(blob_name, prefix).strip("/")
    parent = str(Path(relative).parent)
    return "." if parent == "." else parent


def _bucket_depth(blob_name: str, prefix: str) -> int:
    relative = _bucket_relative_name(blob_name, prefix).strip("/")
    if not relative:
        return 0
    return max(0, len(Path(relative).parts) - 1)


def _folder_names_from_blob(blob_name: str, prefix: str) -> set[str]:
    relative = _bucket_relative_name(blob_name, prefix).strip("/")
    parts = Path(relative).parts[:-1]
    folders: set[str] = set()
    for index in range(1, len(parts) + 1):
        folders.add(str(Path(*parts[:index])))
    return folders


def _matches_glob(blob: object, glob_pattern: str | None) -> bool:
    pattern = glob_pattern or "*"
    name = str(getattr(blob, "name", "") or "")
    return fnmatch.fnmatch(Path(name).name, pattern)


def _content_type_format_issue(blob: object, image_format: str | None) -> str | None:
    content_type = str(getattr(blob, "content_type", "") or "").strip().lower()
    if not content_type or not image_format:
        return None
    expected_fragment = {
        "JPEG": "jpeg",
        "PNG": "png",
        "TIFF": "tiff",
        "WEBP": "webp",
        "BMP": "bmp",
    }.get(image_format.upper())
    if expected_fragment and expected_fragment not in content_type:
        return f"content_type_format_mismatch:{content_type}:{image_format}"
    return None


def _extension_format_issue(blob: object, image_format: str | None) -> str | None:
    if not image_format:
        return None
    expected = FORMAT_EXTENSIONS.get(image_format.upper())
    if not expected:
        return None
    suffix = _blob_extension(blob)
    if suffix not in expected:
        return f"extension_format_mismatch:{suffix}:{image_format}"
    return None


def list_bucket_blobs(bucket, *, prefix: str | None = None) -> list[object]:
    normalized_prefix = normalize_prefix(prefix)
    return sorted(
        (blob for blob in bucket.list_blobs(prefix=normalized_prefix or None)),
        key=lambda item: str(getattr(item, "name", "") or ""),
    )


def select_bucket_image_blobs(
    blobs: list[object],
    *,
    glob_pattern: str | None = None,
    allowed_extensions: set[str] | None = None,
) -> list[object]:
    extensions = allowed_extensions or configured_image_extensions()
    return [
        blob
        for blob in blobs
        if not _is_folder_placeholder(blob)
        and _blob_extension(blob) in extensions
        and _matches_glob(blob, glob_pattern)
    ]


def summarize_bucket_data(
    bucket=None,
    *,
    bucket_name: str | None = None,
    prefix: str | None = None,
    glob_pattern: str | None = None,
    allowed_extensions: set[str] | None = None,
) -> dict[str, Any]:
    bucket_obj = bucket or build_storage_bucket(bucket_name)
    resolved_bucket_name = str(getattr(bucket_obj, "name", None) or resolve_bucket_name(bucket_name))
    normalized_prefix = normalize_prefix(prefix)
    blobs = list_bucket_blobs(bucket_obj, prefix=normalized_prefix)
    image_blobs = select_bucket_image_blobs(
        blobs,
        glob_pattern=glob_pattern,
        allowed_extensions=allowed_extensions,
    )
    extensions = allowed_extensions or configured_image_extensions()
    non_placeholder_blobs = [blob for blob in blobs if not _is_folder_placeholder(blob)]
    folder_placeholders = [blob for blob in blobs if _is_folder_placeholder(blob)]
    non_image_blobs = [
        blob for blob in non_placeholder_blobs if _blob_extension(blob) not in extensions
    ]

    folder_file_counts: dict[str, int] = defaultdict(int)
    extension_counts: Counter[str] = Counter()
    depth_counts: Counter[int] = Counter()
    folders: set[str] = set()
    sizes: list[int] = []

    for blob in image_blobs:
        name = str(getattr(blob, "name", "") or "")
        folder_file_counts[_bucket_parent(name, normalized_prefix)] += 1
        extension_counts[_blob_extension(blob) or "<none>"] += 1
        depth_counts[_bucket_depth(name, normalized_prefix)] += 1
        folders.update(_folder_names_from_blob(name, normalized_prefix))
        size = _blob_size(blob)
        if size is not None:
            sizes.append(size)

    for blob in non_placeholder_blobs:
        folders.update(_folder_names_from_blob(str(getattr(blob, "name", "") or ""), normalized_prefix))

    root_uri = f"gs://{resolved_bucket_name}/{normalized_prefix}".rstrip("/")
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "gcs",
        "root": root_uri,
        "bucket": resolved_bucket_name,
        "prefix": normalized_prefix,
        "glob": glob_pattern or "*",
        "folder_count": len(folders) + 1,
        "child_folder_count": len(folders),
        "empty_folder_count": 0,
        "folder_placeholder_blobs": len(folder_placeholders),
        "folders_with_images": len(folder_file_counts),
        "total_files": len(non_placeholder_blobs),
        "total_blobs": len(blobs),
        "image_files": len(image_blobs),
        "non_image_files": len(non_image_blobs),
        "files_by_extension": dict(sorted(extension_counts.items())),
        "files_by_depth": {
            str(depth): count for depth, count in sorted(depth_counts.items())
        },
        "files_by_folder": dict(sorted(folder_file_counts.items())),
        "image_size_bytes": _numeric_stats(sizes),
    }


def validate_bucket_image(
    blob: object,
    *,
    bucket_name: str,
    prefix: str,
    duplicate_basenames: set[str],
) -> dict[str, Any]:
    name = str(getattr(blob, "name", "") or "")
    issues: list[str] = []
    warnings: list[str] = []
    width: int | None = None
    height: int | None = None
    image_format: str | None = None
    mode: str | None = None
    size_bytes = _blob_size(blob)

    if size_bytes == 0:
        issues.append("empty_file")
    if size_bytes is None:
        warnings.append("missing_blob_size")

    try:
        payload = blob.download_as_bytes()
        with Image.open(BytesIO(payload)) as image:
            image.verify()
        with Image.open(BytesIO(payload)) as image:
            width, height = image.size
            image_format = image.format
            mode = image.mode
            if width <= 0 or height <= 0:
                issues.append("invalid_dimensions")
            extension_issue = _extension_format_issue(blob, image_format)
            if extension_issue:
                warnings.append(extension_issue)
            content_type_issue = _content_type_format_issue(blob, image_format)
            if content_type_issue:
                warnings.append(content_type_issue)
    except Exception as exc:
        issues.append(f"image_open_failed:{type(exc).__name__}")

    if Path(name).name in duplicate_basenames:
        warnings.append("duplicate_basename")

    status = "ok"
    if issues:
        status = "error"
    elif warnings:
        status = "warning"

    relative_name = _bucket_relative_name(name, prefix)
    return {
        "status": status,
        "issues": ";".join(issues),
        "warnings": ";".join(warnings),
        "source": "gcs",
        "bucket": bucket_name,
        "blob_name": name,
        "file_name": Path(name).name,
        "relative_path": relative_name,
        "parent_folder": _bucket_parent(name, prefix),
        "extension": _blob_extension(blob),
        "size_bytes": size_bytes,
        "width": width,
        "height": height,
        "format": image_format,
        "mode": mode,
        "content_type": str(getattr(blob, "content_type", "") or ""),
    }


def validate_bucket_data(
    bucket=None,
    *,
    bucket_name: str | None = None,
    prefix: str | None = None,
    glob_pattern: str | None = None,
    allowed_extensions: set[str] | None = None,
) -> dict[str, Any]:
    bucket_obj = bucket or build_storage_bucket(bucket_name)
    resolved_bucket_name = str(getattr(bucket_obj, "name", None) or resolve_bucket_name(bucket_name))
    normalized_prefix = normalize_prefix(prefix)
    blobs = list_bucket_blobs(bucket_obj, prefix=normalized_prefix)
    image_blobs = select_bucket_image_blobs(
        blobs,
        glob_pattern=glob_pattern,
        allowed_extensions=allowed_extensions,
    )
    basename_counts = Counter(Path(str(getattr(blob, "name", "") or "")).name for blob in image_blobs)
    duplicate_basenames = {name for name, count in basename_counts.items() if count > 1}
    records = [
        validate_bucket_image(
            blob,
            bucket_name=resolved_bucket_name,
            prefix=normalized_prefix,
            duplicate_basenames=duplicate_basenames,
        )
        for blob in tqdm(image_blobs, desc="Validating bucket images", unit="img")
    ]
    status_counts = Counter(record["status"] for record in records)
    report_status = "ok"
    if status_counts.get("error", 0) > 0 or not image_blobs:
        report_status = "error"
    elif status_counts.get("warning", 0) > 0:
        report_status = "warning"

    root_uri = f"gs://{resolved_bucket_name}/{normalized_prefix}".rstrip("/")
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "gcs",
        "root": root_uri,
        "bucket": resolved_bucket_name,
        "prefix": normalized_prefix,
        "glob": glob_pattern or "*",
        "status": report_status,
        "total_images": len(image_blobs),
        "ok_count": status_counts.get("ok", 0),
        "warning_count": status_counts.get("warning", 0),
        "error_count": status_counts.get("error", 0),
        "duplicate_basename_count": len(duplicate_basenames),
        "duplicate_basenames": sorted(duplicate_basenames),
        "records": records,
    }

