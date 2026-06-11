from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable


IDENTITY_COLUMN = "image_name"
LEGACY_SOURCE_COLUMN = "file_name"


def _strip_gcs_bucket(value: str) -> str:
    if not value.startswith("gs://"):
        return value
    remainder = value[5:]
    _bucket, separator, object_name = remainder.partition("/")
    return object_name if separator else remainder


def image_name_from_reference(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    text = _strip_gcs_bucket(text).strip("/")
    if not text:
        return None
    name = Path(text).name
    return name or None


def image_name_from_path(path: str | Path) -> str:
    name = Path(path).name
    if not name:
        raise ValueError(f"Unable to derive image name from path: {path}")
    return name


def image_name_from_gcs_object(object_name: str) -> str:
    name = image_name_from_reference(object_name)
    if not name:
        raise ValueError(f"Unable to derive image name from GCS object: {object_name}")
    return name


def row_image_name(row: dict) -> str | None:
    direct = image_name_from_reference(row.get(IDENTITY_COLUMN))
    if direct:
        return direct
    for column in (
        LEGACY_SOURCE_COLUMN,
        "source_path",
        "gcs_object",
        "gcs_uri",
        "relative_path",
        "blob_name",
    ):
        name = image_name_from_reference(row.get(column))
        if name:
            return name
    return None


def identity_columns(source_reference: object) -> dict[str, str]:
    if not isinstance(source_reference, str) or not source_reference.strip():
        raise ValueError("source reference is empty; cannot build image identity")
    reference = source_reference.strip()
    image_name = image_name_from_reference(reference)
    if not image_name:
        raise ValueError(f"Unable to derive image name from source: {source_reference}")
    return {
        IDENTITY_COLUMN: image_name,
        LEGACY_SOURCE_COLUMN: reference,
    }


def ensure_row_image_name(row: dict) -> str | None:
    image_name = row_image_name(row)
    if image_name and not row.get(IDENTITY_COLUMN):
        row[IDENTITY_COLUMN] = image_name
    return image_name


def build_image_name_set(references: Iterable[object]) -> set[str]:
    names: set[str] = set()
    for value in references:
        image_name = image_name_from_reference(value)
        if image_name:
            names.add(image_name)
    return names


def duplicate_image_names(references: Iterable[object]) -> set[str]:
    counts = Counter(
        image_name
        for value in references
        if (image_name := image_name_from_reference(value))
    )
    return {name for name, count in counts.items() if count > 1}


def ensure_unique_image_names(
    references: Iterable[object],
    *,
    source_label: str = "inputs",
) -> None:
    duplicates = sorted(duplicate_image_names(references))
    if not duplicates:
        return
    examples = ", ".join(duplicates[:10])
    suffix = "..." if len(duplicates) > 10 else ""
    raise ValueError(
        f"Duplicate image_name values detected in {source_label}: "
        f"{examples}{suffix}. Image names must be unique across the dataset."
    )
