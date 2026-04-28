from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from patientjournals.shared.tools import find_newest_dataset


def _normalize_output_format(output_format: str) -> str:
    fmt = output_format.strip().lower().lstrip(".")
    if fmt not in {"csv", "jsonl"}:
        raise ValueError(f"Unsupported output_format: {output_format}")
    return fmt


def normalize_gcs_file_key(
    value: object,
    *,
    bucket_name: str | None = None,
) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None

    if text.startswith("gs://"):
        remainder = text[5:]
        bucket, separator, object_name = remainder.partition("/")
        if not separator or not object_name:
            return None
        if bucket_name and bucket != bucket_name:
            return None
        text = object_name

    text = text.lstrip("/")
    return text or None


def resolve_continue_dataset_path(
    value: str,
    *,
    run_root: str | Path,
    dataset_name: str,
) -> Path:
    if value.strip().lower() == "newest":
        return find_newest_dataset(run_root, dataset_name)
    return Path(value).expanduser()


def load_dataset_key_coverage(
    dataset_path: str | Path,
    *,
    output_format: str | None = None,
    csv_sep: str = "$",
    bucket_name: str | None = None,
) -> tuple[str, set[str], int]:
    path = Path(dataset_path).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"dataset not found or not a file: {path}")

    fmt = _normalize_output_format(output_format or path.suffix.lstrip("."))
    keys: set[str] = set()
    row_count = 0

    if fmt == "jsonl":
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                row_count += 1
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                key = normalize_gcs_file_key(
                    payload.get("file_name"),
                    bucket_name=bucket_name,
                )
                if key:
                    keys.add(key)
        return fmt, keys, row_count

    df = pd.read_csv(path, sep=csv_sep)
    row_count = len(df)
    if "file_name" not in df.columns:
        return fmt, keys, row_count

    for value in df["file_name"].dropna().astype(str):
        key = normalize_gcs_file_key(value, bucket_name=bucket_name)
        if key:
            keys.add(key)
    return fmt, keys, row_count


def copy_dataset_rows_for_keys(
    src_path: str | Path,
    dest_path: str | Path,
    *,
    keys: set[str] | None,
    output_format: str | None = None,
    csv_sep: str = "$",
    bucket_name: str | None = None,
) -> int:
    src = Path(src_path).expanduser()
    dest = Path(dest_path).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fmt = _normalize_output_format(output_format or src.suffix.lstrip("."))

    if fmt == "jsonl":
        kept = 0
        with open(src, "r", encoding="utf-8") as src_handle, open(
            dest,
            "w",
            encoding="utf-8",
        ) as dest_handle:
            for line in src_handle:
                raw = line.strip()
                if not raw:
                    continue
                if keys is not None:
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    key = normalize_gcs_file_key(
                        payload.get("file_name"),
                        bucket_name=bucket_name,
                    )
                    if key not in keys:
                        continue
                dest_handle.write(raw)
                dest_handle.write("\n")
                kept += 1
        return kept

    df = pd.read_csv(src, sep=csv_sep)
    if keys is not None and "file_name" in df.columns:
        mask = df["file_name"].apply(
            lambda value: normalize_gcs_file_key(
                value,
                bucket_name=bucket_name,
            )
            in keys
        )
        df = df[mask]
    df.to_csv(dest, index=False, sep=csv_sep)
    return len(df)
