from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from patientjournals.config import config
from patientjournals.data.bucket import build_storage_bucket, normalize_prefix


VALIDATION_METADATA_FILE = "validation_metadata.json"


def write_validation_metadata(
    *,
    run_dir: str | Path,
    csv_path: str | Path,
    dataset_path: str | Path,
    validator_id: str,
    decision_count: int,
    sampling_mode: str = "",
) -> Path:
    run_path = Path(run_dir).expanduser()
    dataset = Path(dataset_path).expanduser()
    csv_file = Path(csv_path).expanduser()
    metadata = {
        "run_id": run_path.name,
        "validator_id": validator_id,
        "sampling_mode": sampling_mode,
        "dataset_file": dataset.name,
        "dataset_path": str(dataset),
        "validation_file": csv_file.name,
        "decision_count": int(decision_count),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    path = run_path / VALIDATION_METADATA_FILE
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _upload_file(bucket: Any, source: Path, object_name: str, *, content_type: str) -> str:
    blob = bucket.blob(object_name)
    blob.upload_from_filename(str(source), content_type=content_type)
    return f"gs://{getattr(bucket, 'name', config.gcs_bucket_name)}/{object_name}"


def upload_validation_run(
    *,
    run_dir: str | Path,
    csv_path: str | Path,
    metadata_path: str | Path,
    bucket_name: str | None = None,
    prefix: str | None = None,
) -> dict[str, str]:
    if not config.upload_validation_to_gcs:
        return {}
    bucket = build_storage_bucket(bucket_name)
    run_path = Path(run_dir).expanduser()
    csv_file = Path(csv_path).expanduser()
    metadata_file = Path(metadata_path).expanduser()
    base_prefix = normalize_prefix(prefix or config.validations_gcs_prefix)
    object_base = f"{base_prefix}{run_path.name}"
    csv_uri = _upload_file(
        bucket,
        csv_file,
        f"{object_base}/{csv_file.name}",
        content_type="text/csv",
    )
    metadata_uri = _upload_file(
        bucket,
        metadata_file,
        f"{object_base}/{metadata_file.name}",
        content_type="application/json",
    )
    return {"validation_csv_uri": csv_uri, "validation_metadata_uri": metadata_uri}
