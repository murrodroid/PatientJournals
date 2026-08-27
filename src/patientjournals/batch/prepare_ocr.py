from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from tqdm import tqdm

from patientjournals.batch.ocr_context import (
    OcrMetadataPreparation,
    prepare_ocr_metadata_for_blobs,
)
from patientjournals.batch.submit_inputs import _list_input_blobs
from patientjournals.config import config
from patientjournals.data.bucket import build_storage_bucket


@dataclass(frozen=True)
class CloudOcrPreparationSummary:
    selected: int
    prepared: int
    cached: int
    failed: int
    manifest_object: str
    records: tuple[OcrMetadataPreparation, ...]

    @property
    def successful(self) -> bool:
        return self.failed == 0


def _manifest_object_name() -> str:
    value = str(config.batch_ocr_manifest_object or "").strip().strip("/")
    if not value:
        raise ValueError("config.batch_ocr_manifest_object is empty.")
    return value


def _upload_manifest(
    *,
    bucket: object,
    records: Sequence[OcrMetadataPreparation],
    selected: int,
    workers: int,
    api_batch_size: int,
) -> str:
    object_name = _manifest_object_name()
    blob_factory = getattr(bucket, "blob", None)
    if not callable(blob_factory):
        raise TypeError("GCS bucket does not support blob().")
    manifest_blob = blob_factory(object_name)
    upload = getattr(manifest_blob, "upload_from_string", None)
    if not callable(upload):
        raise TypeError("GCS OCR manifest blob does not support upload_from_string().")
    status_counts = {
        status: sum(record.status == status for record in records)
        for status in ("prepared", "cached", "failed")
    }
    payload = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bucket": str(getattr(bucket, "name", "") or ""),
        "selected": selected,
        **status_counts,
        "ocr_backend": str(config.ocr_backend),
        "ocr_language_hints": list(config.ocr_language_hints or ()),
        "ocr_api_batch_size": api_batch_size,
        "ocr_api_batch_max_bytes": int(config.batch_ocr_api_batch_max_bytes),
        "ocr_workers": workers,
        "sidecar_suffix": str(config.ocr_sidecar_suffix),
        "records": [record.manifest_record() for record in records],
    }
    upload(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        content_type="application/json",
    )
    return object_name


def _split_vision_batches(
    blobs: Sequence[object],
    *,
    batch_size: int,
    max_bytes: int,
) -> list[list[object]]:
    """Respect both Vision's 16-image limit and a conservative payload cap."""

    if batch_size <= 0:
        raise ValueError("OCR API batch size must be greater than zero.")
    if batch_size > 16:
        raise ValueError("Google Vision OCR API batches cannot exceed 16 images.")
    if max_bytes <= 0:
        raise ValueError("OCR API batch byte limit must be greater than zero.")

    batches: list[list[object]] = []
    current: list[object] = []
    current_bytes = 0
    for blob in blobs:
        raw_size = getattr(blob, "size", None)
        try:
            blob_size = max(0, int(raw_size)) if raw_size is not None else max_bytes
        except (TypeError, ValueError):
            blob_size = max_bytes
        if current and (
            len(current) >= batch_size or current_bytes + blob_size > max_bytes
        ):
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(blob)
        current_bytes += blob_size
        if blob_size >= max_bytes:
            batches.append(current)
            current = []
            current_bytes = 0
    if current:
        batches.append(current)
    return batches


def prepare_cloud_ocr_metadata(
    *,
    bucket: object | None = None,
    blobs: Sequence[object] | None = None,
    workers: int | None = None,
    api_batch_size: int | None = None,
    force: bool = False,
    limit: int | None = None,
    log=print,
) -> CloudOcrPreparationSummary:
    """Populate GCS OCR sidecars for the configured batch input selection."""

    if not bool(config.ocr_enabled):
        raise ValueError("OCR is disabled. Set config.ocr_enabled=True first.")
    active_bucket = bucket or build_storage_bucket()
    selected_blobs = list(
        blobs
        if blobs is not None
        else _list_input_blobs(active_bucket, log=log)
    )
    selected_blobs.sort(key=lambda item: str(getattr(item, "name", "")))
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be greater than zero (received {limit}).")
        selected_blobs = selected_blobs[:limit]
    if not selected_blobs:
        raise FileNotFoundError("No cloud images matched the configured batch input set.")

    max_workers = max(1, int(workers or config.batch_ocr_workers or 1))
    resolved_batch_size = int(
        api_batch_size or config.batch_ocr_api_batch_size or 1
    )
    batches = _split_vision_batches(
        selected_blobs,
        batch_size=resolved_batch_size,
        max_bytes=int(config.batch_ocr_api_batch_max_bytes),
    )
    log(
        f"Preparing {len(selected_blobs)} image(s) as {len(batches)} Vision "
        f"RPC batch(es), up to {resolved_batch_size} images each, with "
        f"{min(max_workers, len(batches))} concurrent RPC batch(es)."
    )
    records: list[OcrMetadataPreparation] = []
    with ThreadPoolExecutor(
        max_workers=min(max_workers, len(batches))
    ) as executor:
        futures = {
            executor.submit(
                prepare_ocr_metadata_for_blobs,
                batch,
                force=force,
            ): batch
            for batch in batches
        }
        progress = tqdm(
            total=len(selected_blobs),
            desc="Preparing cloud OCR metadata",
            unit="img",
        )
        try:
            for future in as_completed(futures):
                batch_records = future.result()
                records.extend(batch_records)
                progress.update(len(batch_records))
        finally:
            progress.close()

    records.sort(key=lambda record: record.blob_name)
    manifest_object = _upload_manifest(
        bucket=active_bucket,
        records=records,
        selected=len(selected_blobs),
        workers=max_workers,
        api_batch_size=resolved_batch_size,
    )
    prepared = sum(record.status == "prepared" for record in records)
    cached = sum(record.status == "cached" for record in records)
    failed = sum(record.status == "failed" for record in records)
    log(
        "Cloud OCR metadata complete: "
        f"selected={len(selected_blobs)} prepared={prepared} "
        f"cached={cached} failed={failed} manifest=gs://"
        f"{getattr(active_bucket, 'name', '')}/{manifest_object}"
    )
    return CloudOcrPreparationSummary(
        selected=len(selected_blobs),
        prepared=prepared,
        cached=cached,
        failed=failed,
        manifest_object=manifest_object,
        records=tuple(records),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create generation-bound OCR sidecars in GCS for batch input images."
        )
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--api-batch-size", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Exit successfully even when individual images fail OCR.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = prepare_cloud_ocr_metadata(
        workers=args.workers,
        api_batch_size=args.api_batch_size,
        force=args.force,
        limit=args.limit,
    )
    if summary.failed and not args.allow_failures:
        raise SystemExit(
            f"Cloud OCR metadata failed for {summary.failed}/{summary.selected} image(s). "
            f"See gs://{config.gcs_bucket_name}/{summary.manifest_object}."
        )


if __name__ == "__main__":
    main()
