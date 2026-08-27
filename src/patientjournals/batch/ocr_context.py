from __future__ import annotations

from patientjournals.config import config
from patientjournals.shared.ocr import (
    OcrDocument,
    detect_configured_ocr,
    image_identity,
    render_ocr_context,
)


def _sidecar_name(image_name: str) -> str:
    suffix = str(config.ocr_sidecar_suffix or ".ocr.json")
    return f"{image_name}{suffix}"


def _download(blob: object) -> bytes:
    download = getattr(blob, "download_as_bytes", None)
    if not callable(download):
        raise TypeError("GCS blob does not support download_as_bytes().")
    payload = download()
    if not isinstance(payload, bytes):
        payload = bytes(payload)
    return payload


def _cached_document(blob: object, image_sha256: str) -> OcrDocument | None:
    bucket = getattr(blob, "bucket", None)
    image_name = str(getattr(blob, "name", "") or "")
    blob_factory = getattr(bucket, "blob", None)
    if not image_name or not callable(blob_factory):
        return None
    sidecar = blob_factory(_sidecar_name(image_name))
    try:
        document = OcrDocument.from_json(_download(sidecar))
    except Exception:  # noqa: BLE001 - a missing/stale sidecar is a cache miss
        return None
    if document.image_sha256 != image_sha256:
        return None
    return document


def _store_document(blob: object, document: OcrDocument) -> None:
    bucket = getattr(blob, "bucket", None)
    image_name = str(getattr(blob, "name", "") or "")
    blob_factory = getattr(bucket, "blob", None)
    if not image_name or not callable(blob_factory):
        return
    sidecar = blob_factory(_sidecar_name(image_name))
    upload = getattr(sidecar, "upload_from_string", None)
    if not callable(upload):
        return
    try:
        upload(document.to_json(), content_type="application/json")
    except Exception:  # noqa: BLE001 - cache persistence must not fail generation
        return


def ocr_document_for_blob(blob: object) -> OcrDocument | None:
    """OCR the exact current GCS object, reusing only digest-matched sidecars."""

    if not bool(config.ocr_enabled):
        return None
    try:
        image_bytes = _download(blob)
        width, height, digest = image_identity(image_bytes)
        cached = _cached_document(blob, digest)
        if cached is not None and (cached.width, cached.height) == (width, height):
            return cached

        attempt = detect_configured_ocr(image_bytes)
        if attempt.document is not None:
            _store_document(blob, attempt.document)
        return attempt.document
    except Exception as exc:  # noqa: BLE001 - optional OCR is explicitly fail-open
        if bool(config.ocr_required):
            raise RuntimeError(
                f"OCR is required for GCS object {getattr(blob, 'name', '')}: {exc}"
            ) from exc
        return None


def ocr_context_for_blob(blob: object) -> str:
    return render_ocr_context(ocr_document_for_blob(blob))
