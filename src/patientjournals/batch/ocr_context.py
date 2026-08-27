from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Sequence

from patientjournals.config import config
from patientjournals.shared.ocr import (
    OcrDocument,
    detect_configured_ocr,
    detect_configured_ocr_batch,
    render_ocr_context,
)


@dataclass(frozen=True)
class CloudBlobIdentity:
    bucket: str
    name: str
    generation: str
    size: int | None
    crc32c: str | None
    md5_hash: str | None
    etag: str | None

    @classmethod
    def from_blob(cls, blob: object) -> "CloudBlobIdentity":
        bucket = getattr(blob, "bucket", None)
        size = getattr(blob, "size", None)
        try:
            normalized_size = int(size) if size is not None else None
        except (TypeError, ValueError):
            normalized_size = None
        return cls(
            bucket=str(getattr(bucket, "name", "") or ""),
            name=str(getattr(blob, "name", "") or ""),
            generation=str(getattr(blob, "generation", "") or ""),
            size=normalized_size,
            crc32c=str(getattr(blob, "crc32c", "") or "") or None,
            md5_hash=str(getattr(blob, "md5_hash", "") or "") or None,
            etag=str(getattr(blob, "etag", "") or "") or None,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "bucket": self.bucket,
            "name": self.name,
            "generation": self.generation,
            "size": self.size,
            "crc32c": self.crc32c,
            "md5_hash": self.md5_hash,
            "etag": self.etag,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CloudBlobIdentity":
        raw_size = payload.get("size")
        return cls(
            bucket=str(payload.get("bucket") or ""),
            name=str(payload.get("name") or ""),
            generation=str(payload.get("generation") or ""),
            size=int(raw_size) if raw_size is not None else None,
            crc32c=str(payload.get("crc32c") or "") or None,
            md5_hash=str(payload.get("md5_hash") or "") or None,
            etag=str(payload.get("etag") or "") or None,
        )

    def matches(self, current: "CloudBlobIdentity") -> bool:
        if not self.generation or not current.generation:
            return False
        if (
            self.bucket != current.bucket
            or self.name != current.name
            or self.generation != current.generation
        ):
            return False
        for expected, actual in (
            (self.size, current.size),
            (self.crc32c, current.crc32c),
            (self.md5_hash, current.md5_hash),
        ):
            if expected is not None and actual is not None and expected != actual:
                return False
        return True


@dataclass(frozen=True)
class CloudOcrMetadata:
    source: CloudBlobIdentity
    document: OcrDocument
    created_at: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": 2,
                "created_at": self.created_at,
                "source": self.source.to_dict(),
                "ocr": self.document.to_dict(),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, value: str | bytes) -> "CloudOcrMetadata":
        payload = json.loads(value)
        if not isinstance(payload, dict) or payload.get("version") != 2:
            raise ValueError("Unsupported cloud OCR metadata version.")
        source = payload.get("source")
        document = payload.get("ocr")
        if not isinstance(source, dict) or not isinstance(document, dict):
            raise ValueError("Cloud OCR metadata is missing source or OCR data.")
        return cls(
            source=CloudBlobIdentity.from_dict(source),
            document=OcrDocument.from_dict(document),
            created_at=str(payload.get("created_at") or ""),
        )


@dataclass(frozen=True)
class OcrMetadataPreparation:
    blob_name: str
    sidecar_name: str
    status: str
    source: CloudBlobIdentity
    document: OcrDocument | None = None
    error: str | None = None

    def manifest_record(self) -> dict[str, object]:
        return {
            "blob_name": self.blob_name,
            "sidecar_name": self.sidecar_name,
            "status": self.status,
            "source": self.source.to_dict(),
            "image_sha256": self.document.image_sha256 if self.document else None,
            "ocr_backend": self.document.backend if self.document else None,
            "ocr_line_count": len(self.document.lines) if self.document else 0,
            "error": self.error,
        }


@dataclass(frozen=True)
class _PendingOcr:
    blob: object
    source: CloudBlobIdentity
    sidecar_name: str
    image_bytes: bytes


def _sidecar_name(image_name: str) -> str:
    suffix = str(config.ocr_sidecar_suffix or ".ocr.json")
    return f"{image_name}{suffix}"


def _download(blob: object, *, generation: str | None = None) -> bytes:
    download = getattr(blob, "download_as_bytes", None)
    if not callable(download):
        raise TypeError("GCS blob does not support download_as_bytes().")
    kwargs: dict[str, object] = {}
    if generation:
        kwargs["if_generation_match"] = int(generation)
    payload = download(**kwargs)
    if not isinstance(payload, bytes):
        payload = bytes(payload)
    return payload


def _reload_identity(blob: object) -> CloudBlobIdentity:
    identity = CloudBlobIdentity.from_blob(blob)
    if identity.generation:
        return identity
    reload_blob = getattr(blob, "reload", None)
    if callable(reload_blob):
        reload_blob()
        identity = CloudBlobIdentity.from_blob(blob)
    if not identity.generation:
        raise ValueError(
            f"GCS object generation is unavailable for {identity.name!r}; "
            "OCR metadata cannot be bound safely."
        )
    return identity


def _sidecar_blob(blob: object) -> object:
    bucket = getattr(blob, "bucket", None)
    image_name = str(getattr(blob, "name", "") or "")
    blob_factory = getattr(bucket, "blob", None)
    if not image_name or not callable(blob_factory):
        raise TypeError("GCS image blob is missing its bucket or object name.")
    return blob_factory(_sidecar_name(image_name))


_METADATA_CACHE: dict[tuple[str, str, str], CloudOcrMetadata] = {}
_METADATA_CACHE_LOCK = Lock()


def _cache_key(identity: CloudBlobIdentity) -> tuple[str, str, str]:
    return identity.bucket, identity.name, identity.generation


def load_ocr_metadata_for_blob(blob: object) -> CloudOcrMetadata | None:
    """Load a generation-matched sidecar without downloading the image."""

    identity = _reload_identity(blob)
    key = _cache_key(identity)
    with _METADATA_CACHE_LOCK:
        cached = _METADATA_CACHE.get(key)
    if cached is not None and cached.source.matches(identity):
        return cached

    try:
        metadata = CloudOcrMetadata.from_json(_download(_sidecar_blob(blob)))
    except Exception:  # noqa: BLE001 - missing/legacy/invalid sidecars are absent
        return None
    if not metadata.source.matches(identity):
        return None
    with _METADATA_CACHE_LOCK:
        _METADATA_CACHE[key] = metadata
    return metadata


def prepare_ocr_metadata_for_blob(
    blob: object,
    *,
    force: bool = False,
) -> OcrMetadataPreparation:
    """Download one immutable image generation, OCR it, and persist its sidecar."""

    try:
        identity = _reload_identity(blob)
    except Exception as exc:  # noqa: BLE001 - retain failure in the cloud manifest
        identity = CloudBlobIdentity.from_blob(blob)
        return OcrMetadataPreparation(
            blob_name=identity.name,
            sidecar_name=_sidecar_name(identity.name),
            status="failed",
            source=identity,
            error=str(exc),
        )
    sidecar_name = _sidecar_name(identity.name)
    if not force:
        cached = load_ocr_metadata_for_blob(blob)
        if cached is not None:
            return OcrMetadataPreparation(
                blob_name=identity.name,
                sidecar_name=sidecar_name,
                status="cached",
                source=identity,
                document=cached.document,
            )

    try:
        image_bytes = _download(blob, generation=identity.generation)
        attempt = detect_configured_ocr(image_bytes)
        if attempt.document is None:
            raise RuntimeError(attempt.error or "OCR returned no document.")
        metadata = CloudOcrMetadata(
            source=identity,
            document=attempt.document,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        sidecar = _sidecar_blob(blob)
        upload = getattr(sidecar, "upload_from_string", None)
        if not callable(upload):
            raise TypeError("GCS OCR sidecar does not support upload_from_string().")
        upload(metadata.to_json(), content_type="application/json")
        with _METADATA_CACHE_LOCK:
            _METADATA_CACHE[_cache_key(identity)] = metadata
        return OcrMetadataPreparation(
            blob_name=identity.name,
            sidecar_name=sidecar_name,
            status="prepared",
            source=identity,
            document=attempt.document,
        )
    except Exception as exc:  # noqa: BLE001 - recorded per object in cloud manifest
        return OcrMetadataPreparation(
            blob_name=identity.name,
            sidecar_name=sidecar_name,
            status="failed",
            source=identity,
            error=str(exc),
        )


def prepare_ocr_metadata_for_blobs(
    blobs: Sequence[object],
    *,
    force: bool = False,
) -> tuple[OcrMetadataPreparation, ...]:
    """Prepare one provider batch while retaining a result for every image."""

    records: list[OcrMetadataPreparation] = []
    pending: list[_PendingOcr] = []
    for blob in blobs:
        try:
            identity = _reload_identity(blob)
        except Exception as exc:  # noqa: BLE001 - persisted as an image failure
            identity = CloudBlobIdentity.from_blob(blob)
            records.append(
                OcrMetadataPreparation(
                    blob_name=identity.name,
                    sidecar_name=_sidecar_name(identity.name),
                    status="failed",
                    source=identity,
                    error=str(exc),
                )
            )
            continue

        sidecar_name = _sidecar_name(identity.name)
        if not force:
            cached = load_ocr_metadata_for_blob(blob)
            if cached is not None:
                records.append(
                    OcrMetadataPreparation(
                        blob_name=identity.name,
                        sidecar_name=sidecar_name,
                        status="cached",
                        source=identity,
                        document=cached.document,
                    )
                )
                continue
        try:
            image_bytes = _download(blob, generation=identity.generation)
        except Exception as exc:  # noqa: BLE001 - persisted as an image failure
            records.append(
                OcrMetadataPreparation(
                    blob_name=identity.name,
                    sidecar_name=sidecar_name,
                    status="failed",
                    source=identity,
                    error=str(exc),
                )
            )
            continue
        pending.append(
            _PendingOcr(
                blob=blob,
                source=identity,
                sidecar_name=sidecar_name,
                image_bytes=image_bytes,
            )
        )

    attempts = (
        detect_configured_ocr_batch([item.image_bytes for item in pending])
        if pending
        else ()
    )
    for item, attempt in zip(pending, attempts, strict=True):
        document = attempt.document
        if document is None:
            records.append(
                OcrMetadataPreparation(
                    blob_name=item.source.name,
                    sidecar_name=item.sidecar_name,
                    status="failed",
                    source=item.source,
                    error=attempt.error or "OCR returned no document.",
                )
            )
            continue
        try:
            metadata = CloudOcrMetadata(
                source=item.source,
                document=document,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            sidecar = _sidecar_blob(item.blob)
            upload = getattr(sidecar, "upload_from_string", None)
            if not callable(upload):
                raise TypeError(
                    "GCS OCR sidecar does not support upload_from_string()."
                )
            upload(metadata.to_json(), content_type="application/json")
            with _METADATA_CACHE_LOCK:
                _METADATA_CACHE[_cache_key(item.source)] = metadata
            records.append(
                OcrMetadataPreparation(
                    blob_name=item.source.name,
                    sidecar_name=item.sidecar_name,
                    status="prepared",
                    source=item.source,
                    document=document,
                )
            )
        except Exception as exc:  # noqa: BLE001 - persisted as an image failure
            records.append(
                OcrMetadataPreparation(
                    blob_name=item.source.name,
                    sidecar_name=item.sidecar_name,
                    status="failed",
                    source=item.source,
                    error=str(exc),
                )
            )
    return tuple(records)


def ocr_document_for_blob(blob: object) -> OcrDocument | None:
    """Retrieve precomputed cloud metadata; never run OCR or fetch image bytes."""

    if not bool(config.ocr_enabled):
        return None
    try:
        metadata = load_ocr_metadata_for_blob(blob)
    except Exception:
        if bool(config.batch_ocr_metadata_required):
            raise
        return None
    if metadata is not None:
        return metadata.document
    if bool(config.batch_ocr_metadata_required):
        name = str(getattr(blob, "name", "") or "")
        raise RuntimeError(
            f"Missing or stale cloud OCR metadata for {name!r}. "
            "Run `uv run invoke batch.ocr` before batch submission."
        )
    return None


def validate_ocr_metadata_for_blobs(blobs: Sequence[object]) -> int:
    """Fail before request generation when required cloud sidecars are unavailable."""

    if not bool(config.ocr_enabled) or not bool(config.batch_ocr_metadata_required):
        return 0
    if not blobs:
        return 0

    def unavailable_reason(blob: object) -> str | None:
        name = str(getattr(blob, "name", "") or "<unnamed>")
        try:
            metadata = load_ocr_metadata_for_blob(blob)
        except Exception as exc:  # noqa: BLE001 - summarized as a preflight failure
            return f"{name} ({exc})"
        return name if metadata is None else None

    max_workers = min(
        len(blobs),
        max(1, int(config.batch_ocr_workers or 1)),
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        unavailable = sorted(
            reason
            for reason in executor.map(unavailable_reason, blobs)
            if reason is not None
        )

    if unavailable:
        preview = ", ".join(unavailable[:5])
        remainder = len(unavailable) - min(len(unavailable), 5)
        suffix = f" and {remainder} more" if remainder else ""
        raise RuntimeError(
            f"Missing or stale cloud OCR metadata for {len(unavailable)}/"
            f"{len(blobs)} selected image(s): {preview}{suffix}. "
            "Run `uv run invoke batch.ocr` before batch submission."
        )
    return len(blobs)


def ocr_context_for_blob(blob: object) -> str:
    return render_ocr_context(ocr_document_for_blob(blob))
