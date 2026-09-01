"""Generation-bound image and optional OCR manifest for extraction/verification."""

from __future__ import annotations

import hashlib
import json
import mimetypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from patientjournals.batch.ocr_context import (
    CloudBlobIdentity,
    load_ocr_metadata_for_blob,
)
from patientjournals.config import config
from patientjournals.shared.ocr import OcrDocument


INPUT_IMAGE_MANIFEST_FILE_NAME = "input_image_manifest.jsonl"
INPUT_IMAGE_MANIFEST_META_FILE_NAME = "input_image_manifest.meta.json"
EXTRACTION_IMAGE_BINDINGS_FILE_NAME = "extraction_image_bindings.jsonl"
EXTRACTION_IMAGE_BINDINGS_META_FILE_NAME = "extraction_image_bindings.meta.json"


class InputImageManifestRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    mime_type: str
    image_source: dict[str, object]
    ocr_enabled: bool = True
    ocr_sidecar_name: str = ""
    ocr_sidecar_source: dict[str, object] = Field(default_factory=dict)
    ocr_sidecar_sha256: str = ""
    ocr_image_sha256: str = ""
    ocr_document_sha256: str = ""
    ocr_backend: str = ""
    ocr_line_count: int = 0

    @field_validator(
        "key",
        "mime_type",
    )
    @classmethod
    def required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Input image manifest text fields must not be empty.")
        return normalized

    @model_validator(mode="after")
    def validate_optional_ocr_binding(self) -> "InputImageManifestRecord":
        if self.source.name != self.key:
            raise ValueError(
                "Input manifest key must equal the generation-bound image object name."
            )
        if self.ocr_line_count < 0:
            raise ValueError("OCR line count must not be negative.")
        if not self.ocr_enabled:
            return self
        required = {
            "ocr_sidecar_name": self.ocr_sidecar_name,
            "ocr_sidecar_sha256": self.ocr_sidecar_sha256,
            "ocr_image_sha256": self.ocr_image_sha256,
            "ocr_document_sha256": self.ocr_document_sha256,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing or not self.ocr_sidecar_source:
            missing_fields = [
                *missing,
                *(["ocr_sidecar_source"] if not self.ocr_sidecar_source else []),
            ]
            joined = ", ".join(missing_fields)
            raise ValueError(f"OCR-enabled input manifest is missing: {joined}.")
        return self

    @property
    def source(self) -> CloudBlobIdentity:
        return CloudBlobIdentity.from_dict(self.image_source)

    @property
    def sidecar_source(self) -> CloudBlobIdentity:
        return CloudBlobIdentity.from_dict(self.ocr_sidecar_source)


class ExtractionImageBinding(BaseModel):
    """Exact image identity consumed by the asynchronous extraction request."""

    model_config = ConfigDict(extra="forbid")

    key: str
    provider: Literal["gemini", "anthropic"]
    reference_mode: Literal[
        "immutable_staged_uri", "generation_qualified_signed_url"
    ]
    source_image: dict[str, object]
    request_image: dict[str, object]
    request_uri: str

    @property
    def source(self) -> CloudBlobIdentity:
        return CloudBlobIdentity.from_dict(self.source_image)

    @property
    def request_source(self) -> CloudBlobIdentity:
        return CloudBlobIdentity.from_dict(self.request_image)


def ocr_document_sha256(document: OcrDocument) -> str:
    payload = json.dumps(
        document.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mime_type(blob: object) -> str:
    configured = str(getattr(blob, "content_type", "") or "").strip()
    if configured:
        return configured
    guessed, _ = mimetypes.guess_type(str(getattr(blob, "name", "") or ""))
    return guessed or "application/octet-stream"


def _source_identity(blob: object) -> CloudBlobIdentity:
    identity = CloudBlobIdentity.from_blob(blob)
    if not identity.generation:
        reload_blob = getattr(blob, "reload", None)
        if callable(reload_blob):
            reload_blob()
            identity = CloudBlobIdentity.from_blob(blob)
    if not identity.name or not identity.bucket or not identity.generation:
        raise RuntimeError("Input image has no complete generation-bound GCS identity.")
    return identity


def input_manifest_record_for_blob(
    blob: object,
    *,
    include_ocr: bool | None = None,
) -> InputImageManifestRecord:
    use_ocr = bool(config.ocr_enabled) if include_ocr is None else bool(include_ocr)
    source = _source_identity(blob)
    if not use_ocr:
        return InputImageManifestRecord(
            key=source.name,
            mime_type=_mime_type(blob),
            image_source=source.to_dict(),
            ocr_enabled=False,
        )

    metadata = load_ocr_metadata_for_blob(blob)
    name = str(getattr(blob, "name", "") or "")
    if metadata is None:
        raise RuntimeError(
            f"Missing or stale generation-bound OCR metadata for {name!r}."
        )
    suffix = str(config.ocr_sidecar_suffix or ".ocr.json")
    sidecar_name = f"{metadata.source.name}{suffix}"
    bucket = getattr(blob, "bucket", None)
    blob_factory = getattr(bucket, "blob", None)
    if not callable(blob_factory):
        raise TypeError(f"GCS blob {name!r} has no bucket binding.")
    sidecar_blob = blob_factory(sidecar_name)
    reload_sidecar = getattr(sidecar_blob, "reload", None)
    if callable(reload_sidecar):
        reload_sidecar()
    sidecar_source = CloudBlobIdentity.from_blob(sidecar_blob)
    if not sidecar_source.generation:
        raise RuntimeError(f"OCR sidecar generation is unavailable: {sidecar_name}")
    download_sidecar = getattr(sidecar_blob, "download_as_bytes", None)
    if not callable(download_sidecar):
        raise TypeError(f"OCR sidecar cannot be downloaded: {sidecar_name}")
    sidecar_bytes = download_sidecar(
        if_generation_match=int(sidecar_source.generation)
    )
    if not isinstance(sidecar_bytes, bytes):
        sidecar_bytes = bytes(sidecar_bytes)
    return InputImageManifestRecord(
        key=metadata.source.name,
        mime_type=_mime_type(blob),
        image_source=metadata.source.to_dict(),
        ocr_enabled=True,
        ocr_sidecar_name=sidecar_name,
        ocr_sidecar_source=sidecar_source.to_dict(),
        ocr_sidecar_sha256=hashlib.sha256(sidecar_bytes).hexdigest(),
        ocr_image_sha256=metadata.document.image_sha256,
        ocr_document_sha256=ocr_document_sha256(metadata.document),
        ocr_backend=metadata.document.backend,
        ocr_line_count=len(metadata.document.lines),
    )


def write_input_image_manifest(
    blobs: Sequence[object],
    path: str | Path,
    *,
    workers: int | None = None,
    include_ocr: bool | None = None,
) -> tuple[Path, tuple[InputImageManifestRecord, ...]]:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    max_workers = min(
        max(1, len(blobs)),
        max(1, int(workers or config.batch_ocr_workers or 1)),
    )
    use_ocr = bool(config.ocr_enabled) if include_ocr is None else bool(include_ocr)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        records = tuple(
            executor.map(
                lambda blob: input_manifest_record_for_blob(
                    blob, include_ocr=use_ocr
                ),
                blobs,
            )
        )

    keys = [record.key for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Input image manifest contains duplicate page keys.")
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record.model_dump(mode="json"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")
    return destination, records


def _normalized_prefix(value: str) -> str:
    normalized = str(value or "").strip().strip("/")
    return f"{normalized}/" if normalized else ""


def _identity_matches(expected: CloudBlobIdentity, actual: CloudBlobIdentity) -> bool:
    return all(
        expected_value == actual_value
        for expected_value, actual_value in (
            (expected.bucket, actual.bucket),
            (expected.name, actual.name),
            (str(expected.generation), str(actual.generation)),
            (expected.size, actual.size),
            (expected.crc32c, actual.crc32c),
            (expected.md5_hash, actual.md5_hash),
        )
        if expected_value not in {None, ""}
    )


def _binding_for_record(
    *,
    bucket: object,
    record: InputImageManifestRecord,
    provider: Literal["gemini", "anthropic"],
    run_dir_name: str,
) -> ExtractionImageBinding:
    source = record.source
    if provider == "anthropic":
        return ExtractionImageBinding(
            key=record.key,
            provider=provider,
            reference_mode="generation_qualified_signed_url",
            source_image=source.to_dict(),
            request_image=source.to_dict(),
            request_uri=(
                f"gs://{source.bucket}/{source.name}?generation={source.generation}"
            ),
        )

    source_blob = getattr(bucket, "blob")(
        source.name, generation=int(source.generation)
    )
    suffix = Path(source.name).suffix.lower()
    identity_digest = hashlib.sha256(
        (
            f"{source.bucket}/{source.name}#{source.generation}:"
            f"{source.crc32c or ''}:{source.md5_hash or ''}"
        ).encode("utf-8")
    ).hexdigest()[:40]
    object_name = (
        f"{_normalized_prefix(config.batch_requests_gcs_prefix)}"
        f"{run_dir_name}/extraction_images/{identity_digest}{suffix}"
    )
    staged_blob = getattr(bucket, "copy_blob")(
        source_blob,
        bucket,
        new_name=object_name,
        preserve_acl=False,
        source_generation=int(source.generation),
        if_source_generation_match=int(source.generation),
        if_generation_match=0,
    )
    reload_blob = getattr(staged_blob, "reload", None)
    if callable(reload_blob):
        reload_blob()
    staged = CloudBlobIdentity.from_blob(staged_blob)
    content_matches = all(
        expected == actual
        for expected, actual in (
            (source.size, staged.size),
            (source.crc32c, staged.crc32c),
            (source.md5_hash, staged.md5_hash),
        )
        if expected not in {None, ""}
    )
    if (
        not staged.generation
        or staged.bucket != source.bucket
        or staged.name != object_name
        or not content_matches
    ):
        raise RuntimeError(
            f"Staged extraction image bytes do not match source generation: {record.key}"
        )
    return ExtractionImageBinding(
        key=record.key,
        provider=provider,
        reference_mode="immutable_staged_uri",
        source_image=source.to_dict(),
        request_image=staged.to_dict(),
        request_uri=f"gs://{staged.bucket}/{staged.name}",
    )


def write_extraction_image_bindings(
    *,
    bucket: object,
    records: Sequence[InputImageManifestRecord],
    provider: Literal["gemini", "anthropic"],
    run_dir_name: str,
    path: str | Path,
    workers: int | None = None,
) -> tuple[Path, tuple[ExtractionImageBinding, ...]]:
    """Pin every first-pass provider request to the manifest's exact bytes."""

    max_workers = min(
        max(1, len(records)),
        max(1, int(workers or config.batch_ocr_workers or 1)),
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        bindings = tuple(
            executor.map(
                lambda record: _binding_for_record(
                    bucket=bucket,
                    record=record,
                    provider=provider,
                    run_dir_name=run_dir_name,
                ),
                records,
            )
        )
    keys = [binding.key for binding in bindings]
    if len(keys) != len(set(keys)):
        raise ValueError("Extraction image bindings contain duplicate page keys.")
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for binding in bindings:
            handle.write(
                json.dumps(
                    binding.model_dump(mode="json"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")
    return destination, bindings


def read_extraction_image_bindings(
    path: str | Path,
) -> tuple[ExtractionImageBinding, ...]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Extraction image bindings not found: {source}")
    records: list[ExtractionImageBinding] = []
    keys: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                binding = ExtractionImageBinding.model_validate_json(raw)
            except Exception as exc:
                raise ValueError(
                    f"Invalid extraction image binding at {source}:{line_number}: {exc}"
                ) from exc
            if binding.key in keys:
                raise ValueError(
                    f"Duplicate extraction image binding for {binding.key!r}."
                )
            keys.add(binding.key)
            records.append(binding)
    if not records:
        raise ValueError(f"Extraction image binding artifact is empty: {source}")
    return tuple(records)


def verify_extraction_image_bindings(
    *,
    bucket: object,
    bindings: Sequence[ExtractionImageBinding],
    expected_keys: Sequence[str] | None = None,
    workers: int | None = None,
) -> None:
    """Fail closed if the bytes addressable by a first-pass request drifted."""

    keys = {binding.key for binding in bindings}
    if expected_keys is not None and keys != set(expected_keys):
        raise RuntimeError(
            "Extraction image bindings do not exactly cover submitted page keys."
        )
    def verify_one(binding: ExtractionImageBinding) -> None:
        expected = binding.request_source
        if binding.reference_mode == "generation_qualified_signed_url":
            blob = getattr(bucket, "blob")(
                expected.name, generation=int(expected.generation)
            )
        else:
            # Gemini consumes an unqualified URI to a write-once staged object;
            # inspect the current generation to detect delete/recreate drift.
            blob = getattr(bucket, "blob")(expected.name)
        reload_blob = getattr(blob, "reload", None)
        if callable(reload_blob):
            reload_blob()
        actual = CloudBlobIdentity.from_blob(blob)
        if not _identity_matches(expected, actual):
            raise RuntimeError(
                "First-pass extraction image changed after request construction: "
                f"{binding.key}"
            )

    max_workers = min(
        max(1, len(bindings)),
        max(1, int(workers or config.batch_ocr_workers or 1)),
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tuple(executor.map(verify_one, bindings))


def bindings_by_key(
    bindings: Sequence[ExtractionImageBinding],
) -> Mapping[str, ExtractionImageBinding]:
    return {binding.key: binding for binding in bindings}


def read_input_image_manifest(
    path: str | Path,
) -> tuple[InputImageManifestRecord, ...]:
    source = Path(path).expanduser()
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Input image manifest not found: {source}")
    records: list[InputImageManifestRecord] = []
    keys: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = InputImageManifestRecord.model_validate_json(raw)
            except Exception as exc:
                raise ValueError(
                    f"Invalid input image manifest row at {source}:{line_number}: {exc}"
                ) from exc
            if record.key in keys:
                raise ValueError(
                    f"Duplicate input image key {record.key!r} at "
                    f"{source}:{line_number}."
                )
            keys.add(record.key)
            records.append(record)
    if not records:
        raise ValueError(f"Input image manifest is empty: {source}")
    return tuple(records)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_input_manifest_metadata(
    *,
    run_dir: Path,
    manifest_path: Path,
    gcs_uri: str,
    record_count: int,
    ocr_enabled: bool,
) -> Path:
    path = run_dir / INPUT_IMAGE_MANIFEST_META_FILE_NAME
    path.write_text(
        json.dumps(
            {
                "path": manifest_path.name,
                "gcs_uri": gcs_uri,
                "sha256": file_sha256(manifest_path),
                "record_count": int(record_count),
                "ocr_enabled": bool(ocr_enabled),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path
