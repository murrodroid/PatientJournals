"""Immutable, replay-safe dataset version publication.

The validation batch is the publication boundary for model-validation-enabled
jobs.  This module is deliberately outside the app package so the direct batch
CLI and the app use the same vNNN allocator and ledger.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from google.api_core.exceptions import NotFound, PreconditionFailed


DATASET_VERSION_LEDGER_FILE_NAME = "dataset_versions.json"
DATASET_VERSION_LEDGER_SCHEMA_VERSION = 1
DATASET_VERSIONS_DIR_NAME = "dataset_versions"
_VERSION_OBJECT_PATTERN = re.compile(
    r"(?:^|/)v(?P<number>[0-9]+)_model_validation(?:\.[^/]+)?$"
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_component(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in value.strip()
    ).strip("._")
    return cleaned or "extraction"


def _normalize_prefix(value: str) -> str:
    normalized = str(value or "").strip().strip("/")
    return f"{normalized}/" if normalized else ""


def publication_idempotency_key(
    *,
    source_run_id: str,
    verification_run_id: str,
    candidate_hash: str,
    verification_prompt_hash: str,
    dataset_sha256: str,
    publication_provenance_sha256: str,
) -> str:
    payload = {
        "source_run_id": source_run_id,
        "verification_run_id": verification_run_id,
        "candidate_hash": candidate_hash,
        "verification_prompt_hash": verification_prompt_hash,
        "dataset_sha256": dataset_sha256,
        "publication_provenance_sha256": publication_provenance_sha256,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def publication_provenance_sha256(
    metadata: Mapping[str, object] | None,
) -> str:
    """Hash every caller-supplied provenance field into replay identity."""

    encoded = json.dumps(
        dict(metadata or {}),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DatasetVersionPublication:
    version: int
    version_id: str
    local_path: str
    gcs_uri: str
    gcs_generation: str
    sha256: str
    size_bytes: int
    idempotency_key: str
    publication_provenance_sha256: str
    verification_run_id: str
    ledger_path: str
    ledger_gcs_uri: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _empty_ledger(source_run_id: str) -> dict[str, Any]:
    return {
        "schema_version": DATASET_VERSION_LEDGER_SCHEMA_VERSION,
        "source_run_id": source_run_id,
        "versions": [],
    }


def _parse_ledger(payload: object, *, source_run_id: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Dataset version ledger must be a JSON object.")
    if int(payload.get("schema_version") or 0) != DATASET_VERSION_LEDGER_SCHEMA_VERSION:
        raise ValueError("Unsupported dataset version ledger schema.")
    recorded_source = str(payload.get("source_run_id") or "")
    if recorded_source != source_run_id:
        raise ValueError(
            "Dataset version ledger source identity does not match the extraction run."
        )
    versions = payload.get("versions")
    if not isinstance(versions, list) or any(
        not isinstance(item, dict) for item in versions
    ):
        raise ValueError("Dataset version ledger versions must be objects.")
    numbers = [int(item.get("version") or 0) for item in versions]
    if any(number <= 0 for number in numbers) or len(numbers) != len(set(numbers)):
        raise ValueError("Dataset version ledger contains invalid version numbers.")
    return {**payload, "versions": list(versions)}


def _read_local_ledger(path: Path, *, source_run_id: str) -> dict[str, Any]:
    if not path.is_file():
        return _empty_ledger(source_run_id)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid dataset version ledger: {path}") from exc
    return _parse_ledger(payload, source_run_id=source_run_id)


def _not_found(exc: Exception) -> bool:
    return isinstance(exc, NotFound) or getattr(exc, "code", None) == 404


def _precondition_failed(exc: Exception) -> bool:
    return isinstance(exc, PreconditionFailed) or getattr(exc, "code", None) == 412


def _read_cloud_ledger(
    blob: object, *, source_run_id: str
) -> tuple[dict[str, Any] | None, int]:
    try:
        reload_blob = getattr(blob, "reload")
        reload_blob()
        generation = int(getattr(blob, "generation", 0) or 0)
        raw = getattr(blob, "download_as_bytes")(if_generation_match=generation or None)
    except Exception as exc:  # noqa: BLE001 - provider exceptions vary by SDK
        if _not_found(exc):
            return None, 0
        raise
    try:
        payload = json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Cloud dataset version ledger is invalid JSON.") from exc
    return _parse_ledger(payload, source_run_id=source_run_id), generation


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _record_to_publication(
    record: Mapping[str, object], ledger_path: Path
) -> DatasetVersionPublication:
    return DatasetVersionPublication(
        version=int(record["version"]),
        version_id=str(record["version_id"]),
        local_path=str(record["local_path"]),
        gcs_uri=str(record["gcs_uri"]),
        gcs_generation=str(record.get("gcs_generation") or ""),
        sha256=str(record["sha256"]),
        size_bytes=int(record["size_bytes"]),
        idempotency_key=str(record["idempotency_key"]),
        publication_provenance_sha256=str(
            record.get("publication_provenance_sha256")
            or publication_provenance_sha256(
                record.get("metadata")
                if isinstance(record.get("metadata"), dict)
                else {}
            )
        ),
        verification_run_id=str(record["verification_run_id"]),
        ledger_path=str(ledger_path),
        ledger_gcs_uri=str(record["ledger_gcs_uri"]),
    )


@dataclass(frozen=True)
class _CloudVersionObject:
    """Generation-bound snapshot of one immutable cloud dataset object."""

    version: int
    name: str
    generation: int
    metadata: dict[str, str]


def _cloud_version_number(object_name: str) -> int | None:
    match = _VERSION_OBJECT_PATTERN.search(object_name)
    number = int(match.group("number")) if match else 0
    return number if number > 0 else None


def _reload_cloud_version(
    *,
    bucket: object,
    object_name: str,
) -> _CloudVersionObject | None:
    version = _cloud_version_number(object_name)
    if version is None:
        return None
    blob = getattr(bucket, "blob")(object_name)
    try:
        getattr(blob, "reload")()
    except Exception as exc:  # noqa: BLE001 - provider exceptions vary by SDK
        if _not_found(exc):
            return None
        raise
    generation = int(getattr(blob, "generation", 0) or 0)
    if generation <= 0:
        raise RuntimeError(
            f"Immutable dataset object has no GCS generation: {object_name}"
        )
    raw_metadata = getattr(blob, "metadata", None)
    metadata = {
        str(key): str(value)
        for key, value in (
            raw_metadata.items() if isinstance(raw_metadata, dict) else ()
        )
        if value is not None
    }
    return _CloudVersionObject(
        version=version,
        name=object_name,
        generation=generation,
        metadata=metadata,
    )


def _list_cloud_versions(
    *,
    bucket: object,
    cloud_base: str,
) -> list[_CloudVersionObject]:
    """List all immutable version objects, including uncommitted orphans."""

    list_blobs = getattr(bucket, "list_blobs", None)
    if not callable(list_blobs):
        return []
    snapshots: list[_CloudVersionObject] = []
    for listed_blob in list_blobs(prefix=f"{cloud_base}/v"):
        object_name = str(getattr(listed_blob, "name", "") or "")
        snapshot = _reload_cloud_version(
            bucket=bucket,
            object_name=object_name,
        )
        if snapshot is not None:
            snapshots.append(snapshot)
    return snapshots


def _verified_cloud_bytes(
    *,
    bucket: object,
    snapshot: _CloudVersionObject,
    expected_sha256: str,
    expected_size: int,
) -> None:
    blob = getattr(bucket, "blob")(snapshot.name)
    raw = bytes(
        getattr(blob, "download_as_bytes")(if_generation_match=snapshot.generation)
    )
    if len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Immutable dataset object metadata matched this publication, but its "
            "generation-bound bytes did not."
        )


def _materialize_local_version(
    *,
    source_path: Path,
    version_path: Path,
    expected_sha256: str,
) -> None:
    """Create the local immutable mirror without overwriting divergent bytes."""

    if version_path.exists():
        if file_sha256(version_path) != expected_sha256:
            raise RuntimeError(
                f"Local dataset version has divergent immutable bytes: {version_path}"
            )
        return
    temporary = version_path.with_name(f".{version_path.name}.{os.getpid()}.tmp")
    shutil.copy2(source_path, temporary)
    temporary.replace(version_path)


def _ledger_with_record(
    ledger: Mapping[str, object],
    record: Mapping[str, object],
) -> dict[str, object]:
    versions = [
        *(item for item in ledger.get("versions", []) if isinstance(item, dict)),
        dict(record),
    ]
    versions.sort(key=lambda item: int(item.get("version") or 0))
    return {**ledger, "versions": versions}


def publish_dataset_version(
    *,
    dataset_path: str | Path,
    source_run_dir: str | Path,
    verification_run_dir: str | Path,
    bucket: object,
    datasets_prefix: str,
    candidate_hash: str,
    verification_prompt_hash: str,
    metadata: Mapping[str, object] | None = None,
    max_attempts: int = 8,
) -> DatasetVersionPublication:
    """Publish one immutable vNNN dataset and update the shared cloud ledger.

    Cloud generation preconditions serialize competing publishers.  Replaying
    the same verifier run returns its existing record without creating vNNN+1.
    """

    source_path = Path(dataset_path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"Verified dataset is missing: {source_path}")
    extraction_run = Path(source_run_dir).expanduser()
    verification_run = Path(verification_run_dir).expanduser()
    source_run_id = _safe_component(extraction_run.name)
    verification_run_id = _safe_component(verification_run.name)
    dataset_digest = file_sha256(source_path)
    dataset_size = source_path.stat().st_size
    publication_metadata = dict(metadata or {})
    provenance_digest = publication_provenance_sha256(publication_metadata)
    idempotency_key = publication_idempotency_key(
        source_run_id=source_run_id,
        verification_run_id=verification_run_id,
        candidate_hash=str(candidate_hash or ""),
        verification_prompt_hash=str(verification_prompt_hash or ""),
        dataset_sha256=dataset_digest,
        publication_provenance_sha256=provenance_digest,
    )

    versions_dir = extraction_run / DATASET_VERSIONS_DIR_NAME
    versions_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = extraction_run / DATASET_VERSION_LEDGER_FILE_NAME
    lock_path = extraction_run / f".{DATASET_VERSION_LEDGER_FILE_NAME}.lock"
    cloud_base = (
        f"{_normalize_prefix(datasets_prefix)}{source_run_id}/validation_versions"
    )
    ledger_object_name = f"{cloud_base}/{DATASET_VERSION_LEDGER_FILE_NAME}"
    ledger_gcs_uri = f"gs://{getattr(bucket, 'name')}/{ledger_object_name}"
    ledger_blob = getattr(bucket, "blob")(ledger_object_name)
    suffix = source_path.suffix or ".jsonl"
    known_cloud_objects: dict[str, _CloudVersionObject] = {}

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        for _attempt in range(max(1, int(max_attempts))):
            cloud_ledger, ledger_generation = _read_cloud_ledger(
                ledger_blob, source_run_id=source_run_id
            )
            ledger = cloud_ledger or _read_local_ledger(
                ledger_path, source_run_id=source_run_id
            )
            versions = list(ledger["versions"])
            same_verifier_run = next(
                (
                    item
                    for item in versions
                    if str(item.get("verification_run_id") or "") == verification_run_id
                ),
                None,
            )
            existing = None
            if same_verifier_run is not None:
                recorded_metadata = same_verifier_run.get("metadata")
                recorded_provenance = str(
                    same_verifier_run.get("publication_provenance_sha256")
                    or publication_provenance_sha256(
                        recorded_metadata if isinstance(recorded_metadata, dict) else {}
                    )
                )
                replay_matches = all(
                    (
                        str(same_verifier_run.get("sha256") or "") == dataset_digest,
                        str(same_verifier_run.get("candidate_hash") or "")
                        == str(candidate_hash or ""),
                        str(same_verifier_run.get("verification_prompt_hash") or "")
                        == str(verification_prompt_hash or ""),
                        recorded_provenance == provenance_digest,
                    )
                )
                if not replay_matches:
                    raise RuntimeError(
                        "The same verifier run produced different publication bytes "
                        "or provenance; refusing to allocate another dataset version."
                    )
                existing = same_verifier_run
            if existing is None:
                existing = next(
                    (
                        item
                        for item in versions
                        if str(item.get("idempotency_key") or "") == idempotency_key
                    ),
                    None,
                )
            if existing is not None:
                recorded_name = Path(str(existing.get("local_path") or "")).name
                if not recorded_name:
                    suffix = source_path.suffix or ".jsonl"
                    recorded_name = (
                        f"{str(existing.get('version_id') or '')}_"
                        f"model_validation{suffix}"
                    )
                # Local paths are machine-specific hints. Recover the immutable
                # cloud version beneath this worker's extraction run.
                existing_path = versions_dir / recorded_name
                if not existing_path.is_file():
                    existing_path.parent.mkdir(parents=True, exist_ok=True)
                    version_bucket_name = str(existing.get("gcs_bucket") or "")
                    version_object_name = str(existing.get("gcs_object") or "")
                    if (
                        version_bucket_name != str(getattr(bucket, "name"))
                        or not version_object_name
                    ):
                        raise RuntimeError(
                            "Published dataset version is missing locally and has no valid cloud binding."
                        )
                    getattr(bucket, "blob")(version_object_name).download_to_filename(
                        str(existing_path),
                        if_generation_match=int(existing.get("gcs_generation") or 0)
                        or None,
                    )
                if file_sha256(existing_path) != dataset_digest:
                    raise RuntimeError(
                        "Idempotent dataset version replay has different bytes."
                    )
                portable_existing = {**existing, "local_path": str(existing_path)}
                local_versions = [
                    portable_existing if item is existing else item for item in versions
                ]
                _atomic_write_json(ledger_path, {**ledger, "versions": local_versions})
                return _record_to_publication(portable_existing, ledger_path)

            # The ledger is authoritative for committed versions, but create-only
            # dataset objects are allocation reservations too.  Listing them lets a
            # fresh worker recover its own post-upload/pre-ledger-CAS orphan and
            # skip a different worker's orphan without overwriting immutable bytes.
            for snapshot in _list_cloud_versions(
                bucket=bucket,
                cloud_base=cloud_base,
            ):
                known_cloud_objects[snapshot.name] = snapshot
            cloud_versions = list(known_cloud_objects.values())
            versions_by_number: dict[int, list[_CloudVersionObject]] = {}
            for snapshot in cloud_versions:
                versions_by_number.setdefault(snapshot.version, []).append(snapshot)
            ledger_numbers = {int(item.get("version") or 0) for item in versions}

            exact_orphans: list[_CloudVersionObject] = []
            for snapshot in cloud_versions:
                snapshot_key = snapshot.metadata.get("publication_idempotency_key", "")
                snapshot_verifier = snapshot.metadata.get("verification_run_id", "")
                snapshot_source = snapshot.metadata.get("source_run_id", "")
                if snapshot_verifier == verification_run_id:
                    same_publication = all(
                        (
                            snapshot_source == source_run_id,
                            snapshot_key == idempotency_key,
                            snapshot.metadata.get("dataset_sha256", "")
                            == dataset_digest,
                            snapshot.metadata.get("publication_provenance_sha256", "")
                            == provenance_digest,
                        )
                    )
                    if not same_publication:
                        raise RuntimeError(
                            "The same verifier run has an immutable orphan with "
                            "different publication bytes or provenance; refusing "
                            "to allocate another dataset version."
                        )
                    if (
                        snapshot.version not in ledger_numbers
                        and len(versions_by_number[snapshot.version]) == 1
                    ):
                        exact_orphans.append(snapshot)
                elif snapshot_key == idempotency_key:
                    raise RuntimeError(
                        "Immutable dataset object has inconsistent publication "
                        "identity metadata."
                    )

            if exact_orphans:
                snapshot = min(exact_orphans, key=lambda item: item.version)
                _verified_cloud_bytes(
                    bucket=bucket,
                    snapshot=snapshot,
                    expected_sha256=dataset_digest,
                    expected_size=dataset_size,
                )
                version_id = f"v{snapshot.version:03d}"
                local_version_path = versions_dir / Path(snapshot.name).name
                _materialize_local_version(
                    source_path=source_path,
                    version_path=local_version_path,
                    expected_sha256=dataset_digest,
                )
                record: dict[str, object] = {
                    "version": snapshot.version,
                    "version_id": version_id,
                    "created_at": snapshot.metadata.get(
                        "publication_created_at",
                        datetime.now(timezone.utc).isoformat(),
                    ),
                    "local_path": str(local_version_path),
                    "gcs_uri": (f"gs://{getattr(bucket, 'name')}/{snapshot.name}"),
                    "gcs_bucket": str(getattr(bucket, "name")),
                    "gcs_object": snapshot.name,
                    "gcs_generation": str(snapshot.generation),
                    "sha256": dataset_digest,
                    "size_bytes": dataset_size,
                    "idempotency_key": idempotency_key,
                    "publication_provenance_sha256": provenance_digest,
                    "verification_run_id": verification_run_id,
                    "candidate_hash": str(candidate_hash or ""),
                    "verification_prompt_hash": str(verification_prompt_hash or ""),
                    "ledger_gcs_uri": ledger_gcs_uri,
                    "metadata": publication_metadata,
                }
                updated_ledger = _ledger_with_record(ledger, record)
                ledger_payload = json.dumps(
                    updated_ledger,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
                try:
                    ledger_blob.upload_from_string(
                        ledger_payload,
                        content_type="application/json",
                        if_generation_match=ledger_generation,
                    )
                except Exception as exc:  # noqa: BLE001 - provider exceptions vary
                    if _precondition_failed(exc):
                        continue
                    raise
                _atomic_write_json(ledger_path, updated_ledger)
                return _record_to_publication(record, ledger_path)

            occupied_numbers = ledger_numbers | set(versions_by_number)
            next_number = max(occupied_numbers, default=0) + 1
            version_id = f"v{next_number:03d}"
            local_version_path = versions_dir / f"{version_id}_model_validation{suffix}"
            cloud_object_name = f"{cloud_base}/{local_version_path.name}"
            cloud_blob = getattr(bucket, "blob")(cloud_object_name)
            publication_created_at = datetime.now(timezone.utc).isoformat()
            cloud_blob.metadata = {
                **{
                    str(key): str(value)
                    for key, value in publication_metadata.items()
                    if value is not None
                },
                "artifact_kind": "validated_dataset_version",
                "dataset_version": version_id,
                "dataset_sha256": dataset_digest,
                "publication_idempotency_key": idempotency_key,
                "publication_provenance_sha256": provenance_digest,
                "source_run_id": source_run_id,
                "verification_run_id": verification_run_id,
                "publication_created_at": publication_created_at,
            }
            try:
                cloud_blob.upload_from_filename(
                    str(source_path),
                    content_type=(
                        "application/jsonl"
                        if suffix.lower() == ".jsonl"
                        else "text/csv"
                    ),
                    if_generation_match=0,
                )
            except Exception as exc:  # noqa: BLE001 - provider exceptions vary
                if _precondition_failed(exc):
                    occupied = _reload_cloud_version(
                        bucket=bucket,
                        object_name=cloud_object_name,
                    )
                    if occupied is None:
                        continue
                    known_cloud_objects[occupied.name] = occupied
                    # Whether this is the same publication or another worker's,
                    # the next attempt reconciles it against a fresh ledger.
                    continue
                else:
                    raise

            snapshot = _reload_cloud_version(
                bucket=bucket,
                object_name=cloud_object_name,
            )
            if snapshot is None:
                raise RuntimeError(
                    f"Published dataset version disappeared: {cloud_object_name}"
                )
            known_cloud_objects[snapshot.name] = snapshot
            _verified_cloud_bytes(
                bucket=bucket,
                snapshot=snapshot,
                expected_sha256=dataset_digest,
                expected_size=dataset_size,
            )
            _materialize_local_version(
                source_path=source_path,
                version_path=local_version_path,
                expected_sha256=dataset_digest,
            )

            record = {
                "version": next_number,
                "version_id": version_id,
                "created_at": publication_created_at,
                "local_path": str(local_version_path),
                "gcs_uri": f"gs://{getattr(bucket, 'name')}/{cloud_object_name}",
                "gcs_bucket": str(getattr(bucket, "name")),
                "gcs_object": cloud_object_name,
                "gcs_generation": str(snapshot.generation),
                "sha256": dataset_digest,
                "size_bytes": dataset_size,
                "idempotency_key": idempotency_key,
                "publication_provenance_sha256": provenance_digest,
                "verification_run_id": verification_run_id,
                "candidate_hash": str(candidate_hash or ""),
                "verification_prompt_hash": str(verification_prompt_hash or ""),
                "ledger_gcs_uri": ledger_gcs_uri,
                "metadata": publication_metadata,
            }
            updated_ledger = _ledger_with_record(ledger, record)
            ledger_payload = json.dumps(
                updated_ledger, indent=2, ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
            try:
                ledger_blob.upload_from_string(
                    ledger_payload,
                    content_type="application/json",
                    if_generation_match=ledger_generation,
                )
            except Exception as exc:  # noqa: BLE001 - provider exceptions vary
                if _precondition_failed(exc):
                    continue
                raise
            _atomic_write_json(ledger_path, updated_ledger)
            return _record_to_publication(record, ledger_path)

    raise RuntimeError(
        "Could not allocate an immutable dataset version after concurrent updates."
    )
