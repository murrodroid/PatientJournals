"""Batch-first, candidate-aware second-pass model validation.

Submission is one request per complete page candidate. Retrieval always writes
sparse verdict/patch artifacts and publishes an immutable dataset version only
when the complete apply-patches gate succeeds.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import mimetypes
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import storage
from google.genai import types

from patientjournals.batch.client import get_batch_client, resolve_service_account_path
from patientjournals.batch.ocr_context import (
    CloudBlobIdentity,
    CloudOcrMetadata,
    load_ocr_metadata_for_blob,
)
from patientjournals.batch.retrieve import (
    _await_completion,
    _batch_job_state,
    _batch_job_successful,
    _download_from_anthropic_output,
    _download_from_mldev_output,
    _download_from_vertex_gcs_output,
    _extract_anthropic_response_metadata,
    _extract_location_from_batch_name,
    _gemini_output_reference,
    _get_batch_job,
)
from patientjournals.batch.submit import _get_anthropic_client
from patientjournals.batch.submit_requests import (
    _anthropic_custom_id_for_key,
    _anthropic_signed_url_expiration,
    _anthropic_strict_json_schema,
    _output_dest_gcs_uri,
    _upload_requests_to_gcs,
    _vertex_compatible_schema,
)
from patientjournals.config import config
from patientjournals.config.models import resolve_model_spec
from patientjournals.config.prompts import (
    MODEL_VALIDATION_CANDIDATE_HEADING,
    MODEL_VALIDATION_INSTRUCTIONS,
    MODEL_VALIDATION_OCR_HEADING,
    MODEL_VALIDATION_PROMPT_VERSION,
    MODEL_VALIDATION_SCHEMA_HEADING,
    build_model_validation_prompt,
)
from patientjournals.config.schemas import (
    PageModelValidation,
    model_from_json_schema,
    resolve_output_schema,
)
from patientjournals.shared.ocr import render_ocr_context
from patientjournals.shared.output_handler import data_to_rows
from patientjournals.shared.response_parsing import extract_response_metadata
from patientjournals.shared.tools import flush_rows, get_run_logger
from patientjournals.validation.candidates import (
    PAGE_CANDIDATES_FILE_NAME,
    PageCandidateRecord,
    PageCandidateWriter,
    candidate_sha256,
    read_page_candidates,
    source_run_id_from_path,
    write_page_candidates,
)
from patientjournals.validation.input_manifest import (
    INPUT_IMAGE_MANIFEST_FILE_NAME,
    InputImageManifestRecord,
    file_sha256,
    ocr_document_sha256,
    read_input_image_manifest,
)
from patientjournals.validation.publication import publish_dataset_version
from patientjournals.validation.routing import (
    DETERMINISTIC_ROUTING_POLICY_VERSION,
    METADATA_CANDIDATE_SHA256_KEY,
    METADATA_CONTROL_SAMPLE_KEY,
    METADATA_CONTROL_SAMPLE_SHA256_KEY,
    METADATA_METRICS_KEY,
    METADATA_POLICY_VERSION_KEY,
    METADATA_ROUTE_KEY,
    METADATA_RULE_IDS_KEY,
    METADATA_SCHEMA_VALID_KEY,
    METADATA_STATUS_KEY,
    METADATA_THRESHOLDS_KEY,
    RoutingThresholds,
    route_candidates,
    write_routing_decisions,
)


VERIFICATION_RUNS_DIR_NAME = "verifications"
VERIFICATION_BATCH_JOB_FILE_NAME = "batch_job.json"
VERIFICATION_BINDINGS_FILE_NAME = "validation_bindings.jsonl"
VERIFICATION_SCHEMA_FILE_NAME = "extraction_schema.json"
VERIFICATION_REQUEST_CONTRACT_FILE_NAME = "validation_request_contract.json"
VERIFICATION_RESULTS_FILE_NAME = "validation_results.jsonl"
VERIFICATION_FAILURES_FILE_NAME = "validation_failures.jsonl"
VERIFICATION_SUMMARY_FILE_NAME = "validation_summary.json"
VERIFICATION_PATCHED_CANDIDATES_FILE_NAME = "patched_candidates.jsonl"
SOURCE_PAGE_CANDIDATES_FILE_NAME = "source_page_candidates.jsonl"
FINAL_VALIDATION_POLICY_FILE_NAME = "final_validation_policy.json"
FINAL_VALIDATION_POLICY_SCHEMA_VERSION = 4
FINAL_VALIDATION_POLICY_ANCHOR_PREFIX = (
    "_patientjournals/model_validation_policies"
)
FIELD_CORRECTION_METADATA_FILE_NAME = "field_corrections.json"
FIELD_CORRECTION_METADATA_SCHEMA_VERSION = 2
FINAL_VALIDATION_APPLY_MODE = "apply_patches"
AUTOMATIC_CORRECTION_ACCEPTANCE_POLICY = (
    "automatic_schema_validated_verifier_corrections_v1"
)
ANTHROPIC_MAX_BATCH_REQUEST_BYTES = 250_000_000
ANTHROPIC_MAX_BATCH_REQUESTS = 100_000
GEMINI_MAX_BATCH_REQUEST_BYTES = 1_900_000_000
MLDEV_REGISTER_FILES_BATCH_SIZE = 100

ThinkingLevel = Literal["low", "medium", "high"]
VerificationScope = Literal["all", "flagged"]
VerificationApplyMode = Literal["report_only", "apply_patches"]


@dataclass(frozen=True)
class ModelValidationSubmitRequest:
    source_run_dir: str | None = None
    candidate_file: str | None = None
    input_manifest_file: str | None = None
    model: str | None = None
    thinking_level: ThinkingLevel | None = None
    scope: VerificationScope | None = None
    apply_mode: VerificationApplyMode | None = None
    max_output_tokens: int | None = None
    num_chunks: int | None = None


@dataclass(frozen=True)
class ModelValidationSubmitResult:
    run_dir: Path
    provider: str
    model: str
    candidate_count: int
    source_candidate_count: int
    batch_job_names: tuple[str, ...]
    requests_paths: tuple[Path, ...]
    candidates_path: Path
    bindings_path: Path
    input_manifest_path: Path


@dataclass(frozen=True)
class ModelValidationRetrieveRequest:
    run_dir: str
    wait: bool = False
    allow_partial: bool = False


@dataclass(frozen=True)
class ModelValidationRetrieveResult:
    run_dir: Path
    provider: str
    model: str
    results_path: Path
    failures_path: Path
    summary_path: Path
    field_corrections_path: Path
    field_corrections_gcs_uri: str
    field_corrections_gcs_generation: str
    field_corrections_sha256: str
    correction_acceptance_policy: str
    accepted_correction_fields: int
    corrected_fields: int
    patched_candidates_path: Path | None
    dataset_path: Path | None
    dataset_gcs_uri: str
    dataset_gcs_generation: str
    dataset_sha256: str
    dataset_publication_idempotency_key: str
    publication_provenance_sha256: str
    dataset_version: int | None
    dataset_version_id: str
    dataset_version_path: str
    dataset_version_ledger_path: str
    dataset_version_ledger_gcs_uri: str
    dataset_rows: int
    rows_written: int
    expected_pages: int
    completed_pages: int
    successful_pages: int
    model_reviewed_pages: int
    deterministically_cleared_pages: int
    missing_pages: int
    confirmed_pages: int
    needs_correction_pages: int
    unverifiable_pages: int
    failed_pages: int
    success: bool
    publishable: bool
    status: str
    candidate_hash: str
    verification_prompt_hash: str
    artifact_gcs_uris: Mapping[str, str]


@dataclass(frozen=True)
class _PreparedValidationPage:
    record: PageCandidateRecord
    input_record: InputImageManifestRecord
    candidate_digest: str
    mime_type: str
    prompt: str
    provider_image_reference: str
    request_image_source: CloudBlobIdentity
    binding: dict[str, object]


@dataclass(frozen=True)
class _ExtractionSchemaSnapshot:
    schema: dict[str, Any]
    name: str
    version_id: str
    sha256: str


@dataclass(frozen=True)
class _FinalValidationPolicySnapshot:
    apply_mode: VerificationApplyMode
    acceptance_policy: str
    verification_scope: VerificationScope
    source_run_id: str
    datasets_gcs_prefix: str
    validations_gcs_prefix: str
    source_candidates_sha256: str = ""
    selected_candidates_sha256: str = ""
    source_candidate_count: int = 0
    selected_candidate_count: int = 0
    deterministic_routing_policy: str = ""
    deterministic_routing_sha256: str = ""
    deterministic_routing_gcs_uri: str = ""
    deterministic_routing_gcs_generation: str = ""
    verification_request_contract_sha256: str = ""
    verification_request_contract_gcs_uri: str = ""
    verification_request_contract_gcs_generation: str = ""
    artifact_sha256: str = ""
    artifact_gcs_uri: str = ""
    artifact_gcs_generation: str = ""
    legacy: bool = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit or retrieve candidate-aware model validation batches."
    )
    parser.add_argument("--retrieve", action="store_true")
    parser.add_argument("--source-run-dir", "--extraction-run-dir")
    parser.add_argument("--candidate-file")
    parser.add_argument("--input-manifest-file")
    parser.add_argument("--run-dir")
    parser.add_argument("--model")
    parser.add_argument("--thinking-level", choices=("low", "medium", "high"))
    parser.add_argument("--scope", choices=("all", "flagged"))
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--num-chunks", "--num-batches", type=int)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def _storage_client_and_bucket() -> tuple[storage.Client, storage.Bucket]:
    bucket_name = str(config.gcs_bucket_name or "").strip()
    if not bucket_name:
        raise ValueError("config.gcs_bucket_name is empty.")
    service_account_file = str(config.service_account_file or "").strip()
    if not service_account_file:
        raise ValueError("config.service_account_file is empty.")
    service_account_path = resolve_service_account_path(service_account_file)
    client = storage.Client.from_service_account_json(str(service_account_path))
    return client, client.bucket(bucket_name)


def _create_verification_run_dir() -> Path:
    parent = Path(config.output_root).expanduser() / VERIFICATION_RUNS_DIR_NAME
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for _ in range(100):
        candidate = parent / f"{timestamp}_{uuid.uuid4().hex}"
        try:
            # Atomic creation avoids the exists-then-create race between local
            # workers; the UUID component makes collisions globally negligible.
            candidate.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("Could not allocate a unique verification run directory.")


def _latest_verification_run_dir() -> Path:
    parent = Path(config.output_root).expanduser() / VERIFICATION_RUNS_DIR_NAME
    if not parent.is_dir():
        raise FileNotFoundError("No model-validation run directory exists.")
    candidates = [
        path
        for path in parent.iterdir()
        if path.is_dir() and (path / VERIFICATION_BATCH_JOB_FILE_NAME).is_file()
    ]
    if not candidates:
        raise FileNotFoundError("No submitted model-validation run exists.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    normalized = uri.strip()
    if not normalized.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {uri!r}")
    remainder = normalized[5:]
    if "/" not in remainder:
        return remainder, ""
    return tuple(remainder.split("/", 1))  # type: ignore[return-value]


def _download_gcs_file(
    storage_client: storage.Client,
    uri: str,
    destination: Path,
    *,
    generation: str | int | None = None,
) -> Path:
    bucket_name, object_name = _parse_gcs_uri(uri)
    if not object_name:
        raise ValueError(f"GCS artifact URI has no object name: {uri}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    generation_number = int(generation) if generation not in {None, ""} else None
    bucket = storage_client.bucket(bucket_name)
    blob = (
        bucket.blob(object_name, generation=generation_number)
        if generation_number is not None
        else bucket.blob(object_name)
    )
    download_kwargs = (
        {"if_generation_match": generation_number}
        if generation_number is not None
        else {}
    )
    blob.download_to_filename(str(destination), **download_kwargs)
    return destination


def _copy_or_download_candidate_artifact(
    *,
    request: ModelValidationSubmitRequest,
    run_dir: Path,
    storage_client: storage.Client,
) -> Path:
    requested = str(request.candidate_file or "").strip()
    expected_sha256 = ""
    expected_generation = ""
    results_payload: dict[str, Any] = {}
    if not requested:
        source_run = str(request.source_run_dir or "").strip()
        if not source_run:
            raise ValueError(
                "Model validation requires source_run_dir or candidate_file."
            )
        source_run_path = Path(source_run).expanduser()
        results_path = source_run_path / "batch_results.json"
        if results_path.is_file():
            try:
                loaded = json.loads(results_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                loaded = None
            if isinstance(loaded, dict):
                results_payload = loaded
                expected_sha256 = str(
                    loaded.get("page_candidates_sha256") or ""
                ).strip()
                expected_generation = str(
                    loaded.get("page_candidates_gcs_generation") or ""
                ).strip()
        direct_candidate = source_run_path / PAGE_CANDIDATES_FILE_NAME
        if direct_candidate.is_file():
            requested = str(direct_candidate)
        else:
            local_result = str(
                results_payload.get("page_candidates_path") or ""
            ).strip()
            cloud_result = str(
                results_payload.get("page_candidates_gcs_uri") or ""
            ).strip()
            if local_result and Path(local_result).expanduser().is_file():
                requested = local_result
            elif cloud_result:
                requested = cloud_result
            else:
                requested = local_result or str(direct_candidate)

    destination = run_dir / PAGE_CANDIDATES_FILE_NAME
    if requested.startswith("gs://"):
        if expected_generation:
            bucket_name, object_name = _parse_gcs_uri(requested)
            blob = storage_client.bucket(bucket_name).blob(
                object_name, generation=int(expected_generation)
            )
            blob.download_to_filename(
                str(destination), if_generation_match=int(expected_generation)
            )
        else:
            _download_gcs_file(storage_client, requested, destination)
        if expected_sha256 and file_sha256(destination) != expected_sha256:
            raise RuntimeError(
                "Downloaded page candidates do not match retrieval SHA-256."
            )
        return destination
    source = Path(requested).expanduser()
    if not source.is_file():
        raise FileNotFoundError(
            f"Canonical page candidate artifact not found: {source}. "
            "Run batch.retrieve successfully first."
        )
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    if expected_sha256 and file_sha256(destination) != expected_sha256:
        raise RuntimeError(
            "Local page candidates do not match retrieval SHA-256."
        )
    return destination


def _scope_candidates(
    records: Sequence[PageCandidateRecord],
    scope: VerificationScope,
) -> tuple[PageCandidateRecord, ...]:
    if scope == "all":
        return tuple(records)
    confirmed_labels = {"confirmed", "valid", "passed", "success"}
    selected = [
        record
        for record in records
        if str(record.extraction_metadata.get("deterministic_status") or "")
        .strip()
        .lower()
        not in confirmed_labels
    ]
    return tuple(selected)


_ROUTING_METADATA_KEYS = (
    METADATA_STATUS_KEY,
    METADATA_ROUTE_KEY,
    METADATA_POLICY_VERSION_KEY,
    METADATA_RULE_IDS_KEY,
    METADATA_METRICS_KEY,
    METADATA_THRESHOLDS_KEY,
    METADATA_CONTROL_SAMPLE_KEY,
    METADATA_CONTROL_SAMPLE_SHA256_KEY,
    METADATA_CANDIDATE_SHA256_KEY,
    METADATA_SCHEMA_VALID_KEY,
)


def _validate_and_snapshot_deterministic_routing(
    *,
    records: Sequence[PageCandidateRecord],
    extraction_model: type,
    destination: Path,
) -> tuple[str, str]:
    """Recompute every route and bind the exact decision set for validation."""

    if not records:
        raise ValueError("Deterministic routing requires at least one candidate.")
    threshold_payloads = {
        json.dumps(
            record.extraction_metadata.get(METADATA_THRESHOLDS_KEY),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        for record in records
    }
    if len(threshold_payloads) != 1:
        raise RuntimeError(
            "Candidates do not share one deterministic routing threshold snapshot."
        )
    raw_thresholds = records[0].extraction_metadata.get(METADATA_THRESHOLDS_KEY)
    thresholds = RoutingThresholds.model_validate(raw_thresholds)
    policy_versions = {
        str(record.extraction_metadata.get(METADATA_POLICY_VERSION_KEY) or "")
        for record in records
    }
    if policy_versions != {DETERMINISTIC_ROUTING_POLICY_VERSION}:
        raise RuntimeError(
            "Candidates do not carry the supported deterministic routing policy."
        )

    routed = tuple(
        route_candidates(
            records,
            full_model=extraction_model,
            max_collection_items=thresholds.max_collection_items,
            max_populated_leaves=thresholds.max_populated_leaves,
            max_text_chars=thresholds.max_text_chars,
            control_sample_percent=thresholds.control_sample_percent,
            control_sample_seed=thresholds.control_sample_seed,
            policy_version=DETERMINISTIC_ROUTING_POLICY_VERSION,
        )
    )
    for source, recomputed in zip(records, routed, strict=True):
        for key in _ROUTING_METADATA_KEYS:
            if source.extraction_metadata.get(key) != recomputed.candidate.extraction_metadata.get(
                key
            ):
                raise RuntimeError(
                    "Candidate deterministic routing metadata does not match "
                    f"recomputed policy for {source.key!r} ({key})."
                )
    write_routing_decisions(destination, (item.decision for item in routed))
    return DETERMINISTIC_ROUTING_POLICY_VERSION, file_sha256(destination)


def _unique_metadata_value(
    records: Sequence[PageCandidateRecord], key: str
) -> str | None:
    values = {
        str(record.extraction_metadata.get(key) or "").strip()
        for record in records
        if str(record.extraction_metadata.get(key) or "").strip()
    }
    if len(values) > 1:
        raise ValueError(f"Candidates reference multiple {key} values: {sorted(values)}")
    return next(iter(values), None)


def _candidate_source_run_id(record: PageCandidateRecord) -> str:
    metadata = record.extraction_metadata
    explicit = str(metadata.get("source_run_id") or "").strip()
    legacy_path = str(metadata.get("source_run_dir") or "").strip()
    legacy = source_run_id_from_path(legacy_path)
    if explicit and source_run_id_from_path(explicit) != explicit:
        raise RuntimeError(
            f"Candidate {record.key!r} has a non-portable source_run_id: {explicit!r}."
        )
    if legacy_path and not legacy:
        raise RuntimeError(
            f"Candidate {record.key!r} has an invalid source_run_dir provenance."
        )
    if explicit and legacy and explicit != legacy:
        raise RuntimeError(
            f"Candidate {record.key!r} has conflicting source-run provenance: "
            f"source_run_id={explicit!r}, source_run_dir basename={legacy!r}."
        )
    resolved = explicit or legacy
    if not resolved:
        raise RuntimeError(
            f"Candidate {record.key!r} has no source_run_id or legacy "
            "source_run_dir provenance."
        )
    return resolved


def _validate_candidate_source_run(
    *,
    records: Sequence[PageCandidateRecord],
    source_run_dir: str,
) -> str:
    """Bind all candidates to the supplied extraction run before submission."""

    supplied_source_run_id = source_run_id_from_path(source_run_dir)
    if not supplied_source_run_id:
        raise ValueError("Model validation source_run_dir has no portable run id.")
    candidate_source_run_ids = {
        _candidate_source_run_id(record) for record in records
    }
    if len(candidate_source_run_ids) != 1:
        raise RuntimeError(
            "Validation candidates reference multiple extraction source runs: "
            f"{sorted(candidate_source_run_ids)}."
        )
    candidate_source_run_id = next(iter(candidate_source_run_ids))
    if candidate_source_run_id != supplied_source_run_id:
        raise RuntimeError(
            "Supplied source_run_dir does not match candidate provenance: "
            f"supplied={supplied_source_run_id!r}, "
            f"candidate={candidate_source_run_id!r}."
        )
    return candidate_source_run_id


def _metadata_source_run_id(metadata: Mapping[str, Any]) -> str:
    explicit = str(metadata.get("source_run_id") or "").strip()
    legacy = source_run_id_from_path(
        str(metadata.get("source_run_dir") or "")
    )
    if explicit and source_run_id_from_path(explicit) != explicit:
        raise RuntimeError(
            f"Verification metadata has a non-portable source_run_id: {explicit!r}."
        )
    if explicit and legacy and explicit != legacy:
        raise RuntimeError(
            "Verification metadata has conflicting source-run provenance: "
            f"source_run_id={explicit!r}, source_run_dir basename={legacy!r}."
        )
    return explicit or legacy


def _resolve_input_manifest(
    *,
    request: ModelValidationSubmitRequest,
    records: Sequence[PageCandidateRecord],
    run_dir: Path,
    storage_client: storage.Client,
) -> Path:
    requested = str(request.input_manifest_file or "").strip()
    expected_sha = _unique_metadata_value(records, "input_image_manifest_sha256")
    if not expected_sha:
        raise RuntimeError(
            "Every final-validation candidate must bind the extraction input "
            "manifest by SHA-256."
        )
    if not requested:
        source_run_dirs = {
            str(record.extraction_metadata.get("source_run_dir") or "").strip()
            for record in records
            if str(record.extraction_metadata.get("source_run_dir") or "").strip()
        }
        for source_run in sorted(source_run_dirs):
            candidate = Path(source_run).expanduser() / INPUT_IMAGE_MANIFEST_FILE_NAME
            if candidate.is_file():
                requested = str(candidate)
                break
    if not requested and request.source_run_dir:
        candidate = (
            Path(request.source_run_dir).expanduser()
            / INPUT_IMAGE_MANIFEST_FILE_NAME
        )
        if candidate.is_file():
            requested = str(candidate)
    if not requested:
        requested = _unique_metadata_value(records, "input_image_manifest_gcs_uri") or ""
    if not requested:
        raise FileNotFoundError(
            "Generation-bound extraction input manifest is unavailable. Refusing "
            "to validate candidates against an image that may have changed."
        )

    destination = run_dir / INPUT_IMAGE_MANIFEST_FILE_NAME
    if requested.startswith("gs://"):
        _download_gcs_file(storage_client, requested, destination)
    else:
        source = Path(requested).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"Input image manifest not found: {source}")
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
    if expected_sha and file_sha256(destination) != expected_sha:
        raise RuntimeError(
            "Extraction input manifest hash does not match candidate provenance."
        )
    return destination


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_schema_identity(
    records: Sequence[PageCandidateRecord],
) -> tuple[str, str]:
    names = {
        str(record.extraction_metadata.get("schema_name") or "").strip()
        for record in records
    }
    versions = {
        str(record.extraction_metadata.get("schema_version_id") or "").strip()
        for record in records
    }
    if "" in names or len(names) != 1:
        raise RuntimeError(
            "Every validation candidate must carry one identical, non-empty "
            "extraction schema_name."
        )
    if len(versions) != 1:
        raise RuntimeError(
            "Validation candidates reference multiple extraction schema versions."
        )
    return next(iter(names)), next(iter(versions))


def _resolve_extraction_schema(
    *,
    request: ModelValidationSubmitRequest,
    records: Sequence[PageCandidateRecord],
    cache_dir: Path | None = None,
    storage_client: storage.Client | None = None,
) -> _ExtractionSchemaSnapshot:
    candidate_name, candidate_version = _candidate_schema_identity(records)
    run_dirs: list[Path] = []
    if request.source_run_dir:
        run_dirs.append(Path(request.source_run_dir).expanduser())
    for record in records:
        source_run = str(record.extraction_metadata.get("source_run_dir") or "").strip()
        if source_run:
            run_dirs.append(Path(source_run).expanduser())

    metadata_gcs_uri = _unique_metadata_value(records, "source_metadata_gcs_uri")
    metadata_expected_sha = _unique_metadata_value(
        records, "source_metadata_sha256"
    )
    metadata_expected_generation = _unique_metadata_value(
        records, "source_metadata_gcs_generation"
    )
    batch_gcs_uri = _unique_metadata_value(records, "source_batch_job_gcs_uri")
    batch_expected_sha = _unique_metadata_value(
        records, "source_batch_job_sha256"
    )
    batch_expected_generation = _unique_metadata_value(
        records, "source_batch_job_gcs_generation"
    )
    if not metadata_expected_sha:
        raise RuntimeError(
            "Every final-validation candidate must bind extraction metadata.json "
            "by SHA-256."
        )
    if not run_dirs and not (metadata_gcs_uri and batch_gcs_uri):
        raise RuntimeError(
            "Candidate provenance has neither a local extraction source_run_dir "
            "nor cloud-backed immutable extraction metadata."
        )

    def provenance_file(
        *,
        local_path: Path,
        gcs_uri: str | None,
        expected_sha: str | None,
        expected_generation: str | None,
        cache_name: str,
    ) -> Path:
        selected = local_path
        if not selected.is_file():
            if not gcs_uri or storage_client is None or cache_dir is None:
                raise FileNotFoundError(
                    f"Immutable extraction provenance is unavailable: {local_path}"
                )
            selected = cache_dir / cache_name
            if not expected_generation:
                raise RuntimeError(
                    "Cloud extraction provenance has no candidate-bound GCS "
                    "generation."
                )
            _download_gcs_file(
                storage_client,
                gcs_uri,
                selected,
                generation=expected_generation,
            )
        if expected_sha:
            actual_sha = file_sha256(selected)
            if actual_sha != expected_sha:
                raise RuntimeError(
                    "Immutable extraction provenance digest mismatch for "
                    f"{selected.name}."
                )
        elif selected != local_path:
            raise RuntimeError(
                "Cloud extraction provenance has no candidate-bound SHA-256 digest."
            )
        return selected

    snapshots_by_digest: dict[str, _ExtractionSchemaSnapshot] = {}
    seen_paths: set[Path] = set()
    provenance_run_dirs = run_dirs or [Path("cloud-extraction-provenance")]
    for provenance_index, run_dir in enumerate(provenance_run_dirs, start=1):
        try:
            resolved = run_dir.resolve()
        except OSError:
            resolved = run_dir
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        metadata_path = provenance_file(
            local_path=run_dir / "metadata.json",
            gcs_uri=metadata_gcs_uri,
            expected_sha=metadata_expected_sha,
            expected_generation=metadata_expected_generation,
            cache_name=f"source_metadata_{provenance_index:03d}.json",
        )
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid extraction metadata: {metadata_path}") from exc
        schema = payload.get("output_schema") if isinstance(payload, dict) else None
        if not isinstance(schema, dict) or not schema:
            raise RuntimeError(
                f"Extraction metadata has no immutable output_schema: {metadata_path}"
            )
        schema_name = str(payload.get("schema_name") or "").strip()
        schema_version = str(payload.get("schema_version_id") or "").strip()
        if schema_name != candidate_name or schema_version != candidate_version:
            raise RuntimeError(
                "Candidate schema provenance does not match extraction metadata: "
                f"candidate={candidate_name}@{candidate_version or '<unversioned>'}, "
                f"metadata={schema_name or '<missing>'}@"
                f"{schema_version or '<unversioned>'}."
            )

        batch_job_path = provenance_file(
            local_path=run_dir / "batch_job.json",
            gcs_uri=batch_gcs_uri,
            expected_sha=batch_expected_sha,
            expected_generation=batch_expected_generation,
            cache_name=f"source_batch_job_{provenance_index:03d}.json",
        )
        try:
            batch_payload = json.loads(batch_job_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid extraction batch metadata: {batch_job_path}") from exc
        if not isinstance(batch_payload, dict):
            raise ValueError(f"Invalid extraction batch metadata: {batch_job_path}")
        batch_name = str(batch_payload.get("schema_name") or "").strip()
        batch_version = str(batch_payload.get("schema_version_id") or "").strip()
        if batch_name != schema_name or batch_version != schema_version:
            raise RuntimeError(
                "Extraction metadata.json and batch_job.json disagree about the "
                f"schema for {run_dir}."
            )

        digest = _canonical_json_sha256(schema)
        snapshots_by_digest[digest] = _ExtractionSchemaSnapshot(
            schema=copy.deepcopy(schema),
            name=schema_name,
            version_id=schema_version,
            sha256=digest,
        )
    if len(snapshots_by_digest) > 1:
        raise RuntimeError(
            "Candidate provenance resolves to multiple extraction schemas."
        )
    if not snapshots_by_digest:
        raise RuntimeError("No immutable extraction schema could be resolved.")
    return next(iter(snapshots_by_digest.values()))


def _extraction_model_for_snapshot(
    snapshot: _ExtractionSchemaSnapshot,
) -> type:
    """Reuse built-in Pydantic validators only when their schema is exact."""

    try:
        built_in = resolve_output_schema(snapshot.name)
    except ValueError:
        built_in = None
    if (
        built_in is not None
        and _canonical_json_sha256(built_in.model_json_schema()) == snapshot.sha256
    ):
        return built_in
    _assert_dynamic_schema_supported(snapshot.schema)
    return model_from_json_schema(snapshot.name, snapshot.schema)


def _assert_dynamic_schema_supported(schema: Mapping[str, Any]) -> None:
    """Fail closed instead of silently ignoring JSON Schema validation rules."""

    unsupported_keywords = {
        "const",
        "contains",
        "dependentRequired",
        "dependentSchemas",
        "if",
        "maxContains",
        "maxItems",
        "maxProperties",
        "minContains",
        "minItems",
        "minProperties",
        "multipleOf",
        "not",
        "patternProperties",
        "prefixItems",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
        "uniqueItems",
    }
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        definitions = schema.get("definitions")
    if not isinstance(definitions, dict):
        definitions = {}

    def visit(node: object, pointer: str) -> None:
        if isinstance(node, list):
            for index, item in enumerate(node):
                visit(item, f"{pointer}/{index}")
            return
        if not isinstance(node, dict):
            return
        present = sorted(unsupported_keywords.intersection(node))
        if present:
            raise RuntimeError(
                "Managed extraction schema uses locally unsupported validation "
                f"keyword(s) at {pointer or '/'}: {', '.join(present)}."
            )
        ref = node.get("$ref")
        if isinstance(ref, str):
            prefixes = ("#/$defs/", "#/definitions/")
            matched = next((prefix for prefix in prefixes if ref.startswith(prefix)), None)
            if matched is None or ref[len(matched) :] not in definitions:
                raise RuntimeError(
                    f"Managed extraction schema has an unresolved $ref at {pointer or '/'}: {ref}"
                )
        for variant_key in ("anyOf", "oneOf"):
            variants = node.get(variant_key)
            if isinstance(variants, list):
                concrete = [
                    item
                    for item in variants
                    if isinstance(item, dict) and item.get("type") != "null"
                ]
                null_count = sum(
                    1
                    for item in variants
                    if isinstance(item, dict) and item.get("type") == "null"
                )
                if len(concrete) != 1 or null_count != len(variants) - 1:
                    raise RuntimeError(
                        "Managed extraction schema uses a complex union that the "
                        f"local validator cannot enforce at {pointer or '/'} ({variant_key})."
                    )
        all_of = node.get("allOf")
        if isinstance(all_of, list) and len(all_of) != 1:
            raise RuntimeError(
                "Managed extraction schema uses a multi-branch allOf that the "
                f"local validator cannot enforce at {pointer or '/'} ."
            )
        raw_type = node.get("type")
        if isinstance(raw_type, list):
            concrete_types = [item for item in raw_type if item != "null"]
            if len(concrete_types) != 1:
                raise RuntimeError(
                    f"Managed extraction schema uses a complex type union at {pointer or '/'} ."
                )
        additional = node.get("additionalProperties")
        if isinstance(additional, dict):
            raise RuntimeError(
                "Managed extraction schema uses typed additionalProperties, which "
                f"the local validator cannot enforce at {pointer or '/'} ."
            )
        value_format = node.get("format")
        if value_format not in {None, "date"}:
            raise RuntimeError(
                f"Managed extraction schema format {value_format!r} is not locally "
                f"enforced at {pointer or '/'} ."
            )
        for key, value in node.items():
            if key in {"default", "description", "examples", "title"}:
                continue
            visit(value, f"{pointer}/{key}")

    visit(dict(schema), "")


def _validate_extraction_candidate(
    model: type,
    candidate: Mapping[str, Any],
) -> Any:
    """Validate JSON without Python-side scalar coercion."""

    payload = json.dumps(
        dict(candidate), ensure_ascii=False, separators=(",", ":")
    )
    return model.model_validate_json(payload, strict=True)


def _mime_type_for_name(name: str, fallback: str) -> str:
    if fallback.strip():
        return fallback.strip()
    guessed, _ = mimetypes.guess_type(name)
    return guessed or "application/octet-stream"


def _validate_input_manifest_coverage(
    *,
    records: Sequence[PageCandidateRecord],
    input_by_key: Mapping[str, InputImageManifestRecord],
    scope: str,
) -> None:
    candidate_keys = {record.key for record in records}
    manifest_keys = set(input_by_key)
    absent_bindings = sorted(candidate_keys - manifest_keys)
    if absent_bindings:
        preview = ", ".join(absent_bindings[:5])
        raise RuntimeError(
            "Extraction input manifest does not bind every validation candidate: "
            f"{preview}."
        )
    if scope != "all":
        return
    missing_candidates = sorted(manifest_keys - candidate_keys)
    unexpected_candidates = sorted(candidate_keys - manifest_keys)
    if missing_candidates or unexpected_candidates:
        missing_preview = ", ".join(missing_candidates[:5]) or "none"
        unexpected_preview = ", ".join(unexpected_candidates[:5]) or "none"
        raise RuntimeError(
            "Scope 'all' requires exact page-key equality between the complete "
            "extraction input manifest and page candidates. "
            f"missing_candidates={len(missing_candidates)} ({missing_preview}); "
            f"unexpected_candidates={len(unexpected_candidates)} "
            f"({unexpected_preview})."
        )


def _normalize_prefix(value: str) -> str:
    normalized = str(value or "").strip().strip("/")
    return f"{normalized}/" if normalized else ""


def _stage_exact_gemini_image(
    *,
    bucket: storage.Bucket,
    source: CloudBlobIdentity,
    run_dir_name: str,
) -> tuple[str, CloudBlobIdentity]:
    generation = int(source.generation)
    source_blob = bucket.blob(source.name, generation=generation)
    suffix = Path(source.name).suffix.lower()
    object_digest = hashlib.sha256(
        f"{source.bucket}/{source.name}#{source.generation}".encode("utf-8")
    ).hexdigest()[:32]
    object_name = (
        f"{_normalize_prefix(config.batch_requests_gcs_prefix)}"
        f"{run_dir_name}/validation_images/{object_digest}{suffix}"
    )
    staged_blob = bucket.copy_blob(
        source_blob,
        bucket,
        new_name=object_name,
        preserve_acl=False,
        source_generation=generation,
        if_source_generation_match=generation,
        if_generation_match=0,
    )
    reload_blob = getattr(staged_blob, "reload", None)
    if callable(reload_blob):
        reload_blob()
    staged = CloudBlobIdentity.from_blob(staged_blob)
    if not staged.generation:
        raise RuntimeError(f"Staged validation image has no GCS generation: {object_name}")
    for expected, actual, label in (
        (source.size, staged.size, "size"),
        (source.crc32c, staged.crc32c, "crc32c"),
        (source.md5_hash, staged.md5_hash, "md5_hash"),
    ):
        if expected is not None and actual is not None and expected != actual:
            raise RuntimeError(
                f"Staged validation image {label} mismatch for {source.name!r}."
            )
    return f"gs://{bucket.name}/{object_name}", staged


def _register_mldev_validation_images(
    *,
    client: object,
    storage_client: storage.Client,
    pages: Sequence[_PreparedValidationPage],
) -> tuple[_PreparedValidationPage, ...]:
    """Register immutable staged GCS objects with the Gemini Files API."""

    credentials = getattr(storage_client, "_credentials", None)
    if credentials is None:
        raise RuntimeError(
            "Gemini Developer validation requires Google credentials to register "
            "the staged GCS image files."
        )
    files_api = getattr(client, "files", None)
    register_files = getattr(files_api, "register_files", None)
    if not callable(register_files):
        raise RuntimeError(
            "The installed Gemini SDK cannot register GCS images with the Files API. "
            "Use Vertex Gemini or upgrade google-genai."
        )
    staged_uris = [page.provider_image_reference for page in pages]
    if any(not uri.startswith("gs://") for uri in staged_uris):
        raise RuntimeError("Gemini MLDev image registration requires staged GCS URIs.")
    registered_files: list[object] = []
    for start in range(0, len(staged_uris), MLDEV_REGISTER_FILES_BATCH_SIZE):
        uri_batch = staged_uris[start : start + MLDEV_REGISTER_FILES_BATCH_SIZE]
        response = register_files(auth=credentials, uris=uri_batch)
        returned = list(getattr(response, "files", None) or [])
        if len(returned) != len(uri_batch):
            raise RuntimeError(
                "Gemini Files API returned an unexpected registration count: "
                f"{len(returned)}/{len(uri_batch)} for batch starting at {start}."
            )
        registered_files.extend(returned)

    registered_pages: list[_PreparedValidationPage] = []
    for page, registered in zip(pages, registered_files, strict=True):
        uri = str(getattr(registered, "uri", "") or "").strip()
        if not uri:
            raise RuntimeError(
                f"Gemini Files API returned no file URI for {page.record.key!r}."
            )
        registered_sha256 = str(
            getattr(registered, "sha256_hash", "") or ""
        ).strip()
        if registered_sha256 and page.input_record.ocr_image_sha256:
            try:
                registered_digest = base64.b64decode(
                    registered_sha256, validate=True
                ).hex()
            except (ValueError, TypeError) as exc:
                raise RuntimeError(
                    f"Gemini Files API returned an invalid SHA-256 for {page.record.key!r}."
                ) from exc
            if registered_digest != page.input_record.ocr_image_sha256:
                raise RuntimeError(
                    f"Gemini Files API image digest mismatch for {page.record.key!r}."
                )
        binding = dict(page.binding)
        binding["staged_image_uri"] = page.provider_image_reference
        binding["request_image_uri"] = uri
        binding["registered_file_name"] = str(
            getattr(registered, "name", "") or ""
        )
        binding["registered_file_sha256"] = registered_sha256
        registered_pages.append(
            replace(
                page,
                provider_image_reference=uri,
                binding=binding,
            )
        )
    return tuple(registered_pages)


def _prepare_validation_page(
    *,
    record: PageCandidateRecord,
    input_record: InputImageManifestRecord,
    bucket: storage.Bucket,
    provider: str,
    run_dir_name: str,
    extraction_schema: Mapping[str, Any],
) -> _PreparedValidationPage:
    if record.key != input_record.key:
        raise ValueError("Candidate key and input image manifest key differ.")

    current_ocr: CloudOcrMetadata | None = None
    current_sidecar_source: CloudBlobIdentity | None = None
    if input_record.ocr_enabled:
        current_ocr, current_sidecar_source = _load_bound_ocr_evidence(
            bucket=bucket,
            input_record=input_record,
        )

    mime_type = _mime_type_for_name(record.key, input_record.mime_type)
    expected_source = input_record.source
    if provider == "gemini":
        provider_reference, request_source = _stage_exact_gemini_image(
            bucket=bucket,
            source=expected_source,
            run_dir_name=run_dir_name,
        )
    elif provider == "anthropic":
        generation = int(expected_source.generation)
        pinned_blob = bucket.blob(expected_source.name, generation=generation)
        provider_reference = pinned_blob.generate_signed_url(
            version="v4",
            method="GET",
            expiration=_anthropic_signed_url_expiration(),
            query_parameters={"generation": str(generation)},
        )
        request_source = expected_source
    else:
        raise ValueError(f"Unsupported verification provider: {provider}")

    digest = candidate_sha256(record.candidate)
    prompt = build_model_validation_prompt(
        candidate=record.candidate,
        extraction_schema=extraction_schema,
        ocr_context=(
            render_ocr_context(current_ocr.document) if current_ocr is not None else ""
        ),
    )
    binding = {
        "key": record.key,
        "candidate_sha256": digest,
        "extraction_image_source": expected_source.to_dict(),
        "request_image_source": request_source.to_dict(),
        "request_image_uri": (
            provider_reference
            if provider == "gemini"
            else f"gs://{expected_source.bucket}/{expected_source.name}#{expected_source.generation}"
        ),
        "ocr_enabled": bool(input_record.ocr_enabled),
        "ocr_sidecar_name": input_record.ocr_sidecar_name,
        "ocr_sidecar_source": (
            current_sidecar_source.to_dict()
            if current_sidecar_source is not None
            else {}
        ),
        "ocr_sidecar_sha256": input_record.ocr_sidecar_sha256,
        "ocr_image_sha256": input_record.ocr_image_sha256,
        "ocr_document_sha256": input_record.ocr_document_sha256,
        "ocr_backend": input_record.ocr_backend,
        "ocr_line_count": input_record.ocr_line_count,
    }
    return _PreparedValidationPage(
        record=record,
        input_record=input_record,
        candidate_digest=digest,
        mime_type=mime_type,
        prompt=prompt,
        provider_image_reference=provider_reference,
        request_image_source=request_source,
        binding=binding,
    )


def _load_bound_ocr_evidence(
    *,
    bucket: storage.Bucket,
    input_record: InputImageManifestRecord,
) -> tuple[CloudOcrMetadata, CloudBlobIdentity]:
    """Reload and validate the exact image/OCR evidence in an extraction manifest."""

    if not input_record.ocr_enabled:
        raise ValueError("OCR evidence was not enabled for this extraction page.")

    current_blob = bucket.blob(input_record.key)
    current_ocr = load_ocr_metadata_for_blob(current_blob)
    if current_ocr is None:
        raise RuntimeError(
            f"Missing or stale cloud OCR metadata for {input_record.key!r}; "
            "refusing validation."
        )
    expected_source = input_record.source
    if not expected_source.matches(current_ocr.source):
        raise RuntimeError(
            f"GCS generation for {input_record.key!r} changed since extraction; "
            "refusing to validate a candidate against different bytes."
        )
    if current_ocr.document.image_sha256 != input_record.ocr_image_sha256:
        raise RuntimeError(
            f"OCR image digest for {input_record.key!r} differs from extraction manifest."
        )
    if ocr_document_sha256(current_ocr.document) != input_record.ocr_document_sha256:
        raise RuntimeError(
            f"OCR text/position document for {input_record.key!r} changed since extraction."
        )
    sidecar_blob = bucket.blob(input_record.ocr_sidecar_name)
    reload_sidecar = getattr(sidecar_blob, "reload", None)
    if callable(reload_sidecar):
        reload_sidecar()
    current_sidecar_source = CloudBlobIdentity.from_blob(sidecar_blob)
    if not input_record.sidecar_source.matches(current_sidecar_source):
        raise RuntimeError(
            f"OCR sidecar generation for {input_record.key!r} changed since extraction."
        )
    download_sidecar = getattr(sidecar_blob, "download_as_bytes", None)
    if not callable(download_sidecar):
        raise TypeError(
            f"OCR sidecar cannot be downloaded: {input_record.ocr_sidecar_name}"
        )
    sidecar_bytes = download_sidecar(
        if_generation_match=int(current_sidecar_source.generation)
    )
    if not isinstance(sidecar_bytes, bytes):
        sidecar_bytes = bytes(sidecar_bytes)
    if hashlib.sha256(sidecar_bytes).hexdigest() != input_record.ocr_sidecar_sha256:
        raise RuntimeError(
            f"OCR sidecar content for {input_record.key!r} changed since extraction."
        )
    return current_ocr, current_sidecar_source


def _split_evenly[T](items: Sequence[T], num_chunks: int) -> list[list[T]]:
    if num_chunks <= 0:
        raise ValueError("verification_num_chunks must be >= 1.")
    if not items:
        return []
    count = min(len(items), num_chunks)
    base, remainder = divmod(len(items), count)
    chunks: list[list[T]] = []
    start = 0
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        chunks.append(list(items[start : start + size]))
        start += size
    return chunks


def _chunk_file_name(index: int, total: int) -> str:
    return f"validation_requests.part{index:03d}-of-{total:03d}.jsonl"


def _gemini_generation_config(
    *,
    model: str,
    for_vertex: bool,
    thinking_level: str,
    max_output_tokens: int,
) -> dict:
    schema: object = PageModelValidation.model_json_schema()
    if for_vertex:
        schema = _vertex_compatible_schema(schema)
    schema_key = "responseSchema" if for_vertex else "responseJsonSchema"
    if "gemini-2.5-" in model.lower():
        available_budget = max_output_tokens - 256
        if available_budget < 128:
            raise ValueError(
                "Gemini 2.5 verification requires verification_max_output_tokens "
                ">= 384 when thinking is enabled."
            )
        requested_budget = {
            "low": 512,
            "medium": 2048,
            # High is the slider's Maximum position. Preserve an answer reserve
            # and give all remaining output budget to provider reasoning.
            "high": available_budget,
        }[thinking_level]
        thinking_config = {"thinkingBudget": min(requested_budget, available_budget)}
    else:
        thinking_config = {"thinkingLevel": thinking_level}
    return {
        "responseMimeType": "application/json",
        schema_key: schema,
        "maxOutputTokens": max_output_tokens,
        "thinkingConfig": thinking_config,
    }


def _gemini_request_line(
    page: _PreparedValidationPage,
    *,
    model: str,
    for_vertex: bool,
    thinking_level: str,
    max_output_tokens: int,
) -> dict[str, object]:
    return {
        "key": page.record.key,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "fileData": {
                                "fileUri": page.provider_image_reference,
                                "mimeType": page.mime_type,
                            }
                        },
                        {"text": page.prompt},
                    ],
                }
            ],
            "generationConfig": _gemini_generation_config(
                model=model,
                for_vertex=for_vertex,
                thinking_level=thinking_level,
                max_output_tokens=max_output_tokens,
            ),
        },
    }


def _anthropic_thinking_config(thinking_level: str, max_tokens: int) -> dict[str, object]:
    available_budget = max_tokens - 512
    requested = {
        "low": 1024,
        "medium": 2048,
        "high": available_budget,
    }[thinking_level]
    budget = min(requested, available_budget)
    if budget < 1024:
        raise ValueError(
            "Anthropic verification requires verification_max_output_tokens >= 1536 "
            "when thinking is enabled."
        )
    return {"type": "enabled", "budget_tokens": budget}


def _anthropic_request(
    page: _PreparedValidationPage,
    *,
    model: str,
    thinking_level: str,
    max_output_tokens: int,
) -> dict[str, object]:
    return {
        "custom_id": _anthropic_custom_id_for_key(page.record.key),
        "params": {
            "model": model,
            "max_tokens": max_output_tokens,
            "thinking": _anthropic_thinking_config(
                thinking_level, max_output_tokens
            ),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": page.provider_image_reference,
                            },
                        },
                        {"type": "text", "text": page.prompt},
                    ],
                }
            ],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": _anthropic_strict_json_schema(
                        PageModelValidation.model_json_schema()
                    ),
                }
            },
        },
    }


def _split_anthropic_chunks_by_bytes(
    chunks: Sequence[Sequence[_PreparedValidationPage]],
    *,
    model: str,
    thinking_level: str,
    max_output_tokens: int,
    byte_limit: int = ANTHROPIC_MAX_BATCH_REQUEST_BYTES,
    request_limit: int = ANTHROPIC_MAX_BATCH_REQUESTS,
) -> list[list[_PreparedValidationPage]]:
    """Keep each durable/submitted Message Batch safely below 256 MB."""

    refined: list[list[_PreparedValidationPage]] = []
    for requested_chunk in chunks:
        current: list[_PreparedValidationPage] = []
        current_bytes = 0
        for page in requested_chunk:
            row = _anthropic_request(
                page,
                model=model,
                thinking_level=thinking_level,
                max_output_tokens=max_output_tokens,
            )
            row_bytes = len(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            ) + 1
            if row_bytes > byte_limit:
                raise RuntimeError(
                    f"One Anthropic validation request for {page.record.key!r} is "
                    f"{row_bytes} bytes, exceeding the {byte_limit}-byte safety limit."
                )
            if current and (
                current_bytes + row_bytes > byte_limit
                or len(current) >= request_limit
            ):
                refined.append(current)
                current = []
                current_bytes = 0
            current.append(page)
            current_bytes += row_bytes
        if current:
            refined.append(current)
    return refined


def _split_gemini_chunks_by_bytes(
    chunks: Sequence[Sequence[_PreparedValidationPage]],
    *,
    model: str,
    for_vertex: bool,
    thinking_level: str,
    max_output_tokens: int,
    byte_limit: int = GEMINI_MAX_BATCH_REQUEST_BYTES,
) -> list[list[_PreparedValidationPage]]:
    """Keep each Gemini request JSONL safely below its 2 GB file limit."""

    refined: list[list[_PreparedValidationPage]] = []
    for requested_chunk in chunks:
        current: list[_PreparedValidationPage] = []
        current_bytes = 0
        for page in requested_chunk:
            row = _gemini_request_line(
                page,
                model=model,
                for_vertex=for_vertex,
                thinking_level=thinking_level,
                max_output_tokens=max_output_tokens,
            )
            row_bytes = len(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            ) + 1
            if row_bytes > byte_limit:
                raise RuntimeError(
                    f"One Gemini validation request for {page.record.key!r} is "
                    f"{row_bytes} bytes, exceeding the {byte_limit}-byte safety limit."
                )
            if current and current_bytes + row_bytes > byte_limit:
                refined.append(current)
                current = []
                current_bytes = 0
            current.append(page)
            current_bytes += row_bytes
        if current:
            refined.append(current)
    return refined


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> tuple[int, int]:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    return len(rows), path.stat().st_size


def _write_bindings(path: Path, pages: Sequence[_PreparedValidationPage]) -> None:
    _write_jsonl(path, [page.binding for page in pages])


def _write_validation_request_contract(
    *,
    run_dir: Path,
    provider: str,
    client_backend: str,
    model: str,
    thinking_level: str,
    max_output_tokens: int,
    requested_num_chunks: int,
    chunk_records: Sequence[Mapping[str, object]],
) -> tuple[Path, dict[str, object], str]:
    """Freeze the exact provider request files before any batch is submitted."""

    normalized_chunks = [dict(record) for record in chunk_records]
    payload: dict[str, object] = {
        "schema_version": 1,
        "provider": provider,
        "client_backend": client_backend,
        "model": model,
        "thinking_level": thinking_level,
        "max_output_tokens": int(max_output_tokens),
        "requested_num_chunks": int(requested_num_chunks),
        "effective_num_chunks": len(normalized_chunks),
        "chunks": normalized_chunks,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    path = run_dir / VERIFICATION_REQUEST_CONTRACT_FILE_NAME
    path.write_bytes(encoded)
    return path, payload, hashlib.sha256(encoded).hexdigest()


def _read_jsonl_request_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid frozen validation request JSONL at "
                    f"{path.name}:{line_number}."
                ) from exc
            if not isinstance(row, dict):
                raise RuntimeError(
                    f"Validation request row is not an object at "
                    f"{path.name}:{line_number}."
                )
            rows.append(row)
    return rows


def _validate_frozen_request_contract(
    *,
    run_dir: Path,
    payload: Mapping[str, object],
    provider: str,
    client_backend: str,
    model: str,
    thinking_level: str,
    max_output_tokens: int,
) -> tuple[dict[str, object], ...]:
    """Fail closed when policy-bound request bytes or parameters drift."""

    expected_top_level = {
        "schema_version",
        "provider",
        "client_backend",
        "model",
        "thinking_level",
        "max_output_tokens",
        "requested_num_chunks",
        "effective_num_chunks",
        "chunks",
    }
    if set(payload) != expected_top_level or int(payload.get("schema_version") or 0) != 1:
        raise RuntimeError("Frozen validation request contract is invalid.")
    expected_values = {
        "provider": provider,
        "client_backend": client_backend,
        "model": model,
        "thinking_level": thinking_level,
        "max_output_tokens": int(max_output_tokens),
    }
    for key, expected in expected_values.items():
        if payload.get(key) != expected:
            raise RuntimeError(
                f"Frozen validation request contract disagrees on {key}."
            )
    raw_chunks = payload.get("chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise RuntimeError("Frozen validation request contract has no chunks.")
    if int(payload.get("effective_num_chunks") or 0) != len(raw_chunks):
        raise RuntimeError("Frozen validation request chunk count is inconsistent.")
    if int(payload.get("requested_num_chunks") or 0) <= 0:
        raise RuntimeError("Frozen validation requested chunk count is invalid.")

    normalized: list[dict[str, object]] = []
    seen_files: set[str] = set()
    total_chunks = len(raw_chunks)
    expected_chunk_keys = {
        "chunk_index",
        "total_chunks",
        "requests_file",
        "request_count",
        "request_bytes",
        "requests_sha256",
    }
    for expected_index, raw_chunk in enumerate(raw_chunks, start=1):
        if not isinstance(raw_chunk, dict) or set(raw_chunk) != expected_chunk_keys:
            raise RuntimeError("Frozen validation request chunk is invalid.")
        requests_file = str(raw_chunk.get("requests_file") or "").strip()
        digest = str(raw_chunk.get("requests_sha256") or "").strip()
        request_count = int(raw_chunk.get("request_count") or 0)
        request_bytes = int(raw_chunk.get("request_bytes") or 0)
        if (
            int(raw_chunk.get("chunk_index") or 0) != expected_index
            or int(raw_chunk.get("total_chunks") or 0) != total_chunks
            or not requests_file
            or Path(requests_file).name != requests_file
            or requests_file in seen_files
            or len(digest) != 64
            or request_count <= 0
            or request_bytes <= 0
        ):
            raise RuntimeError("Frozen validation request chunk identity is invalid.")
        path = run_dir / requests_file
        if (
            not path.is_file()
            or path.stat().st_size != request_bytes
            or file_sha256(path) != digest
            or len(_read_jsonl_request_rows(path)) != request_count
        ):
            raise RuntimeError(
                f"Frozen validation request bytes changed: {requests_file}."
            )
        seen_files.add(requests_file)
        normalized.append(dict(raw_chunk))
    return tuple(normalized)


def _upload_validation_artifact(
    bucket: storage.Bucket,
    run_dir: Path,
    path: Path,
    *,
    validations_prefix: str | None = None,
) -> str:
    uri, object_name = _validation_artifact_location(
        bucket,
        run_dir,
        path,
        validations_prefix=validations_prefix,
    )
    content_type = (
        "application/jsonl" if path.suffix.lower() == ".jsonl" else "application/json"
    )
    bucket.blob(object_name).upload_from_filename(
        str(path), content_type=content_type
    )
    return uri


def _upload_immutable_validation_artifact(
    bucket: storage.Bucket,
    run_dir: Path,
    path: Path,
    *,
    sha256: str,
    validations_prefix: str | None = None,
) -> tuple[str, str]:
    """Create or verify one content-addressed, write-once audit artifact."""

    prefix = _normalize_prefix(
        config.validations_gcs_prefix
        if validations_prefix is None
        else validations_prefix
    )
    object_name = (
        f"{prefix}model/{run_dir.name}/immutable/{sha256}/{path.name}"
    )
    uri = f"gs://{bucket.name}/{object_name}"
    blob = bucket.blob(object_name)
    blob.metadata = {
        "artifact_kind": (
            "field_correction_metadata"
            if path.name == FIELD_CORRECTION_METADATA_FILE_NAME
            else path.stem
        ),
        "sha256": sha256,
        "verification_run_id": run_dir.name,
    }
    try:
        blob.upload_from_filename(
            str(path),
            content_type=(
                "application/jsonl"
                if path.suffix.lower() == ".jsonl"
                else "application/json"
            ),
            if_generation_match=0,
        )
    except Exception as exc:  # noqa: BLE001 - provider exception wrappers vary
        if not isinstance(exc, PreconditionFailed) and getattr(exc, "code", None) != 412:
            raise
        blob.reload()
        generation = int(getattr(blob, "generation", 0) or 0)
        payload = blob.download_as_bytes(if_generation_match=generation or None)
        if hashlib.sha256(bytes(payload)).hexdigest() != sha256:
            raise RuntimeError(
                "Content-addressed validation artifact has different bytes."
            ) from exc
    blob.reload()
    generation = str(getattr(blob, "generation", "") or "")
    if not generation:
        raise RuntimeError(
            "Immutable validation artifact upload has no GCS generation binding."
        )
    return uri, generation


def _validation_artifact_location(
    bucket: storage.Bucket,
    run_dir: Path,
    path: Path,
    *,
    validations_prefix: str | None = None,
) -> tuple[str, str]:
    prefix = _normalize_prefix(
        config.validations_gcs_prefix
        if validations_prefix is None
        else validations_prefix
    )
    object_name = f"{prefix}model/{run_dir.name}/{path.name}"
    return f"gs://{bucket.name}/{object_name}", object_name


def _final_validation_policy_anchor_object_name(verification_run_id: str) -> str:
    """Return a config-independent cloud anchor for one run's final policy."""

    run_id = str(verification_run_id or "").strip()
    if not run_id or Path(run_id).name != run_id:
        raise ValueError(f"Invalid verification run id: {verification_run_id!r}")
    return (
        f"{FINAL_VALIDATION_POLICY_ANCHOR_PREFIX}/{run_id}/"
        f"{FINAL_VALIDATION_POLICY_FILE_NAME}"
    )


def _final_validation_policy_payload(
    *,
    verification_run_id: str,
    source_run_id: str,
    datasets_gcs_prefix: str,
    validations_gcs_prefix: str,
    verification_scope: VerificationScope = "all",
    source_candidates_sha256: str = "",
    selected_candidates_sha256: str = "",
    source_candidate_count: int = 0,
    selected_candidate_count: int = 0,
    deterministic_routing_policy: str = "",
    deterministic_routing_sha256: str = "",
    deterministic_routing_gcs_uri: str = "",
    deterministic_routing_gcs_generation: str = "",
    verification_provider: str = "",
    verification_client_backend: str = "",
    verification_model: str = "",
    verification_thinking_level: str = "",
    verification_max_output_tokens: int = 0,
    verification_prompt_sha256: str = "",
    verification_request_contract_sha256: str = "",
    verification_request_contract_gcs_uri: str = "",
    verification_request_contract_gcs_generation: str = "",
    input_image_manifest_sha256: str = "",
    extraction_schema_sha256: str = "",
    validation_bindings_sha256: str = "",
) -> dict[str, object]:
    return {
        "schema_version": FINAL_VALIDATION_POLICY_SCHEMA_VERSION,
        "job_kind": "final_validation_policy",
        "verification_run_id": verification_run_id,
        "source_run_id": source_run_id,
        "verification_scope": verification_scope,
        "verification_apply_mode": FINAL_VALIDATION_APPLY_MODE,
        "correction_acceptance_policy": (
            AUTOMATIC_CORRECTION_ACCEPTANCE_POLICY
        ),
        # Prefixes affect the durable evidence and publication destinations, so
        # they are part of the immutable submission policy rather than mutable
        # retrieval-time configuration.
        "datasets_gcs_prefix": datasets_gcs_prefix,
        "validations_gcs_prefix": validations_gcs_prefix,
        "source_candidates_sha256": source_candidates_sha256,
        "selected_candidates_sha256": selected_candidates_sha256,
        "source_candidate_count": int(source_candidate_count),
        "selected_candidate_count": int(selected_candidate_count),
        "deterministic_routing_policy": deterministic_routing_policy,
        "deterministic_routing_sha256": deterministic_routing_sha256,
        "deterministic_routing_gcs_uri": deterministic_routing_gcs_uri,
        "deterministic_routing_gcs_generation": (
            deterministic_routing_gcs_generation
        ),
        "verification_provider": verification_provider,
        "verification_client_backend": verification_client_backend,
        "verification_model": verification_model,
        "verification_thinking_level": verification_thinking_level,
        "verification_max_output_tokens": int(verification_max_output_tokens),
        "verification_prompt_sha256": verification_prompt_sha256,
        "verification_request_contract_sha256": (
            verification_request_contract_sha256
        ),
        "verification_request_contract_gcs_uri": (
            verification_request_contract_gcs_uri
        ),
        "verification_request_contract_gcs_generation": (
            verification_request_contract_gcs_generation
        ),
        "input_image_manifest_sha256": input_image_manifest_sha256,
        "extraction_schema_sha256": extraction_schema_sha256,
        "validation_bindings_sha256": validation_bindings_sha256,
    }


def _write_final_validation_policy(
    *,
    run_dir: Path,
    source_run_dir: str,
    datasets_gcs_prefix: str,
    validations_gcs_prefix: str,
    verification_scope: VerificationScope = "all",
    source_candidates_sha256: str = "",
    selected_candidates_sha256: str = "",
    source_candidate_count: int = 0,
    selected_candidate_count: int = 0,
    deterministic_routing_policy: str = "",
    deterministic_routing_sha256: str = "",
    deterministic_routing_gcs_uri: str = "",
    deterministic_routing_gcs_generation: str = "",
    verification_provider: str = "",
    verification_client_backend: str = "",
    verification_model: str = "",
    verification_thinking_level: str = "",
    verification_max_output_tokens: int = 0,
    verification_prompt_sha256: str = "",
    verification_request_contract_sha256: str = "",
    verification_request_contract_gcs_uri: str = "",
    verification_request_contract_gcs_generation: str = "",
    input_image_manifest_sha256: str = "",
    extraction_schema_sha256: str = "",
    validation_bindings_sha256: str = "",
) -> tuple[Path, dict[str, object], str]:
    source_run_id = source_run_id_from_path(source_run_dir)
    if not source_run_id:
        raise ValueError("Model validation requires a non-empty source_run_dir.")
    payload = _final_validation_policy_payload(
        verification_run_id=run_dir.name,
        source_run_id=source_run_id,
        datasets_gcs_prefix=datasets_gcs_prefix,
        validations_gcs_prefix=validations_gcs_prefix,
        verification_scope=verification_scope,
        source_candidates_sha256=source_candidates_sha256,
        selected_candidates_sha256=selected_candidates_sha256,
        source_candidate_count=source_candidate_count,
        selected_candidate_count=selected_candidate_count,
        deterministic_routing_policy=deterministic_routing_policy,
        deterministic_routing_sha256=deterministic_routing_sha256,
        deterministic_routing_gcs_uri=deterministic_routing_gcs_uri,
        deterministic_routing_gcs_generation=(
            deterministic_routing_gcs_generation
        ),
        verification_provider=verification_provider,
        verification_client_backend=verification_client_backend,
        verification_model=verification_model,
        verification_thinking_level=verification_thinking_level,
        verification_max_output_tokens=verification_max_output_tokens,
        verification_prompt_sha256=verification_prompt_sha256,
        verification_request_contract_sha256=(
            verification_request_contract_sha256
        ),
        verification_request_contract_gcs_uri=(
            verification_request_contract_gcs_uri
        ),
        verification_request_contract_gcs_generation=(
            verification_request_contract_gcs_generation
        ),
        input_image_manifest_sha256=input_image_manifest_sha256,
        extraction_schema_sha256=extraction_schema_sha256,
        validation_bindings_sha256=validation_bindings_sha256,
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    path = run_dir / FINAL_VALIDATION_POLICY_FILE_NAME
    path.write_bytes(encoded)
    return path, payload, hashlib.sha256(encoded).hexdigest()


def _upload_final_validation_policy(
    *,
    bucket: storage.Bucket,
    run_dir: Path,
    policy_path: Path,
    policy_sha256: str,
) -> tuple[str, str]:
    """Create or verify the config-independent immutable policy anchor."""

    object_name = _final_validation_policy_anchor_object_name(run_dir.name)
    uri = f"gs://{bucket.name}/{object_name}"
    blob = bucket.blob(object_name)
    blob.metadata = {
        "artifact_kind": "final_validation_policy",
        "sha256": policy_sha256,
        "verification_run_id": run_dir.name,
    }
    try:
        blob.upload_from_filename(
            str(policy_path),
            content_type="application/json",
            if_generation_match=0,
        )
    except Exception as exc:  # noqa: BLE001 - provider exception wrappers vary
        if not isinstance(exc, PreconditionFailed) and getattr(exc, "code", None) != 412:
            raise
        blob.reload()
        generation = int(getattr(blob, "generation", 0) or 0)
        payload = blob.download_as_bytes(if_generation_match=generation or None)
        if hashlib.sha256(bytes(payload)).hexdigest() != policy_sha256:
            raise RuntimeError(
                "Immutable final-validation policy anchor has different bytes."
            ) from exc
    blob.reload()
    generation = str(getattr(blob, "generation", "") or "")
    if not generation:
        raise RuntimeError(
            "Final-validation policy upload has no GCS generation binding."
        )
    return uri, generation


def _is_not_found(exc: Exception) -> bool:
    return isinstance(exc, NotFound) or getattr(exc, "code", None) == 404


def _resolve_final_validation_policy(
    *,
    bucket: storage.Bucket,
    run_dir: Path,
    metadata: Mapping[str, Any],
) -> _FinalValidationPolicySnapshot:
    """Load the cloud-anchored policy; fail closed for legacy unanchored runs."""

    object_name = _final_validation_policy_anchor_object_name(run_dir.name)
    expected_uri = f"gs://{bucket.name}/{object_name}"
    blob = bucket.blob(object_name)
    try:
        blob.reload()
    except Exception as exc:  # noqa: BLE001 - provider exception wrappers vary
        if not _is_not_found(exc):
            raise
        policy_claimed = any(
            str(metadata.get(key) or "").strip()
            for key in (
                "final_validation_policy_file",
                "final_validation_policy_sha256",
                "final_validation_policy_gcs_uri",
                "final_validation_policy_gcs_generation",
            )
        )
        if policy_claimed:
            raise RuntimeError(
                "Verification metadata claims an immutable final-validation "
                "policy, but its cloud anchor is missing."
            ) from exc
        # Historical runs predate the immutable policy. They remain usable for
        # reports, but mutable legacy metadata can never promote them to an
        # automatically corrected dataset publication.
        return _FinalValidationPolicySnapshot(
            apply_mode="report_only",
            acceptance_policy="legacy_report_only",
            verification_scope="all",
            source_run_id=_metadata_source_run_id(metadata),
            datasets_gcs_prefix=str(
                metadata.get("datasets_gcs_prefix")
                or config.datasets_gcs_prefix
                or "datasets"
            ).strip().strip("/"),
            validations_gcs_prefix=str(
                metadata.get("validations_gcs_prefix")
                if metadata.get("validations_gcs_prefix") is not None
                else config.validations_gcs_prefix
            ).strip().strip("/"),
            legacy=True,
        )

    generation = str(getattr(blob, "generation", "") or "")
    expected_generation = str(
        metadata.get("final_validation_policy_gcs_generation") or ""
    ).strip()
    expected_sha256 = str(
        metadata.get("final_validation_policy_sha256") or ""
    ).strip()
    expected_file = str(metadata.get("final_validation_policy_file") or "").strip()
    recorded_uri = str(
        metadata.get("final_validation_policy_gcs_uri") or ""
    ).strip()
    if (
        expected_file != FINAL_VALIDATION_POLICY_FILE_NAME
        or not expected_sha256
        or recorded_uri != expected_uri
        or not generation
        or expected_generation != generation
    ):
        raise RuntimeError(
            "Mutable verification metadata does not match the immutable "
            "final-validation policy anchor."
        )
    payload_bytes = bytes(
        blob.download_as_bytes(if_generation_match=int(generation))
    )
    actual_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError("Final-validation policy SHA-256 mismatch.")

    policy_path = run_dir / FINAL_VALIDATION_POLICY_FILE_NAME
    if not policy_path.is_file() or policy_path.read_bytes() != payload_bytes:
        raise RuntimeError(
            "Local final-validation policy does not match its immutable cloud anchor."
        )
    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Immutable final-validation policy is invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Immutable final-validation policy is not a JSON object.")

    metadata_source_run_id = _metadata_source_run_id(metadata)
    expected_payload = _final_validation_policy_payload(
        verification_run_id=run_dir.name,
        source_run_id=metadata_source_run_id,
        datasets_gcs_prefix=str(payload.get("datasets_gcs_prefix") or "")
        .strip()
        .strip("/"),
        validations_gcs_prefix=str(payload.get("validations_gcs_prefix") or "")
        .strip()
        .strip("/"),
        verification_scope=str(
            payload.get("verification_scope") or "all"
        ),  # type: ignore[arg-type]
        source_candidates_sha256=str(
            payload.get("source_candidates_sha256") or ""
        ),
        selected_candidates_sha256=str(
            payload.get("selected_candidates_sha256") or ""
        ),
        source_candidate_count=int(payload.get("source_candidate_count") or 0),
        selected_candidate_count=int(
            payload.get("selected_candidate_count") or 0
        ),
        deterministic_routing_policy=str(
            payload.get("deterministic_routing_policy") or ""
        ),
        deterministic_routing_sha256=str(
            payload.get("deterministic_routing_sha256") or ""
        ),
        deterministic_routing_gcs_uri=str(
            payload.get("deterministic_routing_gcs_uri") or ""
        ),
        deterministic_routing_gcs_generation=str(
            payload.get("deterministic_routing_gcs_generation") or ""
        ),
        verification_provider=str(payload.get("verification_provider") or ""),
        verification_client_backend=str(
            payload.get("verification_client_backend") or ""
        ),
        verification_model=str(payload.get("verification_model") or ""),
        verification_thinking_level=str(
            payload.get("verification_thinking_level") or ""
        ),
        verification_max_output_tokens=int(
            payload.get("verification_max_output_tokens") or 0
        ),
        verification_prompt_sha256=str(
            payload.get("verification_prompt_sha256") or ""
        ),
        verification_request_contract_sha256=str(
            payload.get("verification_request_contract_sha256") or ""
        ),
        verification_request_contract_gcs_uri=str(
            payload.get("verification_request_contract_gcs_uri") or ""
        ),
        verification_request_contract_gcs_generation=str(
            payload.get("verification_request_contract_gcs_generation") or ""
        ),
        input_image_manifest_sha256=str(
            payload.get("input_image_manifest_sha256") or ""
        ),
        extraction_schema_sha256=str(
            payload.get("extraction_schema_sha256") or ""
        ),
        validation_bindings_sha256=str(
            payload.get("validation_bindings_sha256") or ""
        ),
    )
    if payload != expected_payload:
        raise RuntimeError("Immutable final-validation policy contract is invalid.")
    routing_sha256 = str(payload.get("deterministic_routing_sha256") or "")
    routing_uri = str(payload.get("deterministic_routing_gcs_uri") or "")
    routing_generation = str(
        payload.get("deterministic_routing_gcs_generation") or ""
    )
    if not routing_sha256 or not routing_uri or not routing_generation:
        raise RuntimeError(
            "Immutable final-validation policy lacks routing artifact identity."
        )
    routing_bucket_name, routing_object_name = _parse_gcs_uri(routing_uri)
    if routing_bucket_name != bucket.name:
        raise RuntimeError("Routing artifact is stored in an unexpected bucket.")
    routing_blob = bucket.blob(routing_object_name)
    routing_blob.reload()
    if str(getattr(routing_blob, "generation", "") or "") != routing_generation:
        raise RuntimeError("Immutable routing artifact generation changed.")
    routing_bytes = bytes(
        routing_blob.download_as_bytes(
            if_generation_match=int(routing_generation)
        )
    )
    if hashlib.sha256(routing_bytes).hexdigest() != routing_sha256:
        raise RuntimeError("Immutable routing artifact SHA-256 mismatch.")
    local_routing_path = run_dir / "deterministic_routing.jsonl"
    if not local_routing_path.is_file() or local_routing_path.read_bytes() != routing_bytes:
        raise RuntimeError(
            "Local deterministic routing does not match immutable cloud evidence."
        )
    request_contract_sha256 = str(
        payload.get("verification_request_contract_sha256") or ""
    )
    request_contract_uri = str(
        payload.get("verification_request_contract_gcs_uri") or ""
    )
    request_contract_generation = str(
        payload.get("verification_request_contract_gcs_generation") or ""
    )
    if (
        not request_contract_sha256
        or not request_contract_uri
        or not request_contract_generation
    ):
        raise RuntimeError(
            "Immutable final-validation policy lacks request-contract identity."
        )
    contract_bucket_name, contract_object_name = _parse_gcs_uri(
        request_contract_uri
    )
    if contract_bucket_name != bucket.name:
        raise RuntimeError(
            "Validation request contract is stored in an unexpected bucket."
        )
    contract_blob = bucket.blob(contract_object_name)
    contract_blob.reload()
    if (
        str(getattr(contract_blob, "generation", "") or "")
        != request_contract_generation
    ):
        raise RuntimeError("Immutable validation request contract generation changed.")
    contract_bytes = bytes(
        contract_blob.download_as_bytes(
            if_generation_match=int(request_contract_generation)
        )
    )
    if hashlib.sha256(contract_bytes).hexdigest() != request_contract_sha256:
        raise RuntimeError("Immutable validation request contract SHA-256 mismatch.")
    local_contract_path = run_dir / VERIFICATION_REQUEST_CONTRACT_FILE_NAME
    if (
        not local_contract_path.is_file()
        or local_contract_path.read_bytes() != contract_bytes
    ):
        raise RuntimeError(
            "Local validation request contract does not match immutable cloud evidence."
        )
    try:
        request_contract_payload = json.loads(contract_bytes)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Immutable validation request contract is invalid JSON.") from exc
    if not isinstance(request_contract_payload, dict):
        raise RuntimeError("Immutable validation request contract is not an object.")
    _validate_frozen_request_contract(
        run_dir=run_dir,
        payload=request_contract_payload,
        provider=str(payload.get("verification_provider") or ""),
        client_backend=str(payload.get("verification_client_backend") or ""),
        model=str(payload.get("verification_model") or ""),
        thinking_level=str(payload.get("verification_thinking_level") or ""),
        max_output_tokens=int(
            payload.get("verification_max_output_tokens") or 0
        ),
    )
    mutable_policy_values = {
        "verification_scope": str(metadata.get("verification_scope") or "").strip(),
        "verification_apply_mode": str(
            metadata.get("verification_apply_mode") or ""
        ).strip(),
        "correction_acceptance_policy": str(
            metadata.get("correction_acceptance_policy") or ""
        ).strip(),
        "datasets_gcs_prefix": str(
            metadata.get("datasets_gcs_prefix") or ""
        ).strip().strip("/"),
        "validations_gcs_prefix": str(
            metadata.get("validations_gcs_prefix") or ""
        ).strip().strip("/"),
        "source_candidates_sha256": str(
            metadata.get("source_candidates_sha256") or ""
        ).strip(),
        "selected_candidates_sha256": str(
            metadata.get("candidates_sha256") or ""
        ).strip(),
        "source_candidate_count": str(
            int(metadata.get("source_candidate_count") or 0)
        ),
        "selected_candidate_count": str(
            int(metadata.get("candidate_count") or 0)
        ),
        "deterministic_routing_policy": str(
            metadata.get("deterministic_routing_policy") or ""
        ).strip(),
        "deterministic_routing_sha256": str(
            metadata.get("deterministic_routing_sha256") or ""
        ).strip(),
        "deterministic_routing_gcs_uri": str(
            metadata.get("deterministic_routing_gcs_uri") or ""
        ).strip(),
        "deterministic_routing_gcs_generation": str(
            metadata.get("deterministic_routing_gcs_generation") or ""
        ).strip(),
        "verification_provider": str(metadata.get("provider") or "").strip(),
        "verification_client_backend": str(
            metadata.get("client_backend") or ""
        ).strip(),
        "verification_model": str(
            metadata.get("verification_model") or ""
        ).strip(),
        "verification_thinking_level": str(
            metadata.get("verification_thinking_level") or ""
        ).strip(),
        "verification_max_output_tokens": str(
            int(metadata.get("verification_max_output_tokens") or 0)
        ),
        "verification_prompt_sha256": str(
            metadata.get("verification_prompt_hash") or ""
        ).strip(),
        "verification_request_contract_sha256": str(
            metadata.get("verification_request_contract_sha256") or ""
        ).strip(),
        "verification_request_contract_gcs_uri": str(
            metadata.get("verification_request_contract_gcs_uri") or ""
        ).strip(),
        "verification_request_contract_gcs_generation": str(
            metadata.get("verification_request_contract_gcs_generation") or ""
        ).strip(),
        "input_image_manifest_sha256": str(
            metadata.get("input_image_manifest_sha256") or ""
        ).strip(),
        "extraction_schema_sha256": str(
            metadata.get("extraction_schema_canonical_sha256") or ""
        ).strip(),
        "validation_bindings_sha256": str(
            metadata.get("bindings_sha256") or ""
        ).strip(),
    }
    for key, mutable_value in mutable_policy_values.items():
        if mutable_value != str(payload[key]):
            raise RuntimeError(
                f"Mutable verification metadata disagrees with immutable policy: {key}."
            )
    policy_scope = str(payload.get("verification_scope") or "")
    if policy_scope not in {"all", "flagged"}:
        raise RuntimeError("Immutable final-validation policy has invalid scope.")
    return _FinalValidationPolicySnapshot(
        apply_mode=FINAL_VALIDATION_APPLY_MODE,
        acceptance_policy=AUTOMATIC_CORRECTION_ACCEPTANCE_POLICY,
        verification_scope=policy_scope,  # type: ignore[arg-type]
        source_run_id=metadata_source_run_id,
        datasets_gcs_prefix=str(payload["datasets_gcs_prefix"]),
        validations_gcs_prefix=str(payload["validations_gcs_prefix"]),
        source_candidates_sha256=str(payload["source_candidates_sha256"]),
        selected_candidates_sha256=str(payload["selected_candidates_sha256"]),
        source_candidate_count=int(payload["source_candidate_count"]),
        selected_candidate_count=int(payload["selected_candidate_count"]),
        deterministic_routing_policy=str(
            payload["deterministic_routing_policy"]
        ),
        deterministic_routing_sha256=str(
            payload["deterministic_routing_sha256"]
        ),
        deterministic_routing_gcs_uri=str(
            payload["deterministic_routing_gcs_uri"]
        ),
        deterministic_routing_gcs_generation=str(
            payload["deterministic_routing_gcs_generation"]
        ),
        verification_request_contract_sha256=str(
            payload["verification_request_contract_sha256"]
        ),
        verification_request_contract_gcs_uri=str(
            payload["verification_request_contract_gcs_uri"]
        ),
        verification_request_contract_gcs_generation=str(
            payload["verification_request_contract_gcs_generation"]
        ),
        artifact_sha256=actual_sha256,
        artifact_gcs_uri=expected_uri,
        artifact_gcs_generation=generation,
    )


def _write_submit_metadata(
    *,
    run_dir: Path,
    jobs: Sequence[Mapping[str, object]],
    provider: str,
    client_backend: str,
    model: str,
    thinking_level: str,
    scope: str,
    apply_mode: str,
    max_output_tokens: int,
    num_chunks: int,
    requested_num_chunks: int,
    source_run_dir: str,
    source_run_id: str,
    source_candidate_count: int,
    candidate_count: int,
    candidates_path: Path,
    source_candidates_path: Path,
    deterministic_routing_policy: str,
    deterministic_routing_path: Path,
    deterministic_routing_gcs_uri: str,
    deterministic_routing_gcs_generation: str,
    request_contract_path: Path,
    request_contract_gcs_uri: str,
    request_contract_gcs_generation: str,
    bindings_path: Path,
    input_manifest_path: Path,
    schema_path: Path,
    schema_name: str,
    schema_version_id: str,
    datasets_gcs_prefix: str,
    validations_gcs_prefix: str,
    final_validation_policy_path: Path,
    final_validation_policy_sha256: str,
    final_validation_policy_gcs_uri: str,
    final_validation_policy_gcs_generation: str,
    artifact_gcs_uris: Mapping[str, str],
) -> None:
    names = [
        str(job.get("batch_job_name") or "")
        for job in jobs
        if str(job.get("batch_job_name") or "")
    ]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_kind": "model_validation",
        "job_group_id": run_dir.name,
        "batch_job_name": names[0] if names else None,
        "batch_job_names": names,
        "batch_jobs": list(jobs),
        "request_count": candidate_count,
        "page_count": candidate_count,
        "provider": provider,
        "client_backend": client_backend,
        "model": model,
        "verification_model": model,
        "verification_thinking_level": thinking_level,
        "verification_scope": scope,
        "verification_apply_mode": apply_mode,
        "correction_acceptance_policy": AUTOMATIC_CORRECTION_ACCEPTANCE_POLICY,
        "verification_max_output_tokens": max_output_tokens,
        "verification_num_chunks": num_chunks,
        "verification_requested_num_chunks": requested_num_chunks,
        "output_format": str(config.output_format or "jsonl"),
        "dataset_file_name": str(config.dataset_file_name or "dataset"),
        "csv_sep": str(config.csv_sep or "$"),
        "datasets_gcs_prefix": datasets_gcs_prefix,
        "validations_gcs_prefix": validations_gcs_prefix,
        "source_run_dir": source_run_dir,
        "source_run_id": source_run_id,
        "source_candidate_count": source_candidate_count,
        "candidate_count": candidate_count,
        "candidates_file": candidates_path.name,
        "candidates_sha256": file_sha256(candidates_path),
        "source_candidates_file": source_candidates_path.name,
        "source_candidates_sha256": file_sha256(source_candidates_path),
        "deterministic_routing_policy": deterministic_routing_policy,
        "deterministic_routing_file": deterministic_routing_path.name,
        "deterministic_routing_sha256": file_sha256(
            deterministic_routing_path
        ),
        "deterministic_routing_gcs_uri": deterministic_routing_gcs_uri,
        "deterministic_routing_gcs_generation": (
            deterministic_routing_gcs_generation
        ),
        "verification_request_contract_file": request_contract_path.name,
        "verification_request_contract_sha256": file_sha256(
            request_contract_path
        ),
        "verification_request_contract_gcs_uri": request_contract_gcs_uri,
        "verification_request_contract_gcs_generation": (
            request_contract_gcs_generation
        ),
        "bindings_file": bindings_path.name,
        "bindings_sha256": file_sha256(bindings_path),
        "input_image_manifest_file": input_manifest_path.name,
        "input_image_manifest_sha256": file_sha256(input_manifest_path),
        "extraction_schema_file": schema_path.name,
        "extraction_schema_sha256": file_sha256(schema_path),
        "extraction_schema_canonical_sha256": _canonical_json_sha256(
            json.loads(schema_path.read_text(encoding="utf-8"))
        ),
        "extraction_schema_name": schema_name,
        "extraction_schema_version_id": schema_version_id,
        "verification_prompt_version": MODEL_VALIDATION_PROMPT_VERSION,
        "verification_prompt_hash": verification_prompt_hash(),
        "final_validation_policy_file": final_validation_policy_path.name,
        "final_validation_policy_sha256": final_validation_policy_sha256,
        "final_validation_policy_gcs_uri": final_validation_policy_gcs_uri,
        "final_validation_policy_gcs_generation": (
            final_validation_policy_gcs_generation
        ),
        "artifact_gcs_uris": dict(artifact_gcs_uris),
        "num_batches_requested": len(jobs),
        "num_batches_submitted": len(jobs),
    }
    (run_dir / VERIFICATION_BATCH_JOB_FILE_NAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "created_at": payload["created_at"],
                "kind": "model_validation",
                "model": model,
                "provider": provider,
                "source_run_dir": source_run_dir,
                "source_run_id": source_run_id,
                "schema_name": schema_name,
                "schema_version_id": schema_version_id,
                "verification_prompt_version": MODEL_VALIDATION_PROMPT_VERSION,
                "output_schema": json.loads(schema_path.read_text(encoding="utf-8")),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def verification_prompt_hash() -> str:
    # Hash the full static request contract, not only the prose. This makes
    # prompt-order/heading/template changes visible when comparing v1/v2 runs.
    rendered_template = build_model_validation_prompt(
        candidate={"__candidate_sentinel__": True},
        extraction_schema={
            "type": "object",
            "properties": {"__schema_sentinel__": {"type": "boolean"}},
        },
        ocr_context="0,0,1,1|__ocr_sentinel__",
    )
    payload = json.dumps(
        {
            "prompt_version": MODEL_VALIDATION_PROMPT_VERSION,
            "instructions": MODEL_VALIDATION_INSTRUCTIONS,
            "schema_heading": MODEL_VALIDATION_SCHEMA_HEADING,
            "ocr_heading": MODEL_VALIDATION_OCR_HEADING,
            "candidate_heading": MODEL_VALIDATION_CANDIDATE_HEADING,
            "rendered_template": rendered_template,
            "response_schema": PageModelValidation.model_json_schema(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def submit_model_validation(
    request: ModelValidationSubmitRequest,
) -> ModelValidationSubmitResult:
    source_run_dir = str(request.source_run_dir or "").strip()
    source_job: dict[str, Any] = {}
    source_job_path = (
        Path(source_run_dir).expanduser() / "batch_job.json"
        if source_run_dir
        else None
    )
    if source_job_path is not None and source_job_path.is_file():
        try:
            loaded_source_job = json.loads(source_job_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid extraction batch metadata: {source_job_path}"
            ) from exc
        if isinstance(loaded_source_job, dict):
            source_job = loaded_source_job

    model = str(
        request.model
        or source_job.get("verification_model")
        or config.verification_model
        or ""
    ).strip()
    spec = resolve_model_spec(model)
    if not spec.supports_batch or spec.provider not in {"gemini", "anthropic"}:
        raise ValueError(
            f"Verification model {model!r} does not have a supported batch adapter."
        )
    provider = spec.provider
    thinking_level = str(
        request.thinking_level
        or source_job.get("verification_thinking_level")
        or config.verification_thinking_level
    ).strip().lower()
    if thinking_level not in {"low", "medium", "high"}:
        raise ValueError(f"Invalid verification thinking level: {thinking_level!r}")
    scope = str(
        request.scope
        or source_job.get("verification_scope")
        or config.verification_scope
        or "flagged"
    ).strip().lower()
    if scope not in {"all", "flagged"}:
        raise ValueError(f"Invalid verification scope: {scope!r}")
    # ``verification_apply_mode`` remains readable in historical snapshots, but
    # it is no longer an input to new runs. An explicit deprecated request value
    # is accepted only when it agrees with the fixed policy.
    requested_apply_mode = str(
        request.apply_mode or FINAL_VALIDATION_APPLY_MODE
    ).strip().lower()
    if requested_apply_mode != FINAL_VALIDATION_APPLY_MODE:
        raise ValueError(
            "Final-stage model validation automatically accepts schema-valid "
            "corrections and publishes the next dataset version; report_only is "
            "not supported for new verification runs."
        )
    apply_mode = FINAL_VALIDATION_APPLY_MODE
    num_chunks = int(
        request.num_chunks
        or source_job.get("verification_num_chunks")
        or config.verification_num_chunks
        or 1
    )
    if num_chunks <= 0:
        raise ValueError("verification_num_chunks must be >= 1.")
    max_output_tokens = int(
        request.max_output_tokens
        or source_job.get("verification_max_output_tokens")
        or config.verification_max_output_tokens
        or 4096
    )
    if max_output_tokens <= 0:
        raise ValueError("verification_max_output_tokens must be >= 1.")
    if not source_run_dir:
        raise ValueError(
            "Model validation requires source_run_dir, including when "
            "candidate_file is provided."
        )

    datasets_gcs_prefix = str(
        config.datasets_gcs_prefix or "datasets"
    ).strip().strip("/")
    validations_gcs_prefix = str(
        config.validations_gcs_prefix or ""
    ).strip().strip("/")

    run_dir = _create_verification_run_dir()
    log = get_run_logger(run_dir)
    storage_client, bucket = _storage_client_and_bucket()
    candidates_path = _copy_or_download_candidate_artifact(
        request=request,
        run_dir=run_dir,
        storage_client=storage_client,
    )
    source_records = read_page_candidates(candidates_path)
    source_run_id = _validate_candidate_source_run(
        records=source_records,
        source_run_dir=source_run_dir,
    )
    input_manifest_path = _resolve_input_manifest(
        request=request,
        records=source_records,
        run_dir=run_dir,
        storage_client=storage_client,
    )
    input_records = read_input_image_manifest(input_manifest_path)
    input_by_key = {record.key: record for record in input_records}
    _validate_input_manifest_coverage(
        records=source_records,
        input_by_key=input_by_key,
        scope="all",
    )

    schema_snapshot = _resolve_extraction_schema(
        request=request,
        records=source_records,
        cache_dir=run_dir,
        storage_client=storage_client,
    )
    extraction_schema = schema_snapshot.schema
    extraction_model = _extraction_model_for_snapshot(schema_snapshot)
    schema_path = run_dir / VERIFICATION_SCHEMA_FILE_NAME
    schema_path.write_text(
        json.dumps(extraction_schema, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    routing_path = run_dir / "deterministic_routing.jsonl"
    (
        deterministic_routing_policy,
        deterministic_routing_sha256,
    ) = _validate_and_snapshot_deterministic_routing(
        records=source_records,
        extraction_model=extraction_model,
        destination=routing_path,
    )
    selected_records = _scope_candidates(source_records, scope)  # type: ignore[arg-type]
    if not selected_records:
        raise ValueError(
            "Risk-routed verification selected zero pages. Use a positive routine "
            "control sample (chosen before extraction retrieval) or all-page scope."
        )
    source_candidates_path = candidates_path
    if len(selected_records) != len(source_records):
        source_candidates_path = run_dir / SOURCE_PAGE_CANDIDATES_FILE_NAME
        shutil.copyfile(candidates_path, source_candidates_path)
        write_page_candidates(candidates_path, selected_records)
        log(
            f"Verification scope={scope} selected {len(selected_records)}/"
            f"{len(source_records)} candidate page(s)."
        )

    workers = min(
        len(selected_records), max(1, int(config.batch_ocr_workers or 1))
    )
    log(
        f"Preparing {len(selected_records)} generation-bound verification request(s) "
        f"with workers={workers}."
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        prepared = tuple(
            executor.map(
                lambda record: _prepare_validation_page(
                    record=record,
                    input_record=input_by_key[record.key],
                    bucket=bucket,
                    provider=provider,
                    run_dir_name=run_dir.name,
                    extraction_schema=extraction_schema,
                ),
                selected_records,
            )
        )

    if provider == "anthropic":
        client = _get_anthropic_client()
        for_vertex = False
        client_backend = "anthropic"
    else:
        location = str(config.vertex_model_location or config.gcp_location or "").strip()
        client = get_batch_client(location=location or None)
        for_vertex = bool(getattr(client, "vertexai", False))
        client_backend = "vertex" if for_vertex else "mldev"

    if provider == "gemini" and not for_vertex:
        prepared = _register_mldev_validation_images(
            client=client,
            storage_client=storage_client,
            pages=prepared,
        )

    bindings_path = run_dir / VERIFICATION_BINDINGS_FILE_NAME
    _write_bindings(bindings_path, prepared)

    chunks = _split_evenly(prepared, num_chunks)
    if provider == "anthropic":
        chunks = _split_anthropic_chunks_by_bytes(
            chunks,
            model=model,
            thinking_level=thinking_level,
            max_output_tokens=max_output_tokens,
        )
    else:
        chunks = _split_gemini_chunks_by_bytes(
            chunks,
            model=model,
            for_vertex=for_vertex,
            thinking_level=thinking_level,
            max_output_tokens=max_output_tokens,
        )
    requests_paths: list[Path] = []
    request_chunk_records: list[dict[str, object]] = []
    for index, chunk in enumerate(chunks, start=1):
        total = len(chunks)
        requests_path = run_dir / _chunk_file_name(index, total)
        request_rows = (
            [
                _gemini_request_line(
                    page,
                    model=model,
                    for_vertex=for_vertex,
                    thinking_level=thinking_level,
                    max_output_tokens=max_output_tokens,
                )
                for page in chunk
            ]
            if provider == "gemini"
            else [
                _anthropic_request(
                    page,
                    model=model,
                    thinking_level=thinking_level,
                    max_output_tokens=max_output_tokens,
                )
                for page in chunk
            ]
        )
        request_count, request_bytes = _write_jsonl(requests_path, request_rows)
        requests_paths.append(requests_path)
        request_chunk_records.append(
            {
                "chunk_index": index,
                "total_chunks": total,
                "requests_file": requests_path.name,
                "request_count": request_count,
                "request_bytes": request_bytes,
                "requests_sha256": file_sha256(requests_path),
            }
        )
    (
        request_contract_path,
        request_contract_payload,
        request_contract_sha256,
    ) = _write_validation_request_contract(
        run_dir=run_dir,
        provider=provider,
        client_backend=client_backend,
        model=model,
        thinking_level=thinking_level,
        max_output_tokens=max_output_tokens,
        requested_num_chunks=num_chunks,
        chunk_records=request_chunk_records,
    )
    _validate_frozen_request_contract(
        run_dir=run_dir,
        payload=request_contract_payload,
        provider=provider,
        client_backend=client_backend,
        model=model,
        thinking_level=thinking_level,
        max_output_tokens=max_output_tokens,
    )
    (
        request_contract_gcs_uri,
        request_contract_gcs_generation,
    ) = _upload_immutable_validation_artifact(
        bucket,
        run_dir,
        request_contract_path,
        sha256=request_contract_sha256,
        validations_prefix=validations_gcs_prefix,
    )
    (
        deterministic_routing_gcs_uri,
        deterministic_routing_gcs_generation,
    ) = _upload_immutable_validation_artifact(
        bucket,
        run_dir,
        routing_path,
        sha256=deterministic_routing_sha256,
        validations_prefix=validations_gcs_prefix,
    )
    # The create-only policy is written only after candidates, full manifest,
    # schema, routing, and exact request bindings have passed their local gates.
    # It binds both the complete population and the selected heavy-review set.
    (
        final_validation_policy_path,
        _final_validation_policy,
        final_validation_policy_sha256,
    ) = _write_final_validation_policy(
        run_dir=run_dir,
        source_run_dir=source_run_dir,
        datasets_gcs_prefix=datasets_gcs_prefix,
        validations_gcs_prefix=validations_gcs_prefix,
        verification_scope=scope,  # type: ignore[arg-type]
        source_candidates_sha256=file_sha256(source_candidates_path),
        selected_candidates_sha256=file_sha256(candidates_path),
        source_candidate_count=len(source_records),
        selected_candidate_count=len(selected_records),
        deterministic_routing_policy=deterministic_routing_policy,
        deterministic_routing_sha256=deterministic_routing_sha256,
        deterministic_routing_gcs_uri=deterministic_routing_gcs_uri,
        deterministic_routing_gcs_generation=(
            deterministic_routing_gcs_generation
        ),
        verification_provider=provider,
        verification_client_backend=client_backend,
        verification_model=model,
        verification_thinking_level=thinking_level,
        verification_max_output_tokens=max_output_tokens,
        verification_prompt_sha256=verification_prompt_hash(),
        verification_request_contract_sha256=request_contract_sha256,
        verification_request_contract_gcs_uri=request_contract_gcs_uri,
        verification_request_contract_gcs_generation=(
            request_contract_gcs_generation
        ),
        input_image_manifest_sha256=file_sha256(input_manifest_path),
        extraction_schema_sha256=schema_snapshot.sha256,
        validation_bindings_sha256=file_sha256(bindings_path),
    )
    (
        final_validation_policy_gcs_uri,
        final_validation_policy_gcs_generation,
    ) = _upload_final_validation_policy(
        bucket=bucket,
        run_dir=run_dir,
        policy_path=final_validation_policy_path,
        policy_sha256=final_validation_policy_sha256,
    )
    artifact_gcs_uris = {
        candidates_path.name: _upload_validation_artifact(
            bucket,
            run_dir,
            candidates_path,
            validations_prefix=validations_gcs_prefix,
        ),
        bindings_path.name: _upload_validation_artifact(
            bucket,
            run_dir,
            bindings_path,
            validations_prefix=validations_gcs_prefix,
        ),
        input_manifest_path.name: _upload_validation_artifact(
            bucket,
            run_dir,
            input_manifest_path,
            validations_prefix=validations_gcs_prefix,
        ),
        schema_path.name: _upload_validation_artifact(
            bucket,
            run_dir,
            schema_path,
            validations_prefix=validations_gcs_prefix,
        ),
        routing_path.name: deterministic_routing_gcs_uri,
        request_contract_path.name: request_contract_gcs_uri,
        final_validation_policy_path.name: final_validation_policy_gcs_uri,
    }
    if source_candidates_path != candidates_path:
        artifact_gcs_uris[source_candidates_path.name] = (
            _upload_validation_artifact(
                bucket,
                run_dir,
                source_candidates_path,
                validations_prefix=validations_gcs_prefix,
            )
        )
    verification_metadata_path = run_dir / VERIFICATION_BATCH_JOB_FILE_NAME
    run_metadata_path = run_dir / "metadata.json"
    artifact_gcs_uris[verification_metadata_path.name] = (
        _validation_artifact_location(
            bucket,
            run_dir,
            verification_metadata_path,
            validations_prefix=validations_gcs_prefix,
        )[0]
    )
    artifact_gcs_uris[run_metadata_path.name] = _validation_artifact_location(
        bucket,
        run_dir,
        run_metadata_path,
        validations_prefix=validations_gcs_prefix,
    )[0]

    jobs: list[dict[str, object]] = []
    for index, chunk in enumerate(chunks, start=1):
        total = len(chunks)
        label = f"chunk_{index:03d}_of_{total:03d}"
        requests_path = requests_paths[index - 1]
        request_contract_chunk = request_chunk_records[index - 1]
        if (
            file_sha256(requests_path)
            != str(request_contract_chunk["requests_sha256"])
        ):
            raise RuntimeError(
                f"Validation request bytes changed before submission: "
                f"{requests_path.name}."
            )
        request_count = int(request_contract_chunk["request_count"])
        request_bytes = int(request_contract_chunk["request_bytes"])
        if provider == "gemini":
            if for_vertex:
                input_reference = _upload_requests_to_gcs(
                    bucket=bucket,
                    run_dir_name=run_dir.name,
                    local_requests_path=requests_path,
                )
                output_destination = _output_dest_gcs_uri(
                    bucket.name,
                    run_dir.name,
                    chunk_label=f"validation/{label}",
                )
                job = client.batches.create(
                    model=model,
                    src=input_reference,
                    config=types.CreateBatchJobConfig(
                        display_name=f"patientjournals-verify-{label}",
                        dest=output_destination,
                    ),
                )
                batch_name = str(job.name)
                input_source = "gcs"
            else:
                uploaded = client.files.upload(
                    file=str(requests_path),
                    config=types.UploadFileConfig(
                        display_name=f"patientjournals-verify-{label}",
                        mime_type="jsonl",
                    ),
                )
                input_reference = str(uploaded.name)
                output_destination = None
                job = client.batches.create(
                    model=model,
                    src=uploaded.name,
                    config=types.CreateBatchJobConfig(
                        display_name=f"patientjournals-verify-{label}"
                    ),
                )
                batch_name = str(job.name)
                input_source = "gemini_files"
        else:
            # Submit the exact policy-bound rows rather than regenerating prompts
            # or model parameters after the immutable contract was anchored.
            requests = _read_jsonl_request_rows(requests_path)
            input_reference = _upload_requests_to_gcs(
                bucket=bucket,
                run_dir_name=run_dir.name,
                local_requests_path=requests_path,
            )
            output_destination = None
            job = client.messages.batches.create(requests=requests)
            batch_name = str(job.id)
            input_source = "anthropic_manifest"

        artifact_gcs_uris[requests_path.name] = (
            input_reference
            if input_reference.startswith("gs://")
            else _upload_validation_artifact(
                bucket,
                run_dir,
                requests_path,
                validations_prefix=validations_gcs_prefix,
            )
        )
        jobs.append(
            {
                "chunk_index": index,
                "total_chunks": total,
                "chunk_label": label,
                "requests_file": requests_path.name,
                "request_count": request_count,
                "request_bytes": request_bytes,
                "provider_batch_byte_limit": (
                    ANTHROPIC_MAX_BATCH_REQUEST_BYTES
                    if provider == "anthropic"
                    else GEMINI_MAX_BATCH_REQUEST_BYTES
                ),
                "provider_batch_request_limit": (
                    ANTHROPIC_MAX_BATCH_REQUESTS
                    if provider == "anthropic"
                    else None
                ),
                "requests_sha256": request_contract_chunk["requests_sha256"],
                "batch_job_name": batch_name,
                "input_file": input_reference,
                "input_source": input_source,
                "output_destination": output_destination,
                "provider": provider,
            }
        )
        _write_submit_metadata(
            run_dir=run_dir,
            jobs=jobs,
            provider=provider,
            client_backend=client_backend,
            model=model,
            thinking_level=thinking_level,
            scope=scope,
            apply_mode=apply_mode,
            max_output_tokens=max_output_tokens,
            num_chunks=len(chunks),
            requested_num_chunks=num_chunks,
            source_run_dir=source_run_dir,
            source_run_id=source_run_id,
            source_candidate_count=len(source_records),
            candidate_count=len(selected_records),
            candidates_path=candidates_path,
            source_candidates_path=source_candidates_path,
            deterministic_routing_policy=deterministic_routing_policy,
            deterministic_routing_path=routing_path,
            deterministic_routing_gcs_uri=deterministic_routing_gcs_uri,
            deterministic_routing_gcs_generation=(
                deterministic_routing_gcs_generation
            ),
            request_contract_path=request_contract_path,
            request_contract_gcs_uri=request_contract_gcs_uri,
            request_contract_gcs_generation=(
                request_contract_gcs_generation
            ),
            bindings_path=bindings_path,
            input_manifest_path=input_manifest_path,
            schema_path=schema_path,
            schema_name=schema_snapshot.name,
            schema_version_id=schema_snapshot.version_id,
            datasets_gcs_prefix=datasets_gcs_prefix,
            validations_gcs_prefix=validations_gcs_prefix,
            final_validation_policy_path=final_validation_policy_path,
            final_validation_policy_sha256=final_validation_policy_sha256,
            final_validation_policy_gcs_uri=final_validation_policy_gcs_uri,
            final_validation_policy_gcs_generation=(
                final_validation_policy_gcs_generation
            ),
            artifact_gcs_uris=artifact_gcs_uris,
        )
        # Upload metadata after every provider chunk so a partially submitted
        # multi-chunk run is still discoverable and recoverable from cloud.
        _upload_validation_artifact(
            bucket,
            run_dir,
            verification_metadata_path,
            validations_prefix=validations_gcs_prefix,
        )
        _upload_validation_artifact(
            bucket,
            run_dir,
            run_metadata_path,
            validations_prefix=validations_gcs_prefix,
        )
        log(f"Submitted verification {label}: {batch_name}")

    log(
        f"Submitted {len(jobs)} model-validation batch job(s) for "
        f"{len(selected_records)} page(s) with model={model}."
    )
    return ModelValidationSubmitResult(
        run_dir=run_dir,
        provider=provider,
        model=model,
        candidate_count=len(selected_records),
        source_candidate_count=len(source_records),
        batch_job_names=tuple(str(job["batch_job_name"]) for job in jobs),
        requests_paths=tuple(requests_paths),
        candidates_path=candidates_path,
        bindings_path=bindings_path,
        input_manifest_path=input_manifest_path,
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _verify_run_artifact_hashes(run_dir: Path, metadata: Mapping[str, Any]) -> None:
    checks = (
        ("candidates_file", "candidates_sha256"),
        ("source_candidates_file", "source_candidates_sha256"),
        ("bindings_file", "bindings_sha256"),
        ("input_image_manifest_file", "input_image_manifest_sha256"),
        ("extraction_schema_file", "extraction_schema_sha256"),
    )
    for file_key, digest_key in checks:
        file_name = str(metadata.get(file_key) or "").strip()
        expected = str(metadata.get(digest_key) or "").strip()
        if not file_name or not expected:
            raise ValueError(f"Verification metadata is missing {file_key}/{digest_key}.")
        path = run_dir / file_name
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"Verification artifact hash mismatch: {path}")
    policy_fields = (
        str(metadata.get("final_validation_policy_file") or "").strip(),
        str(metadata.get("final_validation_policy_sha256") or "").strip(),
        str(metadata.get("final_validation_policy_gcs_uri") or "").strip(),
        str(
            metadata.get("final_validation_policy_gcs_generation") or ""
        ).strip(),
    )
    if any(policy_fields):
        if not all(policy_fields):
            raise ValueError(
                "Verification metadata has an incomplete final-validation "
                "policy binding."
            )
        policy_file, policy_sha256, _, _ = policy_fields
        if policy_file != FINAL_VALIDATION_POLICY_FILE_NAME:
            raise RuntimeError("Unexpected final-validation policy file name.")
        policy_path = run_dir / policy_file
        if not policy_path.is_file() or file_sha256(policy_path) != policy_sha256:
            raise RuntimeError(
                "Local final-validation policy hash does not match submission metadata."
            )
    jobs = metadata.get("batch_jobs")
    if not isinstance(jobs, list):
        raise ValueError("Verification metadata is missing batch_jobs.")
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("Verification metadata contains an invalid batch job.")
        file_name = str(job.get("requests_file") or "").strip()
        expected = str(job.get("requests_sha256") or "").strip()
        path = run_dir / file_name
        if not file_name or not expected or not path.is_file():
            raise ValueError("Verification batch request hash metadata is incomplete.")
        if file_sha256(path) != expected:
            raise RuntimeError(f"Verification request artifact hash mismatch: {path}")


def _read_bindings(path: Path) -> dict[str, dict[str, Any]]:
    bindings: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid binding at {path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid binding at {path}:{line_number}")
            key = str(payload.get("key") or "").strip()
            if not key or key in bindings:
                raise ValueError(f"Missing/duplicate binding key at {path}:{line_number}")
            bindings[key] = payload
    return bindings


def _verify_retrieval_evidence_bindings(
    *,
    bucket: storage.Bucket,
    provider: str,
    records: Sequence[PageCandidateRecord],
    input_by_key: Mapping[str, InputImageManifestRecord],
    bindings: Mapping[str, Mapping[str, Any]],
) -> None:
    """Recheck cloud evidence after batch completion, before trusting output."""

    def check(record: PageCandidateRecord) -> None:
        input_record = input_by_key.get(record.key)
        binding = bindings.get(record.key)
        if input_record is None or binding is None:
            raise RuntimeError(f"Missing validation evidence binding for {record.key!r}.")

        bound_extraction = binding.get("extraction_image_source")
        if not isinstance(bound_extraction, dict):
            raise RuntimeError(f"Invalid extraction image binding for {record.key!r}.")
        if not input_record.source.matches(
            CloudBlobIdentity.from_dict(bound_extraction)
        ):
            raise RuntimeError(f"Extraction image binding mismatch for {record.key!r}.")
        bound_ocr_enabled = binding.get("ocr_enabled")
        if bound_ocr_enabled is not None and bool(bound_ocr_enabled) != bool(
            input_record.ocr_enabled
        ):
            raise RuntimeError(
                f"Validation evidence binding ocr_enabled mismatch for {record.key!r}."
            )
        scalar_checks: dict[str, object] = {}
        if input_record.ocr_enabled:
            scalar_checks.update(
                {
                    "ocr_sidecar_name": input_record.ocr_sidecar_name,
                    "ocr_sidecar_sha256": input_record.ocr_sidecar_sha256,
                    "ocr_image_sha256": input_record.ocr_image_sha256,
                    "ocr_document_sha256": input_record.ocr_document_sha256,
                    "ocr_backend": input_record.ocr_backend,
                    "ocr_line_count": input_record.ocr_line_count,
                }
            )
        for field, expected in scalar_checks.items():
            if binding.get(field) != expected:
                raise RuntimeError(
                    f"Validation evidence binding {field} mismatch for {record.key!r}."
                )
        if input_record.ocr_enabled:
            bound_sidecar = binding.get("ocr_sidecar_source")
            if not isinstance(
                bound_sidecar, dict
            ) or not input_record.sidecar_source.matches(
                CloudBlobIdentity.from_dict(bound_sidecar)
            ):
                raise RuntimeError(f"OCR sidecar binding mismatch for {record.key!r}.")

            # Reload the current source object and sidecar and verify the exact
            # image, OCR document, sidecar generation, and sidecar bytes.
            _load_bound_ocr_evidence(bucket=bucket, input_record=input_record)
        elif binding.get("ocr_sidecar_source") not in ({}, None):
            raise RuntimeError(
                f"Unexpected OCR evidence attached to OCR-disabled page {record.key!r}."
            )

        raw_request_source = binding.get("request_image_source")
        if not isinstance(raw_request_source, dict):
            raise RuntimeError(f"Invalid request image binding for {record.key!r}.")
        request_source = CloudBlobIdentity.from_dict(raw_request_source)
        if provider == "gemini":
            # Vertex fileData GCS URIs are unversioned. MLDev Files API URIs
            # point back to this staged object. Requiring its *latest* identity
            # to equal the submitted identity catches an overwrite at the same
            # object name while the batch was running.
            staged_blob = bucket.blob(request_source.name)
            reload_staged = getattr(staged_blob, "reload", None)
            if callable(reload_staged):
                reload_staged()
            current_staged = CloudBlobIdentity.from_blob(staged_blob)
            if not request_source.matches(current_staged):
                raise RuntimeError(
                    f"Staged Gemini image changed after submission for {record.key!r}."
                )
            staged_uri = str(
                binding.get("staged_image_uri")
                or binding.get("request_image_uri")
                or ""
            )
            bucket_name, object_name = _parse_gcs_uri(staged_uri)
            if bucket_name != request_source.bucket or object_name != request_source.name:
                raise RuntimeError(
                    f"Staged Gemini image URI binding mismatch for {record.key!r}."
                )
        elif provider == "anthropic":
            if not input_record.source.matches(request_source):
                raise RuntimeError(
                    f"Anthropic generation-qualified image binding mismatch for {record.key!r}."
                )
        else:
            raise ValueError(f"Unsupported verification provider: {provider!r}")

    workers = min(len(records), max(1, int(config.batch_ocr_workers or 1)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        tuple(executor.map(check, records))


def _extract_anthropic_text(record: Mapping[str, Any]) -> tuple[str | None, str | None]:
    result = record.get("result")
    if not isinstance(result, dict):
        return None, "missing_result"
    result_type = str(result.get("type") or "").strip().lower()
    if result_type != "succeeded":
        return None, f"batch_{result_type or 'unknown'}"
    message = result.get("message")
    stop_reason = (
        str(message.get("stop_reason") or "").strip().lower()
        if isinstance(message, dict)
        else str(getattr(message, "stop_reason", "") or "").strip().lower()
    )
    if stop_reason != "end_turn":
        return None, f"stop_reason_{stop_reason or 'missing'}"
    metadata = _extract_anthropic_response_metadata(message)
    text = metadata.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, "empty_response_text"
    return text, None


def _extract_gemini_text(record: Mapping[str, Any]) -> tuple[str | None, str | None]:
    if record.get("error"):
        return None, "batch_error"
    response = record.get("response")
    if response is None:
        return None, "missing_response"
    candidates = (
        response.get("candidates") if isinstance(response, dict) else None
    )
    if not isinstance(candidates, list) or not candidates:
        return None, "missing_candidates"
    first = candidates[0]
    finish_reason = (
        str(first.get("finishReason") or first.get("finish_reason") or "")
        if isinstance(first, dict)
        else str(
            getattr(first, "finish_reason", None)
            or getattr(first, "finishReason", None)
            or ""
        )
    ).strip().upper()
    if "." in finish_reason:
        finish_reason = finish_reason.rsplit(".", 1)[-1]
    if finish_reason != "STOP":
        return None, f"finish_reason_{finish_reason.lower() or 'missing'}"
    metadata = extract_response_metadata(response)
    text = metadata.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, "empty_response_text"
    return text, None


def _decode_pointer(path: str) -> list[str]:
    if not path.startswith("/"):
        raise ValueError(f"Patch path is not an RFC 6901 pointer: {path!r}")
    return [part.replace("~1", "/").replace("~0", "~") for part in path[1:].split("/")]


def _array_index(token: str, *, path: str) -> int:
    if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
        raise ValueError(f"Invalid RFC 6902 array index in patch: {path}")
    return int(token)


def _apply_one_patch(document: Any, patch: Any) -> None:
    tokens = _decode_pointer(patch.path)
    if not tokens or tokens == [""]:
        raise ValueError("Root-level validation patches are not supported.")
    parent = document
    for token in tokens[:-1]:
        if isinstance(parent, dict):
            if token not in parent:
                raise ValueError(f"Patch parent does not exist: {patch.path}")
            parent = parent[token]
        elif isinstance(parent, list):
            try:
                parent = parent[_array_index(token, path=patch.path)]
            except IndexError as exc:
                raise ValueError(f"Invalid list path: {patch.path}") from exc
        else:
            raise ValueError(f"Patch parent is not a container: {patch.path}")

    token = tokens[-1]
    value = json.loads(patch.value_json) if patch.value_json is not None else None
    if isinstance(parent, dict):
        exists = token in parent
        if patch.op == "add":
            if exists:
                raise ValueError(
                    f"Add patch must target a missing object member: {patch.path}"
                )
            parent[token] = value
        elif patch.op == "replace":
            if not exists:
                raise ValueError(f"Replace path does not exist: {patch.path}")
            parent[token] = value
        else:
            if not exists:
                raise ValueError(f"Remove path does not exist: {patch.path}")
            del parent[token]
        return
    if isinstance(parent, list):
        if patch.op == "add" and token == "-":
            parent.append(value)
            return
        index = _array_index(token, path=patch.path)
        if patch.op == "add":
            if index < 0 or index > len(parent):
                raise ValueError(f"Add list index out of range: {patch.path}")
            parent.insert(index, value)
        elif patch.op == "replace":
            if index < 0 or index >= len(parent):
                raise ValueError(f"Replace list index out of range: {patch.path}")
            parent[index] = value
        else:
            if index < 0 or index >= len(parent):
                raise ValueError(f"Remove list index out of range: {patch.path}")
            del parent[index]
        return
    raise ValueError(f"Patch target parent is not a container: {patch.path}")


def apply_validation_patches(
    candidate: Mapping[str, Any],
    validation: PageModelValidation,
) -> dict[str, Any]:
    patched = copy.deepcopy(dict(candidate))
    for patch in validation.patches:
        _apply_one_patch(patched, patch)
    return patched


def _pointer_snapshot(document: Any, path: str) -> tuple[bool, Any]:
    current = document
    for token in _decode_pointer(path):
        if isinstance(current, dict):
            if token not in current:
                return False, None
            current = current[token]
            continue
        if isinstance(current, list):
            if token == "-":
                return False, None
            try:
                index = _array_index(token, path=path)
            except ValueError:
                return False, None
            if index >= len(current):
                return False, None
            current = current[index]
            continue
        return False, None
    return True, copy.deepcopy(current)


def field_correction_records(
    candidate: Mapping[str, Any],
    validation: PageModelValidation,
    *,
    corrections_applied: bool,
    included_in_corrected_dataset: bool,
) -> list[dict[str, Any]]:
    """Describe each proposed patch and whether it was actually applied."""

    working = copy.deepcopy(dict(candidate))
    records: list[dict[str, Any]] = []
    for patch in validation.patches:
        original_exists, original_value = _pointer_snapshot(working, patch.path)
        _apply_one_patch(working, patch)
        proposed_exists, proposed_value = _pointer_snapshot(working, patch.path)
        pointer_tokens = _decode_pointer(patch.path)
        record: dict[str, Any] = {
            "path": patch.path,
            "top_level_field": pointer_tokens[0] if pointer_tokens else "",
            "operation": patch.op,
            "issue": patch.issue,
            "accepted": bool(corrections_applied),
            "applied": bool(corrections_applied),
            "corrected": bool(included_in_corrected_dataset),
            "included_in_corrected_dataset": bool(
                included_in_corrected_dataset
            ),
            "acceptance": (
                "automatic_schema_validated"
                if corrections_applied
                else "proposed_only"
            ),
            "original_exists": original_exists,
            "proposed_exists": proposed_exists,
            "evidence": patch.evidence,
            "ocr_box_refs": list(patch.ocr_box_refs),
        }
        if original_exists:
            record["original_value"] = original_value
        if proposed_exists:
            record["proposed_value"] = proposed_value
        records.append(record)
    return records


def build_field_correction_metadata(
    *,
    run_dir: Path,
    candidates_by_key: Mapping[str, PageCandidateRecord],
    validations: Mapping[str, PageModelValidation],
    failures: Sequence[Mapping[str, object]],
    model: str,
    provider: str,
    apply_mode: str,
    candidate_hash: str,
    verification_prompt_hash: str,
    created_at: str = "",
    verification_prompt_version: str = "",
    acceptance_policy: str = "",
    extraction_schema_name: str = "",
    extraction_schema_version_id: str = "",
    extraction_schema_sha256: str = "",
    input_image_manifest_sha256: str = "",
    validation_bindings_sha256: str = "",
    source_run_id: str = "",
    final_validation_policy_sha256: str = "",
    final_validation_policy_gcs_uri: str = "",
    final_validation_policy_gcs_generation: str = "",
    verification_artifact_sha256s: Mapping[str, str] | None = None,
    included_in_corrected_dataset: bool = False,
    corrected_dataset_sha256: str = "",
    deterministically_cleared_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """Build one auditable record per expected page, including failed pages."""

    corrections_applied = apply_mode == "apply_patches"
    failure_reasons: dict[str, list[str]] = {}
    deterministic_keys = set(deterministically_cleared_keys)
    for failure in failures:
        key = str(failure.get("key") or "")
        if key:
            failure_reasons.setdefault(key, []).append(
                str(failure.get("reason") or "validation_failed")
            )

    pages: list[dict[str, Any]] = []
    proposed_fields = 0
    corrected_fields = 0
    proposed_pages = 0
    corrected_pages = 0
    for key in sorted(candidates_by_key):
        validation = validations.get(key)
        original_candidate = candidates_by_key[key].candidate
        fields = (
            field_correction_records(
                original_candidate,
                validation,
                corrections_applied=corrections_applied,
                included_in_corrected_dataset=included_in_corrected_dataset,
            )
            if validation is not None
            else []
        )
        proposed = bool(fields)
        accepted = proposed and corrections_applied
        corrected = proposed and included_in_corrected_dataset
        proposed_fields += len(fields)
        corrected_fields += len(fields) if corrected else 0
        proposed_pages += int(proposed)
        corrected_pages += int(corrected)
        corrected_candidate = (
            apply_validation_patches(original_candidate, validation)
            if validation is not None
            else (dict(original_candidate) if key in deterministic_keys else None)
        )
        pages.append(
            {
                "key": key,
                "original_candidate_sha256": candidate_sha256(original_candidate),
                "result_candidate_sha256": (
                    candidate_sha256(corrected_candidate)
                    if corrected_candidate is not None
                    else None
                ),
                "corrected_candidate_sha256": (
                    candidate_sha256(corrected_candidate)
                    if corrected_candidate is not None and corrected
                    else None
                ),
                "page_status": (
                    validation.page_status
                    if validation is not None
                    else (
                        "deterministic_cleared"
                        if key in deterministic_keys
                        else "validation_failed"
                    )
                ),
                "correction_proposed": proposed,
                "correction_accepted": accepted,
                "included_in_corrected_dataset": bool(
                    included_in_corrected_dataset
                    and (validation is not None or key in deterministic_keys)
                ),
                "corrected": corrected,
                "corrected_field_count": len(fields) if corrected else 0,
                "fields": fields,
                "failure_reasons": sorted(set(failure_reasons.get(key, []))),
            }
        )

    return {
        "schema_version": FIELD_CORRECTION_METADATA_SCHEMA_VERSION,
        # Use the immutable verification submission time rather than retrieval
        # time so replaying the same batch produces identical correction bytes.
        "created_at": str(created_at or ""),
        "job_kind": "field_correction_metadata",
        # Local run paths differ between workers; the directory name is the
        # portable run identity used by the publication ledger.
        "verification_run_id": run_dir.name,
        "source_run_id": source_run_id,
        "verification_model": model,
        "verification_provider": provider,
        "apply_mode": apply_mode,
        "acceptance_policy": str(
            acceptance_policy
            or (
                AUTOMATIC_CORRECTION_ACCEPTANCE_POLICY
                if corrections_applied
                else "legacy_report_only"
            )
        ),
        "automatic_acceptance": bool(corrections_applied),
        "acceptance_basis": (
            "Each accepted patch passed the verifier response schema, applied "
            "cleanly, and the resulting page passed the original extraction schema."
        ),
        "corrections_applied": corrections_applied,
        # This content-addressed artifact is finalized before the shared
        # publisher allocates vNNN. It attests to the complete corrected bytes;
        # dataset_versions.json is the atomic authority for cloud publication.
        "corrected_dataset_built": bool(included_in_corrected_dataset),
        "corrected_dataset_sha256": str(corrected_dataset_sha256 or ""),
        "dataset_version_publication_authority": "dataset_versions.json",
        "candidate_hash": candidate_hash,
        "extraction_schema": {
            "name": extraction_schema_name,
            "version_id": extraction_schema_version_id,
            "canonical_sha256": extraction_schema_sha256,
        },
        "evidence_artifacts": {
            "input_image_manifest_sha256": input_image_manifest_sha256,
            "validation_bindings_sha256": validation_bindings_sha256,
            "final_validation_policy": {
                "sha256": final_validation_policy_sha256,
                "gcs_uri": final_validation_policy_gcs_uri,
                "gcs_generation": final_validation_policy_gcs_generation,
            },
            "verification_artifact_sha256s": dict(
                sorted((verification_artifact_sha256s or {}).items())
            ),
        },
        "verification_prompt_version": str(
            verification_prompt_version or "legacy_unversioned"
        ),
        "verification_prompt_hash": verification_prompt_hash,
        "expected_pages": len(candidates_by_key),
        "completed_pages": len(validations) + len(deterministic_keys),
        "model_reviewed_pages": len(validations),
        "deterministically_cleared_pages": len(deterministic_keys),
        "proposed_correction_pages": proposed_pages,
        "proposed_correction_fields": proposed_fields,
        "corrected_pages": corrected_pages,
        "corrected_fields": corrected_fields,
        "accepted_correction_pages": proposed_pages if corrections_applied else 0,
        "accepted_correction_fields": proposed_fields if corrections_applied else 0,
        "pages": pages,
    }


def _upload_retrieval_artifacts(
    *,
    bucket: storage.Bucket,
    run_dir: Path,
    paths: Sequence[Path],
    validations_prefix: str | None = None,
) -> dict[str, str]:
    return {
        path.name: _upload_validation_artifact(
            bucket,
            run_dir,
            path,
            validations_prefix=validations_prefix,
        )
        for path in paths
        if path.exists()
    }


def _write_patched_candidates(
    *,
    path: Path,
    verification_run_id: str,
    candidates_by_key: Mapping[str, PageCandidateRecord],
    patched_by_key: Mapping[str, dict[str, Any]],
    validations: Mapping[str, PageModelValidation],
    model: str,
    provider: str,
    deterministically_cleared_keys: Sequence[str] = (),
) -> Path:
    """Write portable patched candidates without machine-local verifier paths."""

    with PageCandidateWriter(path) as writer:
        deterministic_keys = set(deterministically_cleared_keys)
        for key in sorted(candidates_by_key):
            original = candidates_by_key[key]
            extraction_metadata = dict(original.extraction_metadata)
            extraction_metadata.pop("verification_run_dir", None)
            validation = validations.get(key)
            extraction_metadata["verification_status"] = (
                validation.page_status
                if validation is not None
                else (
                    "deterministic_cleared"
                    if key in deterministic_keys
                    else "validation_failed"
                )
            )
            extraction_metadata["verification_run_id"] = verification_run_id
            if validation is not None:
                extraction_metadata["verification_model"] = model
                extraction_metadata["verification_provider"] = provider
            writer.write(
                key=key,
                candidate=patched_by_key[key],
                extraction_metadata=extraction_metadata,
            )
    return path


def _build_verified_dataset(
    *,
    run_dir: Path,
    candidates_by_key: Mapping[str, PageCandidateRecord],
    patched_by_key: Mapping[str, dict[str, Any]],
    validations: Mapping[str, PageModelValidation],
    extraction_model: type,
    extraction_schema_name: str,
    verification_model: str,
    verification_provider: str,
    output_format: str,
    dataset_file_name: str,
    csv_sep: str,
) -> tuple[Path, int]:
    suffix = output_format.lstrip(".") or "jsonl"
    path = run_dir / f"{run_dir.name}_{dataset_file_name}.{suffix}"
    path.unlink(missing_ok=True)
    header_written = False
    total_rows = 0
    for key in sorted(candidates_by_key):
        candidate = candidates_by_key[key]
        parsed = _validate_extraction_candidate(
            extraction_model, patched_by_key[key]
        )
        rows = data_to_rows(
            parsed,
            file_name=key,
            schema_name=extraction_schema_name,
        )
        for row in rows:
            row["failed"] = False
            row["failure_reason"] = None
            row["model"] = str(candidate.extraction_metadata.get("model") or "")
            row["provider"] = str(candidate.extraction_metadata.get("provider") or "")
            row["schema_name"] = str(
                candidate.extraction_metadata.get("schema_name") or ""
            )
            row["schema_version_id"] = str(
                candidate.extraction_metadata.get("schema_version_id") or ""
            )
            validation = validations.get(key)
            row["verification_model"] = (
                verification_model if validation is not None else ""
            )
            row["verification_provider"] = (
                verification_provider if validation is not None else ""
            )
            row["verification_status"] = (
                validation.page_status
                if validation is not None
                else "deterministic_cleared"
            )
        if rows:
            header_written = flush_rows(
                rows=rows,
                out_path=str(path),
                header_written=header_written,
                output_format=suffix,
                sep=csv_sep,
            )
            total_rows += len(rows)
    if not path.exists():
        # A schema can validly flatten to zero rows (for example an empty
        # TextPage); create an explicit empty artifact rather than publishing a
        # missing file.
        path.touch()
    return path, total_rows


def _model_validation_outcome(
    *,
    expected_pages: int,
    completed_pages: int,
    failure_records: int,
    unverifiable_pages: int,
    apply_mode: str,
    scope: str,
) -> tuple[bool, bool, str]:
    complete = completed_pages == expected_pages and failure_records == 0
    publishable = (
        apply_mode == "apply_patches" and complete and unverifiable_pages == 0
    )
    if not complete:
        status = "incomplete"
    elif unverifiable_pages:
        status = "unverifiable"
    elif apply_mode == "report_only":
        status = "report_only_complete"
    else:
        status = "publishable"
    return complete, publishable, status


def _recorded_correction_policy(
    metadata: Mapping[str, object],
    *,
    immutable_policy: _FinalValidationPolicySnapshot | None = None,
) -> tuple[VerificationApplyMode, str]:
    """Resolve immutable run policy without accepting retrieval-time promotion."""

    if immutable_policy is not None and not immutable_policy.legacy:
        return immutable_policy.apply_mode, immutable_policy.acceptance_policy

    # batch_job.json was historically mutable. Without the independent cloud
    # policy anchor, its fields are valid report metadata but never sufficient
    # authority to apply corrections or publish a dataset.
    return "report_only", "legacy_report_only"


def retrieve_model_validation(
    request: ModelValidationRetrieveRequest,
) -> ModelValidationRetrieveResult:
    run_dir = Path(request.run_dir).expanduser()
    metadata_path = run_dir / VERIFICATION_BATCH_JOB_FILE_NAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Verification batch metadata not found: {metadata_path}")
    metadata = _read_json_object(metadata_path)
    if metadata.get("job_kind") != "model_validation":
        raise ValueError(f"Run is not a model-validation batch: {run_dir}")
    _verify_run_artifact_hashes(run_dir, metadata)
    log = get_run_logger(run_dir)
    _, bucket = _storage_client_and_bucket()
    policy_snapshot = _resolve_final_validation_policy(
        bucket=bucket,
        run_dir=run_dir,
        metadata=metadata,
    )
    validations_gcs_prefix = policy_snapshot.validations_gcs_prefix
    datasets_gcs_prefix = policy_snapshot.datasets_gcs_prefix
    provider = str(metadata.get("provider") or "").strip().lower()
    model = str(metadata.get("verification_model") or metadata.get("model") or "")
    if provider == "anthropic":
        client = _get_anthropic_client()
    elif provider == "gemini":
        names = [str(name) for name in metadata.get("batch_job_names") or []]
        location = (
            _extract_location_from_batch_name(names[0]) if names else None
        ) or str(config.vertex_model_location or config.gcp_location or "").strip()
        client = get_batch_client(location=location or None)
        expected_backend = str(metadata.get("client_backend") or "").strip().lower()
        current_backend = "vertex" if bool(getattr(client, "vertexai", False)) else "mldev"
        if expected_backend not in {"vertex", "mldev"}:
            raise ValueError("Verification metadata has no valid Gemini client_backend.")
        if current_backend != expected_backend:
            raise RuntimeError(
                "Current Gemini batch backend does not match the submitted "
                f"verification run ({current_backend!r} != {expected_backend!r})."
            )
    else:
        raise ValueError(f"Unsupported verification provider in metadata: {provider!r}")

    jobs = metadata.get("batch_jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Verification metadata has no batch jobs.")
    downloadable: list[tuple[dict[str, Any], object]] = []
    incomplete: list[tuple[str, str]] = []
    for raw_job in jobs:
        if not isinstance(raw_job, dict):
            continue
        batch_name = str(raw_job.get("batch_job_name") or "").strip()
        if not batch_name:
            continue
        batch_job = _get_batch_job(client, batch_name, provider)
        if not _batch_job_successful(batch_job, provider) and request.wait:
            batch_job = _await_completion(client, batch_name, provider, log)
        if not _batch_job_successful(batch_job, provider):
            incomplete.append((batch_name, _batch_job_state(batch_job, provider)))
            continue
        downloadable.append((raw_job, batch_job))
    if incomplete and not request.allow_partial:
        detail = ", ".join(f"{name}={state}" for name, state in incomplete)
        raise RuntimeError(f"Verification batch jobs are not all successful: {detail}")
    if not downloadable:
        raise RuntimeError("No completed verification batch output is available.")

    raw_paths: list[Path] = []
    for index, (job_meta, batch_job) in enumerate(downloadable, start=1):
        batch_name = str(job_meta["batch_job_name"])
        raw_path = run_dir / f"validation_raw_{index:03d}.jsonl"
        if provider == "anthropic":
            _download_from_anthropic_output(client, batch_name, raw_path, log)
        else:
            reference = _gemini_output_reference(
                batch_job,
                metadata_destination=str(job_meta.get("output_destination") or "") or None,
            )
            if reference is None:
                raise RuntimeError(f"No output reference for verification batch {batch_name}.")
            reference_type, reference_value = reference
            if reference_type == "gcs":
                _download_from_vertex_gcs_output(reference_value, raw_path, log)
            else:
                _download_from_mldev_output(client, reference_value, raw_path)
        raw_paths.append(raw_path)

    candidates_path = run_dir / str(metadata["candidates_file"])
    records = read_page_candidates(candidates_path)
    source_candidates_path = run_dir / str(
        metadata.get("source_candidates_file") or metadata["candidates_file"]
    )
    source_records = read_page_candidates(source_candidates_path)
    expected_selected = _scope_candidates(
        source_records, policy_snapshot.verification_scope
    )
    if {record.key for record in expected_selected} != {record.key for record in records}:
        raise RuntimeError(
            "Persisted heavy-review candidates no longer match the immutable "
            "deterministic routing decision."
        )
    if (
        policy_snapshot.source_candidate_count != len(source_records)
        or policy_snapshot.selected_candidate_count != len(records)
        or policy_snapshot.source_candidates_sha256
        != file_sha256(source_candidates_path)
        or policy_snapshot.selected_candidates_sha256 != file_sha256(candidates_path)
    ):
        raise RuntimeError(
            "Immutable validation policy no longer matches source/selected candidates."
        )
    if policy_snapshot.source_run_id:
        _validate_candidate_source_run(
            records=source_records,
            source_run_dir=policy_snapshot.source_run_id,
        )
    candidates_by_key = {record.key: record for record in records}
    source_candidates_by_key = {record.key: record for record in source_records}
    bindings = _read_bindings(run_dir / str(metadata["bindings_file"]))
    if set(bindings) != set(candidates_by_key):
        raise RuntimeError("Validation bindings do not cover the candidate artifact exactly.")
    for key, record in candidates_by_key.items():
        if bindings[key].get("candidate_sha256") != candidate_sha256(record.candidate):
            raise RuntimeError(f"Candidate binding digest mismatch for {key!r}.")

    input_records = read_input_image_manifest(
        run_dir / str(metadata["input_image_manifest_file"])
    )
    input_by_key = {record.key: record for record in input_records}
    _validate_input_manifest_coverage(
        records=source_records,
        input_by_key=input_by_key,
        scope="all",
    )
    _verify_retrieval_evidence_bindings(
        bucket=bucket,
        provider=provider,
        records=records,
        input_by_key=input_by_key,
        bindings=bindings,
    )

    schema_payload = _read_json_object(run_dir / str(metadata["extraction_schema_file"]))
    schema_name = str(metadata.get("extraction_schema_name") or "").strip()
    schema_version_id = str(
        metadata.get("extraction_schema_version_id") or ""
    ).strip()
    candidate_schema_name, candidate_schema_version = _candidate_schema_identity(
        source_records
    )
    if (
        schema_name != candidate_schema_name
        or schema_version_id != candidate_schema_version
    ):
        raise RuntimeError(
            "Verification candidates no longer match the persisted extraction schema identity."
        )
    schema_snapshot = _ExtractionSchemaSnapshot(
        schema=schema_payload,
        name=schema_name,
        version_id=schema_version_id,
        sha256=_canonical_json_sha256(schema_payload),
    )
    expected_schema_digest = str(
        metadata.get("extraction_schema_canonical_sha256") or ""
    ).strip()
    if not expected_schema_digest or schema_snapshot.sha256 != expected_schema_digest:
        raise RuntimeError("Persisted extraction schema canonical digest mismatch.")
    extraction_model = _extraction_model_for_snapshot(schema_snapshot)
    # Preserve historical report-only behavior exactly. The policy is part of
    # immutable submission metadata and cannot be changed during retrieval.
    apply_mode, acceptance_policy = _recorded_correction_policy(
        metadata,
        immutable_policy=policy_snapshot,
    )
    verification_scope = policy_snapshot.verification_scope

    results_path = run_dir / VERIFICATION_RESULTS_FILE_NAME
    failures_path = run_dir / VERIFICATION_FAILURES_FILE_NAME
    summary_path = run_dir / VERIFICATION_SUMMARY_FILE_NAME
    field_corrections_path = run_dir / FIELD_CORRECTION_METADATA_FILE_NAME
    patched_path = (
        run_dir / VERIFICATION_PATCHED_CANDIDATES_FILE_NAME
        if apply_mode == "apply_patches"
        else None
    )
    result_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    validations: dict[str, PageModelValidation] = {}
    patched_by_key: dict[str, dict[str, Any]] = {}
    custom_id_to_key = {
        _anthropic_custom_id_for_key(key): key for key in candidates_by_key
    }

    for raw_path in raw_paths:
        with raw_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    envelope = json.loads(raw)
                except json.JSONDecodeError as exc:
                    failures.append(
                        {
                            "source": raw_path.name,
                            "line_number": line_number,
                            "reason": "invalid_jsonl_line",
                            "detail": str(exc),
                        }
                    )
                    continue
                if not isinstance(envelope, dict):
                    failures.append(
                        {
                            "source": raw_path.name,
                            "line_number": line_number,
                            "reason": "invalid_record_type",
                        }
                    )
                    continue
                if provider == "anthropic":
                    custom_id = str(envelope.get("custom_id") or "")
                    key = custom_id_to_key.get(custom_id)
                    text, rejection = _extract_anthropic_text(envelope)
                else:
                    key = str(envelope.get("key") or "").strip() or None
                    text, rejection = _extract_gemini_text(envelope)
                if not key or key not in candidates_by_key:
                    failures.append(
                        {
                            "key": key,
                            "source": raw_path.name,
                            "line_number": line_number,
                            "reason": "unknown_or_missing_key",
                        }
                    )
                    continue
                if key in validations:
                    failures.append(
                        {
                            "key": key,
                            "source": raw_path.name,
                            "line_number": line_number,
                            "reason": "duplicate_validation_result",
                        }
                    )
                    continue
                if rejection or text is None:
                    failures.append(
                        {
                            "key": key,
                            "source": raw_path.name,
                            "line_number": line_number,
                            "reason": rejection or "empty_response_text",
                        }
                    )
                    continue
                try:
                    validation = PageModelValidation.model_validate_json(text)
                    patched = apply_validation_patches(
                        candidates_by_key[key].candidate, validation
                    )
                    _validate_extraction_candidate(extraction_model, patched)
                except Exception as exc:
                    failures.append(
                        {
                            "key": key,
                            "source": raw_path.name,
                            "line_number": line_number,
                            "reason": "validation_or_patch_schema_failed",
                            "detail": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                validations[key] = validation
                patched_by_key[key] = patched
                result_rows.append(
                    {
                        "key": key,
                        "candidate_sha256": bindings[key]["candidate_sha256"],
                        "page_status": validation.page_status,
                        "patches": [
                            patch.model_dump(mode="json") for patch in validation.patches
                        ],
                        "verification_model": model,
                        "verification_provider": provider,
                    }
                )

    missing = sorted(set(candidates_by_key) - set(validations))
    failed_keys = {
        str(failure.get("key") or "") for failure in failures if failure.get("key")
    }
    for key in missing:
        if key not in failed_keys:
            failures.append({"key": key, "reason": "missing_validation_result"})
    failed_page_keys = failed_keys | set(missing)
    deterministically_cleared_keys = sorted(
        set(source_candidates_by_key) - set(candidates_by_key)
    )
    final_candidates_by_key = {
        key: copy.deepcopy(record.candidate)
        for key, record in source_candidates_by_key.items()
    }
    final_candidates_by_key.update(patched_by_key)

    _write_jsonl(results_path, result_rows)
    _write_jsonl(failures_path, failures)
    if patched_path is not None:
        _write_patched_candidates(
            path=patched_path,
            verification_run_id=run_dir.name,
            candidates_by_key=source_candidates_by_key,
            patched_by_key=final_candidates_by_key,
            validations=validations,
            model=model,
            provider=provider,
            deterministically_cleared_keys=deterministically_cleared_keys,
        )

    counts = {status: 0 for status in ("confirmed", "needs_correction", "unverifiable")}
    for validation in validations.values():
        counts[validation.page_status] += 1
    complete, publishable, status = _model_validation_outcome(
        expected_pages=len(candidates_by_key),
        completed_pages=len(validations),
        failure_records=len(failures),
        unverifiable_pages=counts["unverifiable"],
        apply_mode=apply_mode,
        scope=verification_scope,
    )
    completed_population_pages = len(deterministically_cleared_keys) + len(
        validations
    )

    dataset_path: Path | None = None
    dataset_gcs_uri = ""
    dataset_gcs_generation = ""
    dataset_sha256 = ""
    dataset_publication_idempotency_key = ""
    publication_provenance_sha256 = ""
    dataset_version: int | None = None
    dataset_version_id = ""
    dataset_version_path = ""
    dataset_version_ledger_path = ""
    dataset_version_ledger_gcs_uri = ""
    dataset_rows = 0
    if publishable:
        built_dataset_path, dataset_rows = _build_verified_dataset(
            run_dir=run_dir,
            candidates_by_key=source_candidates_by_key,
            patched_by_key=final_candidates_by_key,
            validations=validations,
            extraction_model=extraction_model,
            extraction_schema_name=schema_name,
            verification_model=model,
            verification_provider=provider,
            output_format=str(metadata.get("output_format") or config.output_format),
            dataset_file_name=str(
                metadata.get("dataset_file_name") or config.dataset_file_name
            ),
            csv_sep=str(metadata.get("csv_sep") or config.csv_sep),
        )
        dataset_path = built_dataset_path
    corrected_dataset_sha256 = (
        file_sha256(dataset_path) if dataset_path is not None else ""
    )
    verification_artifact_sha256s = {
        path.name: file_sha256(path)
        for path in (results_path, failures_path, *raw_paths)
    }
    if policy_snapshot.artifact_sha256:
        verification_artifact_sha256s[FINAL_VALIDATION_POLICY_FILE_NAME] = (
            policy_snapshot.artifact_sha256
        )
    if patched_path is not None:
        verification_artifact_sha256s[patched_path.name] = file_sha256(patched_path)
    field_correction_metadata = build_field_correction_metadata(
        run_dir=run_dir,
        candidates_by_key=source_candidates_by_key,
        validations=validations,
        failures=failures,
        model=model,
        provider=provider,
        apply_mode=apply_mode,
        candidate_hash=str(metadata.get("source_candidates_sha256") or ""),
        verification_prompt_hash=str(
            metadata.get("verification_prompt_hash") or ""
        ),
        created_at=str(metadata.get("created_at") or ""),
        verification_prompt_version=str(
            metadata.get("verification_prompt_version") or "legacy_unversioned"
        ),
        acceptance_policy=acceptance_policy,
        extraction_schema_name=schema_name,
        extraction_schema_version_id=schema_version_id,
        extraction_schema_sha256=schema_snapshot.sha256,
        input_image_manifest_sha256=str(
            metadata.get("input_image_manifest_sha256") or ""
        ),
        validation_bindings_sha256=str(metadata.get("bindings_sha256") or ""),
        source_run_id=policy_snapshot.source_run_id,
        final_validation_policy_sha256=policy_snapshot.artifact_sha256,
        final_validation_policy_gcs_uri=policy_snapshot.artifact_gcs_uri,
        final_validation_policy_gcs_generation=(
            policy_snapshot.artifact_gcs_generation
        ),
        verification_artifact_sha256s=verification_artifact_sha256s,
        included_in_corrected_dataset=publishable,
        corrected_dataset_sha256=corrected_dataset_sha256,
        deterministically_cleared_keys=deterministically_cleared_keys,
    )
    field_corrections_path.write_text(
        json.dumps(field_correction_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    field_corrections_sha256 = file_sha256(field_corrections_path)
    (
        field_corrections_gcs_uri,
        field_corrections_gcs_generation,
    ) = _upload_immutable_validation_artifact(
        bucket,
        run_dir,
        field_corrections_path,
        sha256=field_corrections_sha256,
        validations_prefix=validations_gcs_prefix,
    )
    corrected_fields = int(field_correction_metadata["corrected_fields"])
    accepted_correction_fields = int(
        field_correction_metadata["accepted_correction_fields"]
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_kind": "model_validation_retrieval",
        "run_dir": str(run_dir),
        "source_run_id": policy_snapshot.source_run_id,
        "provider": provider,
        "model": model,
        "apply_mode": apply_mode,
        "correction_acceptance_policy": acceptance_policy,
        "final_validation_policy_sha256": policy_snapshot.artifact_sha256,
        "final_validation_policy_gcs_uri": policy_snapshot.artifact_gcs_uri,
        "final_validation_policy_gcs_generation": (
            policy_snapshot.artifact_gcs_generation
        ),
        "datasets_gcs_prefix": datasets_gcs_prefix,
        "validations_gcs_prefix": validations_gcs_prefix,
        "scope": verification_scope,
        "expected_pages": len(source_candidates_by_key),
        "selected_for_model_review_pages": len(candidates_by_key),
        "completed_pages": completed_population_pages,
        "model_reviewed_pages": len(validations),
        "deterministically_cleared_pages": len(
            deterministically_cleared_keys
        ),
        "confirmed_pages": counts["confirmed"],
        "needs_correction_pages": counts["needs_correction"],
        "unverifiable_pages": counts["unverifiable"],
        "failed_pages": len(failed_page_keys),
        "failure_records": len(failures),
        "candidate_hash": str(
            metadata.get("source_candidates_sha256") or ""
        ),
        "verification_prompt_version": str(
            metadata.get("verification_prompt_version")
            or "legacy_unversioned"
        ),
        "verification_prompt_hash": str(
            metadata.get("verification_prompt_hash") or ""
        ),
        "status": status,
        "success": complete,
        "publishable": publishable,
        "publication_status": "pending" if publishable else "not_applicable",
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "dataset_gcs_uri": dataset_gcs_uri,
        "dataset_gcs_generation": dataset_gcs_generation,
        "dataset_sha256": dataset_sha256,
        "dataset_publication_idempotency_key": (
            dataset_publication_idempotency_key
        ),
        "publication_provenance_sha256": publication_provenance_sha256,
        "dataset_version": dataset_version,
        "dataset_version_id": dataset_version_id,
        "dataset_version_path": dataset_version_path,
        "dataset_version_ledger_path": dataset_version_ledger_path,
        "dataset_version_ledger_gcs_uri": dataset_version_ledger_gcs_uri,
        "dataset_rows": dataset_rows,
        "field_corrections_path": str(field_corrections_path),
        "field_corrections_gcs_uri": field_corrections_gcs_uri,
        "field_corrections_gcs_generation": field_corrections_gcs_generation,
        "field_corrections_sha256": field_corrections_sha256,
        "proposed_correction_fields": int(
            field_correction_metadata["proposed_correction_fields"]
        ),
        "accepted_correction_fields": accepted_correction_fields,
        "corrected_fields": corrected_fields,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    upload_paths = [
        results_path,
        failures_path,
        summary_path,
        *raw_paths,
    ]
    if patched_path is not None:
        upload_paths.append(patched_path)
    if dataset_path is not None:
        upload_paths.append(dataset_path)
    artifact_gcs_uris = _upload_retrieval_artifacts(
        bucket=bucket,
        run_dir=run_dir,
        paths=upload_paths,
        validations_prefix=validations_gcs_prefix,
    )
    artifact_gcs_uris[field_corrections_path.name] = field_corrections_gcs_uri
    if policy_snapshot.artifact_gcs_uri:
        artifact_gcs_uris[FINAL_VALIDATION_POLICY_FILE_NAME] = (
            policy_snapshot.artifact_gcs_uri
        )
    if publishable and dataset_path is not None:
        source_run_dir = str(metadata.get("source_run_dir") or "").strip()
        if not source_run_dir:
            raise RuntimeError(
                "Publishable validation has no extraction source_run_dir; "
                "refusing to allocate a dataset version."
            )
        publication = publish_dataset_version(
            dataset_path=dataset_path,
            source_run_dir=source_run_dir,
            verification_run_dir=run_dir,
            bucket=bucket,
            datasets_prefix=datasets_gcs_prefix,
            candidate_hash=str(
                metadata.get("source_candidates_sha256") or ""
            ),
            verification_prompt_hash=str(
                metadata.get("verification_prompt_hash") or ""
            ),
            metadata={
                "model": model,
                "provider": provider,
                "verification_model": model,
                "verification_provider": provider,
                "source_run_id": policy_snapshot.source_run_id,
                "schema_name": schema_name,
                "schema_version_id": schema_version_id,
                "verification_prompt_version": str(
                    metadata.get("verification_prompt_version")
                    or "legacy_unversioned"
                ),
                "correction_acceptance_policy": acceptance_policy,
                "final_validation_policy_sha256": (
                    policy_snapshot.artifact_sha256
                ),
                "final_validation_policy_gcs_uri": (
                    policy_snapshot.artifact_gcs_uri
                ),
                "final_validation_policy_gcs_generation": (
                    policy_snapshot.artifact_gcs_generation
                ),
                "datasets_gcs_prefix": datasets_gcs_prefix,
                "validations_gcs_prefix": validations_gcs_prefix,
                "rows": dataset_rows,
                "corrected_dataset_sha256": corrected_dataset_sha256,
                "field_corrections_gcs_uri": field_corrections_gcs_uri,
                "field_corrections_gcs_generation": (
                    field_corrections_gcs_generation
                ),
                "field_corrections_sha256": field_corrections_sha256,
                "corrected_fields": corrected_fields,
                "verification_scope": verification_scope,
                "model_reviewed_pages": len(validations),
                "deterministically_cleared_pages": len(
                    deterministically_cleared_keys
                ),
                "deterministic_routing_policy": (
                    policy_snapshot.deterministic_routing_policy
                ),
            },
        )
        dataset_path = Path(publication.local_path)
        dataset_gcs_uri = publication.gcs_uri
        dataset_gcs_generation = publication.gcs_generation
        dataset_sha256 = publication.sha256
        dataset_publication_idempotency_key = publication.idempotency_key
        publication_provenance_sha256 = (
            publication.publication_provenance_sha256
        )
        dataset_version = publication.version
        dataset_version_id = publication.version_id
        dataset_version_path = publication.local_path
        dataset_version_ledger_path = publication.ledger_path
        dataset_version_ledger_gcs_uri = publication.ledger_gcs_uri
        summary.update(
            {
                "dataset_path": str(dataset_path),
                "dataset_gcs_uri": dataset_gcs_uri,
                "dataset_gcs_generation": dataset_gcs_generation,
                "dataset_sha256": dataset_sha256,
                "dataset_publication_idempotency_key": (
                    dataset_publication_idempotency_key
                ),
                "publication_provenance_sha256": (
                    publication_provenance_sha256
                ),
                "dataset_version": dataset_version,
                "dataset_version_id": dataset_version_id,
                "dataset_version_path": dataset_version_path,
                "dataset_version_ledger_path": dataset_version_ledger_path,
                "dataset_version_ledger_gcs_uri": (
                    dataset_version_ledger_gcs_uri
                ),
                "publication_status": "published",
            }
        )
    artifact_gcs_uris[metadata_path.name] = _validation_artifact_location(
        bucket,
        run_dir,
        metadata_path,
        validations_prefix=validations_gcs_prefix,
    )[0]
    summary["artifact_gcs_uris"] = artifact_gcs_uris
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    artifact_gcs_uris[summary_path.name] = _upload_validation_artifact(
        bucket,
        run_dir,
        summary_path,
        validations_prefix=validations_gcs_prefix,
    )
    metadata["retrieval"] = summary
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _upload_validation_artifact(
        bucket,
        run_dir,
        metadata_path,
        validations_prefix=validations_gcs_prefix,
    )
    log(
        "Retrieved model validation: "
        f"model_reviewed={len(validations)}/{len(candidates_by_key)}, "
        f"deterministically_cleared={len(deterministically_cleared_keys)}, "
        f"confirmed={counts['confirmed']}, corrections={counts['needs_correction']}, "
        f"unverifiable={counts['unverifiable']}, failed={len(missing)}."
    )
    if failures:
        log(
            "Model validation reached a terminal incomplete result: "
            f"missing={len(missing)}/{len(candidates_by_key)}, "
            f"failure_records={len(failures)}. See {failures_path}."
        )
    return ModelValidationRetrieveResult(
        run_dir=run_dir,
        provider=provider,
        model=model,
        results_path=results_path,
        failures_path=failures_path,
        summary_path=summary_path,
        field_corrections_path=field_corrections_path,
        field_corrections_gcs_uri=field_corrections_gcs_uri,
        field_corrections_gcs_generation=field_corrections_gcs_generation,
        field_corrections_sha256=field_corrections_sha256,
        correction_acceptance_policy=acceptance_policy,
        accepted_correction_fields=accepted_correction_fields,
        corrected_fields=corrected_fields,
        patched_candidates_path=patched_path,
        dataset_path=dataset_path,
        dataset_gcs_uri=dataset_gcs_uri,
        dataset_gcs_generation=dataset_gcs_generation,
        dataset_sha256=dataset_sha256,
        dataset_publication_idempotency_key=dataset_publication_idempotency_key,
        publication_provenance_sha256=publication_provenance_sha256,
        dataset_version=dataset_version,
        dataset_version_id=dataset_version_id,
        dataset_version_path=dataset_version_path,
        dataset_version_ledger_path=dataset_version_ledger_path,
        dataset_version_ledger_gcs_uri=dataset_version_ledger_gcs_uri,
        dataset_rows=dataset_rows,
        rows_written=dataset_rows,
        expected_pages=len(source_candidates_by_key),
        completed_pages=completed_population_pages,
        successful_pages=completed_population_pages,
        model_reviewed_pages=len(validations),
        deterministically_cleared_pages=len(deterministically_cleared_keys),
        missing_pages=len(missing),
        confirmed_pages=counts["confirmed"],
        needs_correction_pages=counts["needs_correction"],
        unverifiable_pages=counts["unverifiable"],
        failed_pages=len(failed_page_keys),
        success=complete,
        publishable=publishable,
        status=status,
        candidate_hash=str(
            metadata.get("source_candidates_sha256") or ""
        ),
        verification_prompt_hash=str(
            metadata.get("verification_prompt_hash") or ""
        ),
        artifact_gcs_uris=artifact_gcs_uris,
    )


def main() -> None:
    args = _parse_args()
    if args.retrieve:
        run_dir = args.run_dir or str(_latest_verification_run_dir())
        result = retrieve_model_validation(
            ModelValidationRetrieveRequest(
                run_dir=run_dir,
                wait=bool(args.wait),
                allow_partial=bool(args.allow_partial),
            )
        )
        print(
            f"Verification retrieved: {result.completed_pages}/{result.expected_pages} "
            f"page(s), confirmed={result.confirmed_pages}, "
            f"needs_correction={result.needs_correction_pages}, "
            f"corrected_fields={result.corrected_fields}, "
            f"unverifiable={result.unverifiable_pages}, "
            f"dataset_version={result.dataset_version_id or 'none'}. "
            f"correction_metadata={result.field_corrections_path}. [{result.run_dir}]"
        )
        return
    result = submit_model_validation(
        ModelValidationSubmitRequest(
            source_run_dir=args.source_run_dir or args.run_dir,
            candidate_file=args.candidate_file,
            input_manifest_file=args.input_manifest_file,
            model=args.model,
            thinking_level=args.thinking_level,
            scope=args.scope,
            max_output_tokens=args.max_output_tokens,
            num_chunks=args.num_chunks,
        )
    )
    print(
        f"Verification submitted: {result.candidate_count} page(s), "
        f"{len(result.batch_job_names)} batch job(s). [{result.run_dir}]"
    )


if __name__ == "__main__":
    main()
