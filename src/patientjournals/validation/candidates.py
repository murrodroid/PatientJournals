"""Canonical unflattened extraction candidates for second-pass validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, TextIO

from pydantic import BaseModel, ConfigDict, Field, field_validator


PAGE_CANDIDATES_FILE_NAME = "page_candidates.jsonl"

_EXTRACTION_METADATA_KEYS = frozenset(
    {
        "model",
        "provider",
        "schema_name",
        "schema_version_id",
        "batch_name",
        "raw_output_file",
        "line_number",
        "source",
        "source_run_id",
        "source_run_dir",
        "source_metadata_gcs_uri",
        "source_metadata_sha256",
        "source_metadata_gcs_generation",
        "source_batch_job_gcs_uri",
        "source_batch_job_sha256",
        "source_batch_job_gcs_generation",
        "recovered",
        "deterministic_status",
        "deterministic_routing_route",
        "deterministic_routing_policy_version",
        "deterministic_routing_rule_ids",
        "deterministic_routing_metrics",
        "deterministic_routing_thresholds",
        "deterministic_routing_control_sample",
        "deterministic_routing_control_sample_sha256",
        "deterministic_routing_candidate_sha256",
        "deterministic_routing_schema_valid",
        "input_image_manifest_gcs_uri",
        "input_image_manifest_sha256",
        "verification_status",
        "verification_model",
        "verification_provider",
        "verification_run_id",
        # Legacy read/rewrite compatibility only. New verifier artifacts remove
        # this machine-local path before passing metadata to the writer.
        "verification_run_dir",
    }
)


def source_run_id_from_path(value: str | Path) -> str:
    """Return a portable run identifier from a local run-directory path."""

    normalized = str(value or "").strip().rstrip("/\\").replace("\\", "/")
    run_id = normalized.rsplit("/", 1)[-1] if normalized else ""
    return "" if run_id in {".", ".."} else run_id


class PageCandidateRecord(BaseModel):
    """One page-level candidate before one-to-many dataset flattening."""

    model_config = ConfigDict(extra="forbid")

    key: str
    candidate: dict[str, Any]
    extraction_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("key")
    @classmethod
    def key_must_not_be_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Candidate page key is empty.")
        return normalized


def sanitize_extraction_metadata(
    metadata: Mapping[str, Any] | None,
    **provenance: Any,
) -> dict[str, Any]:
    """Keep reproducibility/provenance only; never carry model thoughts forward."""

    combined = dict(metadata or {})
    combined.update({key: value for key, value in provenance.items() if value is not None})
    explicit_source_run_id = str(combined.get("source_run_id") or "").strip()
    if explicit_source_run_id:
        combined["source_run_id"] = explicit_source_run_id
    else:
        derived_source_run_id = source_run_id_from_path(
            str(combined.get("source_run_dir") or "")
        )
        if derived_source_run_id:
            combined["source_run_id"] = derived_source_run_id
    return {
        key: combined[key]
        for key in sorted(_EXTRACTION_METADATA_KEYS)
        if key in combined and combined[key] is not None
    }


def candidate_sha256(candidate: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(candidate),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class PageCandidateWriter:
    """Buffered JSONL writer that enforces one canonical record per page key."""

    path: Path
    _handle: TextIO = field(init=False, repr=False)
    _keys: set[str] = field(default_factory=set, init=False, repr=False)
    records_written: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def write(
        self,
        *,
        key: str,
        candidate: Mapping[str, Any],
        extraction_metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        record = PageCandidateRecord(
            key=key,
            candidate=dict(candidate),
            extraction_metadata=sanitize_extraction_metadata(extraction_metadata),
        )
        if record.key in self._keys:
            return False
        self._keys.add(record.key)
        self._handle.write(
            json.dumps(
                record.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        self._handle.write("\n")
        self.records_written += 1
        return True

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()

    def __enter__(self) -> "PageCandidateWriter":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def read_page_candidates(path: str | Path) -> tuple[PageCandidateRecord, ...]:
    source = Path(path).expanduser()
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Page candidate artifact not found: {source}")

    records: list[PageCandidateRecord] = []
    keys: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = PageCandidateRecord.model_validate_json(raw)
            except Exception as exc:
                raise ValueError(
                    f"Invalid page candidate at {source}:{line_number}: {exc}"
                ) from exc
            if record.key in keys:
                raise ValueError(
                    f"Duplicate page candidate key {record.key!r} at "
                    f"{source}:{line_number}."
                )
            keys.add(record.key)
            records.append(record)
    if not records:
        raise ValueError(f"Page candidate artifact is empty: {source}")
    return tuple(records)


def write_page_candidates(
    path: str | Path,
    records: Iterable[PageCandidateRecord],
) -> Path:
    destination = Path(path).expanduser()
    with PageCandidateWriter(destination) as writer:
        for record in records:
            if not writer.write(
                key=record.key,
                candidate=record.candidate,
                extraction_metadata=record.extraction_metadata,
            ):
                raise ValueError(f"Duplicate page candidate key: {record.key!r}")
    return destination
