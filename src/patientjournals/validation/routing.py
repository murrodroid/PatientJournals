"""Deterministic, auditable routing for page-level extraction candidates.

The router records only stable rule identifiers and aggregate metrics. It never
stores model thoughts or other hidden reasoning. Decisions are suitable for a
JSONL artifact and can be evaluated in one pass over candidate content.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from patientjournals.config.schemas import FrontPage
from patientjournals.validation.candidates import (
    PageCandidateRecord,
    candidate_sha256,
)


ROUTING_DECISION_SCHEMA_VERSION = 1
DETERMINISTIC_ROUTING_POLICY_VERSION = "deterministic-routing-v1"
DEFAULT_CONTROL_SAMPLE_SEED = "patientjournals-control-v1"
DEFAULT_MAX_COLLECTION_ITEMS = 8
DEFAULT_MAX_POPULATED_LEAVES = 40
DEFAULT_MAX_TEXT_CHARS = 1600

# Rule identifiers are artifact API. Do not rename or change their meaning;
# introduce a new identifier and policy version when semantics change.
RULE_STRICT_SCHEMA_VALIDATION_FAILED = "schema.strict_validation_failed"
RULE_RECOVERED_PAGE = "provenance.recovered_page"
RULE_EMPTY_STRING = "content.empty_string"
RULE_LARGE_COLLECTION = "complexity.large_collection"
RULE_HIGH_POPULATED_LEAF_COUNT = "complexity.high_populated_leaf_count"
RULE_LONG_TEXT = "complexity.long_text"
RULE_FRONTPAGE_RELEASE_BEFORE_ADMISSION = (
    "frontpage.release_before_admission"
)
RULE_FRONTPAGE_SERUM_INCONSISTENT = "frontpage.serum_inconsistent"
RULE_FRONTPAGE_DEATH_SEKTION_INCONSISTENT = (
    "frontpage.death_sektion_inconsistent"
)
RULE_CONTROL_SAMPLE = "sampling.control_sample"

ROUTING_RULE_ORDER = (
    RULE_STRICT_SCHEMA_VALIDATION_FAILED,
    RULE_RECOVERED_PAGE,
    RULE_EMPTY_STRING,
    RULE_LARGE_COLLECTION,
    RULE_HIGH_POPULATED_LEAF_COUNT,
    RULE_LONG_TEXT,
    RULE_FRONTPAGE_RELEASE_BEFORE_ADMISSION,
    RULE_FRONTPAGE_SERUM_INCONSISTENT,
    RULE_FRONTPAGE_DEATH_SEKTION_INCONSISTENT,
    RULE_CONTROL_SAMPLE,
)
_ROUTING_RULE_RANK = {
    rule_id: index for index, rule_id in enumerate(ROUTING_RULE_ORDER)
}

METADATA_STATUS_KEY = "deterministic_status"
METADATA_ROUTE_KEY = "deterministic_routing_route"
METADATA_POLICY_VERSION_KEY = "deterministic_routing_policy_version"
METADATA_RULE_IDS_KEY = "deterministic_routing_rule_ids"
METADATA_METRICS_KEY = "deterministic_routing_metrics"
METADATA_THRESHOLDS_KEY = "deterministic_routing_thresholds"
METADATA_CONTROL_SAMPLE_KEY = "deterministic_routing_control_sample"
METADATA_CONTROL_SAMPLE_SHA256_KEY = (
    "deterministic_routing_control_sample_sha256"
)
METADATA_CANDIDATE_SHA256_KEY = "deterministic_routing_candidate_sha256"
METADATA_SCHEMA_VALID_KEY = "deterministic_routing_schema_valid"


class RoutingThresholds(BaseModel):
    """Snapshotted thresholds that give rule identifiers their run context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_collection_items: int | None = Field(
        default=DEFAULT_MAX_COLLECTION_ITEMS, ge=0
    )
    max_populated_leaves: int | None = Field(
        default=DEFAULT_MAX_POPULATED_LEAVES, ge=0
    )
    max_text_chars: int | None = Field(default=DEFAULT_MAX_TEXT_CHARS, ge=0)
    control_sample_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    control_sample_seed: str = DEFAULT_CONTROL_SAMPLE_SEED

    @field_validator("control_sample_percent")
    @classmethod
    def sample_percent_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("control_sample_percent must be finite.")
        return value

    @field_validator("control_sample_seed")
    @classmethod
    def sample_seed_must_not_be_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("control_sample_seed must not be empty.")
        return value


class RoutingMetrics(BaseModel):
    """Non-sensitive aggregate measurements used by deterministic rules."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_validation_error_count: int = Field(ge=0)
    empty_string_count: int = Field(ge=0)
    collection_count: int = Field(ge=0)
    largest_collection_size: int = Field(ge=0)
    populated_leaf_count: int = Field(ge=0)
    text_leaf_count: int = Field(ge=0)
    longest_text_chars: int = Field(ge=0)
    total_text_chars: int = Field(ge=0)


class DeterministicRoutingDecision(BaseModel):
    """One immutable, machine-auditable routing decision for one page."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=ROUTING_DECISION_SCHEMA_VERSION, ge=1)
    policy_version: str = DETERMINISTIC_ROUTING_POLICY_VERSION
    key: str = Field(min_length=1)
    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    route: Literal["routine", "heavy_review"]
    deterministic_status: Literal["confirmed", "flagged"]
    schema_valid: bool
    rule_ids: tuple[str, ...] = ()
    metrics: RoutingMetrics
    thresholds: RoutingThresholds
    control_sample: bool = False
    control_sample_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("policy_version", "key")
    @classmethod
    def nonempty_strings(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Routing identity values must not be empty.")
        return normalized

    @field_validator("rule_ids")
    @classmethod
    def rule_ids_are_known_unique_and_ordered(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("Routing rule IDs must be unique.")
        unknown = [item for item in value if item not in _ROUTING_RULE_RANK]
        if unknown:
            raise ValueError(f"Unknown routing rule ID(s): {', '.join(unknown)}")
        if list(value) != sorted(value, key=_ROUTING_RULE_RANK.__getitem__):
            raise ValueError("Routing rule IDs are not in stable policy order.")
        return value

    @model_validator(mode="after")
    def route_matches_rules_and_status(self) -> "DeterministicRoutingDecision":
        flagged = bool(self.rule_ids)
        if (self.route == "heavy_review") != flagged:
            raise ValueError("heavy_review must correspond to at least one rule ID.")
        if (self.deterministic_status == "flagged") != flagged:
            raise ValueError("flagged status must correspond to at least one rule ID.")
        sampled = RULE_CONTROL_SAMPLE in self.rule_ids
        if self.control_sample != sampled:
            raise ValueError(
                "control_sample must correspond to the control-sample rule ID."
            )
        if self.schema_valid == (
            RULE_STRICT_SCHEMA_VALIDATION_FAILED in self.rule_ids
        ):
            raise ValueError(
                "schema_valid conflicts with the strict-schema routing rule."
            )
        return self


@dataclass(frozen=True)
class RoutedCandidate:
    """A candidate annotated with its matching routing decision."""

    candidate: PageCandidateRecord
    decision: DeterministicRoutingDecision


@dataclass
class _MetricAccumulator:
    empty_string_count: int = 0
    collection_count: int = 0
    largest_collection_size: int = 0
    populated_leaf_count: int = 0
    text_leaf_count: int = 0
    longest_text_chars: int = 0
    total_text_chars: int = 0

    def measure(self, candidate: Mapping[str, Any]) -> None:
        pending: list[Any] = [candidate]
        while pending:
            value = pending.pop()
            if isinstance(value, Mapping):
                self.collection_count += 1
                self.largest_collection_size = max(
                    self.largest_collection_size, len(value)
                )
                pending.extend(value.values())
                continue
            if isinstance(value, (list, tuple)):
                self.collection_count += 1
                self.largest_collection_size = max(
                    self.largest_collection_size, len(value)
                )
                pending.extend(value)
                continue
            if isinstance(value, str):
                length = len(value)
                self.text_leaf_count += 1
                self.longest_text_chars = max(self.longest_text_chars, length)
                self.total_text_chars += length
                if value.strip():
                    self.populated_leaf_count += 1
                else:
                    self.empty_string_count += 1
                continue
            if value is not None:
                # False and zero are meaningful populated JSON values.
                self.populated_leaf_count += 1


def _strict_validate_candidate(
    record: PageCandidateRecord,
    full_model: type[BaseModel],
) -> tuple[BaseModel | None, int]:
    try:
        payload = json.dumps(
            record.candidate,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return full_model.model_validate_json(payload, strict=True), 0
    except ValidationError as exc:
        # Error details can contain source text; record only a count.
        return None, len(exc.errors(include_url=False, include_input=False))
    except (TypeError, ValueError):
        # A candidate that cannot be represented as strict JSON is invalid, but
        # its raw value must not leak into the routing artifact.
        return None, 1


def _is_frontpage_model(full_model: type[BaseModel]) -> bool:
    try:
        return issubclass(full_model, FrontPage)
    except TypeError:
        return False


def _is_populated(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return True


def _frontpage_rule_ids(validated: BaseModel | None) -> tuple[str, ...]:
    if not isinstance(validated, FrontPage):
        return ()

    rules: list[str] = []
    hospital_stay = validated.hospital_stay
    if hospital_stay.release_date < hospital_stay.admission_date:
        rules.append(RULE_FRONTPAGE_RELEASE_BEFORE_ADMISSION)

    serum = validated.serum
    if serum is not None:
        has_details = _is_populated(serum.doses) or _is_populated(serum.type)
        if (serum.given and not has_details) or (not serum.given and has_details):
            rules.append(RULE_FRONTPAGE_SERUM_INCONSISTENT)

    has_sektion = validated.diagnoses.sektion is not None
    if bool(validated.is_dead) != has_sektion:
        rules.append(RULE_FRONTPAGE_DEATH_SEKTION_INCONSISTENT)
    return tuple(rules)


def _control_sample_sha256(*, key: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}\0{key}".encode("utf-8")).hexdigest()


def _sample_selected(digest: str, percent: float) -> bool:
    if percent <= 0:
        return False
    if percent >= 100:
        return True
    numerator = Decimal(str(percent))
    threshold = int((numerator / Decimal(100)) * Decimal(1 << 256))
    return int(digest, 16) < threshold


def _build_thresholds(
    *,
    max_collection_items: int | None,
    max_populated_leaves: int | None,
    max_text_chars: int | None,
    control_sample_percent: float,
    control_sample_seed: str,
) -> RoutingThresholds:
    return RoutingThresholds(
        max_collection_items=max_collection_items,
        max_populated_leaves=max_populated_leaves,
        max_text_chars=max_text_chars,
        control_sample_percent=control_sample_percent,
        control_sample_seed=control_sample_seed,
    )


def _decide_candidate_route_with_thresholds(
    record: PageCandidateRecord,
    *,
    full_model: type[BaseModel],
    thresholds: RoutingThresholds,
    policy_version: str,
) -> DeterministicRoutingDecision:
    validated, schema_error_count = _strict_validate_candidate(record, full_model)
    accumulator = _MetricAccumulator()
    accumulator.measure(record.candidate)
    metrics = RoutingMetrics(
        schema_validation_error_count=schema_error_count,
        empty_string_count=accumulator.empty_string_count,
        collection_count=accumulator.collection_count,
        largest_collection_size=accumulator.largest_collection_size,
        populated_leaf_count=accumulator.populated_leaf_count,
        text_leaf_count=accumulator.text_leaf_count,
        longest_text_chars=accumulator.longest_text_chars,
        total_text_chars=accumulator.total_text_chars,
    )

    rules: list[str] = []
    if validated is None:
        rules.append(RULE_STRICT_SCHEMA_VALIDATION_FAILED)
    recovered = record.extraction_metadata.get("recovered")
    if recovered is True or (
        isinstance(recovered, str) and recovered.strip().lower() == "true"
    ):
        rules.append(RULE_RECOVERED_PAGE)
    if metrics.empty_string_count:
        rules.append(RULE_EMPTY_STRING)
    if (
        thresholds.max_collection_items is not None
        and metrics.largest_collection_size > thresholds.max_collection_items
    ):
        rules.append(RULE_LARGE_COLLECTION)
    if (
        thresholds.max_populated_leaves is not None
        and metrics.populated_leaf_count > thresholds.max_populated_leaves
    ):
        rules.append(RULE_HIGH_POPULATED_LEAF_COUNT)
    if (
        thresholds.max_text_chars is not None
        and metrics.longest_text_chars > thresholds.max_text_chars
    ):
        rules.append(RULE_LONG_TEXT)
    if _is_frontpage_model(full_model):
        rules.extend(_frontpage_rule_ids(validated))

    sample_digest = _control_sample_sha256(
        key=record.key,
        seed=thresholds.control_sample_seed,
    )
    control_sample = not rules and _sample_selected(
        sample_digest, thresholds.control_sample_percent
    )
    if control_sample:
        rules.append(RULE_CONTROL_SAMPLE)

    ordered_rules = tuple(sorted(set(rules), key=_ROUTING_RULE_RANK.__getitem__))
    flagged = bool(ordered_rules)
    return DeterministicRoutingDecision(
        policy_version=policy_version,
        key=record.key,
        candidate_sha256=candidate_sha256(record.candidate),
        route="heavy_review" if flagged else "routine",
        deterministic_status="flagged" if flagged else "confirmed",
        schema_valid=validated is not None,
        rule_ids=ordered_rules,
        metrics=metrics,
        thresholds=thresholds,
        control_sample=control_sample,
        control_sample_sha256=sample_digest,
    )


def decide_candidate_route(
    record: PageCandidateRecord,
    *,
    full_model: type[BaseModel],
    max_collection_items: int | None = DEFAULT_MAX_COLLECTION_ITEMS,
    max_populated_leaves: int | None = DEFAULT_MAX_POPULATED_LEAVES,
    max_text_chars: int | None = DEFAULT_MAX_TEXT_CHARS,
    control_sample_percent: float = 0.0,
    control_sample_seed: str = DEFAULT_CONTROL_SAMPLE_SEED,
    policy_version: str = DETERMINISTIC_ROUTING_POLICY_VERSION,
) -> DeterministicRoutingDecision:
    """Evaluate one page without applying the batch-level minimum sample rule."""

    thresholds = _build_thresholds(
        max_collection_items=max_collection_items,
        max_populated_leaves=max_populated_leaves,
        max_text_chars=max_text_chars,
        control_sample_percent=control_sample_percent,
        control_sample_seed=control_sample_seed,
    )
    return _decide_candidate_route_with_thresholds(
        record,
        full_model=full_model,
        thresholds=thresholds,
        policy_version=policy_version,
    )


def _force_control_sample(
    decision: DeterministicRoutingDecision,
) -> DeterministicRoutingDecision:
    if decision.rule_ids:
        raise ValueError("Only an otherwise-routine decision can become a control sample.")
    payload = decision.model_dump(mode="python")
    payload.update(
        {
            "route": "heavy_review",
            "deterministic_status": "flagged",
            "rule_ids": (RULE_CONTROL_SAMPLE,),
            "control_sample": True,
        }
    )
    return DeterministicRoutingDecision.model_validate(payload)


def rewrite_candidate_routing_metadata(
    record: PageCandidateRecord,
    decision: DeterministicRoutingDecision,
) -> PageCandidateRecord:
    """Return a copy carrying complete deterministic routing provenance."""

    if record.key != decision.key:
        raise ValueError("Routing decision key does not match the candidate key.")
    current_digest = candidate_sha256(record.candidate)
    if current_digest != decision.candidate_sha256:
        raise ValueError("Routing decision digest does not match the candidate bytes.")

    metadata = dict(record.extraction_metadata)
    metadata.update(
        {
            METADATA_STATUS_KEY: decision.deterministic_status,
            METADATA_ROUTE_KEY: decision.route,
            METADATA_POLICY_VERSION_KEY: decision.policy_version,
            METADATA_RULE_IDS_KEY: list(decision.rule_ids),
            METADATA_METRICS_KEY: decision.metrics.model_dump(mode="json"),
            METADATA_THRESHOLDS_KEY: decision.thresholds.model_dump(mode="json"),
            METADATA_CONTROL_SAMPLE_KEY: decision.control_sample,
            METADATA_CONTROL_SAMPLE_SHA256_KEY: decision.control_sample_sha256,
            METADATA_CANDIDATE_SHA256_KEY: decision.candidate_sha256,
            METADATA_SCHEMA_VALID_KEY: decision.schema_valid,
        }
    )
    return record.model_copy(update={"extraction_metadata": metadata})


def route_candidates(
    records: Iterable[PageCandidateRecord],
    *,
    full_model: type[BaseModel],
    max_collection_items: int | None = DEFAULT_MAX_COLLECTION_ITEMS,
    max_populated_leaves: int | None = DEFAULT_MAX_POPULATED_LEAVES,
    max_text_chars: int | None = DEFAULT_MAX_TEXT_CHARS,
    control_sample_percent: float = 0.0,
    control_sample_seed: str = DEFAULT_CONTROL_SAMPLE_SEED,
    policy_version: str = DETERMINISTIC_ROUTING_POLICY_VERSION,
) -> Iterator[RoutedCandidate]:
    """Route a batch and guarantee a nonempty routine control sample when asked.

    The lowest stable SHA-256 is selected only when percentage sampling selected
    no otherwise-routine page. Keeping one compact decision and one record
    reference per input is bounded and practical for 100k+ page batches, while
    avoiding a second candidate-content traversal.
    """

    thresholds = _build_thresholds(
        max_collection_items=max_collection_items,
        max_populated_leaves=max_populated_leaves,
        max_text_chars=max_text_chars,
        control_sample_percent=control_sample_percent,
        control_sample_seed=control_sample_seed,
    )
    prepared: list[tuple[PageCandidateRecord, DeterministicRoutingDecision]] = []
    seen_keys: set[str] = set()
    for record in records:
        if record.key in seen_keys:
            raise ValueError(f"Duplicate candidate key for routing: {record.key!r}")
        seen_keys.add(record.key)
        decision = _decide_candidate_route_with_thresholds(
            record,
            full_model=full_model,
            thresholds=thresholds,
            policy_version=policy_version,
        )
        prepared.append((record, decision))

    if thresholds.control_sample_percent > 0 and not any(
        decision.control_sample for _record, decision in prepared
    ):
        routine_indexes = [
            index
            for index, (_record, decision) in enumerate(prepared)
            if not decision.rule_ids
        ]
        if routine_indexes:
            selected_index = min(
                routine_indexes,
                key=lambda index: prepared[index][1].control_sample_sha256,
            )
            record, decision = prepared[selected_index]
            prepared[selected_index] = (record, _force_control_sample(decision))

    for record, decision in prepared:
        yield RoutedCandidate(
            candidate=rewrite_candidate_routing_metadata(record, decision),
            decision=decision,
        )


def route_candidate(
    record: PageCandidateRecord,
    *,
    full_model: type[BaseModel],
    max_collection_items: int | None = 25,
    max_populated_leaves: int | None = 100,
    max_text_chars: int | None = 1000,
    control_sample_percent: float = 0.0,
    control_sample_seed: str = DEFAULT_CONTROL_SAMPLE_SEED,
    policy_version: str = DETERMINISTIC_ROUTING_POLICY_VERSION,
) -> RoutedCandidate:
    """Route one page, applying the same minimum-sample rule as a one-page batch."""

    return next(
        route_candidates(
            (record,),
            full_model=full_model,
            max_collection_items=max_collection_items,
            max_populated_leaves=max_populated_leaves,
            max_text_chars=max_text_chars,
            control_sample_percent=control_sample_percent,
            control_sample_seed=control_sample_seed,
            policy_version=policy_version,
        )
    )


def write_routing_decisions(
    path: str | Path,
    decisions: Iterable[DeterministicRoutingDecision],
) -> Path:
    """Atomically write canonical decision JSONL without buffering the artifact."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    seen_keys: set[str] = set()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for decision in decisions:
                if decision.key in seen_keys:
                    raise ValueError(
                        f"Duplicate routing decision key: {decision.key!r}"
                    )
                seen_keys.add(decision.key)
                handle.write(
                    json.dumps(
                        decision.model_dump(mode="json"),
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination


def read_routing_decisions(
    path: str | Path,
) -> Iterator[DeterministicRoutingDecision]:
    """Stream and validate one routing decision at a time from JSONL."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Routing decision artifact not found: {source}")
    seen_keys: set[str] = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                decision = DeterministicRoutingDecision.model_validate_json(raw)
            except Exception as exc:
                raise ValueError(
                    f"Invalid routing decision at {source}:{line_number}: {exc}"
                ) from exc
            if decision.key in seen_keys:
                raise ValueError(
                    f"Duplicate routing decision key {decision.key!r} at "
                    f"{source}:{line_number}."
                )
            seen_keys.add(decision.key)
            yield decision
