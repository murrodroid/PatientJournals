from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from patientjournals.config.schemas import FrontPage
from patientjournals.validation.candidates import (
    PageCandidateRecord,
    candidate_sha256,
)
from patientjournals.validation.routing import (
    METADATA_CANDIDATE_SHA256_KEY,
    METADATA_CONTROL_SAMPLE_KEY,
    METADATA_METRICS_KEY,
    METADATA_POLICY_VERSION_KEY,
    METADATA_ROUTE_KEY,
    METADATA_RULE_IDS_KEY,
    METADATA_SCHEMA_VALID_KEY,
    METADATA_STATUS_KEY,
    RULE_CONTROL_SAMPLE,
    RULE_EMPTY_STRING,
    RULE_FRONTPAGE_DEATH_SEKTION_INCONSISTENT,
    RULE_FRONTPAGE_RELEASE_BEFORE_ADMISSION,
    RULE_FRONTPAGE_SERUM_INCONSISTENT,
    RULE_HIGH_POPULATED_LEAF_COUNT,
    RULE_LARGE_COLLECTION,
    RULE_LONG_TEXT,
    RULE_RECOVERED_PAGE,
    RULE_STRICT_SCHEMA_VALIDATION_FAILED,
    decide_candidate_route,
    read_routing_decisions,
    rewrite_candidate_routing_metadata,
    route_candidate,
    route_candidates,
    write_routing_decisions,
)


class FlexiblePage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    items: list[Any]
    fields: dict[str, Any]


class IntegerPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int


def _record(
    key: str = "pages/page.png",
    *,
    candidate: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> PageCandidateRecord:
    return PageCandidateRecord(
        key=key,
        candidate=candidate or {"title": "ok", "items": [], "fields": {}},
        extraction_metadata=metadata or {},
    )


def _frontpage_candidate() -> dict[str, Any]:
    return {
        "is_dead": False,
        "fk_info": "FK",
        "patient": {
            "number": None,
            "name": "Jensen",
            "household_position": "Barn",
            "age": {"number": 3.0, "unit": "Aar", "note": None},
            "address": {"street": "Gade", "number": "1", "apt": None},
        },
        "hospital_stay": {
            "ward": None,
            "admission_date": "1900-01-01",
            "release_date": "1900-01-02",
            "stay_length": "1 Dag",
            "note": None,
        },
        "diagnoses": {
            "top": {"conditions": ["Feber"], "db": None},
            "bottom": {"doctor_name": "Jensen", "diagnosis": "Feber"},
            "sektion": None,
            "severity": None,
        },
        "serum": {"given": False, "doses": None, "type": None},
        "crossed_out": None,
    }


def _decision(
    record: PageCandidateRecord,
    **kwargs: Any,
):
    return decide_candidate_route(
        record,
        full_model=FlexiblePage,
        max_collection_items=None,
        max_populated_leaves=None,
        max_text_chars=None,
        **kwargs,
    )


def test_routine_candidate_passes_strict_schema_and_records_metrics() -> None:
    record = _record(metadata={"source_run_id": "source-run"})

    routed = route_candidate(record, full_model=FlexiblePage)

    decision = routed.decision
    assert decision.route == "routine"
    assert decision.deterministic_status == "confirmed"
    assert decision.schema_valid is True
    assert decision.rule_ids == ()
    assert decision.metrics.populated_leaf_count == 1
    assert decision.metrics.collection_count == 3
    assert decision.candidate_sha256 == candidate_sha256(record.candidate)

    metadata = routed.candidate.extraction_metadata
    assert metadata["source_run_id"] == "source-run"
    assert metadata[METADATA_STATUS_KEY] == "confirmed"
    assert metadata[METADATA_ROUTE_KEY] == "routine"
    assert metadata[METADATA_POLICY_VERSION_KEY] == decision.policy_version
    assert metadata[METADATA_RULE_IDS_KEY] == []
    assert metadata[METADATA_METRICS_KEY]["populated_leaf_count"] == 1
    assert metadata[METADATA_CONTROL_SAMPLE_KEY] is False
    assert metadata[METADATA_CANDIDATE_SHA256_KEY] == decision.candidate_sha256
    assert metadata[METADATA_SCHEMA_VALID_KEY] is True


def test_strict_full_model_validation_does_not_coerce_scalars() -> None:
    record = _record(candidate={"n": "3"})

    decision = decide_candidate_route(
        record,
        full_model=IntegerPage,
        max_collection_items=None,
        max_populated_leaves=None,
        max_text_chars=None,
    )

    assert decision.route == "heavy_review"
    assert decision.schema_valid is False
    assert decision.rule_ids == (RULE_STRICT_SCHEMA_VALIDATION_FAILED,)
    assert decision.metrics.schema_validation_error_count == 1


@pytest.mark.parametrize(
    ("record", "thresholds", "expected_rule"),
    [
        (
            _record(metadata={"recovered": True}),
            {},
            RULE_RECOVERED_PAGE,
        ),
        (
            _record(candidate={"title": " ", "items": [], "fields": {}}),
            {},
            RULE_EMPTY_STRING,
        ),
        (
            _record(candidate={"title": "ok", "items": [1, 2, 3], "fields": {}}),
            {"max_collection_items": 2},
            RULE_LARGE_COLLECTION,
        ),
        (
            _record(candidate={"title": "ok", "items": [1, 2, 3], "fields": {}}),
            {"max_populated_leaves": 3},
            RULE_HIGH_POPULATED_LEAF_COUNT,
        ),
        (
            _record(candidate={"title": "long", "items": [], "fields": {}}),
            {"max_text_chars": 3},
            RULE_LONG_TEXT,
        ),
    ],
)
def test_explicit_complexity_and_provenance_rules(
    record: PageCandidateRecord,
    thresholds: dict[str, int],
    expected_rule: str,
) -> None:
    kwargs: dict[str, Any] = {
        "max_collection_items": None,
        "max_populated_leaves": None,
        "max_text_chars": None,
        **thresholds,
    }

    decision = decide_candidate_route(record, full_model=FlexiblePage, **kwargs)

    assert decision.rule_ids == (expected_rule,)
    assert decision.route == "heavy_review"
    assert decision.deterministic_status == "flagged"


def test_large_mapping_is_a_collection_and_threshold_is_exclusive() -> None:
    at_limit = _record(
        candidate={"title": "ok", "items": [], "fields": {"a": 1, "b": 2}}
    )
    over_limit = _record(
        candidate={
            "title": "ok",
            "items": [],
            "fields": {"a": 1, "b": 2, "c": 3, "d": 4},
        }
    )

    at_limit_decision = decide_candidate_route(
        at_limit,
        full_model=FlexiblePage,
        max_collection_items=3,
        max_populated_leaves=None,
        max_text_chars=None,
    )
    over_limit_decision = decide_candidate_route(
        over_limit,
        full_model=FlexiblePage,
        max_collection_items=3,
        max_populated_leaves=None,
        max_text_chars=None,
    )

    # The root object also has three members, so three is exactly the limit.
    assert at_limit_decision.rule_ids == ()
    assert over_limit_decision.rule_ids == (RULE_LARGE_COLLECTION,)


def test_multiple_rule_ids_use_stable_policy_order() -> None:
    record = _record(
        candidate={"title": "", "items": [1, 2, 3], "fields": {}},
        metadata={"recovered": True},
    )

    decision = decide_candidate_route(
        record,
        full_model=FlexiblePage,
        max_collection_items=2,
        max_populated_leaves=2,
        max_text_chars=None,
    )

    assert decision.rule_ids == (
        RULE_RECOVERED_PAGE,
        RULE_EMPTY_STRING,
        RULE_LARGE_COLLECTION,
        RULE_HIGH_POPULATED_LEAF_COUNT,
    )


@pytest.mark.parametrize(
    ("mutate", "expected_rule"),
    [
        (
            lambda page: page["hospital_stay"].update(
                {"release_date": "1899-12-31"}
            ),
            RULE_FRONTPAGE_RELEASE_BEFORE_ADMISSION,
        ),
        (
            lambda page: page.update(
                {"serum": {"given": False, "doses": "[20]", "type": None}}
            ),
            RULE_FRONTPAGE_SERUM_INCONSISTENT,
        ),
        (
            lambda page: page.update({"is_dead": True}),
            RULE_FRONTPAGE_DEATH_SEKTION_INCONSISTENT,
        ),
    ],
)
def test_frontpage_cross_field_rules(mutate, expected_rule: str) -> None:
    page = _frontpage_candidate()
    mutate(page)

    decision = decide_candidate_route(
        _record(candidate=page),
        full_model=FrontPage,
        max_collection_items=None,
        max_populated_leaves=None,
        max_text_chars=None,
    )

    assert decision.schema_valid is True
    assert decision.rule_ids == (expected_rule,)


def test_frontpage_positive_serum_without_details_is_inconsistent() -> None:
    page = _frontpage_candidate()
    page["serum"] = {"given": True, "doses": None, "type": None}

    decision = decide_candidate_route(
        _record(candidate=page),
        full_model=FrontPage,
        max_collection_items=None,
        max_populated_leaves=None,
        max_text_chars=None,
    )

    assert decision.rule_ids == (RULE_FRONTPAGE_SERUM_INCONSISTENT,)


def test_frontpage_strict_runtime_date_range_is_enforced() -> None:
    page = _frontpage_candidate()
    page["hospital_stay"]["admission_date"] = "1800-01-01"

    decision = decide_candidate_route(
        _record(candidate=page),
        full_model=FrontPage,
        max_collection_items=None,
        max_populated_leaves=None,
        max_text_chars=None,
    )

    assert decision.schema_valid is False
    assert RULE_STRICT_SCHEMA_VALIDATION_FAILED in decision.rule_ids


def test_small_positive_sample_selects_lowest_stable_routine_hash() -> None:
    records = [_record(f"pages/{index}.png") for index in range(4)]
    percent = 1e-70
    independent = [
        _decision(
            record,
            control_sample_percent=percent,
            control_sample_seed="study-seed",
        )
        for record in records
    ]
    assert not any(item.control_sample for item in independent)
    expected_key = min(independent, key=lambda item: item.control_sample_sha256).key

    routed = list(
        route_candidates(
            records,
            full_model=FlexiblePage,
            max_collection_items=None,
            max_populated_leaves=None,
            max_text_chars=None,
            control_sample_percent=percent,
            control_sample_seed="study-seed",
        )
    )

    sampled = [item for item in routed if item.decision.control_sample]
    assert [item.decision.key for item in sampled] == [expected_key]
    assert sampled[0].decision.rule_ids == (RULE_CONTROL_SAMPLE,)
    assert sampled[0].decision.route == "heavy_review"
    assert sampled[0].candidate.extraction_metadata[METADATA_STATUS_KEY] == "flagged"


def test_control_sample_is_reproducible_across_input_order() -> None:
    records = [_record(f"pages/{index}.png") for index in range(5)]
    kwargs = {
        "full_model": FlexiblePage,
        "max_collection_items": None,
        "max_populated_leaves": None,
        "max_text_chars": None,
        "control_sample_percent": 1e-70,
        "control_sample_seed": "same-seed",
    }

    forwards = list(route_candidates(records, **kwargs))
    backwards = list(route_candidates(reversed(records), **kwargs))

    assert {item.decision.key for item in forwards if item.decision.control_sample} == {
        item.decision.key for item in backwards if item.decision.control_sample
    }


def test_forced_control_sample_chooses_only_otherwise_routine_pages() -> None:
    flagged = _record(
        "pages/flagged.png",
        candidate={"title": "", "items": [], "fields": {}},
    )
    routine = _record("pages/routine.png")

    routed = list(
        route_candidates(
            (flagged, routine),
            full_model=FlexiblePage,
            max_collection_items=None,
            max_populated_leaves=None,
            max_text_chars=None,
            control_sample_percent=1e-70,
        )
    )
    by_key = {item.decision.key: item.decision for item in routed}

    assert by_key[flagged.key].rule_ids == (RULE_EMPTY_STRING,)
    assert by_key[flagged.key].control_sample is False
    assert by_key[routine.key].rule_ids == (RULE_CONTROL_SAMPLE,)
    assert by_key[routine.key].control_sample is True


def test_routing_decision_jsonl_round_trip_is_canonical_and_streamed(tmp_path) -> None:
    records = [_record("pages/a.png"), _record("pages/b.png")]
    decisions = [
        item.decision
        for item in route_candidates(records, full_model=FlexiblePage)
    ]
    path = tmp_path / "routing.jsonl"

    write_routing_decisions(path, iter(decisions))
    first_bytes = path.read_bytes()
    loaded = list(read_routing_decisions(path))
    write_routing_decisions(path, loaded)

    assert loaded == decisions
    assert path.read_bytes() == first_bytes
    for line in first_bytes.decode("utf-8").splitlines():
        assert json.dumps(
            json.loads(line),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ) == line


def test_routing_jsonl_duplicate_key_is_rejected_atomically(tmp_path) -> None:
    decision = route_candidate(
        _record(),
        full_model=FlexiblePage,
    ).decision
    path = tmp_path / "routing.jsonl"
    path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate routing decision key"):
        write_routing_decisions(path, (decision, decision))

    assert path.read_text(encoding="utf-8") == "existing\n"


def test_metadata_rewrite_refuses_a_decision_for_changed_candidate() -> None:
    record = _record()
    decision = _decision(record)
    changed = record.model_copy(
        update={
            "candidate": {"title": "changed", "items": [], "fields": {}}
        }
    )

    with pytest.raises(ValueError, match="digest does not match"):
        rewrite_candidate_routing_metadata(changed, decision)


def test_frontpage_death_with_sektion_is_consistent() -> None:
    page = deepcopy(_frontpage_candidate())
    page["is_dead"] = True
    page["diagnoses"]["sektion"] = {"number": 7, "diagnoses": ["Feber"]}

    decision = decide_candidate_route(
        _record(candidate=page),
        full_model=FrontPage,
        max_collection_items=None,
        max_populated_leaves=None,
        max_text_chars=None,
    )

    assert decision.rule_ids == ()
    assert decision.route == "routine"
