from __future__ import annotations

import random

from patientjournals.validation.cli import (
    build_validation_datapoints,
    choose_balanced_ucb_datapoint,
    choose_random_datapoint,
    eligible_flat_fields,
)


def test_validation_candidates_are_schema_fields_only(tmp_path) -> None:
    image = tmp_path / "a.png"
    image.write_bytes(b"image")
    row = {
        "image_name": "a.png",
        "fk_info": "FK",
        "patient": {"name": "A", "age": {"number": 12}},
        "thoughts": "model reasoning",
        "failure_reason": "none",
        "avg_logprobs": -0.2,
        "crossed_out": "not sampled",
        "unknown_column": "not sampled",
    }

    fields = dict(eligible_flat_fields(row))
    datapoints = build_validation_datapoints([row], {"a.png": image})

    assert set(fields) == {"fk_info", "patient.name", "patient.age.number"}
    assert {item.field_name for item in datapoints} == set(fields)


def test_random_sampling_uses_unvalidated_datapoints(tmp_path) -> None:
    image = tmp_path / "a.png"
    image.write_bytes(b"image")
    rows = [
        {"image_name": "a.png", "fk_info": "FK", "patient": {"name": "A"}},
    ]
    datapoints = build_validation_datapoints(rows, {"a.png": image})

    selected = choose_random_datapoint(
        datapoints,
        {("a.png", "fk_info")},
        random.Random(1),
    )

    assert selected is not None
    assert selected.field_name == "patient.name"


def test_balanced_ucb_prioritizes_under_sampled_schema_field(tmp_path) -> None:
    image = tmp_path / "a.png"
    image.write_bytes(b"image")
    rows = [
        {"image_name": "a.png", "fk_info": "FK", "patient": {"name": "A"}},
    ]
    datapoints = build_validation_datapoints(rows, {"a.png": image})

    selected = choose_balanced_ucb_datapoint(
        datapoints,
        validated_pairs=set(),
        selection_counts={"patient.name": 10, "fk_info": 0},
        scored_counts={"patient.name": 10, "fk_info": 0},
        score_sums={"patient.name": 10.0, "fk_info": 0.0},
        rng=random.Random(1),
    )

    assert selected is not None
    assert selected.field_name == "fk_info"
