from __future__ import annotations

import json
import random

from patientjournals.validation.cli import (
    build_validation_datapoints,
    choose_balanced_ucb_datapoint,
    choose_random_datapoint,
    eligible_flat_fields,
)
from patientjournals.validation import browser as browser_validation


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


def test_browser_validation_records_local_decision(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(browser_validation, "upload_validation_run", lambda **_kwargs: {})
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.png").write_bytes(b"image")
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps({"image_name": "a.png", "fk_info": "FK"}) + "\n",
        encoding="utf-8",
    )

    manager = browser_validation.BrowserValidationManager()
    sample = manager.start_session(
        dataset_path=dataset,
        username="alice",
        image_source="local",
        image_root=str(images),
        sampling_mode="random",
    )
    after_mark = manager.mark(sample["session_id"], label="accept")
    session = manager.get(sample["session_id"])

    assert sample["image_url"].startswith("/api/validation/session/image?")
    assert after_mark["decisions"] == 1
    assert session.csv_path.exists()
    assert "accept" in session.csv_path.read_text(encoding="utf-8")


def test_browser_validation_uses_signed_cloud_url_without_persisting_it(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(browser_validation, "upload_validation_run", lambda **_kwargs: {})
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps({"image_name": "a.png", "fk_info": "FK"}) + "\n",
        encoding="utf-8",
    )
    signed_calls = []

    class Blob:
        def __init__(self, name: str) -> None:
            self.name = name

        def generate_signed_url(self, **kwargs) -> str:
            signed_calls.append(kwargs)
            return f"https://signed.example/{self.name}"

    class Bucket:
        name = "encrypted-bucket"

        def blob(self, name: str) -> Blob:
            return Blob(name)

    monkeypatch.setattr(
        browser_validation,
        "build_storage_bucket",
        lambda _name: Bucket(),
    )
    monkeypatch.setattr(
        browser_validation,
        "list_bucket_blobs",
        lambda _bucket, prefix=None: [Blob("pages/run/a.png")],
    )

    manager = browser_validation.BrowserValidationManager()
    sample = manager.start_session(
        dataset_path=dataset,
        username="alice",
        image_source="cloud",
        cloud_prefixes=("pages/run",),
        bucket_name="encrypted-bucket",
        sampling_mode="random",
    )
    manager.mark(sample["session_id"], label="unsure")
    session = manager.get(sample["session_id"])
    output = session.csv_path.read_text(encoding="utf-8")

    assert sample["image_url"] == "https://signed.example/pages/run/a.png"
    assert signed_calls[0]["version"] == "v4"
    assert signed_calls[0]["method"] == "GET"
    assert "https://signed.example" not in output
    assert "gs://encrypted-bucket/pages/run/a.png" in output
