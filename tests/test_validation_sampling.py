from __future__ import annotations

import json
import random
from pathlib import Path

from patientjournals.validation.cli import (
    build_validation_datapoints,
    choose_balanced_ucb_datapoint,
    choose_random_datapoint,
    eligible_flat_fields,
    validation_sampling_group_key,
)
from patientjournals.validation import browser as browser_validation


def test_validation_candidates_are_schema_fields_only(tmp_path) -> None:
    image = tmp_path / "a.png"
    image.write_bytes(b"image")
    row = {
        "image_name": "a.png",
        "fk_info": "FK",
        "patient": {"name": "A", "age": {"number": 12}},
        "diagnoses": {"sektion": None},
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


def test_balanced_ucb_separates_missing_and_present_values(tmp_path) -> None:
    image_index = {}
    rows = []
    for name, value in (
        ("a.png", None),
        ("b.png", None),
        ("c.png", "FK"),
    ):
        image = tmp_path / name
        image.write_bytes(b"image")
        image_index[name] = image
        rows.append({"image_name": name, "fk_info": value})
    datapoints = build_validation_datapoints(rows, image_index)

    selected = choose_balanced_ucb_datapoint(
        datapoints,
        validated_pairs=set(),
        selection_counts={
            validation_sampling_group_key("fk_info", "missing"): 8,
            validation_sampling_group_key("fk_info", "present"): 0,
        },
        scored_counts={
            validation_sampling_group_key("fk_info", "missing"): 8,
            validation_sampling_group_key("fk_info", "present"): 0,
        },
        score_sums={
            validation_sampling_group_key("fk_info", "missing"): 8.0,
            validation_sampling_group_key("fk_info", "present"): 0.0,
        },
        rng=random.Random(1),
    )

    assert [item.value_state for item in datapoints].count("missing") == 2
    assert [item.value_state for item in datapoints].count("present") == 1
    assert selected is not None
    assert selected.image_name == "c.png"
    assert selected.value_state == "present"


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
    output = session.csv_path.read_text(encoding="utf-8")
    assert "accept" in output
    assert "extracted_value" in output
    assert "FK" in output
    assert "value_state" in output


def test_browser_validation_offline_mode_saves_without_uploading(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    uploads = []
    monkeypatch.setattr(
        browser_validation,
        "upload_validation_run",
        lambda **kwargs: uploads.append(kwargs) or {"validation_csv_uri": "gs://bucket/out.csv"},
    )
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
        validator_account="alice@gmail.com",
        image_source="local",
        image_root=str(images),
        sampling_mode="random",
        sync_to_cloud=False,
    )
    manager.mark(sample["session_id"], label="accept")
    result = manager.finish(sample["session_id"])
    session = manager.get(sample["session_id"])
    csv_text = session.csv_path.read_text(encoding="utf-8")
    metadata = json.loads(session.metadata_path.read_text(encoding="utf-8"))

    assert uploads == []
    assert result["offline_mode"] is True
    assert result["upload_skipped_reason"] == "offline_mode"
    assert "alice@gmail.com" in csv_text
    assert metadata["validator_account"] == "alice@gmail.com"
    assert metadata["cloud_sync_enabled"] is False


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
    streamed_blobs = []

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

        def list_blobs(self, prefix=None):
            assert prefix == "pages/run/"
            for name in ("pages/run/a.png", "pages/run/unused.png"):
                streamed_blobs.append(name)
                yield Blob(name)

    monkeypatch.setattr(
        browser_validation,
        "build_storage_bucket",
        lambda _name: Bucket(),
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
    assert streamed_blobs == ["pages/run/a.png"]


def test_browser_validation_cloud_uses_dataset_names_without_selected_folder(
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
    prefixes = []

    class Blob:
        def __init__(self, name: str) -> None:
            self.name = name

        def generate_signed_url(self, **_kwargs) -> str:
            return f"https://signed.example/{self.name}"

    class Bucket:
        name = "encrypted-bucket"

        def blob(self, name: str) -> Blob:
            return Blob(name)

        def list_blobs(self, prefix=None):
            prefixes.append(prefix)
            yield Blob("pages/a.png")

    monkeypatch.setattr(
        browser_validation,
        "build_storage_bucket",
        lambda _name: Bucket(),
    )

    manager = browser_validation.BrowserValidationManager()
    sample = manager.start_session(
        dataset_path=dataset,
        username="alice",
        image_source="cloud",
        bucket_name="encrypted-bucket",
        sampling_mode="random",
    )

    assert sample["image_url"] == "https://signed.example/pages/a.png"
    assert sample["image_uri"] == "gs://encrypted-bucket/pages/a.png"
    assert prefixes[0] == "pages/"


def test_browser_validation_prefetches_lookahead_signed_urls(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(browser_validation, "upload_validation_run", lambda **_kwargs: {})
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"image_name": name, "fk_info": "FK"})
            for name in ("a.png", "b.png", "c.png")
        )
        + "\n",
        encoding="utf-8",
    )
    signed_names = []

    class Blob:
        def __init__(self, name: str) -> None:
            self.name = name

        def generate_signed_url(self, **_kwargs) -> str:
            signed_names.append(self.name)
            return f"https://signed.example/{self.name}"

    class Bucket:
        name = "encrypted-bucket"

        def blob(self, name: str) -> Blob:
            return Blob(name)

        def list_blobs(self, prefix=None):
            assert prefix == "pages/"
            for name in ("pages/a.png", "pages/b.png", "pages/c.png"):
                yield Blob(name)

    monkeypatch.setattr(
        browser_validation,
        "build_storage_bucket",
        lambda _name: Bucket(),
    )

    manager = browser_validation.BrowserValidationManager()
    sample = manager.start_session(
        dataset_path=dataset,
        username="alice",
        image_source="cloud",
        bucket_name="encrypted-bucket",
        sampling_mode="random",
    )
    session = manager.get(sample["session_id"])
    if session.prefetch_future is not None:
        session.prefetch_future.result(timeout=2)

    assert len(set(signed_names)) >= 2


def test_browser_validation_finish_saves_empty_run(tmp_path, monkeypatch) -> None:
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
    result = manager.finish(sample["session_id"])

    assert result["saved"] is True
    assert result["decisions"] == 0
    assert result["csv_path"]
    assert result["metadata_path"]
    assert "validator_account" in Path(result["csv_path"]).read_text(encoding="utf-8")


def test_browser_validation_finish_keeps_local_save_when_upload_fails(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    def fail_upload(**_kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(browser_validation, "upload_validation_run", fail_upload)
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
    manager.mark(sample["session_id"], label="accept")
    result = manager.finish(sample["session_id"])

    assert result["saved"] is True
    assert "network unavailable" in result["upload_error"]
    assert Path(result["csv_path"]).exists()
