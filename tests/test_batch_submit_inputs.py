import json

from patientjournals.batch.submit_inputs import (
    INPUT_DUPLICATES_CSV_NAME,
    INPUT_DUPLICATES_JSONL_NAME,
    _list_input_blobs,
)
from patientjournals.config import config
from patientjournals.shared.processing_metrics import (
    MANIFEST_FILE_NAME,
    read_processing_records,
)


class FakeBlob:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeBucket:
    def __init__(self, blobs: list[FakeBlob]) -> None:
        self._blobs = blobs

    def list_blobs(self, prefix: str | None = None):
        return [
            blob
            for blob in self._blobs
            if prefix is None or blob.name.startswith(prefix)
        ]


def test_list_input_blobs_skips_duplicate_image_names_with_audit(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(config, "batch_input_source", "gcs")
    monkeypatch.setattr(config, "batch_input_prefix", "")
    monkeypatch.setattr(config, "batch_input_prefixes", ("pages/a", "pages/b"))
    monkeypatch.setattr(config, "batch_input_extensions", ("png",))
    monkeypatch.setattr(config, "gcs_pages_prefix", "pages")
    monkeypatch.setattr(config, "batch_year_filter", ())
    logs: list[str] = []
    bucket = FakeBucket(
        [
            FakeBlob("pages/a/273057_001519.png"),
            FakeBlob("pages/b/273057_001519.png"),
            FakeBlob("pages/b/273057_001520.png"),
        ]
    )

    blobs = _list_input_blobs(bucket, log=logs.append, audit_dir=tmp_path)

    assert [blob.name for blob in blobs] == [
        "pages/a/273057_001519.png",
        "pages/b/273057_001520.png",
    ]
    duplicate_rows = [
        json.loads(line)
        for line in (tmp_path / INPUT_DUPLICATES_JSONL_NAME)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert duplicate_rows == [
        {
            "image_name": "273057_001519.png",
            "duplicate_action": "skipped_input_duplicate",
            "kept_object": "pages/a/273057_001519.png",
            "skipped_object": "pages/b/273057_001519.png",
            "kept_index": 1,
            "skipped_index": 2,
        }
    ]
    assert "273057_001519.png" in (tmp_path / INPUT_DUPLICATES_CSV_NAME).read_text(
        encoding="utf-8"
    )
    manifest_records = read_processing_records(tmp_path / MANIFEST_FILE_NAME)
    assert manifest_records[0]["status"] == "duplicate_skipped"
    assert manifest_records[0]["source"] == "batch_input_selection"
    assert manifest_records[0]["image_name"] == "273057_001519.png"
    assert manifest_records[0]["duplicate_action"] == "skipped_input_duplicate"
    assert "Skipped duplicate GCS input image_name values" in "\n".join(logs)


def test_list_input_blobs_scopes_to_restricted_image_names(monkeypatch) -> None:
    monkeypatch.setattr(config, "batch_input_source", "gcs")
    monkeypatch.setattr(config, "batch_input_prefix", "")
    monkeypatch.setattr(config, "batch_input_prefixes", ())
    monkeypatch.setattr(config, "batch_input_extensions", ("png",))
    monkeypatch.setattr(config, "gcs_pages_prefix", "pages")
    monkeypatch.setattr(config, "batch_year_filter", ())
    monkeypatch.setattr(config, "batch_use_local_pdf_folders", False)
    # The bucket holds thousands of pages, but only two were requested.
    monkeypatch.setattr(
        config,
        "batch_restrict_image_names",
        ("273057_000001.png", "273057_000003.png"),
    )
    logs: list[str] = []
    bucket = FakeBucket(
        [
            FakeBlob("pages/273057_000001.png"),
            FakeBlob("pages/273057_000002.png"),
            FakeBlob("pages/273057_000003.png"),
            FakeBlob("pages/273057_000004.png"),
        ]
    )

    blobs = _list_input_blobs(bucket, log=logs.append)

    assert [blob.name for blob in blobs] == [
        "pages/273057_000001.png",
        "pages/273057_000003.png",
    ]
    assert "Applied image-name restriction" in "\n".join(logs)


def test_list_input_blobs_raises_when_restriction_matches_nothing(monkeypatch) -> None:
    monkeypatch.setattr(config, "batch_input_source", "gcs")
    monkeypatch.setattr(config, "batch_input_prefix", "")
    monkeypatch.setattr(config, "batch_input_prefixes", ())
    monkeypatch.setattr(config, "batch_input_extensions", ("png",))
    monkeypatch.setattr(config, "gcs_pages_prefix", "pages")
    monkeypatch.setattr(config, "batch_year_filter", ())
    monkeypatch.setattr(config, "batch_use_local_pdf_folders", False)
    monkeypatch.setattr(config, "batch_restrict_image_names", ("does_not_exist.png",))
    bucket = FakeBucket([FakeBlob("pages/273057_000001.png")])

    import pytest

    with pytest.raises(FileNotFoundError):
        _list_input_blobs(bucket, log=lambda _m: None)
