from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from patientjournals.app.dashboard import analyze_dataset_file
from patientjournals.app import datasets as dataset_module
from patientjournals.app.datasets import read_dataset_page
from patientjournals.app.image_access import ImageAccessService
from patientjournals.app.job_store import JobStore
from patientjournals.app.models import AppSettings
from patientjournals.app.schemas import (
    SchemaService,
    dataset_schema_field_paths,
    flatten_schema_fields,
)
from patientjournals.config.schemas import FrontPage, model_from_json_schema
from patientjournals.validation.cli import build_validation_datapoints


def test_schema_edits_create_immutable_versions_and_can_be_active(tmp_path) -> None:
    service = SchemaService(JobStore(tmp_path / "runs"))
    parent = service.resolve_version("FrontPage")
    original_fields = flatten_schema_fields(parent["schema_json"])

    result = service.create_version(
        name="FrontPage",
        parent_version_id=parent["version_id"],
        fields=[
            *original_fields,
            {
                "path": "research_note",
                "type": "string",
                "required": False,
                "description": "A study-specific transcription field.",
            },
        ],
        created_by="researcher@example.com",
        make_active=True,
    )
    version = result["version"]

    assert version["version_id"].startswith("sv_")
    assert version["version_id"] != parent["version_id"]
    assert version["version_number"] == 2
    assert version["parent_version_id"] == parent["version_id"]
    assert version["is_active"] is True
    assert "research_note" in dataset_schema_field_paths(version)
    assert "research_note" not in dataset_schema_field_paths(
        service.store.schema_version(parent["version_id"])
    )
    assert (
        service.list_versions(sync_cloud=False)["active_version_id"]
        == version["version_id"]
    )


def test_removing_all_nested_leafs_prunes_the_parent_object(tmp_path) -> None:
    service = SchemaService(JobStore(tmp_path / "runs"))
    parent = service.resolve_version("FrontPage")
    fields = [
        field
        for field in flatten_schema_fields(parent["schema_json"])
        if not str(field["path"]).startswith("diagnoses.sektion.")
    ]

    version = service.create_version(
        name="FrontPage",
        parent_version_id=parent["version_id"],
        fields=fields,
        created_by="researcher@example.com",
    )["version"]
    diagnoses = version["schema_json"]["$defs"]["Diagnoses"]

    assert "sektion" not in diagnoses["properties"]


def test_schema_versions_round_trip_through_cloud(tmp_path, monkeypatch) -> None:
    objects: dict[str, str] = {}

    class NotFound(Exception):
        pass

    class Blob:
        def __init__(self, name: str) -> None:
            self.name = name

        def download_as_text(self, encoding: str = "utf-8") -> str:
            del encoding
            if self.name not in objects:
                raise NotFound("404 not found")
            return objects[self.name]

        def upload_from_string(self, value: str, content_type: str = "") -> None:
            del content_type
            objects[self.name] = value

    class Bucket:
        name = "shared-bucket"

        def blob(self, name: str) -> Blob:
            return Blob(name)

    monkeypatch.setattr(
        "patientjournals.app.schemas.build_storage_bucket",
        lambda _name=None: Bucket(),
    )
    first = SchemaService(
        JobStore(tmp_path / "first"),
        bucket_name="shared-bucket",
        schemas_prefix="schemas",
    )
    first.list_versions()
    created = first.create_version(
        name="StudySchema",
        fields=[
            {
                "path": "transcription",
                "type": "string",
                "required": True,
                "description": "Literal transcription.",
            }
        ],
        created_by="first@example.com",
        make_active=True,
    )["version"]

    second = SchemaService(
        JobStore(tmp_path / "second"),
        bucket_name="shared-bucket",
        schemas_prefix="schemas",
    )
    synced = second.list_versions()

    assert synced["cloud_sync"]["status"] == "synced"
    assert synced["active_version_id"] == created["version_id"]
    assert (
        second.resolve_version(created["version_id"])["created_by"]
        == "first@example.com"
    )
    assert f"schemas/versions/{created['version_id']}.json" in objects


def test_managed_json_schema_builds_a_strict_runtime_model() -> None:
    schema = {
        "title": "StudySchema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "score": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        },
        "required": ["name"],
    }
    model = model_from_json_schema("StudySchema", schema)

    parsed = model.model_validate({"name": "A", "score": 2})

    assert parsed.name == "A"
    assert parsed.score == 2.0
    with pytest.raises(ValidationError):
        model.model_validate({"name": "A", "unknown": "not allowed"})


def test_managed_json_schema_preserves_legacy_extra_ignore_behavior() -> None:
    schema = {
        "title": "StudySchema",
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    model = model_from_json_schema("StudySchema", schema)

    parsed = model.model_validate({"name": "A", "unknown": "ignored"})

    assert parsed.model_dump() == {"name": "A"}


def test_managed_json_schema_enforces_date_format_bounds() -> None:
    schema = {
        "title": "DatedStudySchema",
        "type": "object",
        "properties": {
            "inclusive": {
                "type": "string",
                "format": "date",
                "formatMinimum": "1879-01-01",
                "formatMaximum": "1910-12-31",
            },
            "exclusive": {
                "type": "string",
                "format": "date",
                "formatExclusiveMinimum": "1879-01-01",
                "formatExclusiveMaximum": "1910-12-31",
            },
        },
        "required": ["inclusive", "exclusive"],
        "additionalProperties": False,
    }
    model = model_from_json_schema("DatedStudySchema", schema)

    parsed = model.model_validate_json(
        '{"inclusive":"1879-01-01","exclusive":"1900-01-01"}',
        strict=True,
    )

    assert parsed.inclusive.isoformat() == "1879-01-01"
    with pytest.raises(ValidationError):
        model.model_validate_json(
            '{"inclusive":"1878-12-31","exclusive":"1900-01-01"}',
            strict=True,
        )
    with pytest.raises(ValidationError):
        model.model_validate_json(
            '{"inclusive":"1911-01-01","exclusive":"1900-01-01"}',
            strict=True,
        )
    with pytest.raises(ValidationError):
        model.model_validate_json(
            '{"inclusive":"1900-01-01","exclusive":"1879-01-01"}',
            strict=True,
        )
    with pytest.raises(ValidationError):
        model.model_validate_json(
            '{"inclusive":"1900-01-01","exclusive":"1910-12-31"}',
            strict=True,
        )


def test_managed_json_schema_rejects_invalid_date_format_bound() -> None:
    schema = {
        "title": "DatedStudySchema",
        "type": "object",
        "properties": {
            "observed": {
                "type": "string",
                "format": "date",
                "formatMinimum": "not-a-date",
            }
        },
        "required": ["observed"],
    }

    with pytest.raises(ValueError, match="formatMinimum"):
        model_from_json_schema("DatedStudySchema", schema)


def test_builtin_frontpage_retains_exact_legacy_json_schema() -> None:
    schema = FrontPage.model_json_schema()

    assert "serum" in schema["required"]
    assert "crossed_out" in schema["properties"]
    assert "crossed_out" not in schema["required"]
    hospital_stay = schema["$defs"]["HospitalStay"]["properties"]
    for field_name in ("admission_date", "release_date"):
        assert "formatMinimum" not in hospital_stay[field_name]
        assert "formatMaximum" not in hospital_stay[field_name]


def test_versioned_validation_uses_each_rows_schema_and_model(tmp_path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"image")
    rows = [
        {
            "image_name": "page.png",
            "model": "model-a",
            "schema_version_id": "sv_one",
            "first_column": "one",
            "second_column": "not in this version",
            "crossed_out": "support metadata",
        },
        {
            "image_name": "page.png",
            "model": "model-b",
            "schema_version_id": "sv_two",
            "first_column": "not in this version",
            "second_column": "two",
        },
    ]

    datapoints = build_validation_datapoints(
        rows,
        {"page.png": image},
        schema_fields_by_version={
            "sv_one": {"first_column", "crossed_out"},
            "sv_two": {"second_column"},
        },
    )

    assert [(item.model_name, item.field_name) for item in datapoints] == [
        ("model-a", "first_column"),
        ("model-b", "second_column"),
    ]
    assert datapoints[0].sampling_group == "model-a::sv_one::first_column::present"
    assert datapoints[1].sampling_group == "model-b::sv_two::second_column::present"


def test_dashboard_completeness_reports_leafs_not_parent_objects(tmp_path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "image_name": "a.png",
                "diagnoses": {"sektion": {"number": 12, "diagnoses": ["Diagnosis"]}},
                "crossed_out": "support metadata",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    analysis = analyze_dataset_file(
        dataset,
        schema_leaf_fields={
            "diagnoses.sektion.number",
            "diagnoses.sektion.diagnoses",
            "crossed_out",
        },
    )
    schema_columns = {item.column for item in analysis.schema_field_completeness}
    metadata_columns = {item.column for item in analysis.metadata_field_completeness}

    assert "diagnoses.sektion" not in schema_columns
    assert "diagnoses.sektion" not in metadata_columns
    assert "diagnoses.sektion.number" in schema_columns
    assert "diagnoses.sektion.diagnoses" in schema_columns
    assert "crossed_out" not in schema_columns


def test_dashboard_infers_legacy_schema_and_includes_fully_missing_leafs(
    tmp_path,
) -> None:
    dataset = tmp_path / "legacy.jsonl"
    dataset.write_text(
        json.dumps({"image_name": "a.png", "first_column": "value"}) + "\n",
        encoding="utf-8",
    )

    analysis = analyze_dataset_file(
        dataset,
        schema_fields_by_version={
            "sv_matching": {"first_column", "entirely_missing"},
            "sv_unrelated": {"unrelated_column"},
        },
        schema_names_by_version={
            "sv_matching": "Matching",
            "sv_unrelated": "Unrelated",
        },
    )
    completeness = {item.column: item for item in analysis.schema_field_completeness}

    assert completeness["first_column"].completeness == 100.0
    assert completeness["entirely_missing"].completeness == 0.0
    assert "unrelated_column" not in completeness


def test_dataset_inspection_handles_lists_and_prioritizes_provenance(tmp_path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "values": ["one", "two"],
                "image_name": "a.png",
                "model": "model-a",
                "schema_version_id": "sv_one",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    page = read_dataset_page(dataset)

    assert page["rows"][0]["values"] == ["one", "two"]
    assert page["columns"][:3] == ["image_name", "model", "schema_version_id"]


def test_absolute_local_image_hint_does_not_become_a_cloud_object(
    tmp_path, monkeypatch
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    image = images / "a.png"
    image.write_bytes(b"png")

    class Bucket:
        name = "bucket"

        def list_blobs(self, prefix=None):
            del prefix
            return []

    monkeypatch.setattr(
        "patientjournals.app.image_access.build_storage_bucket",
        lambda _name=None: Bucket(),
    )
    service = ImageAccessService(
        AppSettings(
            gcs_bucket_name="bucket",
            gcs_pages_prefix="pages",
            validation_images_root=str(images),
        )
    )

    result = service.dataset_image_link(
        image_name="a.png",
        object_hint=str(image),
    )

    assert result["source"] == "local"
    assert result["uri"] == str(image)


def test_submission_population_counts_configured_cloud_images_by_unique_name(
    monkeypatch,
) -> None:
    class Blob:
        def __init__(self, name: str) -> None:
            self.name = name

    class Bucket:
        name = "bucket"

    blobs_by_prefix = {
        "pages/a/": [Blob("pages/a/one.png"), Blob("pages/a/shared.png")],
        "pages/b/": [Blob("pages/b/two.png"), Blob("pages/b/shared.png")],
    }
    monkeypatch.setattr(
        "patientjournals.app.image_access.build_storage_bucket",
        lambda _name=None: Bucket(),
    )
    monkeypatch.setattr(
        "patientjournals.app.image_access.list_bucket_blobs",
        lambda _bucket, prefix: blobs_by_prefix[prefix],
    )

    result = ImageAccessService(
        AppSettings(gcs_bucket_name="bucket", gcs_pages_prefix="pages")
    ).submission_population(cloud_prefixes=("pages/a", "pages/b"))

    assert result == {
        "source": "cloud",
        "bucket": "bucket",
        "cloud_prefixes": ["pages/a", "pages/b"],
        "selection_count": 3,
    }


def test_cloud_dataset_cache_uses_full_object_path(tmp_path, monkeypatch) -> None:
    downloads: list[str] = []

    class Blob:
        def __init__(self, name: str) -> None:
            self.name = name

        def download_to_filename(self, destination: str) -> None:
            downloads.append(self.name)
            path = Path(destination)
            path.write_text(self.name, encoding="utf-8")

    class Bucket:
        def blob(self, name: str) -> Blob:
            return Blob(name)

    monkeypatch.setattr(
        dataset_module, "build_storage_bucket", lambda _name=None: Bucket()
    )

    first = dataset_module.download_cloud_dataset(
        "gs://bucket/datasets/run-a/current.jsonl",
        destination_root=tmp_path,
        use_cache=True,
    )
    repeated = dataset_module.download_cloud_dataset(
        "gs://bucket/datasets/run-a/current.jsonl",
        destination_root=tmp_path,
        use_cache=True,
    )
    second = dataset_module.download_cloud_dataset(
        "gs://bucket/datasets/run-b/current.jsonl",
        destination_root=tmp_path,
        use_cache=True,
    )

    assert first == repeated
    assert first != second
    assert first.parts[-4:] == ("bucket", "datasets", "run-a", "current.jsonl")
    assert second.parts[-4:] == ("bucket", "datasets", "run-b", "current.jsonl")
    assert downloads == [
        "datasets/run-a/current.jsonl",
        "datasets/run-b/current.jsonl",
    ]
