from __future__ import annotations

import json
from pathlib import Path

from google.api_core.exceptions import NotFound, PreconditionFailed
import pytest

from patientjournals.validation.publication import publish_dataset_version


class _MemoryBlob:
    def __init__(self, bucket: "_MemoryBucket", name: str) -> None:
        self.bucket = bucket
        self.name = name
        self.generation = None
        self.metadata = None

    def reload(self) -> None:
        item = self.bucket.objects.get(self.name)
        if item is None:
            raise NotFound("missing")
        self.generation = item["generation"]
        self.metadata = dict(item.get("metadata") or {})

    def download_as_bytes(self, *, if_generation_match=None) -> bytes:
        self.reload()
        if if_generation_match and int(self.generation) != int(if_generation_match):
            raise PreconditionFailed("generation changed")
        return bytes(self.bucket.objects[self.name]["data"])

    def download_to_filename(self, path: str, *, if_generation_match=None) -> None:
        Path(path).write_bytes(
            self.download_as_bytes(if_generation_match=if_generation_match)
        )

    def upload_from_filename(
        self,
        path: str,
        *,
        content_type: str,
        if_generation_match: int,
    ) -> None:
        assert content_type in {"application/jsonl", "text/csv"}
        collision = self.bucket.create_collision
        if if_generation_match == 0 and collision is not None:
            self.bucket.create_collision = None
            self.bucket.next_generation += 1
            self.bucket.objects[self.name] = {
                "data": bytes(collision["data"]),
                "generation": self.bucket.next_generation,
                "metadata": dict(collision["metadata"]),
            }
        if if_generation_match == 0 and self.name in self.bucket.objects:
            raise PreconditionFailed("already exists")
        self.bucket.next_generation += 1
        self.generation = self.bucket.next_generation
        self.bucket.objects[self.name] = {
            "data": Path(path).read_bytes(),
            "generation": self.generation,
            "metadata": dict(self.metadata or {}),
        }

    def upload_from_string(
        self,
        data: bytes,
        *,
        content_type: str,
        if_generation_match: int,
    ) -> None:
        assert content_type == "application/json"
        if self.bucket.fail_next_ledger_upload:
            self.bucket.fail_next_ledger_upload = False
            raise RuntimeError("simulated crash before ledger commit")
        current = self.bucket.objects.get(self.name)
        current_generation = int(current["generation"]) if current else 0
        if current_generation != int(if_generation_match):
            raise PreconditionFailed("ledger changed")
        self.bucket.next_generation += 1
        self.generation = self.bucket.next_generation
        self.bucket.objects[self.name] = {
            "data": bytes(data),
            "generation": self.generation,
            "metadata": dict(self.metadata or {}),
        }


class _MemoryBucket:
    name = "test-bucket"

    def __init__(self) -> None:
        self.objects: dict[str, dict[str, object]] = {}
        self.next_generation = 100
        self.create_collision: dict[str, object] | None = None
        self.fail_next_ledger_upload = False

    def blob(self, name: str) -> _MemoryBlob:
        return _MemoryBlob(self, name)

    def list_blobs(self, *, prefix: str):
        return [
            _MemoryBlob(self, name)
            for name in sorted(self.objects)
            if name.startswith(prefix)
        ]


def _publish(
    *,
    dataset: Path,
    source_run: Path,
    verification_run: Path,
    bucket: _MemoryBucket,
    metadata: dict[str, object] | None = None,
):
    return publish_dataset_version(
        dataset_path=dataset,
        source_run_dir=source_run,
        verification_run_dir=verification_run,
        bucket=bucket,
        datasets_prefix="datasets",
        candidate_hash="candidate",
        verification_prompt_hash="prompt",
        metadata=metadata or {"verification_model": "gemini-test"},
    )


def test_publication_allocates_immutable_versions_and_replays_once(tmp_path) -> None:
    source_run = tmp_path / "submits" / "source-run"
    source_run.mkdir(parents=True)
    bucket = _MemoryBucket()

    first_dataset = tmp_path / "first.jsonl"
    first_dataset.write_text('{"name":"A"}\n', encoding="utf-8")
    first_run = tmp_path / "verifications" / "first"
    first = _publish(
        dataset=first_dataset,
        source_run=source_run,
        verification_run=first_run,
        bucket=bucket,
    )
    replay = _publish(
        dataset=first_dataset,
        source_run=source_run,
        verification_run=first_run,
        bucket=bucket,
    )

    second_dataset = tmp_path / "second.jsonl"
    second_dataset.write_text('{"name":"B"}\n', encoding="utf-8")
    second = _publish(
        dataset=second_dataset,
        source_run=source_run,
        verification_run=tmp_path / "verifications" / "second",
        bucket=bucket,
    )

    assert first.version_id == "v001"
    assert replay == first
    assert second.version_id == "v002"
    assert Path(first.local_path).read_text(encoding="utf-8") == '{"name":"A"}\n'
    assert first.gcs_uri != second.gcs_uri
    assert first.gcs_generation and second.gcs_generation


def test_publication_recovers_ledger_and_version_from_cloud(tmp_path) -> None:
    source_run = tmp_path / "submits" / "source-run"
    source_run.mkdir(parents=True)
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"name":"A"}\n', encoding="utf-8")
    verification_run = tmp_path / "verifications" / "first"
    bucket = _MemoryBucket()
    first = _publish(
        dataset=dataset,
        source_run=source_run,
        verification_run=verification_run,
        bucket=bucket,
    )

    Path(first.local_path).unlink()
    Path(first.ledger_path).unlink()
    recovered = _publish(
        dataset=dataset,
        source_run=source_run,
        verification_run=verification_run,
        bucket=bucket,
    )

    assert recovered.version_id == "v001"
    assert Path(recovered.local_path).is_file()
    assert Path(recovered.ledger_path).is_file()


def test_publication_recovers_orphan_uploaded_before_ledger_commit(tmp_path) -> None:
    source_run = tmp_path / "submits" / "source-run"
    source_run.mkdir(parents=True)
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"name":"A"}\n', encoding="utf-8")
    verification_run = tmp_path / "verifications" / "first"
    bucket = _MemoryBucket()
    bucket.fail_next_ledger_upload = True

    with pytest.raises(RuntimeError, match="simulated crash"):
        _publish(
            dataset=dataset,
            source_run=source_run,
            verification_run=verification_run,
            bucket=bucket,
        )

    assert not any(name.endswith("dataset_versions.json") for name in bucket.objects)
    orphan_name = next(
        name for name in bucket.objects if name.endswith("v001_model_validation.jsonl")
    )
    orphan_generation = bucket.objects[orphan_name]["generation"]

    recovered = _publish(
        dataset=dataset,
        source_run=source_run,
        verification_run=verification_run,
        bucket=bucket,
    )

    assert recovered.version_id == "v001"
    assert recovered.gcs_generation == str(orphan_generation)
    ledger_name = next(
        name for name in bucket.objects if name.endswith("dataset_versions.json")
    )
    ledger = json.loads(bytes(bucket.objects[ledger_name]["data"]))
    assert [item["version_id"] for item in ledger["versions"]] == ["v001"]


def test_publication_skips_different_worker_orphan_after_create_race(tmp_path) -> None:
    source_run = tmp_path / "submits" / "source-run"
    source_run.mkdir(parents=True)
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"name":"A"}\n', encoding="utf-8")
    bucket = _MemoryBucket()
    competing_bytes = b'{"name":"competing"}\n'
    bucket.create_collision = {
        "data": competing_bytes,
        "metadata": {
            "artifact_kind": "validated_dataset_version",
            "dataset_version": "v001",
            "dataset_sha256": "competing-sha",
            "publication_idempotency_key": "competing-publication",
            "publication_provenance_sha256": "competing-provenance",
            "source_run_id": "source-run",
            "verification_run_id": "other-verifier",
        },
    }

    publication = _publish(
        dataset=dataset,
        source_run=source_run,
        verification_run=tmp_path / "verifications" / "current",
        bucket=bucket,
    )

    orphan_name = next(
        name for name in bucket.objects if name.endswith("v001_model_validation.jsonl")
    )
    assert publication.version_id == "v002"
    assert bytes(bucket.objects[orphan_name]["data"]) == competing_bytes
    ledger_name = next(
        name for name in bucket.objects if name.endswith("dataset_versions.json")
    )
    ledger = json.loads(bytes(bucket.objects[ledger_name]["data"]))
    assert [item["version_id"] for item in ledger["versions"]] == ["v002"]


def test_caller_metadata_cannot_override_immutable_publication_identity(
    tmp_path,
) -> None:
    source_run = tmp_path / "submits" / "source-run"
    source_run.mkdir(parents=True)
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"name":"A"}\n', encoding="utf-8")
    bucket = _MemoryBucket()

    publication = _publish(
        dataset=dataset,
        source_run=source_run,
        verification_run=tmp_path / "verifications" / "real-verifier",
        bucket=bucket,
        metadata={
            "verification_run_id": "forged-verifier",
            "source_run_id": "forged-source",
            "dataset_sha256": "forged-dataset",
            "publication_idempotency_key": "forged-publication",
            "publication_provenance_sha256": "forged-provenance",
        },
    )

    version_name = next(
        name for name in bucket.objects if name.endswith("v001_model_validation.jsonl")
    )
    cloud_metadata = bucket.objects[version_name]["metadata"]
    assert cloud_metadata["verification_run_id"] == "real-verifier"
    assert cloud_metadata["source_run_id"] == "source-run"
    assert cloud_metadata["dataset_sha256"] == publication.sha256
    assert cloud_metadata["publication_idempotency_key"] == publication.idempotency_key
    assert (
        cloud_metadata["publication_provenance_sha256"]
        == publication.publication_provenance_sha256
    )


def test_same_verifier_run_cannot_publish_changed_bytes(tmp_path) -> None:
    source_run = tmp_path / "submits" / "source-run"
    source_run.mkdir(parents=True)
    bucket = _MemoryBucket()
    verification_run = tmp_path / "verifications" / "same"
    first = tmp_path / "first.jsonl"
    first.write_text('{"name":"A"}\n', encoding="utf-8")
    _publish(
        dataset=first,
        source_run=source_run,
        verification_run=verification_run,
        bucket=bucket,
    )
    changed = tmp_path / "changed.jsonl"
    changed.write_text('{"name":"B"}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="same verifier run"):
        _publish(
            dataset=changed,
            source_run=source_run,
            verification_run=verification_run,
            bucket=bucket,
        )


def test_same_verifier_run_cannot_reuse_version_with_changed_provenance(
    tmp_path,
) -> None:
    source_run = tmp_path / "submits" / "source-run"
    source_run.mkdir(parents=True)
    bucket = _MemoryBucket()
    verification_run = tmp_path / "verifications" / "same"
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"name":"A"}\n', encoding="utf-8")
    _publish(
        dataset=dataset,
        source_run=source_run,
        verification_run=verification_run,
        bucket=bucket,
        metadata={
            "verification_model": "gemini-test",
            "field_corrections_sha256": "corrections-a",
        },
    )

    with pytest.raises(RuntimeError, match="different publication bytes or provenance"):
        _publish(
            dataset=dataset,
            source_run=source_run,
            verification_run=verification_run,
            bucket=bucket,
            metadata={
                "verification_model": "gemini-test",
                "field_corrections_sha256": "corrections-b",
            },
        )


def test_same_verifier_run_cannot_replace_orphan_with_changed_provenance(
    tmp_path,
) -> None:
    source_run = tmp_path / "submits" / "source-run"
    source_run.mkdir(parents=True)
    bucket = _MemoryBucket()
    bucket.fail_next_ledger_upload = True
    verification_run = tmp_path / "verifications" / "same"
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"name":"A"}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="simulated crash"):
        _publish(
            dataset=dataset,
            source_run=source_run,
            verification_run=verification_run,
            bucket=bucket,
            metadata={
                "verification_model": "gemini-test",
                "field_corrections_sha256": "corrections-a",
            },
        )

    with pytest.raises(RuntimeError, match="immutable orphan"):
        _publish(
            dataset=dataset,
            source_run=source_run,
            verification_run=verification_run,
            bucket=bucket,
            metadata={
                "verification_model": "gemini-test",
                "field_corrections_sha256": "corrections-b",
            },
        )
