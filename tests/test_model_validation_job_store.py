from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from patientjournals.app.job_store import JobStore
from patientjournals.validation.publication import publication_idempotency_key


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


PUBLICATION_PROVENANCE_SHA256 = hashlib.sha256(
    b"test-publication-provenance"
).hexdigest()


def _registered_store(tmp_path: Path) -> tuple[JobStore, Path]:
    run_dir = tmp_path / "submits" / "20260827_120000"
    run_dir.mkdir(parents=True)
    store = JobStore(tmp_path)
    store.sync_legacy_submit_run(
        run_dir,
        batch_meta={
            "model": "gemini-3.7-flash",
            "provider": "gemini",
            "schema_name": "FrontPage",
            "batch_jobs": [],
        },
        image_count=1,
        status="submitted",
    )
    return store, run_dir


def test_candidate_retrieval_does_not_publish_v1(tmp_path: Path) -> None:
    store, run_dir = _registered_store(tmp_path)
    candidate_dataset = run_dir / "candidate_dataset.jsonl"
    candidate_pages = run_dir / "page_candidates.jsonl"
    candidate_dataset.write_text('{"image_name":"page.png"}\n', encoding="utf-8")
    candidate_pages.write_text(
        '{"key":"pages/page.png","candidate":{"name":"A"}}\n',
        encoding="utf-8",
    )
    routing = run_dir / "deterministic_routing.jsonl"
    routing.write_text('{"key":"pages/page.png","route":"heavy_review"}\n')

    payload = store.record_candidate_retrieval(
        run_dir,
        {
            "dataset_path": str(candidate_dataset),
            "page_candidates_path": str(candidate_pages),
            "deterministic_routing_path": str(routing),
            "expected_pages": 1,
            "successful_pages": 1,
        },
        signature="candidate-signature",
    )

    record = store.record_for_run_dir(run_dir)
    assert payload["model_validation_status"] == "pending"
    assert record["status"] == "validation_pending"
    assert record["dataset"].get("versions", []) == []
    assert not record["dataset"].get("current_path")


def test_completed_model_validation_publishes_immutable_v1(tmp_path: Path) -> None:
    store, run_dir = _registered_store(tmp_path)
    candidate_dataset = run_dir / "candidate_dataset.jsonl"
    candidate_dataset.write_text('{"image_name":"page.png","name":"A"}\n')
    store.record_candidate_retrieval(
        run_dir,
        {
            "dataset_path": str(candidate_dataset),
            "expected_pages": 1,
            "successful_pages": 1,
        },
        signature="candidate-signature",
    )
    verification_run = tmp_path / "verifications" / "20260827_130000"
    verification_run.mkdir(parents=True)
    store.mark_model_validation_submitted(
        run_dir,
        verification_run_dir=verification_run,
        model="gemini-3.1-pro-preview",
        apply_mode="apply_patches",
    )
    verified_dataset = verification_run / "verified_dataset.jsonl"
    verified_dataset.write_text('{"image_name":"page.png","name":"B"}\n')

    store.record_model_validation_result(
        run_dir,
        {
            "dataset_path": str(verified_dataset),
            "verification_run_dir": str(verification_run),
            "verification_model": "gemini-3.1-pro-preview",
            "verification_prompt_hash": "prompt-hash",
            "candidate_hash": "candidate-hash",
            "expected_pages": 1,
            "completed_pages": 1,
            "successful_pages": 1,
            "missing_pages": 0,
            "failed_pages": 0,
            "unverifiable_pages": 0,
            "rows_written": 1,
            "publishable": True,
            "dataset_version": 1,
            "dataset_version_id": "v001",
            "dataset_gcs_uri": "gs://bucket/source/validation_versions/v001.jsonl",
            "dataset_gcs_generation": "101",
            "dataset_sha256": _sha256(verified_dataset),
            "publication_provenance_sha256": PUBLICATION_PROVENANCE_SHA256,
            "dataset_version_ledger_gcs_uri": "gs://bucket/source/versions.json",
        },
        publish_dataset=True,
    )

    record = store.record_for_run_dir(run_dir)
    versions = record["dataset"]["versions"]
    assert record["status"] == "retrieved_complete"
    assert record["model_validation"]["status"] == "published"
    assert len(versions) == 1
    version_path = Path(versions[0]["path"])
    current_path = Path(record["dataset"]["current_path"])
    assert version_path.name == "v001_model_validation.jsonl"
    assert version_path.read_text() == current_path.read_text()

    current_path.write_text('{"image_name":"page.png","name":"later"}\n')
    assert '"name":"B"' in version_path.read_text()


def test_report_only_validation_never_publishes_dataset(tmp_path: Path) -> None:
    store, run_dir = _registered_store(tmp_path)
    store.record_model_validation_result(
        run_dir,
        {
            "model_validation_status": "report_only",
            "verification_run_dir": "runs/verifications/example",
            "expected_pages": 1,
            "successful_pages": 1,
        },
        publish_dataset=False,
    )

    record = store.record_for_run_dir(run_dir)
    assert record["status"] == "validation_report_only"
    assert record["dataset"].get("versions", []) == []


def test_repeated_validation_runs_create_traceable_v1_v2(tmp_path: Path) -> None:
    store, run_dir = _registered_store(tmp_path)
    for attempt, value in enumerate(("A", "B"), start=1):
        verification_run = tmp_path / "verifications" / f"attempt-{attempt}"
        verification_run.mkdir(parents=True)
        store.mark_model_validation_submitted(
            run_dir,
            verification_run_dir=verification_run,
            model="gemini-3.1-pro-preview",
            apply_mode="apply_patches",
            thinking_level="high",
            scope="all",
            max_output_tokens=4096,
            num_chunks=attempt,
        )
        verified_dataset = verification_run / "verified_dataset.jsonl"
        verified_dataset.write_text(
            f'{{"image_name":"page.png","name":"{value}"}}\n',
            encoding="utf-8",
        )
        store.record_model_validation_result(
            run_dir,
            {
                "dataset_path": str(verified_dataset),
                "verification_run_dir": str(verification_run),
                "verification_model": "gemini-3.1-pro-preview",
                "verification_prompt_hash": f"prompt-{attempt}",
                "candidate_hash": "same-candidate",
                "expected_pages": 1,
                "completed_pages": 1,
                "successful_pages": 1,
                "missing_pages": 0,
                "failed_pages": 0,
                "rows_written": 1,
                "confirmed_pages": 1,
                "needs_correction_pages": 0,
                "unverifiable_pages": 0,
                "publishable": True,
                "dataset_version": attempt,
                "dataset_version_id": f"v{attempt:03d}",
                "dataset_gcs_uri": (
                    f"gs://bucket/source/validation_versions/v{attempt:03d}.jsonl"
                ),
                "dataset_gcs_generation": str(100 + attempt),
                "dataset_sha256": _sha256(verified_dataset),
                "publication_provenance_sha256": PUBLICATION_PROVENANCE_SHA256,
                "dataset_version_ledger_gcs_uri": ("gs://bucket/source/versions.json"),
            },
            publish_dataset=True,
        )

    record = store.record_for_run_dir(run_dir)
    versions = record["dataset"]["versions"]
    runs = record["model_validation"]["runs"]
    assert [Path(item["path"]).name for item in versions] == [
        "v001_model_validation.jsonl",
        "v002_model_validation.jsonl",
    ]
    assert [item["dataset_version"] for item in runs] == [1, 2]
    assert [item["verification_prompt_hash"] for item in runs] == [
        "prompt-1",
        "prompt-2",
    ]


def test_job_store_defensively_rejects_unsafe_validation_publication(
    tmp_path: Path,
) -> None:
    store, run_dir = _registered_store(tmp_path)
    dataset = run_dir / "unsafe.jsonl"
    dataset.write_text('{"image_name":"page.png"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="publishable=True"):
        store.record_model_validation_result(
            run_dir,
            {
                "dataset_path": str(dataset),
                "expected_pages": 1,
                "completed_pages": 1,
                "missing_pages": 0,
                "failed_pages": 0,
                "unverifiable_pages": 0,
            },
            publish_dataset=True,
        )

    assert store.record_for_run_dir(run_dir)["dataset"].get("versions", []) == []


def test_retrieving_same_published_verifier_run_is_idempotent(tmp_path: Path) -> None:
    store, run_dir = _registered_store(tmp_path)
    verification_run = tmp_path / "verifications" / "same-run"
    verification_run.mkdir(parents=True)
    store.mark_model_validation_submitted(
        run_dir,
        verification_run_dir=verification_run,
        model="gemini-3.1-pro-preview",
        apply_mode="apply_patches",
    )
    verified_dataset = verification_run / "verified_dataset.jsonl"
    verified_dataset.write_text('{"image_name":"page.png","name":"A"}\n')
    payload = {
        "dataset_path": str(verified_dataset),
        "verification_run_dir": str(verification_run),
        "verification_model": "gemini-3.1-pro-preview",
        "verification_prompt_hash": "prompt",
        "candidate_hash": "candidate",
        "expected_pages": 1,
        "completed_pages": 1,
        "successful_pages": 1,
        "missing_pages": 0,
        "failed_pages": 0,
        "unverifiable_pages": 0,
        "rows_written": 1,
        "publishable": True,
        "dataset_version": 1,
        "dataset_version_id": "v001",
        "dataset_gcs_uri": "gs://bucket/source/v001.jsonl",
        "dataset_gcs_generation": "101",
        "dataset_sha256": _sha256(verified_dataset),
        "publication_provenance_sha256": PUBLICATION_PROVENANCE_SHA256,
        "dataset_version_ledger_gcs_uri": "gs://bucket/source/versions.json",
    }

    first = store.record_model_validation_result(run_dir, payload, publish_dataset=True)
    second = store.record_model_validation_result(
        run_dir, payload, publish_dataset=True
    )

    record = store.record_for_run_dir(run_dir)
    assert len(record["dataset"]["versions"]) == 1
    assert len(record["model_validation"]["runs"]) == 1
    assert first["dataset_version"] == second["dataset_version"] == 1
    assert second["idempotent_replay"] is True


def test_published_verifier_replay_is_portable_across_relocated_runs(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    original_source = tmp_path / "original" / "submits" / "source-run"
    original_source.mkdir(parents=True)
    store = JobStore(state_root)
    store.sync_legacy_submit_run(
        original_source,
        batch_meta={
            "model": "gemini-3.7-flash",
            "provider": "gemini",
            "schema_name": "FrontPage",
            "batch_jobs": [],
        },
        image_count=1,
        status="submitted",
    )
    original_verification = tmp_path / "original" / "verifications" / "verify-run"
    original_verification.mkdir(parents=True)
    store.mark_model_validation_submitted(
        original_source,
        verification_run_dir=original_verification,
        model="gemini-3.1-pro-preview",
        apply_mode="apply_patches",
    )
    original_dataset = original_verification / "verified.jsonl"
    original_dataset.write_text('{"image_name":"page.png","name":"A"}\n')
    payload = {
        "dataset_path": str(original_dataset),
        "verification_run_dir": str(original_verification),
        "verification_model": "gemini-3.1-pro-preview",
        "verification_prompt_hash": "prompt",
        "candidate_hash": "candidate",
        "expected_pages": 1,
        "completed_pages": 1,
        "successful_pages": 1,
        "missing_pages": 0,
        "failed_pages": 0,
        "unverifiable_pages": 0,
        "rows_written": 1,
        "publishable": True,
        "dataset_version": 1,
        "dataset_version_id": "v001",
        "dataset_gcs_uri": "gs://bucket/source-run/v001.jsonl",
        "dataset_gcs_generation": "101",
        "dataset_sha256": _sha256(original_dataset),
        "publication_provenance_sha256": PUBLICATION_PROVENANCE_SHA256,
        "dataset_version_ledger_gcs_uri": (
            "gs://bucket/source-run/dataset_versions.json"
        ),
    }
    first = store.record_model_validation_result(
        original_source, payload, publish_dataset=True
    )

    relocated_source = tmp_path / "relocated" / "submits" / "source-run"
    relocated_source.mkdir(parents=True)
    relocated_verification = tmp_path / "relocated" / "verifications" / "verify-run"
    relocated_verification.mkdir(parents=True)
    relocated_dataset = relocated_verification / "verified.jsonl"
    relocated_dataset.write_bytes(original_dataset.read_bytes())
    replay_payload = {
        **payload,
        "dataset_path": str(relocated_dataset),
        "verification_run_dir": str(relocated_verification),
    }

    replay = store.record_model_validation_result(
        relocated_source, replay_payload, publish_dataset=True
    )

    record = store.record_for_run_dir(relocated_source)
    assert replay["idempotent_replay"] is True
    assert replay["source_run_id"] == "source-run"
    assert replay["verification_run_id"] == "verify-run"
    assert (
        replay["dataset_publication_idempotency_key"]
        == first["dataset_publication_idempotency_key"]
    )
    assert len(record["dataset"]["versions"]) == 1
    assert len(record["model_validation"]["runs"]) == 1


def test_relocated_replay_rejects_changed_cloud_provenance(tmp_path: Path) -> None:
    store, run_dir = _registered_store(tmp_path)
    verification_run = tmp_path / "verifications" / "verify-run"
    verification_run.mkdir(parents=True)
    store.mark_model_validation_submitted(
        run_dir,
        verification_run_dir=verification_run,
        model="gemini-3.1-pro-preview",
        apply_mode="apply_patches",
    )
    dataset = verification_run / "verified.jsonl"
    dataset.write_text('{"image_name":"page.png","name":"A"}\n')
    payload = {
        "dataset_path": str(dataset),
        "verification_run_dir": str(verification_run),
        "verification_model": "gemini-3.1-pro-preview",
        "verification_prompt_hash": "prompt",
        "candidate_hash": "candidate",
        "expected_pages": 1,
        "completed_pages": 1,
        "successful_pages": 1,
        "missing_pages": 0,
        "failed_pages": 0,
        "unverifiable_pages": 0,
        "rows_written": 1,
        "publishable": True,
        "dataset_version": 1,
        "dataset_version_id": "v001",
        "dataset_gcs_uri": "gs://bucket/source/v001.jsonl",
        "dataset_gcs_generation": "101",
        "dataset_sha256": _sha256(dataset),
        "publication_provenance_sha256": PUBLICATION_PROVENANCE_SHA256,
        "dataset_version_ledger_gcs_uri": "gs://bucket/source/versions.json",
    }
    store.record_model_validation_result(run_dir, payload, publish_dataset=True)

    relocated_verification = tmp_path / "elsewhere" / "verify-run"
    relocated_verification.mkdir(parents=True)
    relocated_dataset = relocated_verification / "verified.jsonl"
    relocated_dataset.write_bytes(dataset.read_bytes())
    with pytest.raises(ValueError, match="dataset_gcs_generation changed"):
        store.record_model_validation_result(
            tmp_path / "elsewhere" / run_dir.name,
            {
                **payload,
                "dataset_path": str(relocated_dataset),
                "verification_run_dir": str(relocated_verification),
                "dataset_gcs_generation": "102",
            },
            publish_dataset=True,
        )


def test_retry_reconciles_version_recorded_before_completed_run(
    tmp_path: Path,
) -> None:
    store, run_dir = _registered_store(tmp_path)
    verification_run = tmp_path / "verifications" / "crashed-run"
    verification_run.mkdir(parents=True)
    store.mark_model_validation_submitted(
        run_dir,
        verification_run_dir=verification_run,
        model="gemini-3.1-pro-preview",
        apply_mode="apply_patches",
    )
    dataset = verification_run / "verified.jsonl"
    dataset.write_text('{"image_name":"page.png","name":"A"}\n')
    digest = _sha256(dataset)
    publication_key = publication_idempotency_key(
        source_run_id=run_dir.name,
        verification_run_id=verification_run.name,
        candidate_hash="candidate",
        verification_prompt_hash="prompt",
        dataset_sha256=digest,
        publication_provenance_sha256=PUBLICATION_PROVENANCE_SHA256,
    )
    payload = {
        "dataset_path": str(dataset),
        "verification_run_dir": str(verification_run),
        "source_run_id": run_dir.name,
        "verification_run_id": verification_run.name,
        "verification_model": "gemini-3.1-pro-preview",
        "verification_prompt_hash": "prompt",
        "candidate_hash": "candidate",
        "expected_pages": 1,
        "completed_pages": 1,
        "successful_pages": 1,
        "missing_pages": 0,
        "failed_pages": 0,
        "unverifiable_pages": 0,
        "rows_written": 1,
        "publishable": True,
        "dataset_version": 1,
        "dataset_version_id": "v001",
        "dataset_gcs_uri": "gs://bucket/source/v001.jsonl",
        "dataset_gcs_generation": "101",
        "dataset_sha256": digest,
        "publication_provenance_sha256": PUBLICATION_PROVENANCE_SHA256,
        "dataset_version_ledger_gcs_uri": "gs://bucket/source/versions.json",
        "dataset_publication_idempotency_key": publication_key,
    }

    # Simulate the crash boundary: record_retrieval wrote v001 and the job JSON,
    # but record_model_validation_result did not yet append the completed run.
    store.record_retrieval(
        run_dir,
        payload,
        signature=publication_key,
        operation="model_validation",
        version_number=1,
    )
    with store._connect() as conn:
        conn.execute(
            "DELETE FROM dataset_versions WHERE job_id = ?",
            (run_dir.name,),
        )
    before = store.record_for_run_dir(run_dir)
    assert len(before["dataset"]["versions"]) == 1
    assert before.get("model_validation", {}).get("runs", []) == []

    result = store.record_model_validation_result(
        run_dir, payload, publish_dataset=True
    )

    after = store.record_for_run_dir(run_dir)
    assert result["reconciled_recorded_dataset_version"] is True
    assert len(after["dataset"]["versions"]) == 1
    assert len(after["model_validation"]["runs"]) == 1
    assert len(store.dataset_versions(run_dir.name)) == 1


def test_retry_refuses_occupied_version_with_different_publication(
    tmp_path: Path,
) -> None:
    store, run_dir = _registered_store(tmp_path)
    verification_run = tmp_path / "verifications" / "crashed-run"
    verification_run.mkdir(parents=True)
    store.mark_model_validation_submitted(
        run_dir,
        verification_run_dir=verification_run,
        model="gemini-3.1-pro-preview",
        apply_mode="apply_patches",
    )
    original = verification_run / "original.jsonl"
    original.write_text('{"image_name":"page.png","name":"A"}\n')
    original_digest = _sha256(original)
    original_key = publication_idempotency_key(
        source_run_id=run_dir.name,
        verification_run_id=verification_run.name,
        candidate_hash="candidate",
        verification_prompt_hash="prompt",
        dataset_sha256=original_digest,
        publication_provenance_sha256=PUBLICATION_PROVENANCE_SHA256,
    )
    original_payload = {
        "dataset_path": str(original),
        "verification_run_dir": str(verification_run),
        "source_run_id": run_dir.name,
        "verification_run_id": verification_run.name,
        "verification_model": "gemini-3.1-pro-preview",
        "verification_prompt_hash": "prompt",
        "candidate_hash": "candidate",
        "expected_pages": 1,
        "completed_pages": 1,
        "successful_pages": 1,
        "missing_pages": 0,
        "failed_pages": 0,
        "unverifiable_pages": 0,
        "rows_written": 1,
        "publishable": True,
        "dataset_version": 1,
        "dataset_version_id": "v001",
        "dataset_gcs_uri": "gs://bucket/source/v001.jsonl",
        "dataset_gcs_generation": "101",
        "dataset_sha256": original_digest,
        "publication_provenance_sha256": PUBLICATION_PROVENANCE_SHA256,
        "dataset_version_ledger_gcs_uri": "gs://bucket/source/versions.json",
        "dataset_publication_idempotency_key": original_key,
    }
    store.record_retrieval(
        run_dir,
        original_payload,
        signature=original_key,
        operation="model_validation",
        version_number=1,
    )
    original_version = Path(
        store.record_for_run_dir(run_dir)["dataset"]["versions"][0]["path"]
    )

    conflicting = verification_run / "conflicting.jsonl"
    conflicting.write_text('{"image_name":"page.png","name":"B"}\n')
    conflict_digest = _sha256(conflicting)
    with pytest.raises(RuntimeError, match="dataset_sha256 changed"):
        store.record_model_validation_result(
            run_dir,
            {
                **original_payload,
                "dataset_path": str(conflicting),
                "dataset_sha256": conflict_digest,
                "dataset_publication_idempotency_key": publication_idempotency_key(
                    source_run_id=run_dir.name,
                    verification_run_id=verification_run.name,
                    candidate_hash="candidate",
                    verification_prompt_hash="prompt",
                    dataset_sha256=conflict_digest,
                    publication_provenance_sha256=(PUBLICATION_PROVENANCE_SHA256),
                ),
            },
            publish_dataset=True,
        )

    assert '"name":"A"' in original_version.read_text()
