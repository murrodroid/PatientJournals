from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from patientjournals.batch.ocr_context import CloudBlobIdentity, CloudOcrMetadata
from patientjournals.batch import retrieve as batch_retrieve
from patientjournals.batch import verify
from patientjournals.config.prompts import (
    MODEL_VALIDATION_CANDIDATE_HEADING,
    MODEL_VALIDATION_OCR_HEADING,
    MODEL_VALIDATION_SCHEMA_HEADING,
    build_model_validation_prompt,
)
from patientjournals.config.schemas import (
    PageModelValidation,
    TextPage,
    ValidationIssuePatch,
)
from patientjournals.shared.ocr import OcrDocument, OcrLine
from patientjournals.batch.results import RetrieveBatchResult
from patientjournals.validation.candidates import (
    PageCandidateRecord,
    PageCandidateWriter,
    candidate_sha256,
    read_page_candidates,
    sanitize_extraction_metadata,
    write_page_candidates,
)
from patientjournals.validation.input_manifest import InputImageManifestRecord
from patientjournals.validation.input_manifest import ocr_document_sha256


def _identity(generation: str) -> CloudBlobIdentity:
    return CloudBlobIdentity(
        bucket="bucket",
        name="pages/page.png",
        generation=generation,
        size=123,
        crc32c="crc",
        md5_hash="md5",
        etag=None,
    )


def _ocr_metadata(generation: str) -> CloudOcrMetadata:
    return CloudOcrMetadata(
        source=_identity(generation),
        document=OcrDocument(
            image_sha256="image-digest",
            width=100,
            height=100,
            coordinate_scale=1000,
            backend="test",
            lines=(OcrLine(text="Jensen", box=(1, 2, 3, 4)),),
        ),
        created_at="2026-01-01T00:00:00Z",
    )


def _input_record(generation: str = "1") -> InputImageManifestRecord:
    sidecar_source = CloudBlobIdentity(
        bucket="bucket",
        name="pages/page.png.ocr.json",
        generation="7",
        size=8,
        crc32c="sidecar-crc",
        md5_hash="sidecar-md5",
        etag=None,
    )
    document = _ocr_metadata(generation).document
    return InputImageManifestRecord(
        key="pages/page.png",
        mime_type="image/png",
        image_source=_identity(generation).to_dict(),
        ocr_sidecar_name="pages/page.png.ocr.json",
        ocr_sidecar_source=sidecar_source.to_dict(),
        ocr_sidecar_sha256=__import__("hashlib").sha256(b"sidecar").hexdigest(),
        ocr_image_sha256="image-digest",
        ocr_document_sha256=ocr_document_sha256(document),
        ocr_backend="test",
        ocr_line_count=1,
    )


def _input_record_without_ocr(generation: str = "1") -> InputImageManifestRecord:
    return InputImageManifestRecord(
        key="pages/page.png",
        mime_type="image/png",
        image_source=_identity(generation).to_dict(),
        ocr_enabled=False,
    )


def test_candidate_artifact_is_canonical_and_drops_thoughts(tmp_path) -> None:
    path = tmp_path / "page_candidates.jsonl"
    with PageCandidateWriter(path) as writer:
        assert writer.write(
            key="pages/page.png",
            candidate={"patient": {"name": "Jensen"}},
            extraction_metadata={
                "model": "gemini",
                "provider": "gemini",
                "thoughts": "private chain of thought",
            },
        )
        assert not writer.write(
            key="pages/page.png",
            candidate={"patient": {"name": "Duplicate"}},
        )

    records = read_page_candidates(path)

    assert len(records) == 1
    assert records[0].candidate == {"patient": {"name": "Jensen"}}
    assert records[0].extraction_metadata == {
        "model": "gemini",
        "provider": "gemini",
    }
    assert candidate_sha256(records[0].candidate) == candidate_sha256(
        {"patient": {"name": "Jensen"}}
    )


def test_candidate_artifact_resolves_from_extraction_batch_results(tmp_path) -> None:
    submit_run = tmp_path / "submits" / "source"
    retrieve_run = tmp_path / "retrieves" / "result"
    verification_run = tmp_path / "verifications" / "verify"
    submit_run.mkdir(parents=True)
    retrieve_run.mkdir(parents=True)
    verification_run.mkdir(parents=True)
    source = retrieve_run / "page_candidates.jsonl"
    with PageCandidateWriter(source) as writer:
        writer.write(key="pages/page.png", candidate={"name": "Jensen"})
    (submit_run / "batch_results.json").write_text(
        json.dumps({"page_candidates_path": str(source)}),
        encoding="utf-8",
    )

    resolved = verify._copy_or_download_candidate_artifact(
        request=verify.ModelValidationSubmitRequest(
            source_run_dir=str(submit_run)
        ),
        run_dir=verification_run,
        storage_client=object(),
    )

    assert read_page_candidates(resolved)[0].candidate == {"name": "Jensen"}


def test_candidate_artifact_falls_back_to_cloud_when_local_path_is_stale(
    tmp_path, monkeypatch
) -> None:
    submit_run = tmp_path / "submits" / "source"
    verification_run = tmp_path / "verifications" / "verify"
    submit_run.mkdir(parents=True)
    verification_run.mkdir(parents=True)
    (submit_run / "batch_results.json").write_text(
        json.dumps(
            {
                "page_candidates_path": str(tmp_path / "other-machine" / "missing.jsonl"),
                "page_candidates_gcs_uri": "gs://bucket/validation/page_candidates.jsonl",
            }
        ),
        encoding="utf-8",
    )

    def download(_client, uri: str, destination):
        assert uri.startswith("gs://bucket/")
        with PageCandidateWriter(destination) as writer:
            writer.write(key="pages/page.png", candidate={"name": "Jensen"})
        return destination

    monkeypatch.setattr(verify, "_download_gcs_file", download)
    resolved = verify._copy_or_download_candidate_artifact(
        request=verify.ModelValidationSubmitRequest(
            source_run_dir=str(submit_run)
        ),
        run_dir=verification_run,
        storage_client=object(),
    )

    assert read_page_candidates(resolved)[0].candidate == {"name": "Jensen"}


def test_direct_retrieve_records_candidate_location_on_submit_run(tmp_path) -> None:
    submit_run = tmp_path / "submits" / "source"
    retrieve_run = tmp_path / "retrieves" / "result"
    submit_run.mkdir(parents=True)
    retrieve_run.mkdir(parents=True)
    candidates = retrieve_run / "page_candidates.jsonl"
    candidates.write_text("{}\n", encoding="utf-8")
    dataset = retrieve_run / "result_prevalidation_rows.jsonl"
    dataset.write_text("", encoding="utf-8")
    result = RetrieveBatchResult(
        dataset_path=dataset,
        run_dir=retrieve_run,
        provider="gemini",
        batch_count=1,
        output_file_count=1,
        rows_written=3,
        error_rows=0,
        expected_pages=2,
        observed_pages=2,
        successful_pages=2,
        page_candidates_path=candidates,
        page_candidates_gcs_uri=(
            "gs://bucket/validations/candidates/result/page_candidates.jsonl"
        ),
    )

    path = batch_retrieve._record_candidate_retrieval_for_cli(
        submit_run_dir=submit_run,
        result=result,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["page_candidates_path"] == str(candidates)
    assert payload["page_candidates_gcs_uri"].startswith("gs://bucket/")
    assert payload["missing_pages"] == 0


def test_metadata_sanitizer_keeps_provenance_not_response_evidence() -> None:
    result = sanitize_extraction_metadata(
        {
            "model": "source-model",
            "schema_version_id": "v7",
            "source_metadata_gcs_generation": "101",
            "source_batch_job_gcs_generation": "202",
            "thoughts": "do not retain",
            "field_confidence_by_pointer": {"/name": 0.2},
        }
    )

    assert result == {
        "model": "source-model",
        "schema_version_id": "v7",
        "source_batch_job_gcs_generation": "202",
        "source_metadata_gcs_generation": "101",
    }


def test_candidate_writer_derives_portable_source_run_id(tmp_path) -> None:
    path = tmp_path / "page_candidates.jsonl"
    with PageCandidateWriter(path) as writer:
        writer.write(
            key="pages/page.png",
            candidate={"name": "Jensen"},
            extraction_metadata={
                "source_run_dir": "/machine-a/runs/submits/source-run"
            },
        )

    metadata = read_page_candidates(path)[0].extraction_metadata
    assert metadata["source_run_id"] == "source-run"
    assert metadata["source_run_dir"] == (
        "/machine-a/runs/submits/source-run"
    )


def test_candidate_source_run_binding_accepts_portable_and_legacy_provenance() -> None:
    records = (
        PageCandidateRecord(
            key="pages/one.png",
            candidate={"name": "One"},
            extraction_metadata={"source_run_id": "source-run"},
        ),
        PageCandidateRecord(
            key="pages/two.png",
            candidate={"name": "Two"},
            extraction_metadata={
                "source_run_dir": "/another-machine/submits/source-run"
            },
        ),
    )

    assert verify._validate_candidate_source_run(
        records=records,
        source_run_dir="/relocated/submits/source-run",
    ) == "source-run"


@pytest.mark.parametrize(
    ("records", "source_run_dir", "message"),
    [
        (
            (
                PageCandidateRecord(
                    key="pages/one.png",
                    candidate={"name": "One"},
                    extraction_metadata={"source_run_id": "run-a"},
                ),
                PageCandidateRecord(
                    key="pages/two.png",
                    candidate={"name": "Two"},
                    extraction_metadata={"source_run_id": "run-b"},
                ),
            ),
            "/submits/run-a",
            "multiple extraction source runs",
        ),
        (
            (
                PageCandidateRecord(
                    key="pages/one.png",
                    candidate={"name": "One"},
                    extraction_metadata={"source_run_id": "run-a"},
                ),
            ),
            "/submits/run-b",
            "does not match candidate provenance",
        ),
        (
            (
                PageCandidateRecord(
                    key="pages/one.png",
                    candidate={"name": "One"},
                    extraction_metadata={
                        "source_run_id": "run-a",
                        "source_run_dir": "/submits/run-b",
                    },
                ),
            ),
            "/submits/run-a",
            "conflicting source-run provenance",
        ),
        (
            (
                PageCandidateRecord(
                    key="pages/one.png",
                    candidate={"name": "One"},
                    extraction_metadata={},
                ),
            ),
            "/submits/run-a",
            "has no source_run_id",
        ),
    ],
)
def test_candidate_source_run_binding_fails_closed(
    records, source_run_dir, message
) -> None:
    with pytest.raises(RuntimeError, match=message):
        verify._validate_candidate_source_run(
            records=records,
            source_run_dir=source_run_dir,
        )


def test_validation_prompt_is_evidence_first_and_candidate_last() -> None:
    prompt = build_model_validation_prompt(
        candidate={"name": "Candidate"},
        extraction_schema={"type": "object", "properties": {"name": {"type": "string"}}},
        ocr_context="1,2,3,4|Observed",
    )

    schema_index = prompt.index(MODEL_VALIDATION_SCHEMA_HEADING)
    ocr_index = prompt.index(MODEL_VALIDATION_OCR_HEADING)
    candidate_index = prompt.index(MODEL_VALIDATION_CANDIDATE_HEADING)
    assert schema_index < ocr_index < candidate_index
    assert prompt.rstrip().endswith('{"name":"Candidate"}')
    assert "do not merely flag it" in prompt


def test_sparse_validation_schema_enforces_status_and_patch_contract() -> None:
    patch = ValidationIssuePatch(
        op="replace",
        path="/patient/name",
        value_json='"Jensen"',
        issue="incorrect",
        evidence="OCR and image show Jensen.",
        ocr_box_refs=["1,2,3,4"],
    )
    validation = PageModelValidation(
        page_status="needs_correction",
        patches=[patch],
    )

    assert validation.patches[0].path == "/patient/name"
    with pytest.raises(ValidationError):
        PageModelValidation(page_status="confirmed", patches=[patch])
    with pytest.raises(ValidationError):
        ValidationIssuePatch(
            op="replace",
            path="/patient/name",
            value_json="not-json",
            issue="incorrect",
            evidence="Mismatch.",
        )
    with pytest.raises(ValidationError, match="RFC 6901"):
        ValidationIssuePatch(
            op="replace",
            path="/patient/~2name",
            value_json='"Jensen"',
            issue="incorrect",
            evidence="Mismatch.",
        )


def test_sparse_validation_schema_rejects_overlapping_patch_paths() -> None:
    with pytest.raises(ValidationError, match="ancestor"):
        PageModelValidation(
            page_status="needs_correction",
            patches=[
                ValidationIssuePatch(
                    op="replace",
                    path="/patient",
                    value_json='{"name":"Jensen"}',
                    issue="incorrect",
                    evidence="Patient block is wrong.",
                ),
                ValidationIssuePatch(
                    op="replace",
                    path="/patient/name",
                    value_json='"Jensen"',
                    issue="incorrect",
                    evidence="Name is wrong.",
                ),
            ],
        )


def test_apply_validation_patches_is_rfc6901_aware() -> None:
    validation = PageModelValidation(
        page_status="needs_correction",
        patches=[
            ValidationIssuePatch(
                op="replace",
                path="/patient/name",
                value_json='"Jensen"',
                issue="incorrect",
                evidence="Image shows Jensen.",
            ),
            ValidationIssuePatch(
                op="add",
                path="/conditions/-",
                value_json='"Feber"',
                issue="missing",
                evidence="A second condition is visible.",
            ),
        ],
    )

    patched = verify.apply_validation_patches(
        {"patient": {"name": "Jenson"}, "conditions": ["Hoste"]},
        validation,
    )

    assert patched == {
        "patient": {"name": "Jensen"},
        "conditions": ["Hoste", "Feber"],
    }


def test_field_correction_metadata_records_original_and_applied_values(tmp_path) -> None:
    candidate = PageCandidateRecord(
        key="pages/page.png",
        candidate={"patient": {"name": "Jenson"}},
    )
    validation = PageModelValidation(
        page_status="needs_correction",
        patches=[
            ValidationIssuePatch(
                op="replace",
                path="/patient/name",
                value_json='"Jensen"',
                issue="incorrect",
                evidence="The page reads Jensen.",
                ocr_box_refs=["1,2,3,4"],
            )
        ],
    )

    applied = verify.build_field_correction_metadata(
        run_dir=tmp_path,
        candidates_by_key={candidate.key: candidate},
        validations={candidate.key: validation},
        failures=(),
        model="gemini-3.1-pro-preview",
        provider="gemini",
        apply_mode="apply_patches",
        candidate_hash="candidate-hash",
        verification_prompt_hash="prompt-hash",
        created_at="2026-08-27T12:00:00+00:00",
        verification_prompt_version="v2",
        acceptance_policy=verify.AUTOMATIC_CORRECTION_ACCEPTANCE_POLICY,
        verification_artifact_sha256s={"validation_results.jsonl": "results-hash"},
        included_in_corrected_dataset=True,
        corrected_dataset_sha256="corrected-dataset-hash",
    )
    field = applied["pages"][0]["fields"][0]
    assert applied["schema_version"] == 2
    assert applied["accepted_correction_fields"] == 1
    assert applied["corrected_fields"] == 1
    assert applied["corrected_dataset_sha256"] == "corrected-dataset-hash"
    assert applied["evidence_artifacts"]["verification_artifact_sha256s"] == {
        "validation_results.jsonl": "results-hash"
    }
    assert applied["pages"][0]["corrected"] is True
    assert field["top_level_field"] == "patient"
    assert field["accepted"] is True
    assert field["corrected"] is True
    assert field["original_value"] == "Jenson"
    assert field["proposed_value"] == "Jensen"

    report_only = verify.build_field_correction_metadata(
        run_dir=tmp_path,
        candidates_by_key={candidate.key: candidate},
        validations={candidate.key: validation},
        failures=(),
        model="gemini-3.1-pro-preview",
        provider="gemini",
        apply_mode="report_only",
        candidate_hash="candidate-hash",
        verification_prompt_hash="prompt-hash",
        created_at="2026-08-27T12:00:00+00:00",
        verification_prompt_version="v2",
    )
    assert report_only["proposed_correction_fields"] == 1
    assert report_only["accepted_correction_fields"] == 0
    assert report_only["corrected_fields"] == 0
    assert report_only["pages"][0]["fields"][0]["corrected"] is False
    assert report_only["verification_run_id"] == tmp_path.name
    assert report_only["created_at"] == "2026-08-27T12:00:00+00:00"
    assert report_only["verification_prompt_version"] == "v2"
    assert "verification_run_dir" not in report_only
    assert report_only == verify.build_field_correction_metadata(
        run_dir=tmp_path,
        candidates_by_key={candidate.key: candidate},
        validations={candidate.key: validation},
        failures=(),
        model="gemini-3.1-pro-preview",
        provider="gemini",
        apply_mode="report_only",
        candidate_hash="candidate-hash",
        verification_prompt_hash="prompt-hash",
        created_at="2026-08-27T12:00:00+00:00",
        verification_prompt_version="v2",
    )


def test_field_correction_metadata_distinguishes_accepted_from_complete_dataset(
    tmp_path,
) -> None:
    candidate = PageCandidateRecord(
        key="pages/page.png",
        candidate={"patient": {"name": "Jenson"}},
    )
    validation = PageModelValidation(
        page_status="needs_correction",
        patches=[
            ValidationIssuePatch(
                op="replace",
                path="/patient/name",
                value_json='"Jensen"',
                issue="incorrect",
                evidence="The page reads Jensen.",
            )
        ],
    )

    metadata = verify.build_field_correction_metadata(
        run_dir=tmp_path,
        candidates_by_key={candidate.key: candidate},
        validations={candidate.key: validation},
        failures=(),
        model="gemini-3.1-pro-preview",
        provider="gemini",
        apply_mode="apply_patches",
        candidate_hash="candidate-hash",
        verification_prompt_hash="prompt-hash",
        acceptance_policy=verify.AUTOMATIC_CORRECTION_ACCEPTANCE_POLICY,
        included_in_corrected_dataset=False,
    )

    field = metadata["pages"][0]["fields"][0]
    assert metadata["accepted_correction_fields"] == 1
    assert metadata["corrected_fields"] == 0
    assert metadata["corrected_dataset_built"] is False
    assert metadata["dataset_version_publication_authority"] == (
        "dataset_versions.json"
    )
    assert field["accepted"] is True
    assert field["applied"] is True
    assert field["included_in_corrected_dataset"] is False
    assert field["corrected"] is False


def test_patched_candidate_hash_is_portable_across_run_relocation(tmp_path) -> None:
    key = "pages/page.png"
    candidate = PageCandidateRecord(
        key=key,
        candidate={"name": "Jenson"},
        extraction_metadata={
            "schema_name": "Candidate",
            "source_run_dir": "/original-machine/runs/source-run",
            "verification_run_dir": "/old-machine/runs/previous-verifier",
        },
    )
    validation = PageModelValidation(
        page_status="confirmed",
        patches=[],
    )
    first = tmp_path / "machine-a" / "verify-run" / "patched_candidates.jsonl"
    second = tmp_path / "machine-b" / "verify-run" / "patched_candidates.jsonl"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)

    for path in (first, second):
        verify._write_patched_candidates(
            path=path,
            verification_run_id=path.parent.name,
            candidates_by_key={key: candidate},
            patched_by_key={key: {"name": "Jensen"}},
            validations={key: validation},
            model="gemini-3.1-pro-preview",
            provider="gemini",
        )

    assert first.read_bytes() == second.read_bytes()
    assert verify.file_sha256(first) == verify.file_sha256(second)
    metadata = read_page_candidates(first)[0].extraction_metadata
    assert metadata["verification_run_id"] == "verify-run"
    assert "verification_run_dir" not in metadata
    assert str(tmp_path) not in first.read_text(encoding="utf-8")


def test_apply_validation_patches_rejects_negative_array_indices() -> None:
    validation = PageModelValidation(
        page_status="needs_correction",
        patches=[
            ValidationIssuePatch(
                op="replace",
                path="/items/-1/value",
                value_json='"wrong target"',
                issue="incorrect",
                evidence="A negative index is not RFC 6902.",
            )
        ],
    )

    with pytest.raises(ValueError, match="RFC 6902 array index"):
        verify.apply_validation_patches(
            {"items": [{"value": "first"}, {"value": "last"}]},
            validation,
        )


def test_verification_refuses_generation_changed_since_extraction(monkeypatch) -> None:
    monkeypatch.setattr(
        verify,
        "load_ocr_metadata_for_blob",
        lambda _blob: _ocr_metadata("2"),
    )
    bucket = SimpleNamespace(blob=lambda _name: object())
    record = PageCandidateRecord(
        key="pages/page.png",
        candidate={"name": "Jensen"},
    )

    with pytest.raises(RuntimeError, match="changed since extraction"):
        verify._prepare_validation_page(
            record=record,
            input_record=_input_record("1"),
            bucket=bucket,
            provider="gemini",
            run_dir_name="verify-run",
            extraction_schema={"type": "object"},
        )


def test_verification_refuses_changed_ocr_text_with_same_image_and_line_count(
    monkeypatch,
) -> None:
    changed = CloudOcrMetadata(
        source=_identity("1"),
        document=OcrDocument(
            image_sha256="image-digest",
            width=100,
            height=100,
            coordinate_scale=1000,
            backend="test",
            lines=(OcrLine(text="Different", box=(1, 2, 3, 4)),),
        ),
        created_at="2026-01-01T00:00:00Z",
    )
    monkeypatch.setattr(verify, "load_ocr_metadata_for_blob", lambda _blob: changed)
    bucket = SimpleNamespace(blob=lambda _name: object())

    with pytest.raises(RuntimeError, match="text/position document"):
        verify._prepare_validation_page(
            record=PageCandidateRecord(
                key="pages/page.png",
                candidate={"name": "Jensen"},
            ),
            input_record=_input_record("1"),
            bucket=bucket,
            provider="gemini",
            run_dir_name="verify-run",
            extraction_schema={"type": "object"},
        )


def test_prepared_gemini_request_uses_staged_exact_image_and_sparse_schema(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        verify,
        "load_ocr_metadata_for_blob",
        lambda _blob: _ocr_metadata("1"),
    )
    monkeypatch.setattr(
        verify,
        "_stage_exact_gemini_image",
        lambda **_kwargs: ("gs://bucket/immutable/page.png", _identity("99")),
    )
    class SidecarBlob:
        bucket = SimpleNamespace(name="bucket")
        name = "pages/page.png.ocr.json"
        generation = "7"
        size = 8
        crc32c = "sidecar-crc"
        md5_hash = "sidecar-md5"
        etag = None

        def reload(self) -> None:
            return None

        def download_as_bytes(self, **_kwargs) -> bytes:
            return b"sidecar"

    bucket = SimpleNamespace(
        blob=lambda name: SidecarBlob()
        if name.endswith(".ocr.json")
        else object()
    )
    record = PageCandidateRecord(
        key="pages/page.png",
        candidate={"name": "Jensen"},
    )
    prepared = verify._prepare_validation_page(
        record=record,
        input_record=_input_record("1"),
        bucket=bucket,
        provider="gemini",
        run_dir_name="verify-run",
        extraction_schema={"type": "object"},
    )

    request = verify._gemini_request_line(
        prepared,
        model="gemini-3.1-pro-preview",
        for_vertex=True,
        thinking_level="high",
        max_output_tokens=4096,
    )

    parts = request["request"]["contents"][0]["parts"]
    assert parts[0]["fileData"]["fileUri"] == "gs://bucket/immutable/page.png"
    assert request["key"] == "pages/page.png"
    assert request["request"]["generationConfig"]["responseSchema"]["type"] == "object"
    assert prepared.binding["ocr_image_sha256"] == "image-digest"


def test_mldev_registers_staged_gcs_images_before_request() -> None:
    page = verify._PreparedValidationPage(
        record=PageCandidateRecord(
            key="pages/page.png", candidate={"name": "Jensen"}
        ),
        input_record=_input_record(),
        candidate_digest="candidate-digest",
        mime_type="image/png",
        prompt="prompt",
        provider_image_reference="gs://bucket/staged/page.png",
        request_image_source=CloudBlobIdentity(
            bucket="bucket",
            name="staged/page.png",
            generation="99",
            size=123,
            crc32c="crc",
            md5_hash="md5",
            etag=None,
        ),
        binding={"request_image_uri": "gs://bucket/staged/page.png"},
    )
    captured: dict[str, object] = {}

    class Files:
        def register_files(self, *, auth, uris):
            captured.update(auth=auth, uris=uris)
            return SimpleNamespace(
                files=[
                    SimpleNamespace(
                        uri="https://generativelanguage.googleapis.com/files/one",
                        name="files/one",
                        sha256_hash=None,
                    )
                ]
            )

    credentials = object()
    registered = verify._register_mldev_validation_images(
        client=SimpleNamespace(files=Files()),
        storage_client=SimpleNamespace(_credentials=credentials),
        pages=(page,),
    )

    assert captured == {
        "auth": credentials,
        "uris": ["gs://bucket/staged/page.png"],
    }
    assert registered[0].provider_image_reference.endswith("/files/one")
    assert registered[0].binding["staged_image_uri"] == (
        "gs://bucket/staged/page.png"
    )


def test_verification_uses_exact_image_without_requiring_ocr(monkeypatch) -> None:
    monkeypatch.setattr(
        verify,
        "_stage_exact_gemini_image",
        lambda **_kwargs: ("gs://bucket/immutable/page.png", _identity("99")),
    )

    prepared = verify._prepare_validation_page(
        record=PageCandidateRecord(
            key="pages/page.png", candidate={"name": "Jensen"}
        ),
        input_record=_input_record_without_ocr(),
        bucket=object(),
        provider="gemini",
        run_dir_name="verify-run",
        extraction_schema={"type": "object"},
    )

    assert MODEL_VALIDATION_OCR_HEADING not in prepared.prompt
    assert prepared.binding["ocr_enabled"] is False
    assert prepared.binding["ocr_sidecar_source"] == {}


def test_gemini_generation_config_uses_model_default_temperature() -> None:
    generation_config = verify._gemini_generation_config(
        model="gemini-3.1-pro-preview",
        for_vertex=True,
        thinking_level="high",
        max_output_tokens=4096,
    )

    assert "temperature" not in generation_config


def test_gemini_25_uses_thinking_budget_not_thinking_level() -> None:
    generation_config = verify._gemini_generation_config(
        model="gemini-2.5-pro",
        for_vertex=True,
        thinking_level="high",
        max_output_tokens=4096,
    )

    assert generation_config["thinkingConfig"] == {"thinkingBudget": 3840}


def test_provider_stop_reasons_must_be_clean_before_validation() -> None:
    gemini_text = json.dumps({"page_status": "confirmed", "patches": []})
    gemini_record = {
        "response": {
            "candidates": [
                {
                    "finishReason": "MAX_TOKENS",
                    "content": {"parts": [{"text": gemini_text}]},
                }
            ]
        }
    }
    anthropic_record = {
        "result": {
            "type": "succeeded",
            "message": {
                "stop_reason": "max_tokens",
                "content": [{"type": "text", "text": gemini_text}],
            },
        }
    }

    assert verify._extract_gemini_text(gemini_record) == (
        None,
        "finish_reason_max_tokens",
    )
    assert verify._extract_anthropic_text(anthropic_record) == (
        None,
        "stop_reason_max_tokens",
    )
    assert batch_retrieve._anthropic_stop_reason(
        anthropic_record["result"]["message"]
    ) == "max_tokens"


def test_flagged_scope_is_explicit_and_missing_status_is_conservative() -> None:
    records = (
        PageCandidateRecord(
            key="confirmed.png",
            candidate={"value": 1},
            extraction_metadata={"deterministic_status": "confirmed"},
        ),
        PageCandidateRecord(
            key="flagged.png",
            candidate={"value": 2},
            extraction_metadata={"deterministic_status": "flagged"},
        ),
        PageCandidateRecord(key="unknown.png", candidate={"value": 3}),
    )

    assert [record.key for record in verify._scope_candidates(records, "flagged")] == [
        "flagged.png",
        "unknown.png",
    ]


def test_all_scope_requires_candidates_equal_full_input_manifest() -> None:
    candidate = PageCandidateRecord(
        key="pages/page.png",
        candidate={"value": 1},
    )
    extra = _input_record().model_copy(
        update={"key": "pages/extra.png"}
    )

    with pytest.raises(RuntimeError, match="missing_candidates=1"):
        verify._validate_input_manifest_coverage(
            records=(candidate,),
            input_by_key={
                "pages/page.png": _input_record(),
                "pages/extra.png": extra,
            },
            scope="all",
        )

    verify._validate_input_manifest_coverage(
        records=(candidate,),
        input_by_key={
            "pages/page.png": _input_record(),
            "pages/extra.png": extra,
        },
        scope="flagged",
    )


def test_extraction_schema_must_match_candidate_and_batch_provenance(tmp_path) -> None:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }
    source_run = tmp_path / "source"
    source_run.mkdir()
    (source_run / "metadata.json").write_text(
        json.dumps(
            {
                "schema_name": "Candidate",
                "schema_version_id": "v7",
                "output_schema": schema,
            }
        ),
        encoding="utf-8",
    )
    (source_run / "batch_job.json").write_text(
        json.dumps(
            {"schema_name": "Candidate", "schema_version_id": "v7"}
        ),
        encoding="utf-8",
    )
    record = PageCandidateRecord(
        key="pages/page.png",
        candidate={"name": "Jensen"},
        extraction_metadata={
            "schema_name": "Candidate",
            "schema_version_id": "v7",
            "source_run_dir": str(source_run),
            "source_metadata_sha256": verify.file_sha256(
                source_run / "metadata.json"
            ),
            "source_batch_job_sha256": verify.file_sha256(
                source_run / "batch_job.json"
            ),
        },
    )

    snapshot = verify._resolve_extraction_schema(
        request=verify.ModelValidationSubmitRequest(), records=(record,)
    )

    assert snapshot.schema == schema
    assert snapshot.name == "Candidate"
    assert snapshot.version_id == "v7"
    with pytest.raises(RuntimeError, match="does not match extraction metadata"):
        verify._resolve_extraction_schema(
            request=verify.ModelValidationSubmitRequest(),
            records=(
                record.model_copy(
                    update={
                        "extraction_metadata": {
                            **record.extraction_metadata,
                            "schema_version_id": "v8",
                        }
                    }
                ),
            ),
        )


def test_extraction_schema_recovers_from_candidate_bound_cloud_artifacts(
    tmp_path, monkeypatch
) -> None:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }
    cloud_metadata = tmp_path / "cloud_metadata.json"
    cloud_batch = tmp_path / "cloud_batch.json"
    cloud_metadata.write_text(
        json.dumps(
            {
                "schema_name": "Candidate",
                "schema_version_id": "v7",
                "output_schema": schema,
            }
        ),
        encoding="utf-8",
    )
    cloud_batch.write_text(
        json.dumps({"schema_name": "Candidate", "schema_version_id": "v7"}),
        encoding="utf-8",
    )
    uri_sources = {
        "gs://bucket/metadata.json": cloud_metadata,
        "gs://bucket/batch_job.json": cloud_batch,
    }

    def download(_client, uri: str, destination, *, generation=None) -> Path:
        assert str(generation) == "17"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(uri_sources[uri].read_bytes())
        return destination

    monkeypatch.setattr(verify, "_download_gcs_file", download)
    record = PageCandidateRecord(
        key="pages/page.png",
        candidate={"name": "Jensen"},
        extraction_metadata={
            "schema_name": "Candidate",
            "schema_version_id": "v7",
            "source_run_dir": str(tmp_path / "other-machine" / "source"),
            "source_metadata_gcs_uri": "gs://bucket/metadata.json",
            "source_metadata_sha256": verify.file_sha256(cloud_metadata),
            "source_metadata_gcs_generation": "17",
            "source_batch_job_gcs_uri": "gs://bucket/batch_job.json",
            "source_batch_job_sha256": verify.file_sha256(cloud_batch),
            "source_batch_job_gcs_generation": "17",
        },
    )

    snapshot = verify._resolve_extraction_schema(
        request=verify.ModelValidationSubmitRequest(),
        records=(record,),
        cache_dir=tmp_path / "cache",
        storage_client=object(),
    )

    assert snapshot.schema == schema
    assert snapshot.name == "Candidate"


def test_dynamic_schema_validation_does_not_coerce_scalar_types() -> None:
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"],
        "additionalProperties": False,
    }
    snapshot = verify._ExtractionSchemaSnapshot(
        schema=schema,
        name="ManagedCandidate",
        version_id="v1",
        sha256=verify._canonical_json_sha256(schema),
    )
    model = verify._extraction_model_for_snapshot(snapshot)

    with pytest.raises(ValidationError):
        verify._validate_extraction_candidate(model, {"n": "3"})


def test_managed_schema_fails_closed_for_unenforced_keywords() -> None:
    schema = {
        "type": "object",
        "properties": {
            "items": {"type": "array", "minItems": 1, "items": {"type": "string"}}
        },
        "required": ["items"],
    }
    snapshot = verify._ExtractionSchemaSnapshot(
        schema=schema,
        name="ManagedCandidate",
        version_id="v1",
        sha256=verify._canonical_json_sha256(schema),
    )

    with pytest.raises(RuntimeError, match="minItems"):
        verify._extraction_model_for_snapshot(snapshot)


def test_builtin_schema_snapshot_reuses_original_model_validators() -> None:
    schema = TextPage.model_json_schema()
    snapshot = verify._ExtractionSchemaSnapshot(
        schema=schema,
        name="TextPage",
        version_id="",
        sha256=verify._canonical_json_sha256(schema),
    )

    assert verify._extraction_model_for_snapshot(snapshot) is TextPage


def test_retrieval_rejects_overwritten_unversioned_gemini_stage(monkeypatch) -> None:
    input_record = _input_record()
    staged = CloudBlobIdentity(
        bucket="bucket",
        name="staged/page.png",
        generation="99",
        size=123,
        crc32c="crc",
        md5_hash="md5",
        etag=None,
    )
    changed_blob = SimpleNamespace(
        bucket=SimpleNamespace(name="bucket"),
        name="staged/page.png",
        generation="100",
        size=123,
        crc32c="crc",
        md5_hash="md5",
        etag=None,
        reload=lambda: None,
    )
    monkeypatch.setattr(
        verify,
        "_load_bound_ocr_evidence",
        lambda **_kwargs: (_ocr_metadata("1"), input_record.sidecar_source),
    )
    binding = {
        "candidate_sha256": candidate_sha256({"name": "Jensen"}),
        "extraction_image_source": input_record.source.to_dict(),
        "request_image_source": staged.to_dict(),
        "request_image_uri": "gs://bucket/staged/page.png",
        "ocr_sidecar_name": input_record.ocr_sidecar_name,
        "ocr_sidecar_source": input_record.sidecar_source.to_dict(),
        "ocr_sidecar_sha256": input_record.ocr_sidecar_sha256,
        "ocr_image_sha256": input_record.ocr_image_sha256,
        "ocr_document_sha256": input_record.ocr_document_sha256,
        "ocr_backend": input_record.ocr_backend,
        "ocr_line_count": input_record.ocr_line_count,
    }

    with pytest.raises(RuntimeError, match="Staged Gemini image changed"):
        verify._verify_retrieval_evidence_bindings(
            bucket=SimpleNamespace(blob=lambda _name: changed_blob),
            provider="gemini",
            records=(
                PageCandidateRecord(
                    key=input_record.key, candidate={"name": "Jensen"}
                ),
            ),
            input_by_key={input_record.key: input_record},
            bindings={input_record.key: binding},
        )


def test_anthropic_chunks_refine_by_actual_serialized_request_bytes() -> None:
    def page(key: str) -> verify._PreparedValidationPage:
        record = PageCandidateRecord(key=key, candidate={"name": "Jensen"})
        return verify._PreparedValidationPage(
            record=record,
            input_record=_input_record().model_copy(update={"key": key}),
            candidate_digest=candidate_sha256(record.candidate),
            mime_type="image/png",
            prompt="evidence-first prompt",
            provider_image_reference="https://signed.example/image.png",
            request_image_source=_identity("1"),
            binding={},
        )

    pages = (page("pages/one.png"), page("pages/two.png"))
    one_row = verify._anthropic_request(
        pages[0],
        model="claude-sonnet-4-5",
        thinking_level="high",
        max_output_tokens=4096,
    )
    one_row_bytes = len(
        json.dumps(one_row, ensure_ascii=False, separators=(",", ":")).encode()
    ) + 1

    chunks = verify._split_anthropic_chunks_by_bytes(
        (pages,),
        model="claude-sonnet-4-5",
        thinking_level="high",
        max_output_tokens=4096,
        byte_limit=one_row_bytes + 10,
    )

    assert [[item.record.key for item in chunk] for chunk in chunks] == [
        ["pages/one.png"],
        ["pages/two.png"],
    ]

    count_chunks = verify._split_anthropic_chunks_by_bytes(
        (pages,),
        model="claude-sonnet-4-5",
        thinking_level="high",
        max_output_tokens=4096,
        byte_limit=10_000_000,
        request_limit=1,
    )
    assert [len(chunk) for chunk in count_chunks] == [1, 1]


def test_publishable_dataset_uses_existing_model_to_rows_path(tmp_path) -> None:
    class Candidate(BaseModel):
        name: str

    record = PageCandidateRecord(
        key="pages/page.png",
        candidate={"name": "Jenson"},
        extraction_metadata={
            "model": "source-model",
            "provider": "gemini",
            "schema_name": "Candidate",
            "schema_version_id": "v1",
        },
    )
    validation = PageModelValidation(
        page_status="needs_correction",
        patches=[
            ValidationIssuePatch(
                op="replace",
                path="/name",
                value_json='"Jensen"',
                issue="incorrect",
                evidence="Image shows Jensen.",
            )
        ],
    )

    path, rows = verify._build_verified_dataset(
        run_dir=tmp_path,
        candidates_by_key={record.key: record},
        patched_by_key={record.key: {"name": "Jensen"}},
        validations={record.key: validation},
        extraction_model=Candidate,
        extraction_schema_name="Candidate",
        verification_model="gemini-3.1-pro-preview",
        verification_provider="gemini",
        output_format="jsonl",
        dataset_file_name="dataset",
        csv_sep="$",
    )

    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert rows == 1
    assert payload["name"] == "Jensen"
    assert payload["verification_status"] == "needs_correction"
    assert payload["verification_model"] == "gemini-3.1-pro-preview"


def test_publishable_textpage_dataset_uses_persisted_schema_handler(tmp_path) -> None:
    schema = TextPage.model_json_schema()
    schema["title"] = "Managed Text Page"
    snapshot = verify._ExtractionSchemaSnapshot(
        schema=schema,
        name="TextPage",
        version_id="managed-v1",
        sha256=verify._canonical_json_sha256(schema),
    )
    model = verify._extraction_model_for_snapshot(snapshot)
    candidate_payload = {
        "page_lines": [
            {"text": "First", "metadata": None, "page_line_number": 1},
            {"text": "Second", "metadata": None, "page_line_number": 2},
        ]
    }
    record = PageCandidateRecord(
        key="pages/page.png",
        candidate=candidate_payload,
        extraction_metadata={
            "schema_name": "TextPage",
            "schema_version_id": "managed-v1",
        },
    )
    validation = PageModelValidation(page_status="confirmed", patches=[])

    path, rows = verify._build_verified_dataset(
        run_dir=tmp_path,
        candidates_by_key={record.key: record},
        patched_by_key={record.key: candidate_payload},
        validations={record.key: validation},
        extraction_model=model,
        extraction_schema_name="TextPage",
        verification_model="gemini-3.1-pro-preview",
        verification_provider="gemini",
        output_format="jsonl",
        dataset_file_name="dataset",
        csv_sep="$",
    )

    payloads = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == 2
    assert [row["text"] for row in payloads] == ["First", "Second"]


def test_verification_prompt_hash_is_stable_and_contract_aware(monkeypatch) -> None:
    first = verify.verification_prompt_hash()
    second = verify.verification_prompt_hash()

    assert first == second
    assert len(first) == 64
    monkeypatch.setattr(
        verify,
        "MODEL_VALIDATION_CANDIDATE_HEADING",
        "Changed candidate heading",
    )
    assert verify.verification_prompt_hash() != first


def test_terminal_validation_outcomes_separate_success_from_publication() -> None:
    assert verify._model_validation_outcome(
        expected_pages=2,
        completed_pages=2,
        failure_records=0,
        unverifiable_pages=0,
        apply_mode="report_only",
        scope="all",
    ) == (True, False, "report_only_complete")
    assert verify._model_validation_outcome(
        expected_pages=2,
        completed_pages=1,
        failure_records=1,
        unverifiable_pages=0,
        apply_mode="apply_patches",
        scope="all",
    ) == (False, False, "incomplete")


def test_verification_run_ids_are_unique_under_same_timestamp_and_root(
    tmp_path, monkeypatch
) -> None:
    class _FrozenNow:
        def strftime(self, _format: str) -> str:
            return "20260828_120000"

    class _FrozenDateTime:
        @classmethod
        def now(cls):
            return _FrozenNow()

    ids = iter(
        [
            SimpleNamespace(hex="a" * 32),
            # Force one real atomic mkdir collision on the second allocation.
            SimpleNamespace(hex="a" * 32),
            SimpleNamespace(hex="b" * 32),
        ]
    )
    monkeypatch.setattr(verify.config, "output_root", str(tmp_path))
    monkeypatch.setattr(verify, "datetime", _FrozenDateTime)
    monkeypatch.setattr(verify.uuid, "uuid4", lambda: next(ids))

    first = verify._create_verification_run_dir()
    second = verify._create_verification_run_dir()

    assert first.parent == second.parent == tmp_path / "verifications"
    assert first.name == f"20260828_120000_{'a' * 32}"
    assert second.name == f"20260828_120000_{'b' * 32}"
    assert first.is_dir() and second.is_dir()


def test_source_run_mismatch_fails_before_policy_or_provider_submission(
    tmp_path, monkeypatch
) -> None:
    candidates = tmp_path / "input" / "page_candidates.jsonl"
    candidates.parent.mkdir()
    with PageCandidateWriter(candidates) as writer:
        writer.write(
            key="pages/page.png",
            candidate={"name": "Jensen"},
            extraction_metadata={"source_run_id": "source-a"},
        )
    run_dir = tmp_path / "verifications" / "verify-run"
    run_dir.mkdir(parents=True)
    reached: list[str] = []

    def unexpected_policy(**_kwargs):
        reached.append("policy")
        raise AssertionError("immutable policy must not be created")

    def unexpected_provider(*_args, **_kwargs):
        reached.append("provider")
        raise AssertionError("provider submission must not be reached")

    monkeypatch.setattr(verify, "_create_verification_run_dir", lambda: run_dir)
    monkeypatch.setattr(
        verify,
        "_storage_client_and_bucket",
        lambda: (object(), SimpleNamespace(name="bucket")),
    )
    monkeypatch.setattr(verify, "_write_final_validation_policy", unexpected_policy)
    monkeypatch.setattr(verify, "get_batch_client", unexpected_provider)
    monkeypatch.setattr(verify, "_get_anthropic_client", unexpected_provider)

    with pytest.raises(RuntimeError, match="does not match candidate provenance"):
        verify.submit_model_validation(
            verify.ModelValidationSubmitRequest(
                source_run_dir=str(tmp_path / "submits" / "source-b"),
                candidate_file=str(candidates),
                model="gemini-3.1-pro-preview",
            )
        )

    assert reached == []
    assert not (run_dir / verify.FINAL_VALIDATION_POLICY_FILE_NAME).exists()


@pytest.mark.parametrize(
    ("submit_request", "message"),
    [
        (
            verify.ModelValidationSubmitRequest(
                model="gemini-3.1-pro-preview",
                scope="all",
                apply_mode="report_only",
            ),
            "report_only is not supported",
        ),
        (
            verify.ModelValidationSubmitRequest(
                model="gemini-3.1-pro-preview",
                scope="invalid",  # type: ignore[arg-type]
                apply_mode="apply_patches",
            ),
            "Invalid verification scope",
        ),
    ],
)
def test_new_verification_submission_rejects_nonpublishing_modes_before_cloud_calls(
    monkeypatch, submit_request, message
) -> None:
    cloud_calls: list[str] = []

    def unexpected_run_dir_creation():
        cloud_calls.append("run_dir")
        raise AssertionError("verification run creation must not be reached")

    def unexpected_cloud_setup():
        cloud_calls.append("cloud")
        raise AssertionError("cloud setup must not be reached")

    monkeypatch.setattr(verify, "_create_verification_run_dir", unexpected_run_dir_creation)
    monkeypatch.setattr(verify, "_storage_client_and_bucket", unexpected_cloud_setup)

    with pytest.raises(ValueError, match=message):
        verify.submit_model_validation(submit_request)

    assert cloud_calls == []


def test_new_verification_submission_ignores_stale_report_only_config(
    monkeypatch,
) -> None:
    monkeypatch.setattr(verify.config, "verification_apply_mode", "report_only")
    monkeypatch.setattr(verify.config, "verification_scope", "flagged")

    def reached_run_creation():
        raise RuntimeError("fixed policy reached run creation")

    monkeypatch.setattr(verify, "_create_verification_run_dir", reached_run_creation)

    with pytest.raises(RuntimeError, match="fixed policy reached run creation"):
        verify.submit_model_validation(
            verify.ModelValidationSubmitRequest(
                model="gemini-3.1-pro-preview",
                scope="all",
                source_run_dir="/portable/source-run",
            )
        )


def test_new_verification_requires_source_run_even_with_candidate_file(
    monkeypatch,
) -> None:
    reached: list[str] = []

    def unexpected_run_dir_creation():
        reached.append("run_dir")
        raise AssertionError("verification run creation must not be reached")

    def unexpected_cloud_setup():
        reached.append("cloud")
        raise AssertionError("cloud setup must not be reached")

    monkeypatch.setattr(verify, "_create_verification_run_dir", unexpected_run_dir_creation)
    monkeypatch.setattr(verify, "_storage_client_and_bucket", unexpected_cloud_setup)

    with pytest.raises(ValueError, match="requires source_run_dir"):
        verify.submit_model_validation(
            verify.ModelValidationSubmitRequest(
                candidate_file="gs://bucket/candidates.jsonl",
                model="gemini-3.1-pro-preview",
            )
        )

    assert reached == []


def test_recorded_correction_policy_preserves_historical_runs() -> None:
    assert verify._recorded_correction_policy(
        {"verification_apply_mode": "report_only"}
    ) == ("report_only", "legacy_report_only")
    assert verify._recorded_correction_policy(
        {"verification_apply_mode": "apply_patches"}
    ) == ("report_only", "legacy_report_only")
    assert verify._recorded_correction_policy(
        {
            "verification_apply_mode": "report_only",
            "correction_acceptance_policy": (
                verify.AUTOMATIC_CORRECTION_ACCEPTANCE_POLICY
            ),
        }
    ) == ("report_only", "legacy_report_only")
    immutable = verify._FinalValidationPolicySnapshot(
        apply_mode="apply_patches",
        acceptance_policy=verify.AUTOMATIC_CORRECTION_ACCEPTANCE_POLICY,
        verification_scope="all",
        source_run_id="source-run",
        datasets_gcs_prefix="datasets-at-submit",
        validations_gcs_prefix="validations-at-submit",
    )
    assert verify._recorded_correction_policy(
        {"verification_apply_mode": "report_only"},
        immutable_policy=immutable,
    ) == (
        "apply_patches",
        verify.AUTOMATIC_CORRECTION_ACCEPTANCE_POLICY,
    )


class _FakePreconditionFailure(Exception):
    code = 412


class _ImmutableArtifactBlob:
    def __init__(self, bucket, name: str) -> None:
        self._bucket = bucket
        self.name = name
        self.generation = None
        self.metadata = None

    def upload_from_filename(
        self,
        filename: str,
        *,
        content_type: str,
        if_generation_match: int,
    ) -> None:
        assert content_type in {"application/json", "application/jsonl"}
        assert if_generation_match == 0
        self._bucket.upload_attempts += 1
        if self.name in self._bucket.objects:
            raise _FakePreconditionFailure("object already exists")
        self._bucket.next_generation += 1
        with open(filename, "rb") as artifact_file:
            payload = artifact_file.read()
        self._bucket.objects[self.name] = {
            "payload": payload,
            "generation": self._bucket.next_generation,
            "metadata": dict(self.metadata or {}),
        }
        self.generation = self._bucket.next_generation

    def reload(self) -> None:
        stored = self._bucket.objects[self.name]
        self.generation = stored["generation"]
        self.metadata = dict(stored["metadata"])

    def download_as_bytes(self, *, if_generation_match=None) -> bytes:
        self.reload()
        if if_generation_match is not None:
            assert int(if_generation_match) == self.generation
        return bytes(self._bucket.objects[self.name]["payload"])


class _ImmutableArtifactBucket:
    name = "audit-bucket"

    def __init__(self) -> None:
        self.objects: dict[str, dict[str, object]] = {}
        self.next_generation = 0
        self.upload_attempts = 0

    def blob(self, name: str) -> _ImmutableArtifactBlob:
        return _ImmutableArtifactBlob(self, name)


class _MissingPolicyBlob:
    def reload(self) -> None:
        from google.api_core.exceptions import NotFound

        raise NotFound("missing policy")


class _MissingPolicyBucket:
    name = "audit-bucket"

    def blob(self, _name: str) -> _MissingPolicyBlob:
        return _MissingPolicyBlob()


def test_final_validation_policy_is_create_only_and_anchors_prefixes(tmp_path) -> None:
    run_dir = tmp_path / "verify-run"
    run_dir.mkdir()
    routing_path = run_dir / "deterministic_routing.jsonl"
    routing_path.write_text('{"key":"pages/one.png","status":"confirmed"}\n')
    routing_digest = verify.file_sha256(routing_path)
    bucket = _ImmutableArtifactBucket()
    routing_uri, routing_generation = verify._upload_immutable_validation_artifact(
        bucket,
        run_dir,
        routing_path,
        sha256=routing_digest,
        validations_prefix="submitted/validations",
    )
    requests_path = run_dir / verify._chunk_file_name(1, 1)
    request_count, request_bytes = verify._write_jsonl(
        requests_path,
        ({"key": "pages/one.png", "request": {"contents": []}},),
    )
    contract_path, contract_payload, contract_digest = (
        verify._write_validation_request_contract(
            run_dir=run_dir,
            provider="gemini",
            client_backend="vertex",
            model="gemini-3.1-pro-preview",
            thinking_level="high",
            max_output_tokens=4096,
            requested_num_chunks=1,
            chunk_records=(
                {
                    "chunk_index": 1,
                    "total_chunks": 1,
                    "requests_file": requests_path.name,
                    "request_count": request_count,
                    "request_bytes": request_bytes,
                    "requests_sha256": verify.file_sha256(requests_path),
                },
            ),
        )
    )
    verify._validate_frozen_request_contract(
        run_dir=run_dir,
        payload=contract_payload,
        provider="gemini",
        client_backend="vertex",
        model="gemini-3.1-pro-preview",
        thinking_level="high",
        max_output_tokens=4096,
    )
    contract_uri, contract_generation = (
        verify._upload_immutable_validation_artifact(
            bucket,
            run_dir,
            contract_path,
            sha256=contract_digest,
            validations_prefix="submitted/validations",
        )
    )
    policy_path, payload, digest = verify._write_final_validation_policy(
        run_dir=run_dir,
        source_run_dir=str(tmp_path / "source-run"),
        datasets_gcs_prefix="submitted/datasets",
        validations_gcs_prefix="submitted/validations",
        deterministic_routing_policy="deterministic-routing-v1",
        deterministic_routing_sha256=routing_digest,
        deterministic_routing_gcs_uri=routing_uri,
        deterministic_routing_gcs_generation=routing_generation,
        verification_provider="gemini",
        verification_client_backend="vertex",
        verification_model="gemini-3.1-pro-preview",
        verification_thinking_level="high",
        verification_max_output_tokens=4096,
        verification_request_contract_sha256=contract_digest,
        verification_request_contract_gcs_uri=contract_uri,
        verification_request_contract_gcs_generation=contract_generation,
    )

    first = verify._upload_final_validation_policy(
        bucket=bucket,
        run_dir=run_dir,
        policy_path=policy_path,
        policy_sha256=digest,
    )
    second = verify._upload_final_validation_policy(
        bucket=bucket,
        run_dir=run_dir,
        policy_path=policy_path,
        policy_sha256=digest,
    )

    assert first == second
    assert first[0] == (
        "gs://audit-bucket/_patientjournals/model_validation_policies/"
        "verify-run/final_validation_policy.json"
    )
    assert payload["source_run_id"] == "source-run"
    assert payload["datasets_gcs_prefix"] == "submitted/datasets"
    assert payload["validations_gcs_prefix"] == "submitted/validations"
    assert len(bucket.objects) == 3
    assert bucket.upload_attempts == 4

    metadata = {
        "source_run_dir": str(tmp_path / "relocated" / "source-run"),
        "verification_scope": "all",
        "verification_apply_mode": "apply_patches",
        "correction_acceptance_policy": (
            verify.AUTOMATIC_CORRECTION_ACCEPTANCE_POLICY
        ),
        "datasets_gcs_prefix": "submitted/datasets",
        "validations_gcs_prefix": "submitted/validations",
        "deterministic_routing_policy": "deterministic-routing-v1",
        "deterministic_routing_sha256": routing_digest,
        "deterministic_routing_gcs_uri": routing_uri,
        "deterministic_routing_gcs_generation": routing_generation,
        "provider": "gemini",
        "client_backend": "vertex",
        "verification_model": "gemini-3.1-pro-preview",
        "verification_thinking_level": "high",
        "verification_max_output_tokens": 4096,
        "verification_request_contract_file": contract_path.name,
        "verification_request_contract_sha256": contract_digest,
        "verification_request_contract_gcs_uri": contract_uri,
        "verification_request_contract_gcs_generation": contract_generation,
        "final_validation_policy_file": policy_path.name,
        "final_validation_policy_sha256": digest,
        "final_validation_policy_gcs_uri": first[0],
        "final_validation_policy_gcs_generation": first[1],
    }
    snapshot = verify._resolve_final_validation_policy(
        bucket=bucket,
        run_dir=run_dir,
        metadata=metadata,
    )
    assert snapshot.datasets_gcs_prefix == "submitted/datasets"
    assert snapshot.validations_gcs_prefix == "submitted/validations"
    assert snapshot.artifact_sha256 == digest

    with pytest.raises(RuntimeError, match="disagrees with immutable policy"):
        verify._resolve_final_validation_policy(
            bucket=bucket,
            run_dir=run_dir,
            metadata={**metadata, "verification_apply_mode": "report_only"},
        )

    requests_path.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="request bytes changed"):
        verify._resolve_final_validation_policy(
            bucket=bucket,
            run_dir=run_dir,
            metadata=metadata,
        )


def test_legacy_mutable_metadata_cannot_promote_report_only_run(tmp_path) -> None:
    run_dir = tmp_path / "legacy-run"
    run_dir.mkdir()
    snapshot = verify._resolve_final_validation_policy(
        bucket=_MissingPolicyBucket(),
        run_dir=run_dir,
        metadata={
            # Even editing every legacy mutable policy field cannot authorize
            # correction application without the independent cloud anchor.
            "verification_apply_mode": "apply_patches",
            "correction_acceptance_policy": (
                verify.AUTOMATIC_CORRECTION_ACCEPTANCE_POLICY
            ),
            "verification_scope": "all",
        },
    )

    assert snapshot.legacy is True
    assert verify._recorded_correction_policy(
        {}, immutable_policy=snapshot
    ) == ("report_only", "legacy_report_only")


def test_immutable_validation_artifact_upload_is_content_addressed_and_idempotent(
    tmp_path,
) -> None:
    run_dir = tmp_path / "verification-run"
    run_dir.mkdir()
    artifact = run_dir / "field_corrections.json"
    artifact.write_text('{"corrected":true}\n', encoding="utf-8")
    digest = __import__("hashlib").sha256(artifact.read_bytes()).hexdigest()
    bucket = _ImmutableArtifactBucket()

    first = verify._upload_immutable_validation_artifact(
        bucket, run_dir, artifact, sha256=digest
    )
    second = verify._upload_immutable_validation_artifact(
        bucket, run_dir, artifact, sha256=digest
    )

    assert first == second
    assert first[0].endswith(
        f"/model/{run_dir.name}/immutable/{digest}/{artifact.name}"
    )
    assert first[1] == "1"
    assert len(bucket.objects) == 1
    assert bucket.upload_attempts == 2
    stored = next(iter(bucket.objects.values()))
    assert stored["payload"] == artifact.read_bytes()
    assert stored["metadata"] == {
        "artifact_kind": "field_correction_metadata",
        "sha256": digest,
        "verification_run_id": run_dir.name,
    }


def test_immutable_validation_artifact_rejects_mismatched_existing_bytes(
    tmp_path,
) -> None:
    run_dir = tmp_path / "verification-run"
    run_dir.mkdir()
    artifact = run_dir / "field_corrections.json"
    artifact.write_text('{"corrected":true}\n', encoding="utf-8")
    digest = __import__("hashlib").sha256(artifact.read_bytes()).hexdigest()
    bucket = _ImmutableArtifactBucket()

    verify._upload_immutable_validation_artifact(
        bucket, run_dir, artifact, sha256=digest
    )
    stored = next(iter(bucket.objects.values()))
    stored["payload"] = b'{"tampered":true}\n'

    with pytest.raises(RuntimeError, match="different bytes"):
        verify._upload_immutable_validation_artifact(
            bucket, run_dir, artifact, sha256=digest
        )


def _write_complete_verification_retrieval_run(tmp_path):
    run_dir = tmp_path / "verifications" / "verify-run"
    source_run = tmp_path / "submits" / "source-run"
    run_dir.mkdir(parents=True)
    source_run.mkdir(parents=True)
    key = "pages/page.png"
    candidates_path = run_dir / "page_candidates.jsonl"
    with PageCandidateWriter(candidates_path) as writer:
        writer.write(
            key=key,
            candidate={"name": "Jenson"},
            extraction_metadata={
                "model": "source-model",
                "provider": "gemini",
                "schema_name": "Candidate",
                "schema_version_id": "v1",
                "source_run_id": "source-run",
            },
        )
    candidate = read_page_candidates(candidates_path)[0]
    bindings_path = run_dir / "validation_bindings.jsonl"
    bindings_path.write_text(
        json.dumps(
            {"key": key, "candidate_sha256": candidate_sha256(candidate.candidate)}
        )
        + "\n",
        encoding="utf-8",
    )
    input_manifest_path = run_dir / "input_image_manifest.jsonl"
    input_record = _input_record_without_ocr()
    input_manifest_path.write_text(
        input_record.model_dump_json() + "\n",
        encoding="utf-8",
    )
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }
    schema_path = run_dir / "extraction_schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    requests_path = run_dir / "validation_requests.part001-of-001.jsonl"
    requests_path.write_text("{}\n", encoding="utf-8")
    policy_path, _, policy_sha256 = verify._write_final_validation_policy(
        run_dir=run_dir,
        source_run_dir=str(source_run),
        datasets_gcs_prefix="submitted/datasets",
        validations_gcs_prefix="submitted/validations",
    )
    validation = {
        "page_status": "needs_correction",
        "patches": [
            {
                "op": "replace",
                "path": "/name",
                "value_json": '"Jensen"',
                "issue": "incorrect",
                "evidence": "The page reads Jensen.",
                "ocr_box_refs": [],
            }
        ],
    }
    raw_envelope = {
        "key": key,
        "response": {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "parts": [{"text": json.dumps(validation)}]
                    },
                }
            ]
        },
    }
    metadata = {
        "created_at": "2026-08-28T12:00:00+00:00",
        "job_kind": "model_validation",
        "provider": "gemini",
        "client_backend": "mldev",
        "verification_model": "gemini-3.1-pro-preview",
        "verification_scope": "all",
        "verification_apply_mode": "apply_patches",
        "correction_acceptance_policy": (
            verify.AUTOMATIC_CORRECTION_ACCEPTANCE_POLICY
        ),
        "batch_job_names": ["batches/verify"],
        "batch_jobs": [
            {
                "batch_job_name": "batches/verify",
                "output_destination": "files/output",
                "requests_file": requests_path.name,
                "requests_sha256": verify.file_sha256(requests_path),
            }
        ],
        "source_run_dir": str(source_run),
        "source_run_id": "source-run",
        "candidates_file": candidates_path.name,
        "candidates_sha256": verify.file_sha256(candidates_path),
        "source_candidates_file": candidates_path.name,
        "source_candidates_sha256": verify.file_sha256(candidates_path),
        "source_candidate_count": 1,
        "candidate_count": 1,
        "deterministic_routing_policy": "",
        "bindings_file": bindings_path.name,
        "bindings_sha256": verify.file_sha256(bindings_path),
        "input_image_manifest_file": input_manifest_path.name,
        "input_image_manifest_sha256": verify.file_sha256(input_manifest_path),
        "extraction_schema_file": schema_path.name,
        "extraction_schema_sha256": verify.file_sha256(schema_path),
        "extraction_schema_canonical_sha256": verify._canonical_json_sha256(schema),
        "extraction_schema_name": "Candidate",
        "extraction_schema_version_id": "v1",
        "verification_prompt_version": "v2",
        "verification_prompt_hash": "prompt-hash",
        "output_format": "jsonl",
        "dataset_file_name": "dataset",
        "csv_sep": "$",
        "datasets_gcs_prefix": "submitted/datasets",
        "validations_gcs_prefix": "submitted/validations",
        "final_validation_policy_file": policy_path.name,
        "final_validation_policy_sha256": policy_sha256,
        "final_validation_policy_gcs_uri": (
            "gs://bucket/_patientjournals/model_validation_policies/"
            "verify-run/final_validation_policy.json"
        ),
        "final_validation_policy_gcs_generation": "11",
    }
    (run_dir / verify.VERIFICATION_BATCH_JOB_FILE_NAME).write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    return run_dir, raw_envelope


def _mock_complete_verification_retrieval(
    monkeypatch,
    *,
    raw_envelope,
    publisher,
) -> None:
    bucket = SimpleNamespace(name="bucket")
    monkeypatch.setattr(
        verify, "_storage_client_and_bucket", lambda: (object(), bucket)
    )

    def resolve_policy(**kwargs):
        local_run_dir = Path(kwargs["run_dir"])
        metadata = json.loads(
            (local_run_dir / verify.VERIFICATION_BATCH_JOB_FILE_NAME).read_text()
        )
        return verify._FinalValidationPolicySnapshot(
            apply_mode="apply_patches",
            acceptance_policy=verify.AUTOMATIC_CORRECTION_ACCEPTANCE_POLICY,
            verification_scope=metadata["verification_scope"],
            source_run_id="source-run",
            datasets_gcs_prefix="submitted/datasets",
            validations_gcs_prefix="submitted/validations",
            source_candidates_sha256=metadata["source_candidates_sha256"],
            selected_candidates_sha256=metadata["candidates_sha256"],
            source_candidate_count=metadata["source_candidate_count"],
            selected_candidate_count=metadata["candidate_count"],
            deterministic_routing_policy=str(
                metadata.get("deterministic_routing_policy") or ""
            ),
            artifact_sha256="policy-hash",
            artifact_gcs_uri=(
                "gs://bucket/_patientjournals/model_validation_policies/"
                "verify-run/final_validation_policy.json"
            ),
            artifact_gcs_generation="11",
        )

    monkeypatch.setattr(
        verify,
        "_resolve_final_validation_policy",
        resolve_policy,
    )
    monkeypatch.setattr(
        verify,
        "get_batch_client",
        lambda **_kwargs: SimpleNamespace(vertexai=False),
    )
    monkeypatch.setattr(verify, "_get_batch_job", lambda *_args: object())
    monkeypatch.setattr(verify, "_batch_job_successful", lambda *_args: True)
    monkeypatch.setattr(
        verify,
        "_gemini_output_reference",
        lambda *_args, **_kwargs: ("file", "files/output"),
    )

    def download(_client, _file_name, output_path):
        output_path.write_text(json.dumps(raw_envelope) + "\n", encoding="utf-8")
        return output_path

    monkeypatch.setattr(verify, "_download_from_mldev_output", download)
    monkeypatch.setattr(
        verify, "_verify_retrieval_evidence_bindings", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        verify,
        "_upload_immutable_validation_artifact",
        lambda _bucket, run_dir, path, *, sha256, validations_prefix=None: (
            f"gs://bucket/validations/{run_dir.name}/{sha256}/{path.name}",
            "17",
        ),
    )
    monkeypatch.setattr(
        verify,
        "_upload_retrieval_artifacts",
        lambda *, bucket, run_dir, paths, validations_prefix=None: {
            path.name: f"gs://{bucket.name}/{run_dir.name}/{path.name}"
            for path in paths
        },
    )
    monkeypatch.setattr(
        verify,
        "_upload_validation_artifact",
        lambda bucket, run_dir, path, *, validations_prefix=None: (
            f"gs://{bucket.name}/{run_dir.name}/{path.name}"
        ),
    )
    monkeypatch.setattr(verify, "publish_dataset_version", publisher)


def test_retrieve_model_validation_auto_corrects_and_publishes_vnnn(
    tmp_path,
    monkeypatch,
) -> None:
    run_dir, raw_envelope = _write_complete_verification_retrieval_run(tmp_path)
    publication_calls: list[dict[str, object]] = []

    def publish(**kwargs):
        publication_calls.append(kwargs)
        dataset_path = kwargs["dataset_path"]
        payload = json.loads(dataset_path.read_text(encoding="utf-8").strip())
        assert payload["name"] == "Jensen"
        ledger_path = tmp_path / "submits" / "source-run" / "dataset_versions.json"
        return SimpleNamespace(
            local_path=str(dataset_path),
            gcs_uri="gs://bucket/datasets/source-run/v001_model_validation.jsonl",
            gcs_generation="29",
            sha256=verify.file_sha256(dataset_path),
            idempotency_key="publication-key",
            publication_provenance_sha256="publication-provenance-hash",
            version=1,
            version_id="v001",
            ledger_path=str(ledger_path),
            ledger_gcs_uri="gs://bucket/datasets/source-run/dataset_versions.json",
        )

    _mock_complete_verification_retrieval(
        monkeypatch,
        raw_envelope=raw_envelope,
        publisher=publish,
    )
    monkeypatch.setattr(verify.config, "datasets_gcs_prefix", "mutated/datasets")
    monkeypatch.setattr(verify.config, "validations_gcs_prefix", "mutated/validations")
    validation_prefix_calls: list[str | None] = []

    def upload_immutable(_bucket, local_run_dir, path, *, sha256, validations_prefix):
        validation_prefix_calls.append(validations_prefix)
        return (
            f"gs://bucket/{validations_prefix}/{local_run_dir.name}/"
            f"{sha256}/{path.name}",
            "17",
        )

    def upload_many(*, bucket, run_dir, paths, validations_prefix):
        validation_prefix_calls.append(validations_prefix)
        return {
            path.name: (
                f"gs://{bucket.name}/{validations_prefix}/{run_dir.name}/{path.name}"
            )
            for path in paths
        }

    def upload_one(bucket, run_dir, path, *, validations_prefix):
        validation_prefix_calls.append(validations_prefix)
        return f"gs://{bucket.name}/{validations_prefix}/{run_dir.name}/{path.name}"

    monkeypatch.setattr(verify, "_upload_immutable_validation_artifact", upload_immutable)
    monkeypatch.setattr(verify, "_upload_retrieval_artifacts", upload_many)
    monkeypatch.setattr(verify, "_upload_validation_artifact", upload_one)

    result = verify.retrieve_model_validation(
        verify.ModelValidationRetrieveRequest(run_dir=str(run_dir))
    )

    assert result.publishable is True
    assert result.status == "publishable"
    assert result.dataset_version_id == "v001"
    assert result.dataset_publication_idempotency_key == "publication-key"
    assert result.publication_provenance_sha256 == "publication-provenance-hash"
    assert result.accepted_correction_fields == 1
    assert result.corrected_fields == 1
    assert len(publication_calls) == 1
    assert publication_calls[0]["datasets_prefix"] == "submitted/datasets"
    assert validation_prefix_calls
    assert set(validation_prefix_calls) == {"submitted/validations"}
    assert publication_calls[0]["metadata"]["final_validation_policy_sha256"] == (
        "policy-hash"
    )
    correction_metadata = json.loads(
        result.field_corrections_path.read_text(encoding="utf-8")
    )
    field = correction_metadata["pages"][0]["fields"][0]
    assert field["accepted"] is True
    assert field["corrected"] is True
    assert correction_metadata["corrected_dataset_built"] is True
    assert correction_metadata["corrected_dataset_sha256"] == result.dataset_sha256
    assert correction_metadata["evidence_artifacts"][
        "verification_artifact_sha256s"
    ]
    assert correction_metadata["source_run_id"] == "source-run"
    assert correction_metadata["evidence_artifacts"][
        "final_validation_policy"
    ] == {
        "sha256": "policy-hash",
        "gcs_uri": (
            "gs://bucket/_patientjournals/model_validation_policies/"
            "verify-run/final_validation_policy.json"
        ),
        "gcs_generation": "11",
    }
    assert publication_calls[0]["metadata"]["field_corrections_sha256"] == (
        result.field_corrections_sha256
    )


def test_risk_routed_validation_consolidates_routine_and_corrected_pages(
    tmp_path,
    monkeypatch,
) -> None:
    run_dir, raw_envelope = _write_complete_verification_retrieval_run(tmp_path)
    selected_path = run_dir / "page_candidates.jsonl"
    selected = read_page_candidates(selected_path)[0]
    policy_version = "deterministic-routing-v1"
    selected = selected.model_copy(
        update={
            "extraction_metadata": {
                **selected.extraction_metadata,
                "deterministic_status": "flagged",
                "deterministic_routing_policy_version": policy_version,
            }
        }
    )
    write_page_candidates(selected_path, (selected,))
    routine = PageCandidateRecord(
        key="pages/routine.png",
        candidate={"name": "Routine"},
        extraction_metadata={
            **selected.extraction_metadata,
            "deterministic_status": "confirmed",
        },
    )
    source_path = run_dir / verify.SOURCE_PAGE_CANDIDATES_FILE_NAME
    write_page_candidates(source_path, (selected, routine))

    manifest_path = run_dir / "input_image_manifest.jsonl"
    second_identity = _identity("1").to_dict()
    second_identity["name"] = routine.key
    routine_input = InputImageManifestRecord(
        key=routine.key,
        mime_type="image/png",
        image_source=second_identity,
        ocr_enabled=False,
    )
    manifest_path.write_text(
        _input_record_without_ocr().model_dump_json()
        + "\n"
        + routine_input.model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    metadata_path = run_dir / verify.VERIFICATION_BATCH_JOB_FILE_NAME
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        {
            "verification_scope": "flagged",
            "candidates_sha256": verify.file_sha256(selected_path),
            "source_candidates_file": source_path.name,
            "source_candidates_sha256": verify.file_sha256(source_path),
            "source_candidate_count": 2,
            "candidate_count": 1,
            "deterministic_routing_policy": policy_version,
            "input_image_manifest_sha256": verify.file_sha256(manifest_path),
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    publication_calls: list[dict[str, object]] = []

    def publish(**kwargs):
        publication_calls.append(kwargs)
        dataset_path = Path(kwargs["dataset_path"])
        rows = [json.loads(line) for line in dataset_path.read_text().splitlines()]
        assert {row["name"] for row in rows} == {"Jensen", "Routine"}
        routine_row = next(row for row in rows if row["name"] == "Routine")
        assert routine_row["verification_status"] == "deterministic_cleared"
        return SimpleNamespace(
            local_path=str(dataset_path),
            gcs_uri="gs://bucket/datasets/source-run/v001_model_validation.jsonl",
            gcs_generation="29",
            sha256=verify.file_sha256(dataset_path),
            idempotency_key="publication-key",
            publication_provenance_sha256="publication-provenance-hash",
            version=1,
            version_id="v001",
            ledger_path=str(tmp_path / "dataset_versions.json"),
            ledger_gcs_uri="gs://bucket/dataset_versions.json",
        )

    _mock_complete_verification_retrieval(
        monkeypatch,
        raw_envelope=raw_envelope,
        publisher=publish,
    )
    result = verify.retrieve_model_validation(
        verify.ModelValidationRetrieveRequest(run_dir=str(run_dir))
    )

    assert result.publishable is True
    assert result.expected_pages == 2
    assert result.completed_pages == 2
    assert result.model_reviewed_pages == 1
    assert result.deterministically_cleared_pages == 1
    correction_metadata = json.loads(result.field_corrections_path.read_text())
    assert correction_metadata["expected_pages"] == 2
    assert correction_metadata["deterministically_cleared_pages"] == 1
    routine_page = next(
        page for page in correction_metadata["pages"] if page["key"] == routine.key
    )
    assert routine_page["page_status"] == "deterministic_cleared"
    assert routine_page["included_in_corrected_dataset"] is True
    assert len(publication_calls) == 1
