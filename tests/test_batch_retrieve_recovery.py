import asyncio
import hashlib
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from argparse import Namespace

from pydantic import BaseModel

from patientjournals.batch import retrieve
from patientjournals.batch.ocr_context import CloudBlobIdentity, CloudOcrMetadata
from patientjournals.config import config
from patientjournals.shared.ocr import OcrDocument, OcrLine
from patientjournals.shared.subagents import encode_specialist_request_key
from patientjournals.validation.input_manifest import (
    ExtractionImageBinding,
    InputImageManifestRecord,
    ocr_document_sha256,
)


class SimpleOutput(BaseModel):
    value: str


def test_recovery_api_key_uses_provider_config(monkeypatch) -> None:
    monkeypatch.setattr(config, "provider_api_keys", {"gemini": "provider-key"})
    monkeypatch.setattr(config, "api_key", "")

    assert retrieve._resolve_recovery_api_key() == "provider-key"


def test_recovery_api_key_accepts_api_keys_gemini_alias(monkeypatch) -> None:
    fake_api_keys = types.ModuleType("api_keys")
    fake_api_keys.gemini = "module-gemini-key"
    monkeypatch.setitem(sys.modules, "api_keys", fake_api_keys)
    monkeypatch.setattr(config, "provider_api_keys", {})
    monkeypatch.setattr(config, "api_key", "")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    assert retrieve._resolve_recovery_api_key() == "module-gemini-key"


def test_failed_dataset_row_marks_image_failed() -> None:
    row = retrieve._failed_dataset_row(
        "gs://bucket/pages/folder/273057_001519.png",
        "schema_validation_failed",
    )

    assert row == {
        "image_name": "273057_001519.png",
        "file_name": "gs://bucket/pages/folder/273057_001519.png",
        "failed": True,
        "failure_reason": "schema_validation_failed",
        "model": config.model,
        "schema_name": config.output_schema_name,
        "schema_version_id": config.output_schema_version_id,
    }


def test_restore_submit_semantics_restores_retry_request_configuration(
    tmp_path,
    monkeypatch,
) -> None:
    schema = SimpleOutput.model_json_schema()
    source_values = {
        "model": "gemini-3.1-pro-preview",
        "model_temperature": 0.37,
        "model_max_output_tokens": 8123,
        "thinking_level": "medium",
        "include_thoughts": True,
        "include_confidence_scores": True,
        "include_response_avg_logprobs": False,
        "batch_include_response_schema": False,
        "response_mime_type": "application/json",
        "response_schema_field": "response_schema",
        "output_format": "jsonl",
        "dataset_file_name": "source_dataset",
        "csv_sep": "|",
        "input_prompt_name": "frontpage",
        "ocr_enabled": False,
    }
    submit_run_dir = tmp_path / "submit"
    submit_run_dir.mkdir()
    (submit_run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_name": "SimpleOutput",
                "schema_version_id": "",
                "output_schema": schema,
                "config_values": {"config": source_values},
            }
        ),
        encoding="utf-8",
    )

    for field_name, current_value in {
        "model": "gemini-3.7-flash",
        "model_temperature": 0.0,
        "model_max_output_tokens": 1,
        "thinking_level": "low",
        "include_thoughts": False,
        "include_confidence_scores": False,
        "include_response_avg_logprobs": True,
        "batch_include_response_schema": True,
        "response_mime_type": "text/plain",
        "response_schema_field": "response_json_schema",
        "output_format": "csv",
        "dataset_file_name": "current_dataset",
        "csv_sep": ",",
        "input_prompt_name": "frontpage",
        "ocr_enabled": True,
    }.items():
        monkeypatch.setattr(config, field_name, current_value)
    monkeypatch.setattr(config, "output_model", SimpleOutput)
    monkeypatch.setattr(config, "output_schema", schema)
    monkeypatch.setattr(config, "output_schema_name", "SimpleOutput")
    monkeypatch.setattr(config, "output_schema_version_id", "")
    monkeypatch.setattr(config, "output_schema_override", None)

    retrieve._restore_submit_semantics(
        submit_run_dir=submit_run_dir,
        submit_metadata={
            "schema_name": "SimpleOutput",
            "schema_version_id": "",
        },
        log=lambda _message: None,
    )

    assert {
        field_name: getattr(config, field_name) for field_name in source_values
    } == source_values
    assert config.output_schema == schema


def test_subagent_retry_plan_withholds_non_repairable_join_failures() -> None:
    duplicate_page = "pages/duplicate.png"
    joined_page = "pages/joined.png"
    invalid_page = "pages/invalid.png"
    missing_page = "pages/missing.png"
    absent_page = "pages/absent.png"
    failed_pages = {
        duplicate_page,
        joined_page,
        invalid_page,
        missing_page,
        absent_page,
    }

    retry_keys, retry_reasons, withheld = retrieve._plan_subagent_retry_requests(
        failed_page_keys=failed_pages,
        failed_page_reasons={page: "missing_output" for page in failed_pages},
        subagent_failures=(
            {
                "page_key": duplicate_page,
                "specialist": "patient",
                "reason": "duplicate_valid_specialist",
                "retryable": False,
            },
            {
                "page_key": joined_page,
                "reason": "joined_schema_validation_failed",
                "retryable": False,
            },
            {
                "page_key": invalid_page,
                "specialist": "patient",
                "reason": "specialist_schema_validation_failed",
            },
            {
                "page_key": missing_page,
                "reason": "missing_specialists",
                "missing_specialists": ["diagnosis"],
            },
        ),
    )

    expected_retry_keys = {
        encode_specialist_request_key(invalid_page, "patient"),
        encode_specialist_request_key(missing_page, "diagnosis"),
        absent_page,
    }
    assert retry_keys == expected_retry_keys
    assert retry_reasons == {
        request_key: "missing_output" for request_key in expected_retry_keys
    }
    assert withheld == {
        duplicate_page: ("duplicate_valid_specialist",),
        joined_page: ("joined_schema_validation_failed",),
    }
    assert failed_pages == {
        duplicate_page,
        joined_page,
        invalid_page,
        missing_page,
        absent_page,
    }


def test_failed_page_retry_can_split_into_multiple_chunks(
    tmp_path,
    monkeypatch,
) -> None:
    from patientjournals.batch.retry import _submit_failed_pages_as_batch

    monkeypatch.setattr(config, "output_root", str(tmp_path))
    monkeypatch.setattr(config, "gcs_bucket_name", "bucket")
    monkeypatch.setattr(config, "model", "gemini-3.1-pro-preview")
    monkeypatch.setattr(config, "batch_requests_file_name", "batch_requests.jsonl")
    monkeypatch.setattr(config, "batch_job_display_name", "retry-test")
    monkeypatch.setattr(config, "batch_include_response_schema", False)
    monkeypatch.setattr(
        "patientjournals.batch.retry.ocr_context_for_blob",
        lambda _blob: "",
    )

    parent = tmp_path / "parent_submit"
    parent.mkdir()
    (parent / "batch_job.json").write_text(
        json.dumps(
            {
                "model": "gemini-3.1-pro-preview",
                "job_group_id": "parent_submit",
                "job_group_role": "root",
                "batch_jobs": [
                    {
                        "chunk_index": 1,
                        "total_chunks": 1,
                        "chunk_label": "chunk_001_of_001",
                        "batch_job_name": "original-batch",
                        "request_count": 5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class FakeFiles:
        def __init__(self) -> None:
            self.uploaded: list[str] = []

        def upload(self, *, file: str, config) -> SimpleNamespace:
            self.uploaded.append(Path(file).name)
            return SimpleNamespace(name=f"uploaded-{Path(file).name}")

    class FakeBatches:
        def __init__(self) -> None:
            self.created: list[tuple[str, str]] = []

        def create(self, *, model: str, src: str, config) -> SimpleNamespace:
            self.created.append((model, src))
            return SimpleNamespace(name=f"retry-batch-{len(self.created)}")

    class FakeClient:
        vertexai = False

        def __init__(self) -> None:
            self.files = FakeFiles()
            self.batches = FakeBatches()

    client = FakeClient()
    logs: list[str] = []

    result = _submit_failed_pages_as_batch(
        failed_keys={f"pages/folder/p{index}.png" for index in range(5)},
        failure_reasons={},
        provider="gemini",
        client=client,
        batch_names=["original-batch"],
        submit_run_dir=parent,
        log=logs.append,
        num_batches=3,
    )

    assert result is not None
    retry_run_dir, retry_batch_names, retry_count = result
    assert retry_count == 5
    assert retry_batch_names == ["retry-batch-1", "retry-batch-2", "retry-batch-3"]
    assert client.files.uploaded == [
        "batch_requests.part001-of-003.jsonl",
        "batch_requests.part002-of-003.jsonl",
        "batch_requests.part003-of-003.jsonl",
    ]

    retry_meta = json.loads((retry_run_dir / "batch_job.json").read_text(encoding="utf-8"))
    assert retry_meta["num_batches_requested"] == 3
    assert retry_meta["num_batches_submitted"] == 3
    assert retry_meta["request_count"] == 5
    assert retry_meta["batch_job_names"] == retry_batch_names
    assert [job["request_count"] for job in retry_meta["batch_jobs"]] == [2, 2, 1]
    assert [job["total_chunks"] for job in retry_meta["batch_jobs"]] == [3, 3, 3]

    parent_meta = json.loads((parent / "batch_job.json").read_text(encoding="utf-8"))
    retry_entries = [
        job for job in parent_meta["batch_jobs"] if job.get("is_retry")
    ]
    assert len(retry_entries) == 3
    assert parent_meta["batch_job_names"] == [
        "original-batch",
        "retry-batch-1",
        "retry-batch-2",
        "retry-batch-3",
    ]
    assert parent_meta["retry_runs"][0]["batch_count"] == 3
    assert parent_meta["retry_runs"][0]["request_count"] == 5


def test_request_identity_resolution_loads_each_portable_retry_run(
    tmp_path,
) -> None:
    submits = tmp_path / "submits"
    parent = submits / "root"
    retry_one = submits / "retry-one"
    retry_two = submits / "retry-two"
    parent.mkdir(parents=True)
    retry_one.mkdir()
    retry_two.mkdir()

    original_key = "pages/original.png"
    retry_one_key = encode_specialist_request_key(
        "pages/retry-one.png", "diagnosis"
    )
    retry_two_key = encode_specialist_request_key("pages/retry-two.png", "patient")
    (parent / "original.jsonl").write_text(
        json.dumps({"key": original_key, "custom_id": "original-id"}) + "\n",
        encoding="utf-8",
    )
    # Retry attempts intentionally reuse the same basename. Their run IDs, not
    # that basename, must keep the request populations distinct.
    for retry_dir, request_key, custom_id in (
        (retry_one, retry_one_key, "retry-one-id"),
        (retry_two, retry_two_key, "retry-two-id"),
    ):
        (retry_dir / "retry.jsonl").write_text(
            json.dumps({"key": request_key, "custom_id": custom_id}) + "\n",
            encoding="utf-8",
        )

    (parent / "batch_job.json").write_text(
        json.dumps(
            {
                "batch_jobs": [
                    {
                        "batch_job_name": "original",
                        "requests_file": "original.jsonl",
                    },
                    {
                        "batch_job_name": "retry-one-job",
                        "requests_file": "retry.jsonl",
                        "is_retry": True,
                        "retry_run_id": retry_one.name,
                        "retry_run_dir": "/stale/machine/path/one",
                    },
                    {
                        "batch_job_name": "retry-two-job",
                        "requests_file": "retry.jsonl",
                        "is_retry": True,
                        "retry_run_id": retry_two.name,
                        "retry_run_dir": "/stale/machine/path/two",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    logs: list[str] = []
    expected = retrieve._resolve_expected_request_keys(
        submit_run_dir=parent,
        batch_names=["original", "retry-one-job", "retry-two-job"],
        selected_batch_names=["original", "retry-one-job", "retry-two-job"],
        log=logs.append,
    )
    custom_ids = retrieve._resolve_anthropic_custom_id_to_key(
        submit_run_dir=parent,
        batch_names=["original", "retry-one-job", "retry-two-job"],
        selected_batch_names=["original", "retry-one-job", "retry-two-job"],
        log=logs.append,
    )

    assert expected == {
        original_key,
        "pages/retry-one.png",
        "pages/retry-two.png",
    }
    assert custom_ids == {
        "original-id": original_key,
        "retry-one-id": retry_one_key,
        "retry-two-id": retry_two_key,
    }


def test_request_identity_resolution_recovers_generation_bound_cloud_artifact(
    tmp_path,
    monkeypatch,
) -> None:
    parent = tmp_path / "root"
    parent.mkdir()
    binding = {
        "uri": "gs://bucket/retries/immutable/retry.jsonl",
        "sha256": "a" * 64,
        "generation": "7",
    }
    (parent / "batch_job.json").write_text(
        json.dumps(
            {
                "batch_jobs": [
                    {
                        "batch_job_name": "retry-job",
                        "requests_file": "retry.jsonl",
                        "is_retry": True,
                        "retry_run_id": "missing-retry-run",
                        "requests_gcs_binding": binding,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    request_key = encode_specialist_request_key("pages/cloud.png", "patient")
    downloaded: list[dict[str, str]] = []

    def fake_download(value):
        downloaded.append(dict(value))
        return (json.dumps({"key": request_key}) + "\n").encode()

    monkeypatch.setattr(
        retrieve,
        "_download_bound_request_artifact",
        fake_download,
    )

    assert retrieve._resolve_expected_request_keys(
        submit_run_dir=parent,
        batch_names=["retry-job"],
        selected_batch_names=["retry-job"],
        log=lambda _message: None,
    ) == {"pages/cloud.png"}
    assert downloaded == [binding]


def test_submit_run_discovery_prefers_root_covering_all_retry_jobs(
    tmp_path,
    monkeypatch,
) -> None:
    submits = tmp_path / "submits"
    root = submits / "root"
    retry = submits / "retry"
    root.mkdir(parents=True)
    retry.mkdir()
    (root / "original.jsonl").write_text(
        json.dumps({"key": "pages/original.png"}) + "\n",
        encoding="utf-8",
    )
    (retry / "retry.jsonl").write_text(
        json.dumps({"key": "pages/retry.png"}) + "\n",
        encoding="utf-8",
    )
    (retry / "batch_job.json").write_text(
        json.dumps(
            {
                "job_group_role": "retry",
                "batch_jobs": [
                    {
                        "batch_job_name": "retry-job",
                        "requests_file": "retry.jsonl",
                        "is_retry": True,
                        "retry_run_id": retry.name,
                        "retry_run_dir": str(retry),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "batch_job.json").write_text(
        json.dumps(
            {
                "job_group_role": "root",
                "batch_jobs": [
                    {
                        "batch_job_name": "original-job",
                        "requests_file": "original.jsonl",
                    },
                    {
                        "batch_job_name": "retry-job",
                        "requests_file": "retry.jsonl",
                        "is_retry": True,
                        "retry_run_id": retry.name,
                        "retry_run_dir": str(retry),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "output_root", str(tmp_path))

    assert retrieve._find_submit_run_dir(["original-job", "retry-job"]) == root


class FakeBlob:
    content_type = "image/png"

    def exists(self) -> bool:
        return True

    def download_as_bytes(self) -> bytes:
        return b"image-bytes"


class FakeBucket:
    def blob(self, key: str) -> FakeBlob:
        return FakeBlob()


def recovery_response(value: str = "ok") -> dict:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps({"value": value}),
                        }
                    ]
                }
            }
        ]
    }


def test_api_key_recovery_uses_configured_concurrency(monkeypatch) -> None:
    monkeypatch.setattr(config, "api_concurrent_tasks", 2)
    monkeypatch.setattr(config, "api_max_attempts", 1)
    monkeypatch.setattr(config, "output_model", SimpleOutput)
    monkeypatch.setattr(retrieve, "ocr_context_for_blob", lambda _blob: "")

    class FakeModels:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0

        async def generate_content(self, **kwargs) -> dict:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await asyncio.sleep(0.05)
                return recovery_response()
            finally:
                self.active -= 1

    models = FakeModels()
    client = SimpleNamespace(aio=SimpleNamespace(models=models))

    results = asyncio.run(
        retrieve._recover_missing_pages_via_api_key_async(
            missing_keys={"pages/1.png", "pages/2.png", "pages/3.png"},
            bucket=FakeBucket(),
            recovery_client=client,
            recovery_model="gemini-test",
            generation_config={},
            log=lambda *args, **kwargs: None,
        )
    )

    assert len(results) == 3
    assert all(result.parsed_model == SimpleOutput(value="ok") for result in results)
    assert models.max_active == 2


def test_retrieve_args_support_repeated_batch_names(monkeypatch) -> None:
    monkeypatch.setattr(config, "batch_duplicate_strategy", "first_successful")
    args = Namespace(
        batch_name=["batch-a", "batch-a", "batch-b"],
        duplicate_strategy=None,
    )

    assert retrieve._arg_batch_names(args) == ["batch-a", "batch-b"]
    assert retrieve._effective_duplicate_strategy(args) == "first_successful"


def test_api_key_recovery_retries_transient_errors(monkeypatch) -> None:
    monkeypatch.setattr(config, "api_concurrent_tasks", 1)
    monkeypatch.setattr(config, "api_max_attempts", 2)
    monkeypatch.setattr(config, "api_retry_initial_delay_seconds", 0)
    monkeypatch.setattr(config, "api_retry_max_delay_seconds", 0)
    monkeypatch.setattr(config, "api_retry_jitter_seconds", 0)
    monkeypatch.setattr(config, "output_model", SimpleOutput)
    monkeypatch.setattr(retrieve, "ocr_context_for_blob", lambda _blob: "")

    class FakeModels:
        def __init__(self) -> None:
            self.calls = 0

        async def generate_content(self, **kwargs) -> dict:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("503 unavailable")
            return recovery_response("retried")

    models = FakeModels()
    client = SimpleNamespace(aio=SimpleNamespace(models=models))

    results = asyncio.run(
        retrieve._recover_missing_pages_via_api_key_async(
            missing_keys={"pages/1.png"},
            bucket=FakeBucket(),
            recovery_client=client,
            recovery_model="gemini-test",
            generation_config={},
            log=lambda *args, **kwargs: None,
        )
    )

    assert models.calls == 2
    assert results[0].parsed_model == SimpleOutput(value="retried")


def test_api_key_recovery_failure_reason_includes_exception_detail(monkeypatch) -> None:
    monkeypatch.setattr(config, "api_concurrent_tasks", 1)
    monkeypatch.setattr(config, "api_max_attempts", 1)
    monkeypatch.setattr(retrieve, "ocr_context_for_blob", lambda _blob: "")

    class FakeModels:
        async def generate_content(self, **kwargs) -> dict:
            raise RuntimeError("400 INVALID_ARGUMENT: model gemini-x is not supported")

    client = SimpleNamespace(aio=SimpleNamespace(models=FakeModels()))

    results = asyncio.run(
        retrieve._recover_missing_pages_via_api_key_async(
            missing_keys={"pages/1.png"},
            bucket=FakeBucket(),
            recovery_client=client,
            recovery_model="gemini-test",
            generation_config={},
            log=lambda *args, **kwargs: None,
        )
    )

    assert results[0].parsed_model is None
    assert results[0].failure_reason
    assert results[0].failure_reason.startswith(
        "api_key_recovery_failed:RuntimeError:"
    )
    assert "400 INVALID_ARGUMENT" in results[0].failure_reason
    assert "gemini-x is not supported" in results[0].failure_reason


def test_validation_recovery_uses_exact_request_image_and_bound_ocr_sidecar(
    monkeypatch,
) -> None:
    key = "pages/1.png"
    request_image_bytes = b"exact-first-pass-request-image"
    image_digest = hashlib.sha256(request_image_bytes).hexdigest()
    source = CloudBlobIdentity(
        bucket="bucket",
        name=key,
        generation="101",
        size=len(request_image_bytes),
        crc32c="source-crc",
        md5_hash="source-md5",
        etag="source-etag",
    )
    request_source = CloudBlobIdentity(
        bucket="bucket",
        name="batch/run/extraction_images/staged.png",
        generation="202",
        size=len(request_image_bytes),
        crc32c="source-crc",
        md5_hash="source-md5",
        etag="request-etag",
    )
    document = OcrDocument(
        image_sha256=image_digest,
        width=100,
        height=200,
        backend="test-ocr",
        lines=(OcrLine(text="bound text", box=(10, 20, 30, 40)),),
    )
    sidecar_source = CloudBlobIdentity(
        bucket="bucket",
        name=f"{key}.ocr.json",
        generation="303",
        size=None,
        crc32c=None,
        md5_hash=None,
        etag="sidecar-etag",
    )
    sidecar_bytes = CloudOcrMetadata(
        source=source,
        document=document,
        created_at=datetime.now(timezone.utc).isoformat(),
    ).to_json().encode("utf-8")
    input_record = InputImageManifestRecord(
        key=key,
        mime_type="image/png",
        image_source=source.to_dict(),
        ocr_enabled=True,
        ocr_sidecar_name=sidecar_source.name,
        ocr_sidecar_source=sidecar_source.to_dict(),
        ocr_sidecar_sha256=hashlib.sha256(sidecar_bytes).hexdigest(),
        ocr_image_sha256=image_digest,
        ocr_document_sha256=ocr_document_sha256(document),
        ocr_backend=document.backend,
        ocr_line_count=len(document.lines),
    )
    binding = ExtractionImageBinding(
        key=key,
        provider="gemini",
        reference_mode="immutable_staged_uri",
        source_image=source.to_dict(),
        request_image=request_source.to_dict(),
        request_uri=f"gs://bucket/{request_source.name}",
    )
    evidence = retrieve._FirstPassRecoveryEvidence(
        image_binding=binding,
        input_record=input_record,
    )

    class BoundBlob:
        content_type = "image/png"

        def __init__(self, bucket, name: str, generation: int | None) -> None:
            self.bucket = bucket
            self.name = name
            self.generation = str(generation or "")

        def exists(self) -> bool:
            return (self.name, self.generation) in self.bucket.payloads

        def download_as_bytes(self, **kwargs) -> bytes:
            self.bucket.downloads.append((self.name, self.generation, kwargs))
            return self.bucket.payloads[(self.name, self.generation)]

    class BoundBucket:
        name = "bucket"

        def __init__(self) -> None:
            self.payloads = {
                (request_source.name, request_source.generation): request_image_bytes,
                (sidecar_source.name, sidecar_source.generation): sidecar_bytes,
            }
            self.requests: list[tuple[str, int | None]] = []
            self.downloads: list[tuple[str, str, dict]] = []

        def blob(self, name: str, generation: int | None = None) -> BoundBlob:
            self.requests.append((name, generation))
            return BoundBlob(self, name, generation)

    captured: dict[str, object] = {}

    async def fake_generate_recovery_response(**kwargs) -> dict:
        captured.update(kwargs)
        response = recovery_response("bound")
        response["candidates"][0]["finishReason"] = "STOP"
        return response

    monkeypatch.setattr(config, "model_validation_enabled", True)
    monkeypatch.setattr(config, "api_concurrent_tasks", 1)
    monkeypatch.setattr(config, "api_max_attempts", 1)
    monkeypatch.setattr(config, "output_model", SimpleOutput)
    monkeypatch.setattr(
        retrieve,
        "_generate_recovery_response",
        fake_generate_recovery_response,
    )
    bucket = BoundBucket()

    results = asyncio.run(
        retrieve._recover_missing_pages_via_api_key_async(
            missing_keys={key},
            bucket=bucket,
            recovery_client=object(),
            recovery_model="gemini-test",
            generation_config={},
            log=lambda *args, **kwargs: None,
            first_pass_evidence_by_key={key: evidence},
        )
    )

    assert results[0].parsed_model == SimpleOutput(value="bound")
    assert captured["image_bytes"] == request_image_bytes
    assert captured["mime_type"] == "image/png"
    assert "bound text" in str(captured["ocr_context"])
    assert bucket.requests == [
        (request_source.name, int(request_source.generation)),
        (sidecar_source.name, int(sidecar_source.generation)),
    ]
    assert bucket.downloads == [
        (
            request_source.name,
            request_source.generation,
            {"if_generation_match": int(request_source.generation)},
        ),
        (
            sidecar_source.name,
            sidecar_source.generation,
            {"if_generation_match": int(sidecar_source.generation)},
        ),
    ]


def test_validation_recovery_fails_closed_without_first_pass_evidence(
    monkeypatch,
) -> None:
    monkeypatch.setattr(config, "model_validation_enabled", True)
    monkeypatch.setattr(config, "api_concurrent_tasks", 1)

    results = asyncio.run(
        retrieve._recover_missing_pages_via_api_key_async(
            missing_keys={"pages/1.png"},
            bucket=FakeBucket(),
            recovery_client=object(),
            recovery_model="gemini-test",
            generation_config={},
            log=lambda *args, **kwargs: None,
            first_pass_evidence_by_key={},
        )
    )

    assert results[0].parsed_model is None
    assert results[0].failure_reason == "recovery_first_pass_evidence_not_found"
