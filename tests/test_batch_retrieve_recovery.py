import asyncio
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from argparse import Namespace

from pydantic import BaseModel

from patientjournals.batch import retrieve
from patientjournals.config import config


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
