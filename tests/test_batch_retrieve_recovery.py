import asyncio
import json
from types import SimpleNamespace
from argparse import Namespace

from pydantic import BaseModel

from patientjournals.batch import retrieve
from patientjournals.config import config


class SimpleOutput(BaseModel):
    value: str


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
