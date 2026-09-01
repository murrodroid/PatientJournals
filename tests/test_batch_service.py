import json

import pytest

from patientjournals.batch import retrieve, submit
from patientjournals.batch.service import (
    BatchCollectOutputsRequest,
    BatchRetrieveRequest,
    BatchSubmitRequest,
)
from patientjournals.config import config


def test_batch_submit_request_namespace() -> None:
    namespace = BatchSubmitRequest(
        num_batches=3,
        continue_dataset="newest",
        downscale=0.5,
    ).to_namespace()

    assert namespace.num_batches == 3
    assert namespace.continue_dataset == "newest"
    assert namespace.downscale == 0.5
    assert namespace.rerun is False


def test_batch_retrieve_request_namespace() -> None:
    namespace = BatchRetrieveRequest(
        run_dir="runs/submit_1",
        output_dir="runs/submit_1",
        batch_names=("batch-a", "batch-b"),
        wait=True,
        allow_partial=True,
        recover_missing_with_api=True,
        ignore_failed=True,
        duplicate_strategy="provide_all",
    ).to_namespace()

    assert namespace.run_dir == "runs/submit_1"
    assert namespace.output_dir == "runs/submit_1"
    assert namespace.batch_name == ["batch-a", "batch-b"]
    assert namespace.wait is True
    assert namespace.allow_partial is True
    assert namespace.recover_missing_with_api is True
    assert namespace.ignore_failed is True
    assert namespace.duplicate_strategy == "provide_all"


def test_batch_collect_outputs_request_namespace() -> None:
    namespace = BatchCollectOutputsRequest(
        bucket_name="bucket",
        local_output=("out.jsonl",),
        skip_gcs_outputs=True,
    ).to_namespace()

    assert namespace.bucket_name == "bucket"
    assert namespace.local_output == ["out.jsonl"]
    assert namespace.skip_gcs_outputs is True


def test_validation_rerun_refuses_to_regenerate_missing_request_bytes(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(config, "model_validation_enabled", True)

    with pytest.raises(FileNotFoundError, match="original request JSONL bytes"):
        submit._ensure_requests_files_for_rerun(
            run_dir=tmp_path,
            total_chunks=2,
            existing_files_by_index={1: "batch_requests.part001-of-002.jsonl"},
            bucket=object(),
            provider="gemini",
            client=object(),
            log=lambda _message: None,
        )


def test_rerun_restores_recorded_transport_semantics(tmp_path, monkeypatch) -> None:
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "config_values": {
                    "config": {
                        "gcs_bucket_name": "recorded-bucket",
                        "batch_backend": "vertex",
                        "batch_requests_gcs_prefix": "recorded/requests",
                        "batch_outputs_gcs_prefix": "recorded/outputs",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    def restore_scientific_semantics(**_kwargs) -> None:
        config.model = "gemini-3.1-pro-preview"

    monkeypatch.setattr(
        retrieve,
        "_restore_submit_semantics",
        restore_scientific_semantics,
    )
    monkeypatch.setattr(config, "gcs_bucket_name", "current-bucket")
    monkeypatch.setattr(config, "batch_backend", "mldev")
    monkeypatch.setattr(config, "batch_requests_gcs_prefix", "current/requests")
    monkeypatch.setattr(config, "batch_outputs_gcs_prefix", "current/outputs")

    submit._restore_rerun_semantics(
        run_dir=tmp_path,
        batch_payload={"provider": "gemini"},
        log=lambda _message: None,
    )

    assert config.gcs_bucket_name == "recorded-bucket"
    assert config.batch_backend == "vertex"
    assert config.batch_requests_gcs_prefix == "recorded/requests"
    assert config.batch_outputs_gcs_prefix == "recorded/outputs"
