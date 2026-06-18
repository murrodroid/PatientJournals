from patientjournals.batch.service import (
    BatchCollectOutputsRequest,
    BatchRetrieveRequest,
    BatchSubmitRequest,
)


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
        duplicate_strategy="provide_all",
    ).to_namespace()

    assert namespace.run_dir == "runs/submit_1"
    assert namespace.output_dir == "runs/submit_1"
    assert namespace.batch_name == ["batch-a", "batch-b"]
    assert namespace.wait is True
    assert namespace.allow_partial is True
    assert namespace.recover_missing_with_api is True
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
