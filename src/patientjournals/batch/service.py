from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from google.cloud import storage

from patientjournals.batch import submit as submit_module
from patientjournals.batch.collect_outputs import collect_outputs
from patientjournals.batch.results import CollectOutputsResult, RetrieveBatchResult
from patientjournals.batch.retrieve import retrieve_batch
from patientjournals.batch.submit import (
    _downscale_blobs_randomly,
    _filter_blobs_missing_from_dataset,
    _resolve_downscale,
    _resolve_num_batches,
    _split_blobs_evenly,
)
from patientjournals.batch.submit_inputs import _list_input_blobs
from patientjournals.config import config
from patientjournals.shared.dataset_coverage import resolve_continue_dataset_path


@dataclass(frozen=True)
class BatchSubmitRequest:
    num_batches: int | None = None
    rerun: bool = False
    run_dir: str | None = None
    continue_dataset: str | None = None
    downscale: float | None = None

    def to_namespace(self) -> argparse.Namespace:
        return argparse.Namespace(
            num_batches=self.num_batches,
            rerun=self.rerun,
            run_dir=self.run_dir,
            continue_dataset=self.continue_dataset,
            downscale=self.downscale,
        )


@dataclass(frozen=True)
class BatchChunkPlan:
    chunk_index: int
    total_chunks: int
    request_count: int


@dataclass(frozen=True)
class BatchSubmitPlan:
    input_count: int
    submit_count: int
    covered_existing_count: int
    dataset_rows: int
    chunks: tuple[BatchChunkPlan, ...]
    continue_dataset_path: Path | None = None


class BatchSubmitService:
    def __init__(self, bucket: storage.Bucket, log) -> None:
        self.bucket = bucket
        self.log = log

    def plan(self, request: BatchSubmitRequest) -> BatchSubmitPlan:
        if request.rerun:
            raise ValueError("Rerun planning is handled by existing submit metadata.")

        blobs = _list_input_blobs(self.bucket, log=self.log)
        original_count = len(blobs)
        covered_inputs = 0
        dataset_rows = 0
        continue_dataset_path: Path | None = None

        if request.continue_dataset:
            continue_dataset_path = resolve_continue_dataset_path(
                request.continue_dataset,
                run_root=config.output_root,
                dataset_name=config.dataset_file_name,
            )
            blobs, covered_inputs, dataset_rows = _filter_blobs_missing_from_dataset(
                blobs,
                dataset_path=continue_dataset_path,
                bucket_name=config.gcs_bucket_name,
                log=self.log,
            )

        downscale = _resolve_downscale(request.to_namespace())
        if downscale is not None:
            blobs = _downscale_blobs_randomly(blobs, downscale=downscale)

        num_batches = _resolve_num_batches(request.to_namespace())
        chunks = _split_blobs_evenly(blobs, num_batches)
        chunk_plans = tuple(
            BatchChunkPlan(
                chunk_index=index,
                total_chunks=len(chunks),
                request_count=len(chunk),
            )
            for index, chunk in enumerate(chunks, start=1)
        )
        return BatchSubmitPlan(
            input_count=original_count,
            submit_count=len(blobs),
            covered_existing_count=covered_inputs,
            dataset_rows=dataset_rows,
            chunks=chunk_plans,
            continue_dataset_path=continue_dataset_path,
        )

    def submit(self, request: BatchSubmitRequest) -> None:
        submit_module.submit_batch(args=request.to_namespace())


@dataclass(frozen=True)
class BatchRetrieveRequest:
    batch_name: str | None = None
    batch_names: tuple[str, ...] = ()
    run_dir: str | None = None
    output_dir: str | None = None
    wait: bool = False
    allow_partial: bool = False
    submit_failed: bool = False
    failed_retry_num_batches: int | None = None
    recover_missing_with_api: bool = False
    ignore_failed: bool = False
    duplicate_strategy: str | None = None

    def to_namespace(self) -> argparse.Namespace:
        return argparse.Namespace(
            batch_name=[
                *self.batch_names,
                *([self.batch_name] if self.batch_name else []),
            ],
            run_dir=self.run_dir,
            output_dir=self.output_dir,
            wait=self.wait,
            allow_partial=self.allow_partial,
            submit_failed=self.submit_failed,
            failed_retry_num_batches=self.failed_retry_num_batches,
            recover_missing_with_api=self.recover_missing_with_api,
            ignore_failed=self.ignore_failed,
            duplicate_strategy=self.duplicate_strategy,
        )


@dataclass(frozen=True)
class BatchCollectOutputsRequest:
    bucket_name: str | None = None
    outputs_prefix: str = config.batch_outputs_gcs_prefix
    output_glob: str = "*predictions.jsonl"
    pages_prefix: str = config.gcs_pages_prefix
    pages_glob: str = "*"
    local_output: tuple[str, ...] = ()
    continue_dataset: str | None = None
    skip_gcs_outputs: bool = False
    skip_pages: bool = False
    output_format: str = config.output_format
    run_root: str = config.output_root

    def to_namespace(self) -> argparse.Namespace:
        return argparse.Namespace(
            bucket_name=self.bucket_name,
            outputs_prefix=self.outputs_prefix,
            output_glob=self.output_glob,
            pages_prefix=self.pages_prefix,
            pages_glob=self.pages_glob,
            local_output=list(self.local_output),
            continue_dataset=self.continue_dataset,
            skip_gcs_outputs=self.skip_gcs_outputs,
            skip_pages=self.skip_pages,
            output_format=self.output_format,
            run_root=self.run_root,
        )


class BatchResultService:
    def retrieve(self, request: BatchRetrieveRequest) -> RetrieveBatchResult:
        return retrieve_batch(args=request.to_namespace())

    def collect_outputs(
        self,
        request: BatchCollectOutputsRequest,
    ) -> CollectOutputsResult:
        return collect_outputs(args=request.to_namespace())
