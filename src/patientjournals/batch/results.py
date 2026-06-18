from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CollectOutputsResult:
    dataset_path: Path
    report_path: Path
    selected_outputs_path: Path
    rejected_output_keys_path: Path
    missing_page_keys_path: Path | None
    dataset_rows: int
    new_output_rows_added: int
    pages_total: int | None
    pages_covered: int | None
    missing_pages: int | None
    dataset_gcs_uri: str = ""


@dataclass(frozen=True)
class RetrieveBatchResult:
    dataset_path: Path
    run_dir: Path
    provider: str
    batch_count: int
    output_file_count: int
    rows_written: int
    error_rows: int
    expected_pages: int
    observed_pages: int
    successful_pages: int
    duplicate_rows_skipped: int = 0
    recovered_pages: int = 0
    manifest_path: Path | None = None
    dataset_gcs_uri: str = ""
