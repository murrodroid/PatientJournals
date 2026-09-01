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
    page_candidates_path: Path | None = None
    page_candidates_gcs_uri: str = ""
    page_candidates_sha256: str = ""
    page_candidates_gcs_generation: str = ""
    deterministic_routing_path: Path | None = None
    deterministic_routing_gcs_uri: str = ""
    deterministic_routing_sha256: str = ""
    deterministic_routing_gcs_generation: str = ""
    subagent_combined_gcs_uri: str = ""
    subagent_combined_sha256: str = ""
    subagent_combined_gcs_generation: str = ""
    subagent_failures_gcs_uri: str = ""
    subagent_failures_sha256: str = ""
    subagent_failures_gcs_generation: str = ""
    deterministic_flagged_pages: int = 0
    deterministic_routine_pages: int = 0
    duplicate_rows_skipped: int = 0
    recovered_pages: int = 0
    failed_rows_included: int = 0
    manifest_path: Path | None = None
    dataset_gcs_uri: str = ""
