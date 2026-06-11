from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from patientjournals.config import config


AuthMode = Literal["service_account", "adc", "api_key"]
DatasetSource = Literal["local", "cloud"]
RunMode = Literal["local_api", "cloud_batch"]
OutputFormat = Literal["jsonl", "csv"]
DuplicateStrategy = Literal["first_successful", "provide_all"]


@dataclass(frozen=True)
class AppSettings:
    auth_mode: AuthMode = "service_account"
    batch_backend: str = "vertex"
    service_account_file: str = ""
    gcp_project_id: str = ""
    gcp_location: str = ""
    vertex_model_location: str = ""
    gcs_bucket_name: str = ""
    gcs_pages_prefix: str = ""
    batch_requests_gcs_prefix: str = ""
    batch_outputs_gcs_prefix: str = ""
    local_runs_root: str = "runs"
    gemini_api_key_env: str = "GEMINI_API_KEY"
    validation_images_root: str = ""
    batch_duplicate_strategy: DuplicateStrategy = "first_successful"
    # Optional, user-provided rate used only to show a rough cost estimate before
    # submitting a batch. 0 disables the estimate. This is not a billed figure.
    estimated_cost_per_1k_images: float = 0.0

    @classmethod
    def from_runtime_config(cls) -> "AppSettings":
        return cls(
            auth_mode="service_account",
            batch_backend=str(config.batch_backend or "vertex"),
            service_account_file=str(config.service_account_file or ""),
            gcp_project_id=str(config.gcp_project_id or ""),
            gcp_location=str(config.gcp_location or ""),
            vertex_model_location=str(config.vertex_model_location or ""),
            gcs_bucket_name=str(config.gcs_bucket_name or ""),
            gcs_pages_prefix=str(config.gcs_pages_prefix or ""),
            batch_requests_gcs_prefix=str(config.batch_requests_gcs_prefix or ""),
            batch_outputs_gcs_prefix=str(config.batch_outputs_gcs_prefix or ""),
            local_runs_root=str(config.output_root or "runs"),
            validation_images_root=str(
                config.upload_images_folder or config.target_folder or ""
            ),
            batch_duplicate_strategy=str(
                config.batch_duplicate_strategy or "first_successful"
            ),  # type: ignore[arg-type]
        )

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SchemaOption:
    name: str
    module: str
    field_count: int
    is_top_level: bool

    @property
    def label(self) -> str:
        marker = "" if self.is_top_level else " (nested)"
        return f"{self.name}{marker}"


@dataclass(frozen=True)
class ModelOption:
    name: str
    provider: str
    supports_batch: bool
    supports_confidence_scores: bool
    supports_thoughts: bool

    @property
    def label(self) -> str:
        batch = "batch" if self.supports_batch else "api only"
        return f"{self.name} ({batch})"


@dataclass(frozen=True)
class DatasetSummary:
    source: DatasetSource
    name: str
    root: str
    image_count: int
    duplicate_image_names: tuple[str, ...] = ()
    status: str = "ok"
    detail: str = ""


@dataclass(frozen=True)
class CloudResolution:
    local_root: str
    bucket_name: str
    prefix: str
    local_image_count: int
    cloud_image_count: int
    matched_image_count: int
    missing_image_names: tuple[str, ...] = ()
    duplicate_local_image_names: tuple[str, ...] = ()
    duplicate_cloud_image_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class CloudDatasetChoice:
    prefix: str
    image_count: int
    object_count: int
    updated_at: str = ""

    @property
    def label(self) -> str:
        return f"{self.prefix} ({self.image_count} image(s))"


@dataclass(frozen=True)
class SubmitJobDraft:
    dataset_source: DatasetSource
    run_mode: RunMode
    schema_name: str
    model_name: str
    output_format: OutputFormat = "jsonl"
    local_path: str = ""
    cloud_prefix: str = ""
    cloud_prefixes: tuple[str, ...] = ()
    continue_dataset: str = ""
    num_batches: int | None = None


@dataclass(frozen=True)
class CommandSpec:
    module: str
    args: tuple[str, ...] = ()
    config_overrides: dict[str, object] = field(default_factory=dict)

    def argv(self) -> list[str]:
        return [sys.executable, "-m", self.module, *self.args]

    def display(self) -> str:
        parts = ["python", "-m", self.module, *self.args]
        return " ".join(parts)


@dataclass(frozen=True)
class JobSummary:
    job_id: str
    source: str
    kind: str
    status: str
    created_at: str = ""
    model: str = ""
    run_dir: str = ""
    command: str = ""
    detail: str = ""
    input_location: str = ""
    image_count: int = 0
    chunk_count: int = 0
    retrieved: bool = False
    succeeded: int | None = None
    failed: int | None = None


@dataclass(frozen=True)
class BatchChunkSummary:
    chunk_index: int
    total_chunks: int
    chunk_label: str
    batch_job_name: str
    request_count: int
    status: str = "unknown"
    output_destination: str = ""
    requests_file: str = ""
    provider: str = ""


def app_settings_path(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root).expanduser() / "app_config.json"
    return Path.home() / ".patientjournals" / "app_config.json"
