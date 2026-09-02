import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel

from patientjournals.config.prompts import PAGE_PROMPTS
from patientjournals.config.schemas import FrontPage, model_from_json_schema, output_schema_name
from patientjournals.shared.local_secrets import load_local_api_keys


_PROVIDER_NAMES: tuple[str, ...] = ("gemini", "openai", "anthropic")


def _load_provider_api_keys() -> dict[str, str]:
    keys = {name: "" for name in _PROVIDER_NAMES}

    try:
        import api_keys as key_module
    except Exception:
        key_module = None

    aliases: dict[str, tuple[str, ...]] = {
        "gemini": ("gemini_maarten", "gemini", "google", "google_gemini"),
        "openai": ("openai", "openai_api_key", "gpt"),
        "anthropic": ("anthropic", "anthropic_api_key", "claude"),
    }
    if key_module is not None:
        for provider, names in aliases.items():
            for name in names:
                value = getattr(key_module, name, "")
                if isinstance(value, str) and value.strip():
                    keys[provider] = value.strip()
                    break

    for provider, value in load_local_api_keys().items():
        if provider in keys and value:
            keys[provider] = value

    env_aliases: dict[str, tuple[str, ...]] = {
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "openai": ("OPENAI_API_KEY",),
        "anthropic": ("ANTHROPIC_API_KEY",),
    }
    for provider, names in env_aliases.items():
        for name in names:
            try:
                value = os.getenv(name, "")
            except Exception:
                value = ""
            if isinstance(value, str) and value.strip():
                keys[provider] = value.strip()
                break
    return keys


def _default_api_key() -> str:
    return _load_provider_api_keys().get("gemini", "")


@dataclass
class Config:
    model: str = "gemini-3.1-pro"
    input_prompt_name: str = "frontpage"  # change to correct prompt
    output_model: type[BaseModel] = FrontPage  # change to correct schema in schemas.py
    output_schema_name: str = "FrontPage"
    output_schema_version_id: str = ""
    output_schema_override: dict[str, Any] | None = None
    target_folder: str = "/Volumes/Expansion/patientjournaler_1889-1897_jpg"
    fp_mode: Literal["all", "only_fp", "exclude_fp"] = "all"
    output_format: str = "jsonl"
    csv_sep: str = "$"

    model_temperature: float = 0.0
    model_max_output_tokens: int = 4096
    thinking_level: Optional[Literal["low", "medium", "high"]] = "high"
    include_thoughts: bool = False
    include_confidence_scores: bool = False
    include_response_avg_logprobs: bool = True
    # Fan one page out into one model request per top-level schema field, then
    # validate and join the specialist outputs before dataset conversion.
    subagents: bool = False
    ocr_enabled: bool = True
    ocr_required: bool = False
    ocr_backend: Literal["google_vision"] = "google_vision"
    ocr_language_hints: tuple[str, ...] = ("da",)
    ocr_sidecar_suffix: str = ".ocr.json"
    provider_api_keys: dict[str, str] = field(default_factory=_load_provider_api_keys)
    api_key: str = field(default_factory=_default_api_key)
    api_concurrent_tasks: int = 8
    api_max_attempts: int = 10
    api_retry_initial_delay_seconds: float = 2.0
    api_retry_max_delay_seconds: float = 30.0
    api_retry_jitter_seconds: float = 0.5
    # Candidate-aware second-pass model validation. The first extraction and
    # deterministic checks remain unchanged unless this is explicitly enabled.
    model_validation_enabled: bool = False
    verification_model: str = "gemini-3.1-pro-preview"
    verification_thinking_level: Literal["low", "medium", "high"] = "high"
    verification_max_output_tokens: int = 4096
    # ``flagged`` is the batch-efficient production path: deterministic routing
    # escalates risky pages plus a stable control sample. ``all`` remains
    # selectable for full-population experiments.
    verification_scope: Literal["all", "flagged"] = "flagged"
    verification_control_sample_percent: float = 2.0
    # Final-stage verifier corrections are accepted automatically after the
    # corrected page validates against the original extraction schema.
    verification_apply_mode: Literal["report_only", "apply_patches"] = "apply_patches"
    verification_num_chunks: int = 1
    batch_size: int = 2048
    flush_every: int = 1
    dataset_file_name: str = "dataset"
    input_glob: str = "*.png"
    recursive: bool = True
    fp_suffix: str = "_fp"
    output_root: str = "runs"

    # Batch + cloud backend settings
    batch_backend: Literal["vertex", "mldev"] = "vertex"
    batch_job_display_name: str = "patientjournals-batch"
    batch_job_name: str = ""
    batch_poll_interval_seconds: int = 20
    batch_requests_file_name: str = "batch_requests.jsonl"
    batch_num_chunks: int = 1
    batch_input_source: Literal["gcs"] = "gcs"
    batch_ocr_metadata_required: bool = True
    batch_ocr_workers: int = 8
    batch_ocr_api_batch_size: int = 16
    batch_ocr_api_batch_max_bytes: int = 8_000_000
    batch_ocr_manifest_object: str = "batch/ocr/metadata_manifest.json"
    batch_input_prefix: str = ""
    batch_input_prefixes: tuple[str, ...] = ()
    # When non-empty, restrict batch input selection to exactly these image
    # names (basenames). Used to scope a submission to a specific local folder
    # so it cannot accidentally fan out to the entire bucket prefix.
    batch_restrict_image_names: tuple[str, ...] = ()
    # Deterministic experimental subset of the selected input population.
    # None submits the complete population; a percentage requires a seed.
    batch_submission_type: Literal["complete", "sample"] = "complete"
    batch_sample_percent: float | None = None
    batch_sample_seed: str = "42"
    batch_input_extensions: tuple[str, ...] = ("png", "jpg", "jpeg", "webp", "tiff")
    batch_date_mapping_file: str = "date_mapping.csv"
    batch_year_filter: tuple[int | str, ...] = ()
    batch_input_max_bytes: int = 0
    batch_include_response_schema: bool = True
    batch_use_local_pdf_folders: bool = True
    batch_auto_upload_missing: bool = True
    anthropic_signed_url_ttl_hours: int = 48

    response_mime_type: str = "application/json"
    response_schema_field: Literal["response_json_schema", "response_schema"] = (
        "response_json_schema"
    )

    # Validation/recovery controls for batch retrieval
    require_all_expected_pages: bool = True
    require_all_pages_successful: bool = False
    page_validation_sample_size: int = 5
    require_headers_for_all_rows: bool = False
    header_validation_sample_size: int = 5
    api_recovery_enabled: bool = True
    api_recovery_max_missing_pages: int = 50
    api_recovery_model: str = "gemini-3.1-pro-preview"
    batch_submit_failed_pages: bool = False
    batch_duplicate_strategy: Literal["first_successful", "provide_all"] = (
        "first_successful"
    )

    # GCP/GCS settings
    gcp_auth_mode: Literal["service_account", "adc"] = "service_account"
    service_account_file: str = "service-account.json"
    gcp_project_id: str = "gen-lang-client-0854332640"
    gcp_location: str = "europe-north1"
    vertex_model_location: str = "global"
    gcs_bucket_name: str = "data-blegdamsjournaler"
    gcs_pages_prefix: str = "pages"
    batch_requests_gcs_prefix: str = "batch/requests"
    batch_outputs_gcs_prefix: str = "batch/outputs"
    datasets_gcs_prefix: str = "datasets"
    upload_dataset_to_gcs: bool = True
    validations_gcs_prefix: str = "validations"
    upload_validation_to_gcs: bool = True
    schemas_gcs_prefix: str = "schemas"

    # Upload/render settings for PDF to GCS image pages
    upload_source: Literal["pdf", "images", "auto"] = "images"
    upload_images_folder: str = "/Volumes/Expansion/patientjournaler_1889-1897_jpg"
    upload_images_recursive: bool = True
    upload_images_glob: str = "*.png"
    upload_auto_tune: bool = True
    upload_profile: Literal["light", "normal", "aggressive"] = "normal"
    upload_max_workers: int = 0
    upload_timeout_seconds: float = 300.0
    upload_retry_attempts: int = 8
    upload_retry_initial_delay_seconds: float = 1.5
    upload_retry_max_delay_seconds: float = 30.0
    batch_upload_limit: int = 100
    upload_workers: int = 35
    pdf_render_dpi: int = 300
    page_number_digits: int = 4
    image_settings: dict[str, Any] = field(
        default_factory=lambda: {
            "max_dim": 3000,
            "contrast_factor": 1.1,
            "margins": (
                150,  # left
                0,  # top
                0,  # right
                0,  # bottom
            ),
            "output_format": "PNG",
        }
    )

    prompts: dict[str, str] = field(default_factory=lambda: dict(PAGE_PROMPTS))

    # backend for config
    output_schema: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.provider_api_keys = {
            str(provider).strip().lower(): str(value).strip()
            for provider, value in (self.provider_api_keys or {}).items()
            if str(provider).strip()
        }
        raw_prefixes = self.batch_input_prefixes or ()
        if isinstance(raw_prefixes, str):
            raw_prefixes = (raw_prefixes,)
        self.batch_input_prefixes = tuple(
            str(prefix).strip()
            for prefix in raw_prefixes
            if str(prefix).strip()
        )
        raw_restrict = self.batch_restrict_image_names or ()
        if isinstance(raw_restrict, str):
            raw_restrict = (raw_restrict,)
        self.batch_restrict_image_names = tuple(
            str(name).strip()
            for name in raw_restrict
            if str(name).strip()
        )
        if self.batch_submission_type not in {"complete", "sample"}:
            raise ValueError("batch_submission_type must be complete or sample.")
        if self.batch_submission_type == "sample":
            if self.batch_sample_percent is None:
                raise ValueError(
                    "batch_sample_percent is required when sampling is enabled."
                )
            self.batch_sample_percent = float(self.batch_sample_percent)
            if not 0.0 < self.batch_sample_percent <= 100.0:
                raise ValueError(
                    "batch_sample_percent must be greater than 0 and at most 100."
                )
            self.batch_sample_seed = str(self.batch_sample_seed or "").strip()
            if not self.batch_sample_seed:
                raise ValueError(
                    "batch_sample_seed must not be empty when sampling is enabled."
                )
        raw_ocr_hints = self.ocr_language_hints or ()
        if isinstance(raw_ocr_hints, str):
            raw_ocr_hints = (raw_ocr_hints,)
        self.ocr_language_hints = tuple(
            str(hint).strip()
            for hint in raw_ocr_hints
            if str(hint).strip()
        )
        self.verification_control_sample_percent = max(
            0.0, min(100.0, float(self.verification_control_sample_percent or 0.0))
        )
        if not (self.api_key or "").strip():
            self.api_key = self.provider_api_keys.get("gemini", "")
        if self.output_schema_override:
            self.output_schema = deepcopy(self.output_schema_override)
            self.output_model = model_from_json_schema(
                self.output_schema_name or "ManagedSchema",
                self.output_schema,
            )
        else:
            self.output_schema_name = output_schema_name(self.output_model)
            self.output_schema = self.output_model.model_json_schema()
        _ = self.input_prompt

    def api_key_for_provider(self, provider: str) -> str:
        provider_name = str(provider or "").strip().lower()
        if not provider_name:
            raise ValueError("Provider name is empty while resolving API key.")

        api_key = (self.provider_api_keys.get(provider_name) or "").strip()
        if api_key:
            return api_key

        if provider_name == "gemini" and (self.api_key or "").strip():
            return self.api_key.strip()

        raise ValueError(
            f"No API key configured for provider '{provider_name}'. "
            "Set config.provider_api_keys[...] or update api_keys.py."
        )

    @property
    def input_prompt(self) -> str:
        prompt = self.prompts.get(self.input_prompt_name)
        if prompt is None:
            available = ", ".join(sorted(self.prompts))
            raise ValueError(
                f"Unknown input_prompt_name '{self.input_prompt_name}'. "
                f"Available prompts: {available}"
            )
        return prompt

config = Config()


def _apply_external_json_config(cfg: Config) -> None:
    config_path = os.getenv("PATIENTJOURNALS_CONFIG_JSON", "").strip()
    if not config_path:
        return

    path = Path(config_path).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"PATIENTJOURNALS_CONFIG_JSON not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid PATIENTJOURNALS_CONFIG_JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid PATIENTJOURNALS_CONFIG_JSON payload: {path}")

    direct_fields = {
        "batch_backend",
        "gcp_auth_mode",
        "service_account_file",
        "gcp_project_id",
        "gcp_location",
        "vertex_model_location",
        "gcs_bucket_name",
        "gcs_pages_prefix",
        "batch_requests_gcs_prefix",
        "batch_outputs_gcs_prefix",
        "datasets_gcs_prefix",
        "upload_dataset_to_gcs",
        "validations_gcs_prefix",
        "upload_validation_to_gcs",
        "schemas_gcs_prefix",
        "batch_input_prefix",
        "batch_input_prefixes",
        "batch_ocr_metadata_required",
        "batch_ocr_workers",
        "batch_ocr_api_batch_size",
        "batch_ocr_api_batch_max_bytes",
        "batch_ocr_manifest_object",
        "target_folder",
        "upload_images_folder",
        "model",
        "thinking_level",
        "output_format",
        "batch_duplicate_strategy",
        "batch_submission_type",
        "batch_sample_percent",
        "batch_sample_seed",
        "output_schema_name",
        "output_schema_version_id",
        "output_schema_override",
        "subagents",
        "ocr_enabled",
        "ocr_required",
        "ocr_backend",
        "ocr_language_hints",
        "ocr_sidecar_suffix",
        "model_validation_enabled",
        "verification_model",
        "verification_thinking_level",
        "verification_max_output_tokens",
        "verification_scope",
        "verification_control_sample_percent",
        "verification_apply_mode",
        "verification_num_chunks",
    }
    aliases = {
        "auth_mode": "gcp_auth_mode",
        "local_runs_root": "output_root",
    }
    for key in direct_fields:
        if key in payload and payload[key] is not None:
            setattr(cfg, key, payload[key])
    for source_key, target_key in aliases.items():
        if source_key in payload and payload[source_key] is not None:
            setattr(cfg, target_key, payload[source_key])

    schema_payload = payload.get("schema_payload") or payload.get("output_schema_override")
    if isinstance(schema_payload, dict) and schema_payload:
        cfg.output_schema_override = schema_payload
        cfg.output_schema_name = str(
            payload.get("schema_name") or payload.get("output_schema_name") or "ManagedSchema"
        )
        cfg.output_schema_version_id = str(
            payload.get("schema_version_id")
            or payload.get("output_schema_version_id")
            or ""
        )
    schema_name = payload.get("schema_name")
    if (
        not cfg.output_schema_override
        and isinstance(schema_name, str)
        and schema_name.strip()
    ):
        from patientjournals.config.schemas import resolve_output_schema

        cfg.output_model = resolve_output_schema(schema_name)
        cfg.output_schema_name = schema_name.strip()

    api_key_env = payload.get("gemini_api_key_env")
    if isinstance(api_key_env, str) and api_key_env.strip():
        api_key = os.getenv(api_key_env.strip(), "").strip()
        if api_key:
            cfg.provider_api_keys["gemini"] = api_key
            cfg.api_key = api_key

    cfg.__post_init__()


_apply_external_json_config(config)
