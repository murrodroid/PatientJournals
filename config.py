from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from pydantic import BaseModel
from schemas import *


_PROVIDER_NAMES: tuple[str, ...] = ("gemini", "openai", "anthropic")


def _load_provider_api_keys() -> dict[str, str]:
    keys = {name: "" for name in _PROVIDER_NAMES}
    env_aliases: dict[str, tuple[str, ...]] = {
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "openai": ("OPENAI_API_KEY",),
        "anthropic": ("ANTHROPIC_API_KEY",),
    }
    for provider, names in env_aliases.items():
        for name in names:
            try:
                import os

                value = os.getenv(name, "")
            except Exception:
                value = ""
            if isinstance(value, str) and value.strip():
                keys[provider] = value.strip()
                break

    try:
        import api_keys as key_module
    except Exception:
        return keys

    aliases: dict[str, tuple[str, ...]] = {
        "gemini": ("gemini_maarten", "gemini", "google", "google_gemini"),
        "openai": ("openai", "openai_api_key", "gpt"),
        "anthropic": ("anthropic", "anthropic_api_key", "claude"),
    }
    for provider, names in aliases.items():
        for name in names:
            value = getattr(key_module, name, "")
            if isinstance(value, str) and value.strip():
                keys[provider] = value.strip()
                break
    return keys


def _default_api_key() -> str:
    return _load_provider_api_keys().get("gemini", "")


@dataclass
class Config:
    model: str = "claude-opus-4-7"  # gemini-3.1-pro-preview
    input_prompt_name: str = "frontpage"  # change to correct prompt
    output_model: type[BaseModel] = FrontPage  # change to correct schema in schemas.py
    target_folder: str = "/Volumes/Expansion/patientjournaler_1889-1897_jpg"
    fp_mode: Literal["all", "only_fp", "exclude_fp"] = "all"
    output_format: str = "csv"
    csv_sep: str = "$"

    model_temperature: float = 0.0
    model_max_output_tokens: int = 4096
    thinking_level: Optional[Literal["low", "medium", "high"]] = "high"
    include_thoughts: bool = False
    include_confidence_scores: bool = False
    provider_api_keys: dict[str, str] = field(default_factory=_load_provider_api_keys)
    api_key: str = field(default_factory=_default_api_key)
    api_concurrent_tasks: int = 8
    api_max_attempts: int = 10
    api_retry_initial_delay_seconds: float = 2.0
    api_retry_max_delay_seconds: float = 30.0
    api_retry_jitter_seconds: float = 0.5
    verification_model: str = ""
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
    batch_input_prefix: str = ""
    batch_input_extensions: tuple[str, ...] = ("png", "jpg", "jpeg", "webp", "tiff")
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
    api_recovery_model: str = ""

    # GCP/GCS settings
    service_account_file: str = "service-account.json"
    gcp_project_id: str = "gen-lang-client-0854332640"
    gcp_location: str = "europe-north1"
    vertex_model_location: str = "global"
    gcs_bucket_name: str = "data-blegdamsjournaler"
    gcs_pages_prefix: str = "pages"
    batch_requests_gcs_prefix: str = "batch/requests"
    batch_outputs_gcs_prefix: str = "batch/outputs"
    datasets_gcs_prefix: str = "datasets"
    upload_dataset_to_gcs: bool = False

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

    prompts: dict[str, str] = field(
        default_factory=lambda: {
            "frontpage": f"""
                Context:
                You are given a scanned page from a Danish hospital patient journal from the late 1800s.
                Your task is to extract data from the content on the page.

                Objective:
                Fill each column with the information found in the image.
                Not all columns are present within an image, meaning it isn't necessary to fill out all.

                Guidelines:
                - Examples are always written as 'Examples: [example1,example2,example3]'
                - Use only what is visible in the image.
                - Do not infer or guess beyond the evidence on the page.
                - Preserve spellings exactly as written, even if archaic or non-standard. Only exception is numbers, which should be written as float-values.
                - If nothing fits a Field, output an empty field for that position.
                - If a line is crossed out, it should not be included in the datapoint of which it's relevant to.
                """,
            "textpage": """
                **Role:**
                You are an expert archivist specializing in late 19th-century Danish medical manuscripts. Your task is to transcribe the provided handwritten journal page into a structured JSON format, maintaining strict fidelity to the original text.
                
                **Scope & Focus:**
                *   **Primary Page Only:** Transcribe **ONLY** the single page that is centered and in focus.
                *   **Ignore Surroundings:** Strictly ignore any text visible on the facing page (across the binding/gutter) or any text cut off at the far edges of the image.
                *   **Visual Boundaries:** The page usually has a vertical fold or red line separating the left-hand date margin from the main body. Do not transcribe text found outside the physical boundaries of the current page.
                
                **Transcription Rules:**
                1.  **Line-by-Line:** Output a JSON object for every distinct vertical line of writing. Do not merge lines.
                2.  **Margins:** If a date (e.g., "18/12") appears in the left margin, capture it in the `metadata` field. If the margin is blank for that line, leave it as a `None`-value.
                3.  **Vital Signs Columns:** The text frequently breaks into columns of numbers (Time | Temp | Pulse). Transcribe these exactly as they appear visually within the `text` field, preserving spaces between numbers (e.g., `12   39,6   39`).
                4.  **Language & Spelling:**
                    *   Preserve archaic Danish spelling exactly (e.g., write "The" not "Te", "Smerter", "aa" instead of "å").
                    *   Keep all medical abbreviations (e.g., "Rp.", "Tp.", "P.", "Steth.", "dgl.").
                """,
        }
    )

    # backend for config
    output_schema: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.provider_api_keys = {
            str(provider).strip().lower(): str(value).strip()
            for provider, value in (self.provider_api_keys or {}).items()
            if str(provider).strip()
        }
        if not (self.api_key or "").strip():
            self.api_key = self.provider_api_keys.get("gemini", "")
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
