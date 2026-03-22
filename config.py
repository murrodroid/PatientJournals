from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from pydantic import BaseModel
from schemas import *
from api_keys import gemini_maarten as api_key


@dataclass
class Config:
    model: str = "gemini-3.1-pro-preview"
    input_prompt_name: str = "frontpage"          # change to correct prompt
    output_model: type[BaseModel] = FrontPage     # change to correct schema in schemas.py
    target_folder: str = "data"
    fp_mode: Literal["all", "only_fp", "exclude_fp"] = "only_fp"
    output_format: str = "jsonl"
    csv_sep: str = "$"
    
    model_temperature: float = 0.0
    thinking_level: Optional[Literal["low", "medium", "high"]] = "high"
    include_thoughts: bool = True
    include_confidence_scores: bool = False
    api_key: str = api_key
    api_concurrent_tasks: int = 8
    api_max_attempts: int = 6
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
    batch_input_source: Literal["gcs"] = "gcs"
    batch_input_prefix: str = ""
    batch_input_extensions: tuple[str, ...] = ("png", "jpg", "jpeg", "webp", "tiff")
    batch_input_max_bytes: int = 0
    batch_include_response_schema: bool = True
    batch_use_local_pdf_folders: bool = True
    batch_auto_upload_missing: bool = True

    response_mime_type: str = "application/json"
    response_schema_field: Literal["response_json_schema", "response_schema"] = "response_json_schema"

    # Validation/recovery controls for batch retrieval
    require_all_expected_pages: bool = True
    require_all_pages_successful: bool = True
    page_validation_sample_size: int = 5
    require_headers_for_all_rows: bool = False
    header_validation_sample_size: int = 5
    api_recovery_enabled: bool = True
    api_recovery_max_missing_pages: int = 5
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
    batch_upload_limit: int = 20
    upload_workers: int = 4
    pdf_render_dpi: int = 300
    page_number_digits: int = 4
    image_settings: dict[str, Any] = field(
        default_factory=lambda: {
            "max_dim": 3000,
            "contrast_factor": 1.1,
            "margins": (
                0,    # left
                0,    # top
                0,    # right
                0,    # bottom
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
                """
        }
    )
    
    # backend for config
    output_schema: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.output_schema = self.output_model.model_json_schema()
        _ = self.input_prompt

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
