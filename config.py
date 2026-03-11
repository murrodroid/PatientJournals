from dataclasses import dataclass, field
from typing import Any, Literal
from pydantic import BaseModel
from schemas import *


@dataclass
class Config:
    model: str = "gemini-3-pro-preview"
    model_temperature: float = 0.0
    api_concurrent_tasks: int = 8
    api_max_attempts: int = 6
    api_retry_initial_delay_seconds: float = 2.0
    api_retry_max_delay_seconds: float = 30.0
    api_retry_jitter_seconds: float = 0.5
    verification_model: str = ""
    batch_size: int = 2048
    flush_every: int = 1
    dataset_file_name: str = "dataset"
    target_folder: str = "data"
    input_glob: str = "*.png"
    recursive: bool = True
    fp_mode: Literal["all", "only_fp", "exclude_fp"] = "exclude_fp"
    fp_suffix: str = "_fp"
    output_format: str = "jsonl"
    output_root: str = "runs"
    batch_upload_limit: int = 20
    input_prompt_name: str = "textpage"
    output_model: type[BaseModel] = TextPage
    output_schema: dict[str, Any] = field(init=False)
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
