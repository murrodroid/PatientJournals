from dataclasses import dataclass, field
from typing import Any, Literal
from pydantic import BaseModel
from schemas import Journal, TextPage


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
    input_series: str | None = "8dec96"
    fp_mode: Literal["all", "only_fp", "exclude_fp"] = "exclude_fp"
    fp_suffix: str = "_fp"
    output_format: str = "jsonl"
    output_root: str = "runs"
    batch_upload_limit: int = 20
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
            "primary": f"""
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
        }
    )

    def __post_init__(self) -> None:
        self.output_schema = self.output_model.model_json_schema()


config = Config()
