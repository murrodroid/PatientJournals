from dataclasses import dataclass, field
from typing import Any
from schemas import Journal


@dataclass
class Config:
    model: str = "gemini-3-pro-preview"
    concurrent_tasks: int = 6
    verification_model: str = ""
    batch_size: int = 2048
    flush_every: int = 1
    dataset_file_name: str = "dataset"
    target_folder: str = "data"
    input_glob: str = "*.png"
    recursive: bool = True
    output_format: str = "jsonl"
    output_root: str = "runs"
    batch_upload_limit: int = 20
    output_schema: dict[str, Any] = field(
        default_factory=lambda: Journal.model_json_schema()
    )
    image_settings: dict[str, Any] = field(
        default_factory=lambda: {
            "max_dim": 3000,
            "contrast_factor": 1.1,
            "margins": (
                300,  # left
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


config = Config()
