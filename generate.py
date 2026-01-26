import asyncio
from google import genai
from google.genai import types
import time

from preprocess import preprocess_image
from config import config
from tools import data_to_row
from schemas import Journal

async def generate_data(client: genai.Client, model: str, file_name: str) -> tuple[Journal,float]:
    image_bytes, mime_type = await asyncio.to_thread(
        preprocess_image,
        file_name,
        max_dim=config.image_settings.get("max_dim", 3000),
        margins=tuple(config.image_settings.get("margins", (0, 0, 0, 0))),
        contrast_factor=config.image_settings.get("contrast_factor", 1.0),
        output_format=config.image_settings.get("output_format", "PNG"),
    )

    start_time = time.perf_counter()
    output = await client.aio.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            config.prompts.get("primary"),
        ],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": config.output_schema,
        },
    )
    end_time = time.perf_counter()
    duration = end_time - start_time

    return Journal.model_validate_json(output.text), duration

async def process_file(sem, client, model, file_name, log):
    async with sem:
        try:
            journal_data,duration = await generate_data(client=client, model=model, file_name=file_name)
            row = data_to_row(data=journal_data, file_name=file_name)
            row['generation_seconds'] = duration
            return row
        except Exception as e:
            log(f"Error processing {file_name}", exc=e)
            if _is_fatal_api_error(e):
                raise
            return None


def _is_fatal_api_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    fatal_markers = (
        "token limit",
        "quota",
        "resource_exhausted",
        "rate limit",
        "permission",
        "unauthorized",
        "forbidden",
        "invalid api key",
        "invalid_api_key",
        "billing",
    )
    return any(marker in text for marker in fatal_markers)
