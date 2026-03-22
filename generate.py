import asyncio
from google import genai
import time
import random
from typing import Any
from pydantic import BaseModel

from generation_spec import (
    build_live_generation_config,
    build_live_request_contents,
)
from preprocess import preprocess_image
from config import config
from response_parsing import extract_response_metadata
from tools import data_to_rows

async def generate_data(
    client: genai.Client,
    model: str,
    file_name: str,
) -> tuple[BaseModel, float, dict[str, Any]]:
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
        contents=build_live_request_contents(image_bytes=image_bytes, mime_type=mime_type),
        config=build_live_generation_config(
            include_schema=True,
            include_temperature=True,
            include_thinking_level=True,
        ),
    )
    end_time = time.perf_counter()
    duration = end_time - start_time
    metadata = extract_response_metadata(output)
    payload_text = metadata.get("text")
    if not isinstance(payload_text, str) or not payload_text.strip():
        raise ValueError("Empty response text from API.")
    return config.output_model.model_validate_json(payload_text), duration, metadata

async def process_file(sem, client, model, file_name, log):
    async with sem:
        max_attempts = max(1, int(config.api_max_attempts))
        for attempt in range(1, max_attempts + 1):
            try:
                journal_data, duration, metadata = await generate_data(
                    client=client,
                    model=model,
                    file_name=file_name,
                )
                rows = data_to_rows(data=journal_data, file_name=file_name)
                for row in rows:
                    row["generation_seconds"] = duration
                    row["thoughts"] = metadata.get("thoughts") or None
                    row["response_confidence_logprobs"] = metadata.get("response_confidence_logprobs")
                    row["response_confidence_ratio"] = metadata.get("response_confidence_ratio")
                return rows
            except Exception as e:
                retryable = _is_retryable_api_error(e)
                if retryable and attempt < max_attempts:
                    delay = _retry_delay_seconds(attempt)
                    log(
                        f"Transient API error for {file_name} "
                        f"(attempt {attempt}/{max_attempts}). "
                        f"Retrying in {delay:.1f}s. Error: {e}"
                    )
                    await asyncio.sleep(delay)
                    continue

                log(f"Error processing {file_name}", exc=e)
                if _is_fatal_api_error(e) or retryable:
                    raise
                return None


def _is_fatal_api_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    fatal_markers = (
        "token limit",
        "quota",
        "permission",
        "unauthorized",
        "forbidden",
        "invalid api key",
        "invalid_api_key",
        "billing",
    )
    return any(marker in text for marker in fatal_markers)


def _is_retryable_api_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    retryable_markers = (
        "503",
        "unavailable",
        "resource_exhausted",
        "rate limit",
        "429",
        "internal",
        "deadline",
        "timed out",
        "timeout",
        "connection reset",
        "connection error",
        "temporarily",
        "backend error",
    )
    return any(marker in text for marker in retryable_markers)


def _retry_delay_seconds(attempt: int) -> float:
    initial = max(0.0, float(config.api_retry_initial_delay_seconds))
    maximum = max(initial, float(config.api_retry_max_delay_seconds))
    jitter = max(0.0, float(config.api_retry_jitter_seconds))

    delay = min(initial * (2 ** max(0, attempt - 1)), maximum)
    if jitter > 0:
        delay += random.uniform(0, jitter)
    return delay
