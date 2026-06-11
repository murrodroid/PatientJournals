import asyncio
import time
from dataclasses import dataclass
from typing import Any
from pydantic import BaseModel

from patientjournals.shared.preprocess import preprocess_image_with_metadata
from patientjournals.config import config
from patientjournals.local.model_client import LocalModelClient
from patientjournals.shared.api_retry import (
    is_fatal_api_error as _is_fatal_api_error,
    is_retryable_api_error as _is_retryable_api_error,
    retry_delay_seconds as _retry_delay_seconds,
)
from patientjournals.shared.processing_metrics import base_image_record, utc_now_iso
from patientjournals.shared.tools import data_to_rows


@dataclass(frozen=True)
class ProcessedFileResult:
    rows: list[dict]
    metrics: dict[str, Any]


async def generate_data(
    model_client: LocalModelClient,
    file_name: str,
) -> tuple[BaseModel, float, dict[str, Any], dict[str, Any]]:
    image_bytes, mime_type, preprocessing = await asyncio.to_thread(
        preprocess_image_with_metadata,
        file_name,
        max_dim=config.image_settings.get("max_dim", 3000),
        margins=tuple(config.image_settings.get("margins", (0, 0, 0, 0))),
        contrast_factor=config.image_settings.get("contrast_factor", 1.0),
        output_format=config.image_settings.get("output_format", "PNG"),
    )

    start_time = time.perf_counter()
    output = await model_client.generate_json(
        image_bytes=image_bytes,
        mime_type=mime_type,
    )
    end_time = time.perf_counter()
    duration = end_time - start_time
    metadata = {
        "text": output.text,
        "thoughts": output.thoughts,
        "field_confidence_by_pointer": output.field_confidence_by_pointer or {},
    }
    payload_text = metadata.get("text")
    if not isinstance(payload_text, str) or not payload_text.strip():
        raise ValueError("Empty response text from API.")
    return (
        config.output_model.model_validate_json(payload_text),
        duration,
        metadata,
        preprocessing,
    )


async def process_file(sem, model_client, file_name, log):
    async with sem:
        max_attempts = max(1, int(config.api_max_attempts))
        started_at = utc_now_iso()
        total_started = time.perf_counter()
        for attempt in range(1, max_attempts + 1):
            try:
                journal_data, duration, metadata, preprocessing = await generate_data(
                    model_client=model_client,
                    file_name=file_name,
                )
                rows = data_to_rows(
                    data=journal_data,
                    file_name=file_name,
                    field_confidence_by_pointer=metadata.get("field_confidence_by_pointer"),
                )
                for row in rows:
                    row["generation_seconds"] = duration
                    row["thoughts"] = metadata.get("thoughts") or None
                completed_at = utc_now_iso()
                metrics = base_image_record(
                    image_reference=file_name,
                    source="local_api",
                    status="success",
                    model=model_client.model_name,
                    provider=model_client.provider,
                    attempts=attempt,
                    max_attempts=max_attempts,
                    started_at=started_at,
                    completed_at=completed_at,
                    generation_seconds=duration,
                    total_seconds=time.perf_counter() - total_started,
                    rows_written=len(rows),
                    preprocessing=preprocessing,
                )
                return ProcessedFileResult(rows=rows, metrics=metrics)
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
                metrics = base_image_record(
                    image_reference=file_name,
                    source="local_api",
                    status="failed",
                    model=model_client.model_name,
                    provider=model_client.provider,
                    attempts=attempt,
                    max_attempts=max_attempts,
                    started_at=started_at,
                    completed_at=utc_now_iso(),
                    total_seconds=time.perf_counter() - total_started,
                    rows_written=0,
                    failure_reason="fatal_api_error" if _is_fatal_api_error(e) else "processing_error",
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
                if _is_fatal_api_error(e) or retryable:
                    raise
                return ProcessedFileResult(rows=[], metrics=metrics)
