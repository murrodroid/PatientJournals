from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Iterator

from pydantic import BaseModel

from patientjournals.config import config
from patientjournals.shared.response_parsing import extract_response_metadata


@dataclass
class GeminiOutputParseResult:
    key: str | None
    parsed_model: BaseModel | None
    metadata: dict[str, object]
    reason: str | None
    detail: str | None = None
    source: str | None = None
    line_number: int | None = None

    @property
    def is_valid(self) -> bool:
        return (
            self.reason is None
            and self.key is not None
            and self.parsed_model is not None
        )


def normalize_output_key(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def response_has_value(response: object) -> bool:
    if response is None:
        return False
    if isinstance(response, str):
        return bool(response.strip())
    if isinstance(response, (dict, list, tuple, set)):
        return bool(response)
    return True


def parse_gemini_output_record(
    record: object,
    *,
    source: str | None = None,
    line_number: int | None = None,
) -> GeminiOutputParseResult:
    if not isinstance(record, dict):
        return GeminiOutputParseResult(
            key=None,
            parsed_model=None,
            metadata={},
            reason="invalid_record_type",
            source=source,
            line_number=line_number,
        )

    key = normalize_output_key(record.get("key"))
    if not key:
        return GeminiOutputParseResult(
            key=None,
            parsed_model=None,
            metadata={},
            reason="missing_key",
            source=source,
            line_number=line_number,
        )

    if record.get("error"):
        return GeminiOutputParseResult(
            key=key,
            parsed_model=None,
            metadata={},
            reason="batch_error",
            detail=str(record.get("error")),
            source=source,
            line_number=line_number,
        )

    response = record.get("response")
    if not response_has_value(response):
        return GeminiOutputParseResult(
            key=key,
            parsed_model=None,
            metadata={},
            reason="missing_response",
            source=source,
            line_number=line_number,
        )

    metadata = extract_response_metadata(response)
    text_payload = metadata.get("text")
    if not isinstance(text_payload, str) or not text_payload.strip():
        return GeminiOutputParseResult(
            key=key,
            parsed_model=None,
            metadata=metadata,
            reason="empty_response_text",
            source=source,
            line_number=line_number,
        )

    try:
        parsed_model = config.output_model.model_validate_json(text_payload)
    except Exception as exc:
        return GeminiOutputParseResult(
            key=key,
            parsed_model=None,
            metadata=metadata,
            reason="schema_validation_failed",
            detail=f"{type(exc).__name__}: {exc}",
            source=source,
            line_number=line_number,
        )

    return GeminiOutputParseResult(
        key=key,
        parsed_model=parsed_model,
        metadata=metadata,
        reason=None,
        source=source,
        line_number=line_number,
    )


def iter_gemini_jsonl_results(
    lines: Iterable[str],
    *,
    source: str,
) -> Iterator[GeminiOutputParseResult]:
    for line_number, line in enumerate(lines, start=1):
        raw = line.strip()
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            yield GeminiOutputParseResult(
                key=None,
                parsed_model=None,
                metadata={},
                reason="invalid_jsonl_line",
                detail=f"{type(exc).__name__}: {exc}",
                source=source,
                line_number=line_number,
            )
            continue
        yield parse_gemini_output_record(
            record,
            source=source,
            line_number=line_number,
        )


def add_response_metadata_columns(
    rows: list[dict],
    metadata: dict[str, object],
) -> None:
    for row in rows:
        row["thoughts"] = metadata.get("thoughts") or None
        if bool(getattr(config, "include_response_avg_logprobs", False)):
            row["avg_logprobs"] = metadata.get("avg_logprobs")
