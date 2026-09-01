from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from patientjournals.config import config
from patientjournals.batch.output_records import gemini_finish_reason
from patientjournals.shared.response_parsing import extract_response_metadata
from patientjournals.shared.subagents import (
    decode_specialist_request_key,
    merge_specialist_metadata,
    merge_specialist_payloads,
    schema_specialists,
    validate_specialist_json,
)


@dataclass(frozen=True)
class CombinedSubagentOutputs:
    records: tuple[dict[str, object], ...]
    failures: tuple[dict[str, object], ...]
    stats: dict[str, int]


def _anthropic_metadata(response: object) -> dict[str, object]:
    if not isinstance(response, dict):
        return {"text": None, "thoughts": None, "field_confidence_by_pointer": {}}
    content = response.get("content")
    if not isinstance(content, list):
        return {"text": None, "thoughts": None, "field_confidence_by_pointer": {}}
    text_chunks: list[str] = []
    thought_chunks: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "").strip().lower()
        value = block.get("text") if block_type == "text" else block.get("thinking")
        if not isinstance(value, str) or not value.strip():
            continue
        if block_type == "text":
            text_chunks.append(value)
        elif block_type == "thinking":
            thought_chunks.append(value)
    return {
        "text": "".join(text_chunks).strip() or None,
        "thoughts": "\n\n".join(thought_chunks).strip() or None,
        "field_confidence_by_pointer": {},
        "avg_logprobs": None,
    }


def _request_key_and_metadata(
    record: object,
    *,
    provider: str,
    anthropic_custom_id_to_key: Mapping[str, str],
) -> tuple[str | None, dict[str, object], str | None]:
    if not isinstance(record, dict):
        return None, {}, "invalid_record_type"

    if provider == "anthropic":
        custom_id = record.get("custom_id")
        request_key = (
            anthropic_custom_id_to_key.get(custom_id, custom_id)
            if isinstance(custom_id, str) and custom_id.strip()
            else None
        )
        result = record.get("result")
        if not isinstance(result, dict):
            return request_key, {}, "missing_result"
        result_type = str(result.get("type") or "").strip().lower()
        if result_type != "succeeded":
            return request_key, {}, f"batch_{result_type or 'unknown'}"
        response = result.get("message")
        if response is None:
            return request_key, {}, "missing_response"
        if bool(getattr(config, "model_validation_enabled", False)):
            stop_reason = str(response.get("stop_reason") or "").strip().lower()
            if stop_reason not in {"end_turn", "stop_sequence"}:
                return request_key, {}, f"stop_reason_{stop_reason or 'missing'}"
        metadata = _anthropic_metadata(response)
        return request_key, metadata, None

    request_key = record.get("key")
    if not isinstance(request_key, str) or not request_key.strip():
        request_key = None
    if record.get("error"):
        return request_key, {}, "batch_error"
    response = record.get("response")
    if response is None:
        return request_key, {}, "missing_response"
    if bool(getattr(config, "model_validation_enabled", False)):
        finish_reason = gemini_finish_reason(response)
        if finish_reason != "STOP":
            return (
                request_key,
                {},
                f"finish_reason_{finish_reason.lower() or 'missing'}",
            )
    return request_key, extract_response_metadata(response), None


def combine_subagent_jsonl_sources(
    sources: Iterable[tuple[str, Iterable[str]]],
    *,
    provider: str = "gemini",
    anthropic_custom_id_to_key: Mapping[str, str] | None = None,
) -> CombinedSubagentOutputs:
    """Validate specialist results and join them into ordinary page records."""

    specialists = schema_specialists(config.output_schema)
    specialists_by_name = {item.name: item for item in specialists}
    custom_ids = anthropic_custom_id_to_key or {}
    partials_by_page: dict[str, dict[str, dict[str, object]]] = {}
    metadata_by_page: dict[str, dict[str, dict[str, object]]] = {}
    pages_seen: set[str] = set()
    pages_with_duplicate_valid_specialists: set[str] = set()
    failures: list[dict[str, object]] = []
    stats: Counter[str] = Counter()

    for source, lines in sources:
        stats["output_files"] += 1
        for line_number, line in enumerate(lines, start=1):
            raw = line.strip()
            if not raw:
                continue
            stats["output_rows"] += 1
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                stats["rejected"] += 1
                failures.append(
                    {
                        "source": source,
                        "line_number": line_number,
                        "reason": "invalid_jsonl_line",
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            request_key, metadata, rejection = _request_key_and_metadata(
                record,
                provider=provider,
                anthropic_custom_id_to_key=custom_ids,
            )
            decoded = decode_specialist_request_key(request_key)
            if decoded is None:
                stats["rejected"] += 1
                failures.append(
                    {
                        "request_key": request_key,
                        "source": source,
                        "line_number": line_number,
                        "reason": rejection or "invalid_subagent_request_key",
                    }
                )
                continue
            page_key, specialist_name = decoded
            pages_seen.add(page_key)
            if rejection is not None:
                stats["rejected"] += 1
                failures.append(
                    {
                        "page_key": page_key,
                        "specialist": specialist_name,
                        "request_key": request_key,
                        "source": source,
                        "line_number": line_number,
                        "reason": rejection,
                    }
                )
                continue
            specialist = specialists_by_name.get(specialist_name)
            if specialist is None:
                stats["rejected"] += 1
                failures.append(
                    {
                        "page_key": page_key,
                        "specialist": specialist_name,
                        "request_key": request_key,
                        "source": source,
                        "line_number": line_number,
                        "reason": "unknown_specialist",
                    }
                )
                continue
            text_payload = metadata.get("text")
            if not isinstance(text_payload, str) or not text_payload.strip():
                stats["rejected"] += 1
                failures.append(
                    {
                        "page_key": page_key,
                        "specialist": specialist_name,
                        "request_key": request_key,
                        "source": source,
                        "line_number": line_number,
                        "reason": "empty_response_text",
                    }
                )
                continue
            page_partials = partials_by_page.setdefault(page_key, {})
            try:
                validated_payload = validate_specialist_json(
                    specialist,
                    text_payload,
                    parent_name=config.output_schema_name or "Page",
                )
            except Exception as exc:  # noqa: BLE001
                stats["rejected"] += 1
                failures.append(
                    {
                        "page_key": page_key,
                        "specialist": specialist_name,
                        "request_key": request_key,
                        "source": source,
                        "line_number": line_number,
                        "reason": "specialist_schema_validation_failed",
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            if specialist_name in page_partials:
                pages_with_duplicate_valid_specialists.add(page_key)
                stats["duplicate_valid_specialists"] += 1
                stats["rejected"] += 1
                failures.append(
                    {
                        "page_key": page_key,
                        "specialist": specialist_name,
                        "request_key": request_key,
                        "source": source,
                        "line_number": line_number,
                        "reason": "duplicate_valid_specialist",
                        "retryable": False,
                    }
                )
                continue
            page_partials[specialist_name] = validated_payload
            metadata_by_page.setdefault(page_key, {})[specialist_name] = metadata
            stats["valid_specialists"] += 1

    records: list[dict[str, object]] = []
    expected_names = {item.name for item in specialists}
    for page_key in sorted(pages_seen):
        if page_key in pages_with_duplicate_valid_specialists:
            stats["duplicate_pages"] += 1
            continue
        partials = partials_by_page.get(page_key, {})
        missing = sorted(expected_names - set(partials))
        if missing:
            failures.append(
                {
                    "page_key": page_key,
                    "reason": "missing_specialists",
                    "missing_specialists": missing,
                }
            )
            stats["incomplete_pages"] += 1
            continue
        try:
            parsed = merge_specialist_payloads(
                full_model=config.output_model,
                specialists=specialists,
                payloads=partials,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "page_key": page_key,
                    "reason": "joined_schema_validation_failed",
                    "retryable": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
            stats["invalid_joined_pages"] += 1
            continue

        payload_text = parsed.model_dump_json()
        records.append(
            {
                "key": page_key,
                "response": {
                    "candidates": [
                        {
                            "finishReason": "STOP",
                            "content": {"parts": [{"text": payload_text}]},
                        }
                    ]
                },
                "_patientjournals_metadata": merge_specialist_metadata(
                    {
                        specialist.name: metadata_by_page.get(page_key, {}).get(
                            specialist.name, {}
                        )
                        for specialist in specialists
                    }
                ),
            }
        )
        stats["complete_pages"] += 1

    stats["observed_pages"] = len(pages_seen)
    return CombinedSubagentOutputs(
        records=tuple(records),
        failures=tuple(failures),
        stats=dict(stats),
    )


def write_combined_subagent_outputs(
    combined: CombinedSubagentOutputs,
    *,
    output_path: Path,
    failures_path: Path,
) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in combined.records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    with open(failures_path, "w", encoding="utf-8") as handle:
        for failure in combined.failures:
            handle.write(json.dumps(failure, ensure_ascii=False))
            handle.write("\n")
