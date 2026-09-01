from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from patientjournals.batch.output_records import parse_gemini_output_record
from patientjournals.batch.subagent_outputs import combine_subagent_jsonl_sources
from patientjournals.batch.submit_requests import (
    _build_request_line,
    _build_request_lines,
)
from patientjournals.config import config
from patientjournals.config.schemas import model_from_json_schema
from patientjournals.shared.subagents import (
    decode_specialist_request_key,
    encode_specialist_request_key,
    merge_specialist_payloads,
    page_key_from_request_key,
    schema_specialists,
    specialist_prompt,
)


SIMPLE_SCHEMA = {
    "title": "SimplePage",
    "type": "object",
    "properties": {
        "patient": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
        "diagnosis": {"type": "string"},
    },
    "required": ["patient", "diagnosis"],
    "additionalProperties": False,
}


@pytest.fixture
def simple_config(monkeypatch):
    monkeypatch.setattr(config, "output_schema", SIMPLE_SCHEMA)
    monkeypatch.setattr(
        config,
        "output_model",
        model_from_json_schema("SimplePage", SIMPLE_SCHEMA),
    )
    monkeypatch.setattr(config, "output_schema_name", "SimplePage")
    monkeypatch.setattr(config, "subagents", True)
    monkeypatch.setattr(config, "input_prompt_name", "frontpage")


def _gemini_line(key: str, payload: dict) -> str:
    return json.dumps(
        {
            "key": key,
            "response": {
                "candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]
            },
        }
    )


def test_schema_specialists_are_one_top_level_field_each():
    specialists = schema_specialists(SIMPLE_SCHEMA)

    assert [item.name for item in specialists] == ["patient", "diagnosis"]
    assert list(specialists[0].schema["properties"]) == ["patient"]
    assert specialists[0].schema["required"] == ["patient"]
    assert specialists[0].schema["additionalProperties"] is False


def test_schema_specialist_keeps_only_transitively_referenced_definitions():
    schema = {
        "type": "object",
        "properties": {
            "patient": {"$ref": "#/$defs/Patient"},
            "diagnosis": {"type": "string"},
        },
        "$defs": {
            "Patient": {
                "type": "object",
                "properties": {"address": {"$ref": "#/$defs/Address"}},
            },
            "Address": {"type": "object", "properties": {"street": {"type": "string"}}},
            "Unused": {"type": "string"},
        },
    }

    patient, diagnosis = schema_specialists(schema)

    assert set(patient.schema["$defs"]) == {"Patient", "Address"}
    assert "$defs" not in diagnosis.schema


def test_specialist_prompt_explicitly_limits_role_and_scope():
    specialist = schema_specialists(
        {
            "type": "object",
            "properties": {
                "diagnosis": {
                    "type": "string",
                    "description": "Find only the diagnosis written on the page.",
                },
                "patient": {"type": "string"},
            },
        }
    )[0]

    prompt = specialist_prompt("FULL PAGE INSTRUCTIONS", specialist, specialist_count=2)

    assert "one transcription sub-agent" in prompt
    assert "sole assignment is the top-level field `diagnosis`" in prompt
    assert "Do not extract or return any other top-level field" in prompt
    assert "Other sub-agents are responsible" in prompt
    assert "Task brief: Find only the diagnosis written on the page." in prompt
    assert "FULL PAGE INSTRUCTIONS" not in prompt


def test_single_specialist_keeps_base_prompt_and_gets_subagent_brief():
    specialist = schema_specialists(
        {
            "type": "object",
            "properties": {"page_lines": {"type": "array", "items": {}}},
        }
    )[0]

    prompt = specialist_prompt("FULL PAGE INSTRUCTIONS", specialist, specialist_count=1)

    assert prompt.startswith("FULL PAGE INSTRUCTIONS")
    assert "Sub-agent role:" in prompt
    assert "sole assignment is the top-level field `page_lines`" in prompt
    assert "Other sub-agents are responsible" not in prompt


def test_specialist_request_key_round_trip():
    encoded = encode_specialist_request_key("pages/folder/page 1.png", "patient")

    assert decode_specialist_request_key(encoded) == (
        "pages/folder/page 1.png",
        "patient",
    )
    assert page_key_from_request_key(encoded) == "pages/folder/page 1.png"
    assert page_key_from_request_key("pages/plain.png") == "pages/plain.png"


def test_merge_specialists_validates_full_page(simple_config):
    specialists = schema_specialists(SIMPLE_SCHEMA)

    parsed = merge_specialist_payloads(
        full_model=config.output_model,
        specialists=specialists,
        payloads={
            "patient": {"patient": {"name": "Ada"}},
            "diagnosis": {"diagnosis": "Feber"},
        },
    )

    assert parsed.model_dump() == {
        "patient": {"name": "Ada"},
        "diagnosis": "Feber",
    }


def test_combiner_joins_out_of_order_specialist_results(simple_config):
    page_key = "pages/001.png"
    lines = [
        _gemini_line(
            encode_specialist_request_key(page_key, "diagnosis"),
            {"diagnosis": "Feber"},
        ),
        _gemini_line(
            encode_specialist_request_key(page_key, "patient"),
            {"patient": {"name": "Ada"}},
        ),
    ]

    combined = combine_subagent_jsonl_sources([("chunk", lines)])

    assert combined.stats["complete_pages"] == 1
    assert combined.failures == ()
    parsed = parse_gemini_output_record(combined.records[0])
    assert parsed.is_valid
    assert parsed.key == page_key
    assert parsed.parsed_model.model_dump() == {
        "patient": {"name": "Ada"},
        "diagnosis": "Feber",
    }
    assert parsed.metadata["subagent_fields"] == ["patient", "diagnosis"]


def test_combiner_withholds_page_with_duplicate_valid_specialist(simple_config):
    page_key = "pages/001.png"
    lines = [
        _gemini_line(
            encode_specialist_request_key(page_key, "patient"),
            {"patient": {"name": "Ada"}},
        ),
        _gemini_line(
            encode_specialist_request_key(page_key, "patient"),
            {"patient": {"name": "Else"}},
        ),
        _gemini_line(
            encode_specialist_request_key(page_key, "diagnosis"),
            {"diagnosis": "Feber"},
        ),
    ]

    combined = combine_subagent_jsonl_sources([("chunk", lines)])

    assert combined.records == ()
    assert combined.stats["duplicate_valid_specialists"] == 1
    assert combined.stats["duplicate_pages"] == 1
    failure = next(
        item
        for item in combined.failures
        if item.get("reason") == "duplicate_valid_specialist"
        and item.get("specialist") == "patient"
    )
    assert failure["retryable"] is False


def test_combiner_marks_join_schema_failure_non_retryable(
    simple_config, monkeypatch
) -> None:
    page_key = "pages/001.png"
    lines = [
        _gemini_line(
            encode_specialist_request_key(page_key, "patient"),
            {"patient": {"name": "Ada"}},
        ),
        _gemini_line(
            encode_specialist_request_key(page_key, "diagnosis"),
            {"diagnosis": "Feber"},
        ),
    ]

    def reject_join(**_kwargs):
        raise ValueError("cross-field constraint failed")

    monkeypatch.setattr(
        "patientjournals.batch.subagent_outputs.merge_specialist_payloads",
        reject_join,
    )

    combined = combine_subagent_jsonl_sources([("chunk", lines)])

    assert combined.records == ()
    failure = next(
        item
        for item in combined.failures
        if item.get("reason") == "joined_schema_validation_failed"
    )
    assert failure["retryable"] is False


def test_preversion_combiner_rejects_truncated_specialist(
    simple_config, monkeypatch
) -> None:
    monkeypatch.setattr(config, "model_validation_enabled", True)
    page_key = "pages/001.png"

    def line(specialist: str, payload: dict, reason: str) -> str:
        return json.dumps(
            {
                "key": encode_specialist_request_key(page_key, specialist),
                "response": {
                    "candidates": [
                        {
                            "finishReason": reason,
                            "content": {
                                "parts": [{"text": json.dumps(payload)}]
                            },
                        }
                    ]
                },
            }
        )

    combined = combine_subagent_jsonl_sources(
        [
            (
                "chunk",
                [
                    line("patient", {"patient": {"name": "Ada"}}, "MAX_TOKENS"),
                    line("diagnosis", {"diagnosis": "Feber"}, "STOP"),
                ],
            )
        ]
    )

    assert combined.records == ()
    assert any(
        item.get("reason") == "finish_reason_max_tokens"
        for item in combined.failures
    )


def test_combiner_withholds_page_when_specialist_is_missing(simple_config):
    page_key = "pages/001.png"
    lines = [
        _gemini_line(
            encode_specialist_request_key(page_key, "patient"),
            {"patient": {"name": "Ada"}},
        )
    ]

    combined = combine_subagent_jsonl_sources([("chunk", lines)])

    assert combined.records == ()
    assert combined.stats["incomplete_pages"] == 1
    assert combined.failures[-1]["missing_specialists"] == ["diagnosis"]


@dataclass
class _FakeBlob:
    name: str = "pages/001.png"
    content_type: str = "image/png"


def test_batch_request_fanout_and_disabled_compatibility(simple_config, monkeypatch):
    blob = _FakeBlob()
    monkeypatch.setattr(
        "patientjournals.batch.submit_requests.ocr_context_for_blob",
        lambda _blob: "OCR",
    )

    fanout = _build_request_lines(blob, "bucket", for_vertex=False)
    assert len(fanout) == 2
    assert {decode_specialist_request_key(item["key"])[1] for item in fanout} == {
        "patient",
        "diagnosis",
    }
    for item in fanout:
        specialist_name = decode_specialist_request_key(item["key"])[1]
        request = item["request"]
        schema = request["generationConfig"]["responseJsonSchema"]
        assert list(schema["properties"]) == [specialist_name]
        assert request["contents"][0]["parts"][0]["fileData"]["fileUri"] == (
            "gs://bucket/pages/001.png"
        )

    monkeypatch.setattr(config, "subagents", False)
    assert _build_request_lines(blob, "bucket", for_vertex=False) == [
        _build_request_line(blob, "bucket", for_vertex=False)
    ]
