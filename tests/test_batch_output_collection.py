import json

from pydantic import BaseModel

from patientjournals.batch.collect_outputs import (
    collect_valid_outputs_from_jsonl_sources,
    write_collected_dataset,
)
from patientjournals.batch.output_records import parse_gemini_output_record
from patientjournals.config import config


class SimpleOutput(BaseModel):
    value: str


def gemini_response(payload: dict) -> dict:
    return {
        "candidates": [
            {
                "avgLogprobs": -0.25,
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(payload),
                        }
                    ]
                }
            }
        ]
    }


def output_line(key: str, payload: dict | None) -> str:
    response = {} if payload is None else gemini_response(payload)
    return json.dumps({"key": key, "response": response})


def test_parse_gemini_output_record_validates_configured_schema(monkeypatch) -> None:
    monkeypatch.setattr(config, "output_model", SimpleOutput)
    record = {"key": "pages/a.png", "response": gemini_response({"value": "ok"})}

    result = parse_gemini_output_record(record, source="source.jsonl", line_number=1)

    assert result.is_valid
    assert result.key == "pages/a.png"
    assert result.parsed_model == SimpleOutput(value="ok")


def test_collect_outputs_uses_later_valid_candidate_for_same_key(monkeypatch) -> None:
    monkeypatch.setattr(config, "output_model", SimpleOutput)
    sources = [
        (
            "first.jsonl",
            [
                output_line("pages/a.png", {"other": "invalid"}),
                output_line("pages/b.png", None),
            ],
        ),
        (
            "second.jsonl",
            [
                output_line("pages/a.png", {"value": "valid"}),
                output_line("pages/a.png", {"value": "duplicate"}),
            ],
        ),
    ]

    collected = collect_valid_outputs_from_jsonl_sources(sources)

    assert sorted(collected.selected) == ["pages/a.png"]
    assert collected.selected["pages/a.png"].parsed_model == SimpleOutput(value="valid")
    assert collected.stats["rejected:schema_validation_failed"] == 1
    assert collected.stats["rejected:missing_response"] == 1
    assert collected.stats["duplicate_valid_candidates"] == 1


def test_write_collected_dataset_sorts_by_key(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "output_model", SimpleOutput)
    collected = collect_valid_outputs_from_jsonl_sources(
        [
            (
                "source.jsonl",
                [
                    output_line("pages/b.png", {"value": "b"}),
                    output_line("pages/a.png", {"value": "a"}),
                ],
            )
        ]
    )
    out_path = tmp_path / "dataset.jsonl"

    _, rows = write_collected_dataset(
        collected,
        out_path=out_path,
        output_format="jsonl",
    )

    assert rows == 2
    lines = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [line["file_name"] for line in lines] == ["pages/a.png", "pages/b.png"]
    assert lines[0]["avg_logprobs"] == -0.25


def test_write_collected_dataset_can_append_only_new_keys(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "output_model", SimpleOutput)
    collected = collect_valid_outputs_from_jsonl_sources(
        [
            (
                "source.jsonl",
                [
                    output_line("pages/a.png", {"value": "a"}),
                    output_line("pages/b.png", {"value": "b"}),
                ],
            )
        ]
    )
    out_path = tmp_path / "dataset.jsonl"
    out_path.write_text(
        json.dumps({"file_name": "pages/a.png", "value": "existing"}) + "\n",
        encoding="utf-8",
    )

    _, rows = write_collected_dataset(
        collected,
        out_path=out_path,
        output_format="jsonl",
        keys={"pages/b.png"},
        header_written=True,
    )

    assert rows == 1
    lines = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [line["file_name"] for line in lines] == ["pages/a.png", "pages/b.png"]
