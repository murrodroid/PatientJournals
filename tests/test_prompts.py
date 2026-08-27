from __future__ import annotations

from patientjournals.config import config
from patientjournals.config.prompts import (
    OCR_CONTEXT_HEADER,
    PAGE_PROMPTS,
    SUBAGENT_ROLE,
    build_subagent_prompt,
)
from patientjournals.shared.ocr import OcrDocument, OcrLine, render_ocr_context


def test_runtime_page_prompts_are_copied_from_prompt_registry():
    assert config.prompts == dict(PAGE_PROMPTS)
    assert config.input_prompt == PAGE_PROMPTS[config.input_prompt_name]


def test_subagent_prompt_is_rendered_from_central_prompt_definitions():
    rendered = build_subagent_prompt(
        base_prompt="page prompt",
        field_name="diagnoses",
        field_description="Find diagnosis information.",
        specialist_count=2,
    )

    assert rendered.startswith(SUBAGENT_ROLE)
    assert "`diagnoses`" in rendered
    assert "Task brief: Find diagnosis information." in rendered


def test_ocr_context_uses_central_header_template():
    document = OcrDocument(
        image_sha256="digest",
        width=100,
        height=100,
        coordinate_scale=1000,
        backend="test",
        lines=(OcrLine(text="Patient", box=(1, 2, 3, 4)),),
    )

    rendered = render_ocr_context(document)

    assert rendered.splitlines()[0] == OCR_CONTEXT_HEADER.format(
        coordinate_scale=1000
    )
    assert rendered.splitlines()[1] == "1,2,3,4|Patient"
