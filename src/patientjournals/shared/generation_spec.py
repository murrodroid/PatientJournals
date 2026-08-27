from __future__ import annotations

from google.genai import types

from patientjournals.config import config
from patientjournals.shared.ocr import OcrDocument, render_ocr_context


def prompt_text(ocr_context: OcrDocument | str | None = None) -> str:
    if isinstance(ocr_context, OcrDocument):
        rendered = render_ocr_context(ocr_context)
    else:
        rendered = str(ocr_context or "").strip()
    if not rendered:
        return config.input_prompt
    return f"{config.input_prompt.rstrip()}\n\n{rendered}"


def _thinking_level() -> str | None:
    value = str(config.thinking_level or "").strip().lower()
    if not value:
        return None
    return value


def _thinking_config_live() -> dict[str, object] | None:
    payload: dict[str, object] = {}
    thinking_level = _thinking_level()
    if thinking_level:
        payload["thinking_level"] = thinking_level
    if bool(config.include_thoughts):
        payload["include_thoughts"] = True
    return payload or None


def _thinking_config_batch() -> dict[str, object] | None:
    payload: dict[str, object] = {}
    thinking_level = _thinking_level()
    if thinking_level:
        payload["thinkingLevel"] = thinking_level
    if bool(config.include_thoughts):
        payload["includeThoughts"] = True
    return payload or None


def build_live_request_contents(
    image_bytes: bytes,
    mime_type: str,
    ocr_context: OcrDocument | str | None = None,
) -> list[object]:
    return [
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        prompt_text(ocr_context),
    ]


def build_live_generation_config(
    *,
    include_schema: bool,
    include_temperature: bool,
    include_thinking_level: bool,
    schema_payload: object | None = None,
) -> dict[str, object]:
    cfg: dict[str, object] = {
        "response_mime_type": config.response_mime_type,
    }
    if include_schema:
        schema_field = config.response_schema_field
        if schema_field:
            cfg[schema_field] = config.output_schema if schema_payload is None else schema_payload
    if include_temperature:
        cfg["temperature"] = config.model_temperature
    if bool(config.include_confidence_scores):
        cfg["response_logprobs"] = True
        cfg["logprobs"] = 1
    if include_thinking_level:
        thinking_cfg = _thinking_config_live()
        if thinking_cfg:
            cfg["thinking_config"] = thinking_cfg
    return cfg


def build_batch_generation_config(
    *,
    for_vertex: bool,
    include_schema: bool,
    include_temperature: bool,
    include_thinking_level: bool,
    schema_payload: object | None = None,
) -> dict[str, object]:
    cfg: dict[str, object] = {"responseMimeType": config.response_mime_type}

    if include_schema:
        payload = config.output_schema if schema_payload is None else schema_payload
        schema_field = config.response_schema_field
        if schema_field:
            if for_vertex and schema_field == "response_json_schema":
                cfg["responseSchema"] = payload
            elif schema_field == "response_json_schema":
                cfg["responseJsonSchema"] = payload
            elif schema_field == "response_schema":
                cfg["responseSchema"] = payload
            else:
                cfg[schema_field] = payload

    if include_temperature:
        cfg["temperature"] = config.model_temperature
    if bool(config.include_confidence_scores):
        cfg["responseLogprobs"] = True
        cfg["logprobs"] = 1

    if include_thinking_level:
        thinking_cfg = _thinking_config_batch()
        if thinking_cfg:
            cfg["thinkingConfig"] = thinking_cfg

    return cfg
