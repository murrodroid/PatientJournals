from __future__ import annotations

import base64
import copy
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from pydantic import BaseModel

from patientjournals.config.schemas import model_from_json_schema


_REQUEST_KEY_PREFIX = "pj-subagent-v1."


def _referenced_definitions(
    field_schema: Mapping[str, Any],
    definitions: Mapping[str, Any],
    *,
    definitions_key: str,
) -> dict[str, Any]:
    prefix = f"#/{definitions_key}/"
    pending: list[object] = [field_schema]
    names: set[str] = set()
    while pending:
        node = pending.pop()
        if isinstance(node, Mapping):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith(prefix):
                name = ref[len(prefix) :].split("/", 1)[0]
                if name and name not in names and name in definitions:
                    names.add(name)
                    pending.append(definitions[name])
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
    return {name: copy.deepcopy(definitions[name]) for name in sorted(names)}


@dataclass(frozen=True)
class SchemaSpecialist:
    """One independently solvable top-level part of a page schema."""

    field_name: str
    schema: dict[str, Any]
    description: str = ""

    @property
    def name(self) -> str:
        return self.field_name

    def validation_model(self, *, parent_name: str = "Page") -> type[BaseModel]:
        return model_from_json_schema(
            f"{parent_name}_{self.field_name}_Specialist",
            self.schema,
        )


def schema_specialists(schema: Mapping[str, Any]) -> tuple[SchemaSpecialist, ...]:
    """Split an object schema into stable top-level field specialists."""

    properties = schema.get("properties")
    if not isinstance(properties, Mapping) or not properties:
        raise ValueError(
            "Sub-agent decomposition requires an object schema with top-level properties."
        )

    required = {
        str(value) for value in (schema.get("required") or []) if isinstance(value, str)
    }
    specialists: list[SchemaSpecialist] = []
    for field_name, raw_field_schema in properties.items():
        if not isinstance(field_name, str) or not isinstance(raw_field_schema, Mapping):
            continue

        partial_schema: dict[str, Any] = {
            "title": f"{schema.get('title') or 'Page'}_{field_name}_Specialist",
            "type": "object",
            "properties": {field_name: copy.deepcopy(dict(raw_field_schema))},
            "additionalProperties": False,
        }
        if field_name in required:
            partial_schema["required"] = [field_name]
        for definitions_key in ("$defs", "definitions"):
            definitions = schema.get(definitions_key)
            if isinstance(definitions, Mapping):
                referenced = _referenced_definitions(
                    raw_field_schema,
                    definitions,
                    definitions_key=definitions_key,
                )
                if referenced:
                    partial_schema[definitions_key] = referenced
        if isinstance(schema.get("$schema"), str):
            partial_schema["$schema"] = schema["$schema"]

        description = raw_field_schema.get("description")
        specialists.append(
            SchemaSpecialist(
                field_name=field_name,
                schema=partial_schema,
                description=(
                    description.strip() if isinstance(description, str) else ""
                ),
            )
        )

    if not specialists:
        raise ValueError("Sub-agent decomposition found no usable schema properties.")
    return tuple(specialists)


def specialist_by_name(
    schema: Mapping[str, Any],
    name: str,
) -> SchemaSpecialist:
    for specialist in schema_specialists(schema):
        if specialist.name == name:
            return specialist
    raise KeyError(f"Unknown schema specialist '{name}'.")


def specialist_prompt(
    base_prompt: str,
    specialist: SchemaSpecialist,
    *,
    specialist_count: int,
) -> str:
    """Build a compact search brief; the response schema carries field detail."""

    description = " ".join(specialist.description.split())
    if len(description) > 240:
        description = description[:237].rsplit(" ", 1)[0].rstrip() + "..."
    role_brief = (
        "You are one transcription sub-agent in a larger page-extraction job.\n"
        f"Your sole assignment is the top-level field `{specialist.field_name}`. "
        "Do not extract or return any other top-level field.\n"
    )
    if specialist_count > 1:
        role_brief += "Other sub-agents are responsible for the remaining fields.\n"
    if description:
        role_brief += f"Task brief: {description}\n"
    role_brief += (
        "Search the entire page for this assignment. "
        "Use the image as primary evidence and the supplied OCR as positional reading aid. "
        "Do not infer missing facts. Preserve source spelling unless the schema says otherwise.\n"
        "Return only JSON matching the supplied one-field schema."
    )
    if specialist_count == 1:
        return f"{base_prompt.rstrip()}\n\nSub-agent role:\n{role_brief}"
    return role_brief


def encode_specialist_request_key(page_key: str, specialist_name: str) -> str:
    payload = json.dumps(
        {"page": str(page_key), "specialist": str(specialist_name)},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    return f"{_REQUEST_KEY_PREFIX}{encoded}"


def decode_specialist_request_key(value: object) -> tuple[str, str] | None:
    if not isinstance(value, str) or not value.startswith(_REQUEST_KEY_PREFIX):
        return None
    encoded = value[len(_REQUEST_KEY_PREFIX) :]
    if not encoded:
        return None
    try:
        padding = "=" * (-len(encoded) % 4)
        payload = json.loads(base64.urlsafe_b64decode(encoded + padding))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    page_key = payload.get("page")
    specialist_name = payload.get("specialist")
    if not isinstance(page_key, str) or not page_key.strip():
        return None
    if not isinstance(specialist_name, str) or not specialist_name.strip():
        return None
    return page_key.strip(), specialist_name.strip()


def page_key_from_request_key(value: object) -> str | None:
    decoded = decode_specialist_request_key(value)
    if decoded is not None:
        return decoded[0]
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def validate_specialist_json(
    specialist: SchemaSpecialist,
    payload_text: str,
    *,
    parent_name: str = "Page",
) -> dict[str, Any]:
    parsed = specialist.validation_model(parent_name=parent_name).model_validate_json(
        payload_text
    )
    return parsed.model_dump(mode="json", exclude_unset=True)


def merge_specialist_payloads(
    *,
    full_model: type[BaseModel],
    specialists: Iterable[SchemaSpecialist],
    payloads: Mapping[str, Mapping[str, Any]],
) -> BaseModel:
    expected = tuple(item.name for item in specialists)
    missing = [name for name in expected if name not in payloads]
    if missing:
        raise ValueError(
            "Cannot join page; missing specialist output(s): " + ", ".join(missing)
        )

    merged: dict[str, Any] = {}
    for name in expected:
        partial = payloads[name]
        unexpected = set(partial) - {name}
        if unexpected:
            raise ValueError(
                f"Specialist '{name}' returned unexpected top-level field(s): "
                + ", ".join(sorted(unexpected))
            )
        if name in partial:
            merged[name] = partial[name]
    return full_model.model_validate(merged)


def merge_specialist_metadata(
    metadata_by_specialist: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    thoughts: list[str] = []
    confidence: dict[str, object] = {}
    avg_logprobs: list[float] = []
    for name, metadata in metadata_by_specialist.items():
        thought = metadata.get("thoughts")
        if isinstance(thought, str) and thought.strip():
            thoughts.append(f"[{name}]\n{thought.strip()}")
        field_confidence = metadata.get("field_confidence_by_pointer")
        if isinstance(field_confidence, Mapping):
            confidence.update(
                {
                    str(pointer): value
                    for pointer, value in field_confidence.items()
                    if str(pointer)
                }
            )
        avg_logprobs_value = metadata.get("avg_logprobs")
        if isinstance(avg_logprobs_value, (int, float)):
            avg_logprobs.append(float(avg_logprobs_value))
    return {
        "thoughts": "\n\n".join(thoughts) or None,
        "field_confidence_by_pointer": confidence,
        "avg_logprobs": (
            sum(avg_logprobs) / len(avg_logprobs) if avg_logprobs else None
        ),
        "subagent_fields": list(metadata_by_specialist),
    }
