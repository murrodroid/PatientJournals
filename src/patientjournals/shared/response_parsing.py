from __future__ import annotations

import json
import math
import re
from typing import Any


def _pick_value(obj: object, *names: str) -> object | None:
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
        return None
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _iter_candidates(response: object) -> list[object]:
    candidates = _pick_value(response, "candidates")
    if isinstance(candidates, list):
        return candidates
    return []


def _first_candidate(response: object) -> object | None:
    candidates = _iter_candidates(response)
    if not candidates:
        return None
    return candidates[0]


def _iter_parts(candidate: object) -> list[object]:
    content = _pick_value(candidate, "content")
    if content is None:
        return []
    parts = _pick_value(content, "parts")
    if isinstance(parts, list):
        return parts
    return []


def _part_text(part: object) -> str | None:
    text = _pick_value(part, "text")
    return text.strip() if isinstance(text, str) and text.strip() else None


def _part_is_thought(part: object) -> bool:
    value = _pick_value(part, "thought")
    return bool(value)


def extract_response_text(response: object) -> str | None:
    for candidate in _iter_candidates(response):
        chunks: list[str] = []
        for part in _iter_parts(candidate):
            text = _part_text(part)
            if text and not _part_is_thought(part):
                chunks.append(text)
        if chunks:
            return "".join(chunks)

    top_text = _pick_value(response, "text")
    if isinstance(top_text, str) and top_text.strip():
        return top_text.strip()
    return None


def extract_response_thoughts(response: object) -> str | None:
    thoughts: list[str] = []
    for candidate in _iter_candidates(response):
        for part in _iter_parts(candidate):
            text = _part_text(part)
            if text and _part_is_thought(part):
                thoughts.append(text)
    if not thoughts:
        return None
    return "\n\n".join(thoughts)


def extract_response_avg_logprobs(response: object) -> float | None:
    candidate = _first_candidate(response)
    value = _pick_value(candidate, "avg_logprobs", "avgLogprobs")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _chosen_token_logprobs(candidate: object | None) -> list[tuple[str, float]]:
    if candidate is None:
        return []
    logprobs_result = _pick_value(candidate, "logprobs_result", "logprobsResult")
    chosen = _pick_value(logprobs_result, "chosen_candidates", "chosenCandidates")
    if not isinstance(chosen, list):
        return []

    token_logprobs: list[tuple[str, float]] = []
    for item in chosen:
        token = _pick_value(item, "token")
        logprob = _pick_value(item, "log_probability", "logProbability")
        if isinstance(token, str) and token and isinstance(logprob, (int, float)):
            token_logprobs.append((token, float(logprob)))
    return token_logprobs


def _escape_pointer_segment(segment: str) -> str:
    return segment.replace("~", "~0").replace("/", "~1")


def _pointer_from_path(path: tuple[str, ...]) -> str:
    if not path:
        return ""
    return "/" + "/".join(_escape_pointer_segment(segment) for segment in path)


def _skip_ws(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    return index


def _parse_json_string(text: str, index: int) -> int:
    if index >= len(text) or text[index] != '"':
        raise ValueError(f"Expected string at index {index}.")
    index += 1
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == '"':
            return index + 1
        index += 1
    raise ValueError("Unterminated JSON string.")


def _parse_json_key(text: str, index: int) -> tuple[str, int]:
    end = _parse_json_string(text, index)
    key = json.loads(text[index:end])
    if not isinstance(key, str):
        raise ValueError("Object key is not a string.")
    return key, end


_NUMBER_RE = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _parse_json_number(text: str, index: int) -> int:
    match = _NUMBER_RE.match(text, index)
    if not match:
        raise ValueError(f"Expected number at index {index}.")
    return match.end()


def _parse_json_value_spans(
    text: str,
    index: int,
    path: tuple[str, ...],
    spans: dict[str, tuple[int, int]],
) -> int:
    index = _skip_ws(text, index)
    if index >= len(text):
        raise ValueError("Unexpected end while parsing JSON value.")

    char = text[index]
    if char == "{":
        return _parse_json_object_spans(text, index, path, spans)
    if char == "[":
        return _parse_json_array_spans(text, index, path, spans)
    if char == '"':
        end = _parse_json_string(text, index)
        spans[_pointer_from_path(path)] = (index + 1, max(index + 1, end - 1))
        return end
    if char in "-0123456789":
        end = _parse_json_number(text, index)
        spans[_pointer_from_path(path)] = (index, end)
        return end

    for literal in ("true", "false", "null"):
        if text.startswith(literal, index):
            end = index + len(literal)
            spans[_pointer_from_path(path)] = (index, end)
            return end

    raise ValueError(f"Unsupported JSON token at index {index}.")


def _parse_json_object_spans(
    text: str,
    index: int,
    path: tuple[str, ...],
    spans: dict[str, tuple[int, int]],
) -> int:
    if text[index] != "{":
        raise ValueError(f"Expected object at index {index}.")
    index += 1
    index = _skip_ws(text, index)
    if index < len(text) and text[index] == "}":
        return index + 1

    while index < len(text):
        index = _skip_ws(text, index)
        key, index = _parse_json_key(text, index)
        index = _skip_ws(text, index)
        if index >= len(text) or text[index] != ":":
            raise ValueError(f"Expected ':' after key at index {index}.")
        index += 1
        index = _parse_json_value_spans(text, index, path + (key,), spans)
        index = _skip_ws(text, index)
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        if index < len(text) and text[index] == "}":
            return index + 1
        raise ValueError(f"Expected ',' or '}}' at index {index}.")

    raise ValueError("Unterminated JSON object.")


def _parse_json_array_spans(
    text: str,
    index: int,
    path: tuple[str, ...],
    spans: dict[str, tuple[int, int]],
) -> int:
    if text[index] != "[":
        raise ValueError(f"Expected array at index {index}.")
    index += 1
    index = _skip_ws(text, index)
    if index < len(text) and text[index] == "]":
        return index + 1

    item_index = 0
    while index < len(text):
        index = _parse_json_value_spans(
            text=text,
            index=index,
            path=path + (str(item_index),),
            spans=spans,
        )
        item_index += 1
        index = _skip_ws(text, index)
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        if index < len(text) and text[index] == "]":
            return index + 1
        raise ValueError(f"Expected ',' or ']' at index {index}.")

    raise ValueError("Unterminated JSON array.")


def _collect_leaf_value_spans(text: str) -> dict[str, tuple[int, int]]:
    spans: dict[str, tuple[int, int]] = {}
    start = _skip_ws(text, 0)
    end = _parse_json_value_spans(text=text, index=start, path=(), spans=spans)
    end = _skip_ws(text, end)
    if end != len(text):
        raise ValueError("Unexpected trailing characters after JSON value.")
    return spans


def _token_spans(token_logprobs: list[tuple[str, float]]) -> list[tuple[int, int, float]]:
    spans: list[tuple[int, int, float]] = []
    position = 0
    for token, logprob in token_logprobs:
        end = position + len(token)
        spans.append((position, end, logprob))
        position = end
    return spans


def _find_payload_offset(token_text: str, payload_text: str) -> int | None:
    if token_text == payload_text:
        return 0
    direct_index = token_text.find(payload_text)
    if direct_index >= 0:
        return direct_index
    stripped_payload = payload_text.strip()
    if not stripped_payload:
        return None
    stripped_index = token_text.find(stripped_payload)
    if stripped_index >= 0:
        return stripped_index
    return None


def _collect_logprobs_by_pointer(
    payload_text: str,
    token_logprobs: list[tuple[str, float]],
    leaf_spans: dict[str, tuple[int, int]],
) -> dict[str, list[float]]:
    if not payload_text or not token_logprobs or not leaf_spans:
        return {}

    token_text = "".join(token for token, _ in token_logprobs)
    offset = _find_payload_offset(token_text=token_text, payload_text=payload_text)
    if offset is None:
        return {}

    pointer_spans = list(leaf_spans.items())
    by_pointer: dict[str, list[float]] = {}
    payload_len = len(payload_text)

    for token_start, token_end, logprob in _token_spans(token_logprobs):
        local_start = max(0, token_start - offset)
        local_end = min(payload_len, token_end - offset)
        if local_start >= local_end:
            continue

        best_pointer: str | None = None
        best_overlap = 0
        for pointer, (value_start, value_end) in pointer_spans:
            overlap = min(local_end, value_end) - max(local_start, value_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_pointer = pointer

        if best_pointer is None or best_overlap <= 0:
            continue
        by_pointer.setdefault(best_pointer, []).append(logprob)

    return by_pointer


def confidence_from_avg_logprobs(avg_logprobs: float | None) -> float | None:
    if avg_logprobs is None:
        return None
    try:
        confidence = math.exp(avg_logprobs)
    except (OverflowError, ValueError):
        return None
    if confidence < 0:
        return 0.0
    if confidence > 1:
        return 1.0
    return confidence


def extract_field_confidence_by_pointer(
    response: object,
    payload_text: str | None = None,
) -> dict[str, dict[str, float | None]]:
    text = payload_text if isinstance(payload_text, str) else extract_response_text(response)
    if not isinstance(text, str) or not text.strip():
        return {}

    try:
        leaf_spans = _collect_leaf_value_spans(text)
    except ValueError:
        return {}

    token_logprobs = _chosen_token_logprobs(_first_candidate(response))
    if not token_logprobs:
        return {}
    by_pointer_logprobs = _collect_logprobs_by_pointer(
        payload_text=text,
        token_logprobs=token_logprobs,
        leaf_spans=leaf_spans,
    )

    out: dict[str, dict[str, float | None]] = {}
    has_any_confidence = False
    for pointer in leaf_spans:
        values = by_pointer_logprobs.get(pointer, [])
        avg_logprobs = (sum(values) / len(values)) if values else None
        if avg_logprobs is not None:
            has_any_confidence = True
        out[pointer] = {
            "field_confidence_logprobs": avg_logprobs,
            "field_confidence_ratio": confidence_from_avg_logprobs(avg_logprobs),
        }
    if not has_any_confidence:
        return {}
    return out


def extract_response_metadata(response: object) -> dict[str, Any]:
    text = extract_response_text(response)
    return {
        "text": text,
        "thoughts": extract_response_thoughts(response),
        "avg_logprobs": extract_response_avg_logprobs(response),
        "field_confidence_by_pointer": extract_field_confidence_by_pointer(
            response=response,
            payload_text=text,
        ),
    }
