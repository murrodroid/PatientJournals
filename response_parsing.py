from __future__ import annotations

import math
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


def _avg_logprobs_from_candidate(candidate: object) -> float | None:
    avg = _pick_value(candidate, "avg_logprobs", "avgLogprobs")
    if isinstance(avg, (int, float)):
        return float(avg)

    logprobs_result = _pick_value(candidate, "logprobs_result", "logprobsResult")
    chosen = _pick_value(logprobs_result, "chosen_candidates", "chosenCandidates")
    if not isinstance(chosen, list):
        return None
    values: list[float] = []
    for item in chosen:
        val = _pick_value(item, "log_probability", "logProbability")
        if isinstance(val, (int, float)):
            values.append(float(val))
    if not values:
        return None
    return sum(values) / len(values)


def extract_response_avg_logprobs(response: object) -> float | None:
    for candidate in _iter_candidates(response):
        avg = _avg_logprobs_from_candidate(candidate)
        if avg is not None:
            return avg
    return None


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


def extract_response_metadata(response: object) -> dict[str, Any]:
    avg_logprobs = extract_response_avg_logprobs(response)
    confidence_ratio = confidence_from_avg_logprobs(avg_logprobs)
    return {
        "text": extract_response_text(response),
        "thoughts": extract_response_thoughts(response),
        "response_confidence_logprobs": avg_logprobs,
        "response_confidence_ratio": confidence_ratio,
        # Backward-compatible aliases for existing call sites.
        "response_avg_logprobs": avg_logprobs,
        "response_confidence": confidence_ratio,
    }
