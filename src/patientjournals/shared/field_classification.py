from __future__ import annotations

import re


_IDENTIFIER_FIELDS = {
    "image_name",
    "file_name",
    "key",
    "gcs_uri",
    "source_path",
}

_METADATA_FIELDS = {
    "attempts",
    "avg_logprobs",
    "batch_job_name",
    "confidence",
    "corrected_field",
    "crossed_out",
    "duplicate_action",
    "failed",
    "failure_reason",
    "field_confidence",
    "field_confidence_logprobs",
    "field_confidence_ratio",
    "generation_seconds",
    "logprobs",
    "model",
    "provider",
    "raw_response",
    "recovered",
    "rows_written",
    "source",
    "status",
    "thought",
    "thoughts",
    "total_seconds",
}

_METADATA_FRAGMENTS = (
    "avg_logprob",
    "confidence_logprob",
    "failure_reason",
    "field_confidence",
    "generation_second",
    "logprob",
    "thought",
)


def normalize_field_name(field_name: str) -> str:
    value = field_name.strip().lower()
    value = re.sub(r"[\s-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def field_leaf_name(field_name: str) -> str:
    return normalize_field_name(field_name).split(".")[-1]


def is_identifier_field(field_name: str) -> bool:
    normalized = normalize_field_name(field_name)
    leaf = field_leaf_name(field_name)
    return normalized in _IDENTIFIER_FIELDS or leaf in _IDENTIFIER_FIELDS


def is_metadata_field(field_name: str) -> bool:
    normalized = normalize_field_name(field_name)
    leaf = field_leaf_name(field_name)
    if normalized in _METADATA_FIELDS or leaf in _METADATA_FIELDS:
        return True
    return any(fragment in normalized or fragment in leaf for fragment in _METADATA_FRAGMENTS)


def is_schema_data_field(field_name: str) -> bool:
    return not is_identifier_field(field_name) and not is_metadata_field(field_name)
