from __future__ import annotations

import copy
import hashlib
import json
import mimetypes
import re
from datetime import timedelta
from pathlib import Path
from typing import Any

from google.cloud import storage
from tqdm import tqdm

from patientjournals.batch.submit_inputs import _normalize_prefix
from patientjournals.config import config
from patientjournals.shared.generation_spec import (
    build_batch_generation_config,
    prompt_text,
)


_ANTHROPIC_CUSTOM_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _anthropic_custom_id_for_key(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:40]
    return f"gcs_{digest}"


def _guess_mime_type(blob: storage.Blob) -> str:
    if blob.content_type:
        return blob.content_type
    guess, _ = mimetypes.guess_type(blob.name)
    return guess or "application/octet-stream"


def _schema_has_refs(schema: object) -> bool:
    if isinstance(schema, dict):
        if "$ref" in schema or "$defs" in schema:
            return True
        return any(_schema_has_refs(value) for value in schema.values())
    if isinstance(schema, list):
        return any(_schema_has_refs(item) for item in schema)
    return False


def _resolve_json_pointer(root: object, pointer: str) -> object | None:
    if not isinstance(pointer, str) or not pointer.startswith("#/"):
        return None
    current = root
    for token in pointer[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or token not in current:
            return None
        current = current[token]
    return current


def _inline_json_refs(
    node: object, root: object, stack: set[str] | None = None
) -> object:
    if stack is None:
        stack = set()

    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            if ref in stack:
                return {}
            target = _resolve_json_pointer(root, ref)
            if target is None:
                return {}
            resolved = _inline_json_refs(target, root, stack | {ref})
            extras = {
                key: _inline_json_refs(value, root, stack)
                for key, value in node.items()
                if key != "$ref"
            }
            if extras and isinstance(resolved, dict):
                merged = dict(resolved)
                merged.update(extras)
                return merged
            if extras:
                return extras
            return resolved

        output: dict[str, object] = {}
        for key, value in node.items():
            if key == "$defs":
                continue
            if key.startswith("$"):
                continue
            output[key] = _inline_json_refs(value, root, stack)
        return output

    if isinstance(node, list):
        return [_inline_json_refs(item, root, stack) for item in node]

    return node


def _vertex_compatible_schema(raw_schema: object) -> object:
    return _inline_json_refs(raw_schema, raw_schema)


def _build_request_config(*, for_vertex: bool) -> dict:
    schema_payload: object = config.output_schema
    if for_vertex:
        schema_payload = _vertex_compatible_schema(config.output_schema)

    return build_batch_generation_config(
        for_vertex=for_vertex,
        include_schema=bool(config.batch_include_response_schema),
        include_temperature=True,
        include_thinking_level=True,
        schema_payload=schema_payload,
    )


def _build_request_line(
    blob: storage.Blob,
    bucket_name: str,
    *,
    for_vertex: bool,
) -> dict:
    mime_type = _guess_mime_type(blob)
    media_part = {
        "fileData": {
            "fileUri": f"gs://{bucket_name}/{blob.name}",
            "mimeType": mime_type,
        }
    }

    parts = [media_part]
    prompt = prompt_text()
    if prompt:
        parts.append({"text": prompt})

    return {
        "key": blob.name,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
            "generationConfig": _build_request_config(for_vertex=for_vertex),
        },
    }


def _build_anthropic_manifest_line(blob: storage.Blob) -> dict[str, str]:
    key = blob.name
    return {
        "key": key,
        "mime_type": _guess_mime_type(blob),
        "custom_id": _anthropic_custom_id_for_key(key),
    }


def _iter_anthropic_manifest_entries(path: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in anthropic manifest {path} at line {line_number}."
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Invalid anthropic manifest row at line {line_number}: expected object."
                )
            key = payload.get("key")
            if not isinstance(key, str) or not key.strip():
                raise ValueError(
                    f"Missing 'key' in anthropic manifest at line {line_number}."
                )
            key = key.strip()
            mime_type = payload.get("mime_type")
            if not isinstance(mime_type, str) or not mime_type.strip():
                mime_type = "application/octet-stream"
            custom_id_raw = payload.get("custom_id")
            if isinstance(custom_id_raw, str) and custom_id_raw.strip():
                custom_id = custom_id_raw.strip()
            else:
                custom_id = _anthropic_custom_id_for_key(key)
            if not _ANTHROPIC_CUSTOM_ID_PATTERN.fullmatch(custom_id):
                raise ValueError(
                    f"Invalid custom_id in anthropic manifest at line {line_number}: "
                    f"{custom_id!r}"
                )
            entries.append(
                {
                    "key": key,
                    "mime_type": mime_type.strip(),
                    "custom_id": custom_id,
                }
            )
    return entries


def _anthropic_signed_url_expiration() -> timedelta:
    ttl_hours = max(1, int(config.anthropic_signed_url_ttl_hours or 48))
    return timedelta(hours=ttl_hours)


def _anthropic_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise TypeError("Expected output schema to be a dictionary.")

    normalized = copy.deepcopy(schema)

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            node_type = node.get("type")
            if node_type == "object" or (
                isinstance(node_type, list) and "object" in node_type
            ):
                node["additionalProperties"] = False
            for value in node.values():
                _walk(value)
            return
        if isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(normalized)
    return normalized


def _build_anthropic_batch_requests(
    *,
    bucket: storage.Bucket,
    requests_path: Path,
) -> list[dict[str, Any]]:
    manifest = _iter_anthropic_manifest_entries(requests_path)
    if not manifest:
        raise ValueError(f"Anthropic manifest is empty: {requests_path}")

    prompt = prompt_text()
    include_schema = bool(config.batch_include_response_schema)
    max_tokens = max(1, int(config.model_max_output_tokens))
    expiration = _anthropic_signed_url_expiration()
    schema_payload = (
        _anthropic_strict_json_schema(config.output_schema) if include_schema else None
    )

    requests: list[dict[str, Any]] = []
    seen_custom_ids: set[str] = set()
    for row in manifest:
        key = row["key"]
        custom_id = row["custom_id"]
        if custom_id in seen_custom_ids:
            raise ValueError(
                "Duplicate custom_id generated for Anthropic batch requests: "
                f"{custom_id}"
            )
        seen_custom_ids.add(custom_id)
        signed_url = bucket.blob(key).generate_signed_url(
            version="v4",
            method="GET",
            expiration=expiration,
        )
        content: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": signed_url,
                },
            }
        ]
        if prompt:
            content.append({"type": "text", "text": prompt})

        params: dict[str, Any] = {
            "model": config.model,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }
        if include_schema:
            params["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": schema_payload,
                }
            }
        requests.append(
            {
                "custom_id": custom_id,
                "params": params,
            }
        )
    return requests


def _upload_requests_to_gcs(
    bucket: storage.Bucket,
    run_dir_name: str,
    local_requests_path: Path,
) -> str:
    prefix = _normalize_prefix(config.batch_requests_gcs_prefix or "")
    object_name = f"{prefix}{run_dir_name}/{local_requests_path.name}"
    blob = bucket.blob(object_name)
    blob.upload_from_filename(
        str(local_requests_path),
        content_type="application/jsonl",
    )
    return f"gs://{bucket.name}/{object_name}"


def _output_dest_gcs_uri(
    bucket_name: str,
    run_dir_name: str,
    *,
    chunk_label: str | None = None,
) -> str:
    prefix = _normalize_prefix(config.batch_outputs_gcs_prefix or "")
    base = f"{prefix}{run_dir_name}".rstrip("/")
    if chunk_label:
        base = f"{base}/{chunk_label.strip('/')}"
    return f"gs://{bucket_name}/{base}"


def _write_requests_file(
    blobs: list[storage.Blob],
    bucket_name: str,
    output_path: Path,
    log,
    *,
    for_vertex: bool,
    provider: str,
) -> tuple[int, int]:
    max_bytes = int(config.batch_input_max_bytes or 0)
    total_bytes = 0
    count = 0

    with open(output_path, "w", encoding="utf-8") as handle:
        for blob in tqdm(blobs, desc="Building batch JSONL", unit="img"):
            if provider == "anthropic":
                line_obj = _build_anthropic_manifest_line(blob)
            else:
                line_obj = _build_request_line(
                    blob,
                    bucket_name,
                    for_vertex=for_vertex,
                )
            line = json.dumps(line_obj, ensure_ascii=False)
            handle.write(line)
            handle.write("\n")
            count += 1

            total_bytes += len(line.encode("utf-8")) + 1
            if max_bytes and total_bytes > max_bytes:
                raise ValueError(
                    "Batch request file exceeded batch_input_max_bytes. "
                    "Reduce inputs or split into multiple batch jobs."
                )

    log(
        f"Wrote {count} requests to {output_path.name} "
        f"({total_bytes} bytes, source=gcs)."
    )
    if provider == "anthropic":
        log(
            "Anthropic request file stores key/mime manifest only; "
            "signed image URLs are generated at submit time."
        )
    if provider == "gemini" and for_vertex and not config.batch_include_response_schema:
        log(
            "Vertex batch input omits response schema (config.batch_include_response_schema=False)."
        )
    if (
        provider == "gemini"
        and for_vertex
        and config.batch_include_response_schema
        and _schema_has_refs(config.output_schema)
    ):
        log(
            "Vertex batch input inlines schema refs and strips $-keys for compatibility."
        )
    return count, total_bytes


def _count_requests_file(path: Path) -> tuple[int, int]:
    total_bytes = path.stat().st_size if path.exists() else 0
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count, total_bytes

