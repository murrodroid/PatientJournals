from __future__ import annotations

import copy
import hashlib
import json
import mimetypes
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from google.cloud import storage
from google.genai import types

from patientjournals.batch.client import resolve_service_account_path
from patientjournals.config import config
from patientjournals.shared.generation_spec import (
    build_batch_generation_config,
    prompt_text,
)
from patientjournals.shared.tools import create_subfolder, get_run_logger


def _normalize_key(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def _anthropic_custom_id_for_key(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:40]
    return f"gcs_{digest}"


def _normalize_prefix(prefix: str) -> str:
    value = prefix.strip()
    if not value:
        return ""
    return f"{value.strip('/')}/"


def _parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Expected GCS URI, received: {gcs_uri}")
    without_scheme = gcs_uri[len("gs://") :]
    bucket, _, path = without_scheme.partition("/")
    if not bucket or not path:
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    return bucket, path


def _require_bucket_name() -> str:
    bucket_name = (config.gcs_bucket_name or "").strip()
    if not bucket_name:
        raise ValueError("config.gcs_bucket_name is empty.")
    return bucket_name


def _require_service_account_file() -> str:
    service_account_file = (config.service_account_file or "").strip()
    if not service_account_file:
        raise ValueError("config.service_account_file is empty.")
    return service_account_file


def _read_batch_job_payload(path: Path) -> dict | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _extract_location_from_batch_name(batch_name: str) -> str | None:
    parts = [part for part in batch_name.split("/") if part]
    for index, part in enumerate(parts):
        if part == "locations" and index + 1 < len(parts):
            return parts[index + 1]
    return None


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
    node: object,
    root: object,
    stack: set[str] | None = None,
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


def _build_retry_batch_generation_config(*, for_vertex: bool) -> dict:
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


def _guess_key_mime_type(key: str) -> str:
    guess, _ = mimetypes.guess_type(key)
    return guess or "application/octet-stream"


def _normalize_retry_object_key(raw_key: str, *, bucket_name: str) -> str | None:
    key = _normalize_key(raw_key)
    if not key:
        return None
    if key.startswith("gs://"):
        try:
            source_bucket, object_path = _parse_gcs_uri(key)
        except ValueError:
            return None
        if source_bucket != bucket_name:
            return None
        key = object_path
    key = key.lstrip("/")
    return key or None


def _build_retry_gemini_request_line(
    key: str,
    *,
    bucket_name: str,
    for_vertex: bool,
) -> dict[str, object]:
    media_part = {
        "fileData": {
            "fileUri": f"gs://{bucket_name}/{key}",
            "mimeType": _guess_key_mime_type(key),
        }
    }
    parts: list[dict[str, object]] = [media_part]
    prompt = prompt_text()
    if prompt:
        parts.append({"text": prompt})

    return {
        "key": key,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
            "generationConfig": _build_retry_batch_generation_config(
                for_vertex=for_vertex
            ),
        },
    }


def _build_retry_anthropic_manifest_line(key: str) -> dict[str, str]:
    return {
        "key": key,
        "mime_type": _guess_key_mime_type(key),
        "custom_id": _anthropic_custom_id_for_key(key),
    }


def _write_retry_requests_file(
    *,
    keys: list[str],
    output_path: Path,
    provider: str,
    bucket_name: str,
    for_vertex: bool,
) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        for key in keys:
            if provider == "anthropic":
                payload: dict[str, object] = _build_retry_anthropic_manifest_line(key)
            else:
                payload = _build_retry_gemini_request_line(
                    key,
                    bucket_name=bucket_name,
                    for_vertex=for_vertex,
                )
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")


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
            key = _normalize_key(payload.get("key"))
            if not key:
                raise ValueError(
                    f"Missing 'key' in anthropic manifest at line {line_number}."
                )
            mime_type = _normalize_key(payload.get("mime_type")) or _guess_key_mime_type(
                key
            )
            custom_id = _normalize_key(
                payload.get("custom_id")
            ) or _anthropic_custom_id_for_key(key)
            entries.append(
                {
                    "key": key,
                    "mime_type": mime_type,
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


def _build_anthropic_batch_requests_for_retry(
    *,
    bucket: storage.Bucket,
    requests_path: Path,
    model_name: str,
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
            "model": model_name,
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
    *,
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
    *,
    bucket_name: str,
    run_dir_name: str,
    chunk_label: str | None = None,
) -> str:
    prefix = _normalize_prefix(config.batch_outputs_gcs_prefix or "")
    base = f"{prefix}{run_dir_name}".rstrip("/")
    if chunk_label:
        base = f"{base}/{chunk_label.strip('/')}"
    return f"gs://{bucket_name}/{base}"


def _count_requests_file(path: Path) -> tuple[int, int]:
    total_bytes = path.stat().st_size if path.exists() else 0
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count, total_bytes


def _split_keys_evenly(keys: list[str], num_batches: int) -> list[list[str]]:
    if not keys:
        return []
    if num_batches <= 1:
        return [keys]

    target_batches = min(num_batches, len(keys))
    base_size, remainder = divmod(len(keys), target_batches)
    chunks: list[list[str]] = []
    start = 0
    for index in range(target_batches):
        size = base_size + (1 if index < remainder else 0)
        end = start + size
        if size > 0:
            chunks.append(keys[start:end])
        start = end
    return chunks


def _retry_requests_file_name(
    base_name: str,
    *,
    chunk_index: int,
    total_chunks: int,
) -> str:
    if total_chunks <= 1:
        return base_name
    base = Path(base_name)
    suffix = base.suffix or ".jsonl"
    return f"{base.stem}.part{chunk_index:03d}-of-{total_chunks:03d}{suffix}"


def _chunk_label(*, chunk_index: int, total_chunks: int) -> str:
    return f"chunk_{chunk_index:03d}_of_{total_chunks:03d}"


def _write_retry_keys_file(
    *,
    path: Path,
    keys: list[str],
    reasons_by_key: dict[str, str],
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for key in keys:
            payload = {
                "key": key,
                "reason": reasons_by_key.get(key, "unknown_failure"),
            }
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")


def _write_retry_batch_job_meta(
    *,
    run_dir: Path,
    jobs: list[dict[str, object]],
    client_backend: str,
    vertex_location: str | None,
    provider: str,
    retry_source_run: str | None,
    retry_source_run_id: str | None,
    job_group_id: str,
    retry_source_batch_names: list[str],
    retry_failed_keys_file: str,
    num_batches_requested: int,
) -> None:
    if not jobs:
        raise ValueError("Cannot write retry metadata without batch jobs.")

    request_count = sum(int(job.get("request_count") or 0) for job in jobs)
    request_bytes = sum(int(job.get("request_bytes") or 0) for job in jobs)
    first = jobs[0]

    meta: dict[str, object] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "job_group_id": job_group_id,
        "job_group_role": "retry",
        "batch_job_name": first.get("batch_job_name"),
        "batch_job_names": [
            job.get("batch_job_name") for job in jobs if job.get("batch_job_name")
        ],
        "batch_jobs": jobs,
        "request_count": request_count,
        "request_bytes": request_bytes,
        "input_file": first.get("input_file"),
        "input_source": first.get("input_source"),
        "output_destination": first.get("output_destination"),
        "model": first.get("model") or config.model,
        "provider": provider,
        "client_backend": client_backend,
        "vertex_location": vertex_location,
        "num_batches_requested": int(num_batches_requested),
        "num_batches_submitted": len(jobs),
        "retry_failed_keys_file": retry_failed_keys_file,
        "retry_source_run_id": retry_source_run_id,
        "retry_source_batch_names": retry_source_batch_names,
    }
    if retry_source_run:
        meta["retry_source_run"] = retry_source_run

    (run_dir / "batch_job.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _source_job_group_id(source_payload: dict, submit_run_dir: Path | None) -> str:
    value = source_payload.get("job_group_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    if submit_run_dir is not None:
        return submit_run_dir.name
    return datetime.now().strftime("batch_%Y%m%d_%H%M%S")


def _retry_attempt_label(source_payload: dict) -> str:
    retry_runs = source_payload.get("retry_runs")
    count = len(retry_runs) if isinstance(retry_runs, list) else 0
    return f"retry_failed_{count + 1:03d}"


def _append_retry_to_source_metadata(
    *,
    submit_run_dir: Path | None,
    retry_run_dir: Path,
    jobs_to_append: list[dict[str, object]],
    retry_failed_keys_file: str,
) -> None:
    if submit_run_dir is None:
        return
    source_path = submit_run_dir / "batch_job.json"
    source_payload = _read_batch_job_payload(source_path)
    if not source_payload:
        return

    retry_jobs = [job for job in jobs_to_append if job.get("batch_job_name")]
    if not retry_jobs:
        return

    source_payload.setdefault("job_group_id", _source_job_group_id(source_payload, submit_run_dir))
    source_payload.setdefault("job_group_role", "root")
    jobs = source_payload.get("batch_jobs")
    if not isinstance(jobs, list):
        jobs = []
        source_payload["batch_jobs"] = jobs

    existing_names = {
        item.get("batch_job_name")
            for item in jobs
            if isinstance(item, dict) and item.get("batch_job_name")
    }

    attempt_label = _retry_attempt_label(source_payload)
    total_retry_chunks = len(retry_jobs)
    chunk_indices = [
        int(item.get("chunk_index") or 0)
        for item in jobs
        if isinstance(item, dict)
    ]
    next_chunk_index = max(chunk_indices or [0]) + 1
    appended_names: list[object] = []
    for retry_index, job in enumerate(retry_jobs, start=1):
        job_name = job.get("batch_job_name")
        if job_name in existing_names:
            continue
        retry_entry = dict(job)
        retry_label = (
            attempt_label
            if total_retry_chunks == 1
            else f"{attempt_label}_{retry_index:03d}_of_{total_retry_chunks:03d}"
        )
        retry_entry.update(
            {
                "chunk_index": next_chunk_index,
                "total_chunks": next_chunk_index,
                "chunk_label": retry_label,
                "is_retry": True,
                "retry_run_dir": str(retry_run_dir),
                "retry_run_id": retry_run_dir.name,
                "retry_source_run": str(submit_run_dir),
                "retry_failed_keys_file": retry_failed_keys_file,
            }
        )
        jobs.append(retry_entry)
        existing_names.add(job_name)
        appended_names.append(job_name)
        next_chunk_index += 1

    names = [
        item.get("batch_job_name")
        for item in jobs
        if isinstance(item, dict) and item.get("batch_job_name")
    ]
    source_payload["batch_job_names"] = list(dict.fromkeys(names))
    retry_runs = source_payload.get("retry_runs")
    if not isinstance(retry_runs, list):
        retry_runs = []
        source_payload["retry_runs"] = retry_runs
    if not any(
        isinstance(item, dict) and item.get("run_dir") == str(retry_run_dir)
        for item in retry_runs
    ):
        retry_runs.append(
            {
                "run_dir": str(retry_run_dir),
                "run_id": retry_run_dir.name,
                "batch_job_name": retry_jobs[0].get("batch_job_name"),
                "batch_job_names": appended_names
                or [job.get("batch_job_name") for job in retry_jobs],
                "batch_count": total_retry_chunks,
                "request_count": sum(
                    int(job.get("request_count") or 0) for job in retry_jobs
                ),
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
        )

    source_path.write_text(
        json.dumps(source_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _collect_failed_retry_keys(
    *,
    expected_success_keys: set[str],
    successful_page_keys: set[str],
    failures: dict[str, str],
) -> tuple[set[str], dict[str, str]]:
    failed_keys = set(expected_success_keys - successful_page_keys)
    if not failed_keys:
        failed_keys = {
            key
            for key in failures
            if isinstance(key, str) and key and not key.startswith("<")
        }

    reasons: dict[str, str] = {}
    for key in failed_keys:
        reason = failures.get(key)
        if not reason:
            if key in expected_success_keys:
                reason = "missing_success_output"
            else:
                reason = "unknown_failure"
        reasons[key] = reason
    return failed_keys, reasons


def _submit_failed_pages_as_batch(
    *,
    failed_keys: set[str],
    failure_reasons: dict[str, str],
    provider: str,
    client,
    batch_names: list[str],
    submit_run_dir: Path | None,
    log,
    num_batches: int = 1,
) -> tuple[Path, list[str], int] | None:
    if not failed_keys:
        return None
    if num_batches <= 0:
        raise ValueError(f"num_batches must be >= 1 (received {num_batches}).")

    bucket_name = _require_bucket_name()
    normalized_reasons: dict[str, str] = {}
    skipped_keys = 0
    for key in sorted(failed_keys):
        normalized = _normalize_retry_object_key(key, bucket_name=bucket_name)
        if not normalized:
            skipped_keys += 1
            continue
        if normalized not in normalized_reasons:
            normalized_reasons[normalized] = failure_reasons.get(
                key, "unknown_failure"
            )

    normalized_keys = sorted(normalized_reasons)
    if not normalized_keys:
        log("Failed-page retry batch skipped: no valid GCS object keys available.")
        return None
    if skipped_keys:
        log(
            "Failed-page retry batch skipped invalid/non-matching keys: "
            f"{skipped_keys} key(s)."
        )

    run_dir = create_subfolder(config.output_root, category="submit")
    retry_log = get_run_logger(run_dir)
    model_name = str(config.model).strip()
    source_payload: dict = {}
    if submit_run_dir is not None:
        source_payload = _read_batch_job_payload(submit_run_dir / "batch_job.json")
        source_model = _normalize_key((source_payload or {}).get("model"))
        if source_model:
            model_name = source_model
    if not model_name:
        raise ValueError("Retry batch model is empty. Set config.model.")
    job_group_id = _source_job_group_id(source_payload, submit_run_dir)
    key_chunks = _split_keys_evenly(normalized_keys, num_batches)
    total_chunks = len(key_chunks)

    retry_keys_file = run_dir / "failed_keys.jsonl"
    _write_retry_keys_file(
        path=retry_keys_file,
        keys=normalized_keys,
        reasons_by_key=normalized_reasons,
    )

    bucket: storage.Bucket | None = None
    if provider == "anthropic" or bool(getattr(client, "vertexai", False)):
        service_account_path = resolve_service_account_path(
            _require_service_account_file()
        )
        storage_client = storage.Client.from_service_account_json(
            str(service_account_path)
        )
        bucket = storage_client.bucket(bucket_name)

    jobs: list[dict[str, object]] = []
    client_backend = "anthropic" if provider == "anthropic" else "mldev"
    vertex_location = None
    if provider != "anthropic" and bool(getattr(client, "vertexai", False)):
        client_backend = "vertex"
        vertex_location = (
            _extract_location_from_batch_name(batch_names[0])
            if batch_names
            else (config.vertex_model_location or "").strip()
            or (config.gcp_location or "").strip()
            or None
        )

    for chunk_index, chunk_keys in enumerate(key_chunks, start=1):
        chunk_label = _chunk_label(
            chunk_index=chunk_index,
            total_chunks=total_chunks,
        )
        requests_file_name = _retry_requests_file_name(
            config.batch_requests_file_name,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
        )
        requests_path = run_dir / requests_file_name
        _write_retry_requests_file(
            keys=chunk_keys,
            output_path=requests_path,
            provider=provider,
            bucket_name=bucket_name,
            for_vertex=bool(getattr(client, "vertexai", False)),
        )
        request_count, request_bytes = _count_requests_file(requests_path)
        retry_log(
            "Prepared failed-page retry request file "
            f"{requests_file_name} with {request_count} key(s)."
        )

        display_name = f"{config.batch_job_display_name}-failed-retry"
        if total_chunks > 1:
            display_name = f"{display_name}-{chunk_label}"
        input_source = "gcs"
        input_ref = ""
        output_destination: str | None = None

        if provider == "anthropic":
            if bucket is None:
                raise RuntimeError(
                    "Anthropic failed-page retry submission requires GCS bucket access."
                )
            requests = _build_anthropic_batch_requests_for_retry(
                bucket=bucket,
                requests_path=requests_path,
                model_name=model_name,
            )
            batch_job = client.messages.batches.create(requests=requests)
            batch_job_name = batch_job.id
            input_source = "anthropic_manifest"
            input_ref = requests_path.name
            retry_log(
                "Submitted failed-page retry Anthropic batch with "
                f"{len(requests)} request(s). batch_id={batch_job_name}"
            )
        else:
            if bool(getattr(client, "vertexai", False)):
                if bucket is None:
                    raise RuntimeError(
                        "Vertex failed-page retry submission requires GCS bucket access."
                    )
                input_ref = _upload_requests_to_gcs(
                    bucket=bucket,
                    run_dir_name=run_dir.name,
                    local_requests_path=requests_path,
                )
                output_destination = _output_dest_gcs_uri(
                    bucket_name=bucket_name,
                    run_dir_name=run_dir.name,
                    chunk_label=f"retry_failed/{chunk_label}",
                )
                batch_job = client.batches.create(
                    model=model_name,
                    src=input_ref,
                    config=types.CreateBatchJobConfig(
                        display_name=display_name,
                        dest=output_destination,
                    ),
                )
                batch_job_name = batch_job.name
                retry_log(
                    f"Submitted failed-page retry Vertex batch_id={batch_job_name} "
                    f"input={input_ref} output={output_destination}"
                )
            else:
                uploaded_file = client.files.upload(
                    file=str(requests_path),
                    config=types.UploadFileConfig(
                        display_name=f"{display_name}-requests",
                        mime_type="jsonl",
                    ),
                )
                input_source = "gemini_files"
                input_ref = uploaded_file.name
                batch_job = client.batches.create(
                    model=model_name,
                    src=uploaded_file.name,
                    config=types.CreateBatchJobConfig(display_name=display_name),
                )
                batch_job_name = batch_job.name
                retry_log(
                    f"Submitted failed-page retry mldev batch_id={batch_job_name} "
                    f"input={input_ref}"
                )

        jobs.append(
            {
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "chunk_label": chunk_label,
                "requests_file": requests_file_name,
                "request_count": request_count,
                "request_bytes": request_bytes,
                "batch_job_name": batch_job_name,
                "input_file": input_ref,
                "input_source": input_source,
                "output_destination": output_destination,
                "provider": provider,
                "model": model_name,
                "is_retry": True,
                "retry_run_dir": str(run_dir),
                "retry_run_id": run_dir.name,
                "retry_source_run": str(submit_run_dir) if submit_run_dir else None,
                "retry_source_run_id": submit_run_dir.name if submit_run_dir else None,
                "job_group_id": job_group_id,
            }
        )

    _append_retry_to_source_metadata(
        submit_run_dir=submit_run_dir,
        retry_run_dir=run_dir,
        jobs_to_append=jobs,
        retry_failed_keys_file=retry_keys_file.name,
    )
    _write_retry_batch_job_meta(
        run_dir=run_dir,
        jobs=jobs,
        client_backend=client_backend,
        vertex_location=vertex_location,
        provider=provider,
        retry_source_run=str(submit_run_dir) if submit_run_dir else None,
        retry_source_run_id=submit_run_dir.name if submit_run_dir else None,
        job_group_id=job_group_id,
        retry_source_batch_names=batch_names,
        retry_failed_keys_file=retry_keys_file.name,
        num_batches_requested=num_batches,
    )
    batch_job_names = [
        str(job.get("batch_job_name")) for job in jobs if job.get("batch_job_name")
    ]
    total_request_count = sum(int(job.get("request_count") or 0) for job in jobs)
    log(
        f"Submitted failed-page retry batch with {total_request_count} key(s) "
        f"across {len(batch_job_names)} chunk(s): "
        f"{', '.join(batch_job_names)} (run_dir={run_dir})"
    )
    return run_dir, batch_job_names, total_request_count
