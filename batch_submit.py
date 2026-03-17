from __future__ import annotations

import json
import mimetypes
from datetime import datetime
from pathlib import Path

from google.genai import types
from google.cloud import storage
from tqdm import tqdm

from batch_client import get_batch_client, resolve_service_account_path
from config import config
from tools import create_subfolder, get_run_logger
from upload import upload_missing_pdfs


def _pages_prefix() -> str:
    return _normalize_prefix(config.gcs_pages_prefix or "")


def _list_input_blobs(bucket: storage.Bucket) -> list[storage.Blob]:
    _assert_gcs_input_source(config.batch_input_source)
    override_prefix = _normalize_prefix(config.batch_input_prefix or "")
    pages_prefix = _pages_prefix()
    allowed = _allowed_extensions()

    if override_prefix:
        return _list_image_blobs_for_prefix(bucket, override_prefix, allowed)

    if not config.batch_use_local_pdf_folders:
        return _list_image_blobs_for_prefix(bucket, pages_prefix, allowed)

    local_pdf_paths = _list_local_pdf_paths(config.target_folder)
    if not local_pdf_paths:
        return _list_image_blobs_for_prefix(bucket, pages_prefix, allowed)

    collected: list[storage.Blob] = []
    missing: list[str] = []
    for pdf_path in local_pdf_paths:
        folder_prefix = f"{pages_prefix}{pdf_path.name}/"
        blobs = _list_image_blobs_for_prefix(bucket, folder_prefix, allowed)
        if blobs:
            collected.extend(blobs)
        else:
            missing.append(folder_prefix)

    if not collected:
        missing_text = ", ".join(missing) if missing else "(none)"
        raise FileNotFoundError(
            f"No uploaded page images found for local PDFs in bucket "
            f"{config.gcs_bucket_name}. Missing prefixes: {missing_text}"
        )

    return sorted(collected, key=lambda b: b.name)


def _assert_gcs_input_source(source: str) -> None:
    if (source or "").strip().lower() != "gcs":
        raise ValueError(
            "Only GCS input source is supported. Set config.batch_input_source = 'gcs'."
        )


def _normalize_prefix(prefix: str) -> str:
    value = prefix.strip()
    if not value:
        return ""
    value = value.strip("/")
    return f"{value}/"


def _list_local_pdf_paths(target_folder: str | None) -> list[Path]:
    if not target_folder:
        return []
    folder = Path(target_folder).expanduser()
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(path for path in folder.glob("*.pdf") if path.is_file())


def _allowed_extensions() -> set[str]:
    return {ext.lower().lstrip(".") for ext in config.batch_input_extensions}


def _ensure_uploaded_sources(bucket: storage.Bucket, log) -> list[str]:
    if not config.batch_auto_upload_missing:
        return []
    if not config.batch_use_local_pdf_folders:
        return []

    local_pdf_paths = _list_local_pdf_paths(config.target_folder)
    if not local_pdf_paths:
        return []

    uploaded = upload_missing_pdfs(pdf_paths=local_pdf_paths, bucket=bucket)
    if uploaded:
        log(f"Uploaded missing page images for PDF folders: {', '.join(uploaded)}")
    return uploaded


def _list_image_blobs_for_prefix(
    bucket: storage.Bucket,
    prefix: str,
    allowed_extensions: set[str],
) -> list[storage.Blob]:
    blobs = list(bucket.list_blobs(prefix=prefix or None))
    filtered: list[storage.Blob] = []
    for blob in blobs:
        name = blob.name
        if name.endswith("/"):
            continue
        suffix = Path(name).suffix.lower().lstrip(".")
        if suffix in allowed_extensions:
            filtered.append(blob)
    return filtered


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


def _inline_json_refs(node: object, root: object, stack: set[str] | None = None) -> object:
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
    cfg = {"responseMimeType": config.response_mime_type}

    if not config.batch_include_response_schema:
        return cfg

    schema_payload: object = config.output_schema
    if for_vertex:
        schema_payload = _vertex_compatible_schema(config.output_schema)

    schema_field = config.response_schema_field
    if schema_field:
        if for_vertex and schema_field == "response_json_schema":
            cfg["responseSchema"] = schema_payload
        elif schema_field == "response_json_schema":
            cfg["responseJsonSchema"] = schema_payload
        elif schema_field == "response_schema":
            cfg["responseSchema"] = schema_payload
        else:
            cfg[schema_field] = schema_payload
    return cfg


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
    prompt = config.input_prompt
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


def _output_dest_gcs_uri(bucket_name: str, run_dir_name: str) -> str:
    prefix = _normalize_prefix(config.batch_outputs_gcs_prefix or "")
    return f"gs://{bucket_name}/{prefix}{run_dir_name}"


def _write_requests_file(
    blobs: list[storage.Blob],
    bucket_name: str,
    output_path: Path,
    log,
    *,
    for_vertex: bool,
) -> tuple[int, int]:
    max_bytes = int(config.batch_input_max_bytes or 0)
    total_bytes = 0
    count = 0

    with open(output_path, "w", encoding="utf-8") as handle:
        for blob in tqdm(blobs, desc="Building batch JSONL", unit="img"):
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
    if for_vertex and not config.batch_include_response_schema:
        log("Vertex batch input omits response schema (config.batch_include_response_schema=False).")
    if for_vertex and config.batch_include_response_schema and _schema_has_refs(config.output_schema):
        log("Vertex batch input inlines schema refs and strips $-keys for compatibility.")
    return count, total_bytes


def submit_batch() -> None:
    if not (config.service_account_file or "").strip():
        raise ValueError(
            "config.service_account_file is empty. "
            "Set it to your GCP service account JSON path."
        )
    if not (config.gcs_bucket_name or "").strip():
        raise ValueError(
            "config.gcs_bucket_name is empty. "
            "Set the GCS bucket used for batch request/input/output files."
        )

    service_account_path = resolve_service_account_path(
        config.service_account_file
    )
    storage_client = storage.Client.from_service_account_json(
        str(service_account_path)
    )
    bucket = storage_client.bucket(config.gcs_bucket_name)
    run_dir = create_subfolder(config.output_root, prefix="submit_")
    log = get_run_logger(run_dir)

    _ensure_uploaded_sources(bucket, log)

    blobs = _list_input_blobs(bucket)
    if not blobs:
        raise FileNotFoundError(
            f"No input images found in bucket {config.gcs_bucket_name} "
            f"with prefix '{config.batch_input_prefix}'."
        )

    client = get_batch_client()

    requests_path = run_dir / config.batch_requests_file_name
    request_count, request_bytes = _write_requests_file(
        blobs=blobs,
        bucket_name=config.gcs_bucket_name,
        output_path=requests_path,
        log=log,
        for_vertex=bool(client.vertexai),
    )

    input_source = "gcs"
    input_ref = ""
    dest_ref = ""

    if client.vertexai:
        input_ref = _upload_requests_to_gcs(
            bucket=bucket,
            run_dir_name=run_dir.name,
            local_requests_path=requests_path,
        )
        dest_ref = _output_dest_gcs_uri(
            bucket_name=config.gcs_bucket_name,
            run_dir_name=run_dir.name,
        )
        batch_job = client.batches.create(
            model=config.model,
            src=input_ref,
            config=types.CreateBatchJobConfig(
                display_name=config.batch_job_display_name,
                dest=dest_ref,
            ),
        )
        log(
            f"Uploaded batch request file to {input_ref} "
            f"with output destination {dest_ref}."
        )
    else:
        uploaded_file = client.files.upload(
            file=str(requests_path),
            config=types.UploadFileConfig(
                display_name=f"{config.batch_job_display_name}-requests",
                mime_type="jsonl",
            ),
        )
        input_ref = uploaded_file.name
        input_source = "gemini_files"
        batch_job = client.batches.create(
            model=config.model,
            src=uploaded_file.name,
            config=types.CreateBatchJobConfig(
                display_name=config.batch_job_display_name
            ),
        )

    meta = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "batch_job_name": batch_job.name,
        "input_file": input_ref,
        "request_count": request_count,
        "request_bytes": request_bytes,
        "input_source": input_source,
        "output_destination": dest_ref or None,
        "model": config.model,
        "client_backend": "vertex" if client.vertexai else "mldev",
    }
    (run_dir / "batch_job.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log(f"Submitted batch job {batch_job.name}.")
    print(f"Batch job submitted: {batch_job.name}")


if __name__ == "__main__":
    submit_batch()
