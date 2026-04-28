from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime
from pathlib import Path

from google.genai import types
from google.cloud import storage

from patientjournals.batch.client import get_batch_client, resolve_service_account_path
from patientjournals.config import config
from patientjournals.config.models import resolve_model_spec
from patientjournals.batch.submit_requests import (
    _build_anthropic_batch_requests,
    _count_requests_file,
    _output_dest_gcs_uri,
    _upload_requests_to_gcs,
    _write_requests_file,
)
from patientjournals.batch.submit_inputs import (
    _ensure_uploaded_sources,
    _list_input_blobs,
)
from patientjournals.shared.dataset_coverage import (
    load_dataset_key_coverage,
    normalize_gcs_file_key,
    resolve_continue_dataset_path,
)
from patientjournals.shared.tools import create_subfolder, get_run_logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit batch job(s) from GCS image inputs."
    )
    parser.add_argument(
        "-n",
        "--num-batches",
        dest="num_batches",
        type=int,
        help=(
            "Split inputs into N smaller batch jobs. Overrides config.batch_num_chunks."
        ),
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help=(
            "Only resubmit chunks that are not successful from the latest "
            "submit run (or --run-dir)."
        ),
    )
    parser.add_argument(
        "--run-dir",
        dest="run_dir",
        help=(
            "Submit run directory to resume. "
            "Used with --rerun to choose which run to resume."
        ),
    )
    parser.add_argument(
        "--continue-dataset",
        "--missing-from-dataset",
        dest="continue_dataset",
        help=(
            "Submit only GCS input pages not already covered by this dataset "
            "(by file_name), or pass 'newest'."
        ),
    )
    parser.add_argument(
        "--downscale",
        dest="downscale",
        type=float,
        help=(
            "Randomly sample this fraction of GCS inputs before batching. "
            "Must be > 0 and <= 1. Example: --downscale 0.1"
        ),
    )
    return parser.parse_args()


def _resolve_num_batches(args: argparse.Namespace) -> int:
    value = args.num_batches
    if value is None:
        value = int(config.batch_num_chunks or 1)
    if value <= 0:
        raise ValueError(f"num_batches must be >= 1 (received {value}).")
    return value


def _validate_batch_model_support() -> str:
    spec = resolve_model_spec(config.model)
    if not spec.supports_batch:
        raise ValueError(
            "Configured model does not support batch jobs in this pipeline. "
            f"Resolved model='{config.model}' provider='{spec.provider}' "
            f"supports_batch={spec.supports_batch}."
        )
    if spec.provider not in {"gemini", "anthropic"}:
        raise ValueError(
            "Batch submission currently supports provider-specific batch paths "
            f"for Gemini and Anthropic only (resolved provider='{spec.provider}')."
        )
    return spec.provider


def _get_anthropic_client():
    try:
        import anthropic
    except Exception as exc:
        raise RuntimeError(
            "Anthropic provider requested but the 'anthropic' package is unavailable."
        ) from exc

    api_key = config.api_key_for_provider("anthropic")
    return anthropic.Anthropic(api_key=api_key)


def _resolve_downscale(args: argparse.Namespace) -> float | None:
    value = args.downscale
    if value is None:
        return None
    if value <= 0.0 or value > 1.0:
        raise ValueError(f"downscale must be > 0 and <= 1 (received {value}).")
    return float(value)


def _warn_if_confidence_scores_unsupported(*, provider: str, log) -> None:
    if not bool(config.include_confidence_scores):
        return
    if provider == "anthropic":
        log(
            "include_confidence_scores=True requested, but Anthropic Messages API "
            "does not expose token logprobs in responses. "
            "Field-level confidence output will be empty."
        )


def _downscale_blobs_randomly(
    blobs: list[storage.Blob],
    *,
    downscale: float,
) -> list[storage.Blob]:
    if not blobs or downscale >= 1.0:
        return blobs

    total = len(blobs)
    target = int(round(total * downscale))
    target = max(1, min(total, target))
    if target >= total:
        return blobs

    sampled = random.sample(blobs, k=target)
    return sorted(sampled, key=lambda item: item.name)


def _filter_blobs_missing_from_dataset(
    blobs: list[storage.Blob],
    *,
    dataset_path: Path,
    bucket_name: str,
    log,
) -> tuple[list[storage.Blob], int, int]:
    _, covered_keys, dataset_rows = load_dataset_key_coverage(
        dataset_path,
        csv_sep=config.csv_sep,
        bucket_name=bucket_name,
    )
    missing = [
        blob
        for blob in blobs
        if normalize_gcs_file_key(blob.name, bucket_name=bucket_name) not in covered_keys
    ]
    covered_inputs = len(blobs) - len(missing)
    log(
        f"Continuing from dataset {dataset_path}: rows={dataset_rows}, "
        f"covered_inputs={covered_inputs}/{len(blobs)}, missing={len(missing)}."
    )
    return missing, covered_inputs, dataset_rows


def _split_blobs_evenly(
    blobs: list[storage.Blob],
    num_batches: int,
) -> list[list[storage.Blob]]:
    if not blobs:
        return []

    if num_batches <= 1:
        return [blobs]

    target_batches = min(num_batches, len(blobs))
    base_size, remainder = divmod(len(blobs), target_batches)
    chunks: list[list[storage.Blob]] = []
    start = 0
    for index in range(target_batches):
        size = base_size + (1 if index < remainder else 0)
        end = start + size
        if size > 0:
            chunks.append(blobs[start:end])
        start = end
    return chunks


def _chunk_requests_file_name(
    base_name: str,
    *,
    chunk_index: int,
    total_chunks: int,
) -> str:
    base = Path(base_name)
    suffix = base.suffix or ".jsonl"
    stem = base.stem
    return f"{stem}.part{chunk_index:03d}-of-{total_chunks:03d}{suffix}"


def _chunk_label(*, chunk_index: int, total_chunks: int) -> str:
    return f"chunk_{chunk_index:03d}_of_{total_chunks:03d}"


def _read_batch_job_payload(path: Path) -> dict:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"batch job metadata file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid batch metadata in {path}: expected object")
    return payload


def _read_batch_job_payload_if_exists(path: Path) -> dict | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _parse_chunk_file_name(file_name: str, *, base_name: str) -> tuple[int, int] | None:
    base = Path(base_name)
    suffix = base.suffix or ".jsonl"
    stem = base.stem
    pattern = re.compile(
        rf"^{re.escape(stem)}\.part(?P<index>\d+)-of-(?P<total>\d+){re.escape(suffix)}$"
    )
    match = pattern.fullmatch(file_name)
    if not match:
        return None
    try:
        chunk_index = int(match.group("index"))
        total_chunks = int(match.group("total"))
    except Exception:
        return None
    if chunk_index <= 0 or total_chunks <= 0:
        return None
    return chunk_index, total_chunks


def _discover_request_files_in_run_dir(run_dir: Path) -> tuple[dict[int, str], int]:
    by_index: dict[int, str] = {}
    inferred_total = 0
    base_name = config.batch_requests_file_name

    candidate = run_dir / base_name
    if candidate.exists() and candidate.is_file():
        by_index[1] = base_name
        inferred_total = max(inferred_total, 1)

    for path in sorted(run_dir.glob("*.jsonl")):
        parsed = _parse_chunk_file_name(path.name, base_name=base_name)
        if not parsed:
            continue
        chunk_index, total_chunks = parsed
        by_index[chunk_index] = path.name
        inferred_total = max(inferred_total, total_chunks)

    if by_index:
        inferred_total = max(inferred_total, max(by_index))
    return by_index, inferred_total


def _submitted_batch_ids_by_chunk_from_run_log(
    run_dir: Path,
) -> tuple[dict[int, str], int]:
    path = run_dir / "run.log"
    if not path.exists() or not path.is_file():
        return {}, 0

    by_index: dict[int, str] = {}
    inferred_total = 0
    pattern = re.compile(
        r"\[(chunk_(?P<index>\d+)_of_(?P<total>\d+))\].*batch_id=(?P<batch_id>\S+)"
    )
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line)
            if not match:
                continue
            try:
                chunk_index = int(match.group("index"))
                total_chunks = int(match.group("total"))
            except Exception:
                continue
            batch_id = str(match.group("batch_id") or "").strip()
            if chunk_index <= 0 or total_chunks <= 0 or not batch_id:
                continue
            by_index[chunk_index] = batch_id
            inferred_total = max(inferred_total, total_chunks)
    return by_index, inferred_total


def _latest_submit_run_dir(output_root: str) -> Path | None:
    root = Path(output_root).expanduser()
    if not root.exists() or not root.is_dir():
        return None
    run_dirs = sorted(
        (
            item
            for item in root.iterdir()
            if item.is_dir() and item.name.startswith("submit_")
        ),
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


def _resolve_rerun_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser()
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(
                f"--run-dir not found or not a directory: {run_dir}"
            )
        return run_dir

    latest = _latest_submit_run_dir(config.output_root)
    if latest is None:
        raise FileNotFoundError(
            "No previous submit run found to rerun. "
            "Run `uv run invoke batch.submit` first (without --rerun)."
        )
    return latest


def _normalize_job_entries(payload: dict) -> list[dict]:
    entries: list[dict] = []
    raw_jobs = payload.get("batch_jobs")
    if isinstance(raw_jobs, list):
        for index, item in enumerate(raw_jobs, start=1):
            if not isinstance(item, dict):
                continue
            entry = dict(item)
            if "chunk_index" not in entry:
                entry["chunk_index"] = index
            entries.append(entry)

    if entries:
        return entries

    single_name = payload.get("batch_job_name")
    if isinstance(single_name, str) and single_name.strip():
        requests_file = payload.get("requests_file")
        if not isinstance(requests_file, str) or not requests_file.strip():
            requests_file = config.batch_requests_file_name
        return [
            {
                "chunk_index": 1,
                "total_chunks": 1,
                "chunk_label": "chunk_001_of_001",
                "requests_file": requests_file,
                "batch_job_name": single_name.strip(),
                "request_count": int(payload.get("request_count") or 0),
                "request_bytes": int(payload.get("request_bytes") or 0),
                "input_file": payload.get("input_file"),
                "input_source": payload.get("input_source"),
                "output_destination": payload.get("output_destination"),
                "provider": payload.get("provider"),
            }
        ]
    return []


def _infer_rerun_total_chunks(
    *,
    payload: dict | None,
    payload_entries: list[dict],
    request_files_total: int,
    run_log_total: int,
) -> int:
    totals: list[int] = []
    if payload is not None:
        try:
            value = int(payload.get("num_batches_requested") or 0)
        except Exception:
            value = 0
        if value > 0:
            totals.append(value)

    if payload_entries:
        for entry in payload_entries:
            try:
                value = int(entry.get("total_chunks") or 0)
            except Exception:
                value = 0
            if value > 0:
                totals.append(value)

    if request_files_total > 0:
        totals.append(int(request_files_total))
    if run_log_total > 0:
        totals.append(int(run_log_total))

    return max(totals) if totals else 0


def _extract_downscale_from_run_log(run_dir: Path) -> float | None:
    path = run_dir / "run.log"
    if not path.exists() or not path.is_file():
        return None
    pattern = re.compile(r"Downscaled input set with fraction=(?P<value>[0-9eE+\-.]+)")
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line)
            if not match:
                continue
            try:
                value = float(match.group("value"))
            except Exception:
                continue
            if value > 0.0 and value <= 1.0:
                return value
    return None


def _ensure_requests_files_for_rerun(
    *,
    run_dir: Path,
    total_chunks: int,
    existing_files_by_index: dict[int, str],
    bucket: storage.Bucket,
    provider: str,
    client,
    log,
) -> dict[int, str]:
    if total_chunks <= 0:
        return dict(existing_files_by_index)

    expected_missing = [
        index
        for index in range(1, total_chunks + 1)
        if index not in existing_files_by_index
    ]
    if not expected_missing:
        return dict(existing_files_by_index)

    downscale = _extract_downscale_from_run_log(run_dir)
    if downscale is not None and downscale < 1.0:
        raise ValueError(
            "Cannot safely regenerate missing chunk request files for a downscaled run. "
            "Please rerun submit with the same --downscale value and stable input set, "
            "or manually recover from existing request files."
        )

    blobs = _list_input_blobs(bucket, log=log)
    if not blobs:
        raise FileNotFoundError(
            f"No input images found in bucket {config.gcs_bucket_name} "
            f"with prefix '{config.batch_input_prefix}'."
        )

    chunks = _split_blobs_evenly(blobs, total_chunks)
    if len(chunks) != total_chunks:
        raise RuntimeError(
            "Unable to regenerate missing request files for rerun because chunk "
            "reconstruction did not match expected total chunks."
        )

    files_by_index = dict(existing_files_by_index)
    for chunk_index in expected_missing:
        requests_file_name = (
            config.batch_requests_file_name
            if total_chunks == 1
            else _chunk_requests_file_name(
                config.batch_requests_file_name,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
            )
        )
        requests_path = run_dir / requests_file_name
        _write_requests_file(
            blobs=chunks[chunk_index - 1],
            bucket_name=config.gcs_bucket_name,
            output_path=requests_path,
            log=log,
            for_vertex=bool(getattr(client, "vertexai", False)),
            provider=provider,
        )
        files_by_index[chunk_index] = requests_file_name
        log(
            f"Regenerated missing request file for rerun: "
            f"chunk={chunk_index}/{total_chunks} file={requests_file_name}"
        )

    return files_by_index


def _build_rerun_entries(
    *,
    run_dir: Path,
    provider: str,
    payload_entries: list[dict],
    total_chunks: int,
    files_by_index: dict[int, str],
    submitted_by_chunk: dict[int, str],
) -> list[dict]:
    if total_chunks <= 0:
        return []

    by_chunk_payload: dict[int, dict] = {}
    for fallback_index, entry in enumerate(payload_entries, start=1):
        try:
            index = int(entry.get("chunk_index") or fallback_index)
        except Exception:
            index = fallback_index
        if index <= 0:
            continue
        by_chunk_payload[index] = dict(entry)

    entries: list[dict] = []
    for chunk_index in range(1, total_chunks + 1):
        entry = by_chunk_payload.get(chunk_index, {})

        requests_file = files_by_index.get(chunk_index)
        if not requests_file:
            requests_file = (
                config.batch_requests_file_name
                if total_chunks == 1
                else _chunk_requests_file_name(
                    config.batch_requests_file_name,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                )
            )

        requests_path = run_dir / requests_file
        if not requests_path.exists() or not requests_path.is_file():
            raise FileNotFoundError(
                f"Missing request file while preparing rerun metadata: {requests_path}"
            )

        request_count, request_bytes = _count_requests_file(requests_path)
        previous_name = str(entry.get("batch_job_name") or "").strip()
        if not previous_name:
            previous_name = submitted_by_chunk.get(chunk_index, "")

        input_source = str(entry.get("input_source") or "").strip()
        if not input_source:
            input_source = "anthropic_manifest" if provider == "anthropic" else "gcs"

        input_file = str(entry.get("input_file") or "").strip()
        if not input_file:
            input_file = requests_file if provider == "anthropic" else ""

        output_destination = entry.get("output_destination")
        if isinstance(output_destination, str):
            output_destination = output_destination.strip() or None

        rebuilt = _build_chunk_entry(
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            requests_file=requests_file,
            request_count=request_count,
            request_bytes=request_bytes,
            batch_job_name=previous_name,
            input_file=input_file,
            input_source=input_source,
            output_destination=output_destination,
            provider=provider,
        )
        history = entry.get("rerun_history")
        if isinstance(history, list):
            rebuilt["rerun_history"] = history

        entries.append(rebuilt)

    return entries


def _entries_with_replacement(
    entries: list[dict],
    *,
    chunk_index: int,
    replacement: dict,
) -> list[dict]:
    updated: list[dict] = []
    replaced = False
    for item in entries:
        try:
            current_index = int(item.get("chunk_index") or 0)
        except Exception:
            current_index = 0
        if current_index == chunk_index:
            updated.append(dict(replacement))
            replaced = True
        else:
            updated.append(dict(item))

    if not replaced:
        updated.append(dict(replacement))

    return sorted(updated, key=lambda item: int(item.get("chunk_index") or 0))


def _build_chunk_entry(
    *,
    chunk_index: int,
    total_chunks: int,
    requests_file: str,
    request_count: int,
    request_bytes: int,
    batch_job_name: str,
    input_file: str,
    input_source: str,
    output_destination: str | None,
    provider: str,
) -> dict:
    return {
        "chunk_index": int(chunk_index),
        "total_chunks": int(total_chunks),
        "chunk_label": _chunk_label(chunk_index=chunk_index, total_chunks=total_chunks),
        "requests_file": requests_file,
        "request_count": int(request_count),
        "request_bytes": int(request_bytes),
        "batch_job_name": batch_job_name,
        "input_file": input_file,
        "input_source": input_source,
        "output_destination": output_destination,
        "provider": provider,
    }


def _write_batch_job_meta(
    *,
    run_dir: Path,
    jobs: list[dict],
    num_batches_requested: int,
    client_backend: str,
    vertex_location: str | None,
    provider: str,
) -> None:
    if not jobs:
        raise ValueError("Cannot write batch metadata without jobs.")

    total_request_count = sum(int(item.get("request_count") or 0) for item in jobs)
    total_request_bytes = sum(int(item.get("request_bytes") or 0) for item in jobs)

    first = jobs[0]
    meta = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "batch_job_name": first.get("batch_job_name"),
        "batch_job_names": [
            item.get("batch_job_name") for item in jobs if item.get("batch_job_name")
        ],
        "batch_jobs": jobs,
        "request_count": total_request_count,
        "request_bytes": total_request_bytes,
        "input_file": first.get("input_file"),
        "input_source": first.get("input_source"),
        "output_destination": first.get("output_destination"),
        "model": config.model,
        "provider": provider,
        "client_backend": client_backend,
        "vertex_location": vertex_location,
        "num_batches_requested": int(num_batches_requested),
        "num_batches_submitted": len(jobs),
    }
    (run_dir / "batch_job.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _submit_chunk_job(
    *,
    client,
    provider: str,
    bucket: storage.Bucket,
    run_dir_name: str,
    requests_path: Path,
    chunk_index: int,
    total_chunks: int,
    attempt_tag: str | None,
    log,
) -> tuple[str, str, str, str | None]:
    input_source = "gcs"
    input_ref = ""
    dest_ref = None
    label = _chunk_label(chunk_index=chunk_index, total_chunks=total_chunks)
    destination_label = label
    if attempt_tag:
        destination_label = f"{label}/{attempt_tag.strip('/')}"

    display_name = (
        f"{config.batch_job_display_name}-{label}"
        if total_chunks > 1
        else config.batch_job_display_name
    )

    if provider == "anthropic":
        requests = _build_anthropic_batch_requests(
            bucket=bucket,
            requests_path=requests_path,
        )
        batch_job = client.messages.batches.create(requests=requests)
        input_source = "anthropic_manifest"
        input_ref = requests_path.name
        log(
            f"[{label}] Submitted Anthropic batch with {len(requests)} request(s). "
            f"signed_url_ttl_hours={max(1, int(config.anthropic_signed_url_ttl_hours or 48))} "
            f"batch_id={batch_job.id}"
        )
        return batch_job.id, input_source, input_ref, None

    if getattr(client, "vertexai", False):
        input_ref = _upload_requests_to_gcs(
            bucket=bucket,
            run_dir_name=run_dir_name,
            local_requests_path=requests_path,
        )
        dest_ref = _output_dest_gcs_uri(
            bucket_name=config.gcs_bucket_name,
            run_dir_name=run_dir_name,
            chunk_label=destination_label,
        )
        batch_job = client.batches.create(
            model=config.model,
            src=input_ref,
            config=types.CreateBatchJobConfig(
                display_name=display_name,
                dest=dest_ref,
            ),
        )
        log(
            f"[{label}] Uploaded request file to {input_ref} "
            f"with output destination {dest_ref}."
        )
        return batch_job.name, input_source, input_ref, dest_ref

    uploaded_file = client.files.upload(
        file=str(requests_path),
        config=types.UploadFileConfig(
            display_name=f"{display_name}-requests",
            mime_type="jsonl",
        ),
    )
    input_ref = uploaded_file.name
    input_source = "gemini_files"
    batch_job = client.batches.create(
        model=config.model,
        src=uploaded_file.name,
        config=types.CreateBatchJobConfig(display_name=display_name),
    )
    return batch_job.name, input_source, input_ref, None


def _batch_state_and_success(
    *,
    client,
    provider: str,
    batch_job_name: str,
) -> tuple[str, bool]:
    if provider == "anthropic":
        batch_job = client.messages.batches.retrieve(batch_job_name)
        status = (
            str(getattr(batch_job, "processing_status", "") or "unknown")
            .strip()
            .lower()
        )
        counts = getattr(batch_job, "request_counts", None)
        succeeded = int(getattr(counts, "succeeded", 0) or 0)
        errored = int(getattr(counts, "errored", 0) or 0)
        canceled = int(getattr(counts, "canceled", 0) or 0)
        expired = int(getattr(counts, "expired", 0) or 0)
        processing = int(getattr(counts, "processing", 0) or 0)
        state_text = (
            f"{status}(succeeded={succeeded},errored={errored},"
            f"canceled={canceled},expired={expired},processing={processing})"
        )
        successful = (
            status == "ended"
            and errored == 0
            and canceled == 0
            and expired == 0
            and processing == 0
        )
        return state_text, successful

    batch_job = client.batches.get(name=batch_job_name)
    state = str(getattr(batch_job, "state", None) or "UNKNOWN")
    return state, state == "JOB_STATE_SUCCEEDED"


def submit_batch() -> None:
    args = _parse_args()
    if args.rerun and args.continue_dataset:
        raise ValueError("--continue-dataset cannot be combined with --rerun.")

    provider = _validate_batch_model_support()
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

    service_account_path = resolve_service_account_path(config.service_account_file)
    storage_client = storage.Client.from_service_account_json(str(service_account_path))
    bucket = storage_client.bucket(config.gcs_bucket_name)

    vertex_location = (config.vertex_model_location or "").strip() or (
        config.gcp_location or ""
    ).strip()
    if provider == "anthropic":
        client = _get_anthropic_client()
        backend_name = "anthropic"
    else:
        client = get_batch_client(location=vertex_location)
        backend_name = "vertex" if getattr(client, "vertexai", False) else "mldev"

    if args.rerun:
        if args.num_batches is not None:
            print("--num-batches is ignored when --rerun is set.")
        if args.downscale is not None:
            print("--downscale is ignored when --rerun is set.")
        run_dir = _resolve_rerun_run_dir(args)
        log = get_run_logger(run_dir)
        _warn_if_confidence_scores_unsupported(provider=provider, log=log)
        payload = _read_batch_job_payload_if_exists(run_dir / "batch_job.json")
        payload_provider = str((payload or {}).get("provider") or "").strip().lower()
        if payload_provider in {"gemini", "anthropic"} and payload_provider != provider:
            provider = payload_provider
            if provider == "anthropic":
                client = _get_anthropic_client()
                backend_name = "anthropic"
            else:
                client = get_batch_client(location=vertex_location)
                backend_name = (
                    "vertex" if getattr(client, "vertexai", False) else "mldev"
                )
            log(
                f"Rerun provider override from metadata: provider={provider} "
                f"(model in config is '{config.model}')."
            )
            _warn_if_confidence_scores_unsupported(provider=provider, log=log)

        payload_entries = _normalize_job_entries(payload or {})
        files_by_index, files_total = _discover_request_files_in_run_dir(run_dir)
        submitted_by_chunk, run_log_total = _submitted_batch_ids_by_chunk_from_run_log(
            run_dir
        )
        total_chunks = _infer_rerun_total_chunks(
            payload=payload,
            payload_entries=payload_entries,
            request_files_total=files_total,
            run_log_total=run_log_total,
        )
        if total_chunks <= 0:
            raise ValueError(
                "Could not infer rerun chunk layout. Expected batch_job.json, "
                "chunked request files, or chunk submission lines in run.log."
            )

        files_by_index = _ensure_requests_files_for_rerun(
            run_dir=run_dir,
            total_chunks=total_chunks,
            existing_files_by_index=files_by_index,
            bucket=bucket,
            provider=provider,
            client=client,
            log=log,
        )
        entries = _build_rerun_entries(
            run_dir=run_dir,
            provider=provider,
            payload_entries=payload_entries,
            total_chunks=total_chunks,
            files_by_index=files_by_index,
            submitted_by_chunk=submitted_by_chunk,
        )
        if not entries:
            raise ValueError(f"No batch job entries found or derived in {run_dir}.")

        log(
            f"Rerun preparation complete for {run_dir.name}: "
            f"total_chunks={total_chunks}, discovered_request_files={len(files_by_index)}, "
            f"known_submitted_chunks={len(submitted_by_chunk)}."
        )
        print(
            f"Rerun prepared: {total_chunks} chunk(s), "
            f"{len(submitted_by_chunk)} with known batch IDs."
        )

        _write_batch_job_meta(
            run_dir=run_dir,
            jobs=entries,
            num_batches_requested=int(
                (payload or {}).get("num_batches_requested") or total_chunks
            ),
            client_backend=backend_name,
            vertex_location=vertex_location
            if getattr(client, "vertexai", False)
            else None,
            provider=provider,
        )

        rerun_attempt_tag = datetime.now().strftime("rerun_%Y%m%d_%H%M%S")
        current_entries = sorted(
            [dict(item) for item in entries],
            key=lambda item: int(item.get("chunk_index") or 0),
        )
        rerun_count = 0
        for fallback_index, entry in enumerate(current_entries, start=1):
            try:
                chunk_index = int(entry.get("chunk_index") or fallback_index)
            except Exception:
                chunk_index = fallback_index
            try:
                total_chunks = int(entry.get("total_chunks") or len(entries))
            except Exception:
                total_chunks = len(entries)
            label = _chunk_label(chunk_index=chunk_index, total_chunks=total_chunks)
            print(f"Checking {label} ({fallback_index}/{len(current_entries)})...")

            requests_file = entry.get("requests_file")
            if not isinstance(requests_file, str) or not requests_file.strip():
                if len(entries) == 1:
                    requests_file = config.batch_requests_file_name
                else:
                    requests_file = _chunk_requests_file_name(
                        config.batch_requests_file_name,
                        chunk_index=chunk_index,
                        total_chunks=total_chunks,
                    )
            requests_path = run_dir / requests_file
            if not requests_path.exists() or not requests_path.is_file():
                raise FileNotFoundError(
                    f"Missing request file for {label}: {requests_path}"
                )

            previous_name = str(entry.get("batch_job_name") or "").strip()
            state_text = "UNKNOWN"
            was_successful = False
            if previous_name:
                try:
                    state_text, was_successful = _batch_state_and_success(
                        client=client,
                        provider=provider,
                        batch_job_name=previous_name,
                    )
                except Exception as exc:
                    log(
                        f"[{label}] Could not resolve previous job {previous_name}; "
                        f"will resubmit chunk.",
                        exc=exc,
                    )

            if was_successful:
                log(
                    f"[{label}] Already successful. Keeping existing batch id {previous_name}."
                )
                _write_batch_job_meta(
                    run_dir=run_dir,
                    jobs=current_entries,
                    num_batches_requested=int(
                        (payload or {}).get("num_batches_requested") or total_chunks
                    ),
                    client_backend=backend_name,
                    vertex_location=vertex_location
                    if getattr(client, "vertexai", False)
                    else None,
                    provider=provider,
                )
                continue
            if previous_name:
                log(
                    f"[{label}] Previous job not successful: {previous_name} state={state_text}"
                )
            else:
                log(f"[{label}] No previous batch id recorded; submitting chunk.")

            print(f"Submitting {label}...")

            request_count, request_bytes = _count_requests_file(requests_path)
            batch_job_name, input_source, input_ref, dest_ref = _submit_chunk_job(
                client=client,
                provider=provider,
                bucket=bucket,
                run_dir_name=run_dir.name,
                requests_path=requests_path,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                attempt_tag=rerun_attempt_tag,
                log=log,
            )
            rerun_count += 1
            rebuilt = _build_chunk_entry(
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                requests_file=requests_file,
                request_count=request_count,
                request_bytes=request_bytes,
                batch_job_name=batch_job_name,
                input_file=input_ref,
                input_source=input_source,
                output_destination=dest_ref,
                provider=provider,
            )
            if previous_name and previous_name != batch_job_name:
                history = entry.get("rerun_history")
                if not isinstance(history, list):
                    history = []
                rebuilt["rerun_history"] = [*history, previous_name]
            current_entries = _entries_with_replacement(
                current_entries,
                chunk_index=chunk_index,
                replacement=rebuilt,
            )

            _write_batch_job_meta(
                run_dir=run_dir,
                jobs=current_entries,
                num_batches_requested=int(
                    (payload or {}).get("num_batches_requested") or total_chunks
                ),
                client_backend=backend_name,
                vertex_location=vertex_location
                if getattr(client, "vertexai", False)
                else None,
                provider=provider,
            )

        _write_batch_job_meta(
            run_dir=run_dir,
            jobs=current_entries,
            num_batches_requested=int(
                (payload or {}).get("num_batches_requested") or total_chunks
            ),
            client_backend=backend_name,
            vertex_location=vertex_location
            if getattr(client, "vertexai", False)
            else None,
            provider=provider,
        )
        if rerun_count == 0:
            log("Rerun requested, but all chunk jobs were already successful.")
            print("No chunks needed rerun; all jobs already succeeded.")
            return

        job_names = ", ".join(
            str(item.get("batch_job_name"))
            for item in current_entries
            if item.get("batch_job_name")
        )
        log(f"Rerun submitted {rerun_count} chunk job(s). Active jobs: {job_names}")
        print(f"Resubmitted {rerun_count} chunk job(s).")
        return

    run_dir = create_subfolder(config.output_root, prefix="submit_")
    log = get_run_logger(run_dir)
    _warn_if_confidence_scores_unsupported(provider=provider, log=log)
    _ensure_uploaded_sources(bucket, log)

    blobs = _list_input_blobs(bucket, log=log)
    if not blobs:
        raise FileNotFoundError(
            f"No input images found in bucket {config.gcs_bucket_name} "
            f"with prefix '{config.batch_input_prefix}'."
        )
    if args.continue_dataset:
        continue_dataset_path = resolve_continue_dataset_path(
            args.continue_dataset,
            run_root=config.output_root,
            dataset_name=config.dataset_file_name,
        )
        original_count = len(blobs)
        blobs, covered_inputs, _ = _filter_blobs_missing_from_dataset(
            blobs,
            dataset_path=continue_dataset_path,
            bucket_name=config.gcs_bucket_name,
            log=log,
        )
        print(
            "Continue dataset coverage: "
            f"{covered_inputs}/{original_count} input page(s) already covered; "
            f"submitting {len(blobs)} missing page(s)."
        )
        if not blobs:
            log(
                f"Continue dataset {continue_dataset_path} covers all "
                f"{original_count} selected input page(s); no batch submitted."
            )
            print("No missing pages to submit; dataset already covers selected inputs.")
            return

    downscale = _resolve_downscale(args)
    if downscale is not None:
        original_count = len(blobs)
        blobs = _downscale_blobs_randomly(blobs, downscale=downscale)
        sampled_count = len(blobs)
        log(
            f"Downscaled input set with fraction={downscale:.6g}. "
            f"Selected {sampled_count}/{original_count} input blob(s)."
        )
        print(
            f"Downscale applied ({downscale:.6g}): "
            f"{sampled_count}/{original_count} blob(s) selected."
        )

    num_batches_requested = _resolve_num_batches(args)
    chunks = _split_blobs_evenly(blobs, num_batches_requested)
    if not chunks:
        raise RuntimeError("No request chunks could be prepared for submission.")
    if len(chunks) < num_batches_requested:
        log(
            f"Requested {num_batches_requested} chunks, "
            f"but only {len(chunks)} non-empty chunks are possible for {len(blobs)} inputs."
        )

    chunk_jobs: list[dict] = []
    total_chunks = len(chunks)
    for chunk_index, chunk_blobs in enumerate(chunks, start=1):
        if total_chunks == 1:
            requests_file_name = config.batch_requests_file_name
        else:
            requests_file_name = _chunk_requests_file_name(
                config.batch_requests_file_name,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
            )
        requests_path = run_dir / requests_file_name
        request_count, request_bytes = _write_requests_file(
            blobs=chunk_blobs,
            bucket_name=config.gcs_bucket_name,
            output_path=requests_path,
            log=log,
            for_vertex=bool(getattr(client, "vertexai", False)),
            provider=provider,
        )
        batch_job_name, input_source, input_ref, dest_ref = _submit_chunk_job(
            client=client,
            provider=provider,
            bucket=bucket,
            run_dir_name=run_dir.name,
            requests_path=requests_path,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            attempt_tag=None,
            log=log,
        )
        chunk_jobs.append(
            _build_chunk_entry(
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                requests_file=requests_file_name,
                request_count=request_count,
                request_bytes=request_bytes,
                batch_job_name=batch_job_name,
                input_file=input_ref,
                input_source=input_source,
                output_destination=dest_ref,
                provider=provider,
            )
        )

        # Persist progress immediately so batch IDs are not lost on mid-run failure.
        _write_batch_job_meta(
            run_dir=run_dir,
            jobs=chunk_jobs,
            num_batches_requested=num_batches_requested,
            client_backend=backend_name,
            vertex_location=vertex_location
            if getattr(client, "vertexai", False)
            else None,
            provider=provider,
        )

    _write_batch_job_meta(
        run_dir=run_dir,
        jobs=chunk_jobs,
        num_batches_requested=num_batches_requested,
        client_backend=backend_name,
        vertex_location=vertex_location if getattr(client, "vertexai", False) else None,
        provider=provider,
    )

    job_names = ", ".join(
        str(item.get("batch_job_name"))
        for item in chunk_jobs
        if item.get("batch_job_name")
    )
    log(f"Submitted {len(chunk_jobs)} batch job(s): {job_names}")
    print(
        f"Batch jobs submitted ({len(chunk_jobs)}): "
        + ", ".join(
            str(item.get("batch_job_name"))
            for item in chunk_jobs
            if item.get("batch_job_name")
        )
    )


if __name__ == "__main__":
    submit_batch()
