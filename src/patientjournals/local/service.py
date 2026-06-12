from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from patientjournals.config import config
from patientjournals.local.generate import process_file
from patientjournals.local.model_client import create_local_model_client
from patientjournals.shared.identity import ensure_unique_image_names, image_name_from_reference
from patientjournals.shared.processing_metrics import (
    MANIFEST_FILE_NAME,
    append_processing_record,
    write_processing_summary,
)
from patientjournals.shared.tools import (
    build_image_name_id_set,
    create_subfolder,
    filter_dataset_by_image_names,
    find_newest_dataset,
    flush_rows,
    get_run_logger,
    list_input_files,
    load_existing_dataset,
    write_run_error,
)


@dataclass(frozen=True)
class LocalRunRequest:
    data_folder: str | None = None
    continue_dataset: str | None = None
    verbose: bool = False


@dataclass(frozen=True)
class LocalRunProgress:
    event: str
    message: str = ""
    processed_images: int = 0
    total_images: int = 0
    rows_written: int = 0


@dataclass(frozen=True)
class LocalRunResult:
    status: str
    run_dir: Path
    dataset_path: Path
    total_images: int
    processed_images: int
    rows_written: int
    covered_before: int = 0
    covered_after: int = 0
    skipped_images: int = 0
    error: str = ""


ProgressCallback = Callable[[LocalRunProgress], None]


def resolve_data_folder(
    folder_arg: str | None,
    default_folder: str | Path,
) -> Path:
    default_path = Path(default_folder).expanduser()
    if not folder_arg:
        return default_path

    requested = Path(folder_arg).expanduser()
    candidates = [requested]
    if not requested.is_absolute():
        candidates.append(default_path / requested)

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()

    readable_candidates = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Unable to resolve data folder '{folder_arg}'. "
        f"Tried: {readable_candidates}"
    )


def _emit(callback: ProgressCallback | None, progress: LocalRunProgress) -> None:
    if callback is not None:
        callback(progress)


def _input_without_existing(data: list[str], existing_names: set[str]) -> list[str]:
    return [
        path
        for path in data
        if image_name_from_reference(path) not in existing_names
    ]


async def run_local_job(
    request: LocalRunRequest,
    *,
    progress_callback: ProgressCallback | None = None,
) -> LocalRunResult:
    model = config.model
    model_client = create_local_model_client(model)

    target_folder = resolve_data_folder(request.data_folder, config.target_folder)
    selection_cfg = {
        "target_folder": str(target_folder),
        "input_glob": config.input_glob,
        "recursive": config.recursive,
        "fp_mode": config.fp_mode,
        "fp_suffix": config.fp_suffix,
    }
    data = list_input_files(selection_cfg)
    ensure_unique_image_names(data, source_label="selected local input files")

    flush_every = max(1, int(config.flush_every))
    rows: list[dict] = []
    header_written = False
    total_written = 0

    out_name = config.dataset_file_name
    output_format = config.output_format
    run_dir = create_subfolder(config.output_root, category="local")
    log = get_run_logger(run_dir)
    out_path = run_dir / f'{run_dir.name}_{out_name}.{output_format.lstrip(".")}'
    manifest_path = run_dir / MANIFEST_FILE_NAME

    covered_before = 0
    skipped = 0
    normalized_existing: set[str] = set()
    input_ids = build_image_name_id_set(data)
    if request.continue_dataset:
        continue_path = request.continue_dataset
        if continue_path.lower() == "newest":
            continue_path = str(
                find_newest_dataset(config.output_root, config.dataset_file_name)
            )
            log(f"Resolved newest dataset to {continue_path}")
        existing_format, existing_files, existing_count = load_existing_dataset(
            continue_path
        )
        if output_format.strip().lower().lstrip(".") != existing_format:
            log(
                "Overriding output_format to match existing dataset "
                f"({existing_format})."
            )
        output_format = existing_format
        out_path = run_dir / f'{run_dir.name}_{out_name}.{output_format.lstrip(".")}'
        normalized_existing = existing_files & input_ids
        filtered_count = filter_dataset_by_image_names(
            continue_path,
            out_path,
            image_names=input_ids,
            output_format=output_format,
        )
        original_count = len(data)
        data = _input_without_existing(data, normalized_existing)
        skipped = original_count - len(data)
        header_written = filtered_count > 0
        total_written = filtered_count
        covered_before = len(input_ids & normalized_existing)
        log(
            f"Continuing dataset {continue_path} -> {out_path}. "
            f"Existing rows={existing_count} Kept rows={filtered_count} "
            f"Skipped files={skipped}"
        )

    _emit(
        progress_callback,
        LocalRunProgress(
            event="started",
            message=f"Processing {len(data)} image(s).",
            total_images=len(input_ids),
            rows_written=total_written,
        ),
    )

    sem = asyncio.Semaphore(config.api_concurrent_tasks)
    tasks: list[asyncio.Task] = []
    processed_images = 0
    try:
        log(
            f"Local model provider resolved: model={model_client.model_name} "
            f"provider={model_client.provider}"
        )
        for warning in model_client.capability_warnings():
            log(f"Capability warning: {warning}")

        log(
            "Starting run. "
            f"Files={len(data)} "
            f"Folder={target_folder} "
            f"fp_mode={selection_cfg['fp_mode']} "
            f"Output={out_path.name}"
        )
        tasks = [
            asyncio.create_task(process_file(sem, model_client, path, log))
            for path in data
        ]

        for task in asyncio.as_completed(tasks):
            result = await task
            processed_images += 1
            append_processing_record(manifest_path, result.metrics)
            generated_rows = result.rows
            if generated_rows:
                rows.extend(generated_rows)
            else:
                log("Received empty row from processing step.")

            _emit(
                progress_callback,
                LocalRunProgress(
                    event="image_processed",
                    processed_images=processed_images,
                    total_images=len(data),
                    rows_written=total_written,
                ),
            )

            if len(rows) >= flush_every:
                flush_count = len(rows)
                header_written = flush_rows(
                    rows=rows,
                    out_path=str(out_path),
                    header_written=header_written,
                    output_format=output_format,
                )
                total_written += flush_count
                log(
                    f"Flushed {flush_count} row(s) to {out_path.name} "
                    f"(flush_every={flush_every})."
                )
                rows.clear()

    except Exception as exc:
        for task in tasks:
            task.cancel()
        write_run_error(run_dir, exc)
        log("Stopping early due to error.", exc=exc)
        try:
            summary_path = write_processing_summary(run_dir)
            log(f"Wrote image processing manifest: {manifest_path.name}")
            log(f"Wrote image processing summary: {summary_path.name}")
        except Exception as summary_exc:  # noqa: BLE001
            log("Failed to write image processing summary.", exc=summary_exc)
        return LocalRunResult(
            status="failed",
            run_dir=run_dir,
            dataset_path=out_path,
            total_images=len(input_ids),
            processed_images=processed_images,
            rows_written=total_written,
            covered_before=covered_before,
            covered_after=covered_before + processed_images,
            skipped_images=skipped,
            error=str(exc),
        )

    finally:
        await model_client.aclose()

    if rows:
        header_written = flush_rows(
            rows=rows,
            out_path=str(out_path),
            header_written=header_written,
            output_format=output_format,
        )
        total_written += len(rows)
        log(f"Wrote final batch of {len(rows)} rows.")
    elif total_written == 0:
        log("No rows written; output file may be missing.")

    summary_path = write_processing_summary(run_dir)
    log(f"Wrote image processing manifest: {manifest_path.name}")
    log(f"Wrote image processing summary: {summary_path.name}")

    covered_after = total_written
    try:
        _, final_files, _ = load_existing_dataset(out_path, output_format)
        final_ids = final_files
        covered_after = len(input_ids & final_ids)
    except FileNotFoundError:
        covered_after = len(input_ids & normalized_existing) if request.continue_dataset else 0

    result = LocalRunResult(
        status="finished",
        run_dir=run_dir,
        dataset_path=out_path,
        total_images=len(input_ids),
        processed_images=processed_images,
        rows_written=total_written,
        covered_before=covered_before,
        covered_after=covered_after,
        skipped_images=skipped,
    )
    _emit(
        progress_callback,
        LocalRunProgress(
            event="finished",
            message=str(out_path),
            processed_images=processed_images,
            total_images=len(input_ids),
            rows_written=total_written,
        ),
    )
    return result
