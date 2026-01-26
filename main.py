import asyncio
import argparse
from pathlib import Path
from google import genai
from tqdm.asyncio import tqdm_asyncio

from api_keys import gemini_maarten as api_key
from config import config
from tools import *
from generate import process_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process journal images and write a dataset."
    )
    parser.add_argument(
        "--continue-dataset",
        dest="continue_dataset",
        help=(
            "Path to an existing dataset file to append to. "
            "Rows already in the dataset (by file_name) will be skipped."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress details about dataset coverage.",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    model = config.model
    client = genai.Client(api_key=api_key)
    
    data = list_input_files(config)
    target_folder = config.target_folder
    
    flush_every = config.flush_every or config.batch_size
    rows: list[dict] = []
    header_written = False
    total_written = 0

    out_name = config.dataset_file_name
    output_format = config.output_format
    run_dir = create_subfolder(config.output_root)
    log = get_run_logger(run_dir)
    out_path = run_dir / f'{run_dir.name}_{out_name}.{output_format.lstrip(".")}'

    covered_before = 0
    normalized_existing = set()
    input_ids = build_path_id_set(data, target_folder)
    if args.continue_dataset:
        existing_format, existing_files, existing_count = load_existing_dataset(
            args.continue_dataset
        )
        if output_format.strip().lower().lstrip(".") != existing_format:
            log(
                "Overriding output_format to match existing dataset "
                f"({existing_format})."
            )
        output_format = existing_format
        out_path = run_dir / f'{run_dir.name}_{out_name}.{output_format.lstrip(".")}'
        copy_dataset(args.continue_dataset, out_path)

        normalized_existing = build_path_id_set(existing_files, target_folder)
        original_count = len(data)
        data = [
            f for f in data
            if normalize_path(f) not in normalized_existing
        ]
        skipped = original_count - len(data)
        header_written = existing_count > 0
        total_written = existing_count
        covered_before = len(input_ids & normalized_existing)
        log(
            f"Continuing dataset {args.continue_dataset} -> {out_path}. "
            f"Existing rows={existing_count} Skipped files={skipped}"
        )
        if args.verbose:
            print(
                f"Preloaded dataset covers {covered_before}/{original_count} images."
            )
    elif args.verbose:
        print("Preloaded dataset covers 0/0 images.")

    sem = asyncio.Semaphore(config.concurrent_tasks)

    tasks = []
    try:
        log(f"Starting run. Files={len(data)} Output={out_path.name}")
        tasks = [asyncio.create_task(process_file(sem, client, model, f, log)) for f in data]

        for coro in tqdm_asyncio.as_completed(tasks, desc='Processing images', unit='img'):
            journal_row = await coro
            
            if journal_row:
                rows.append(journal_row)
            else:
                log("Received empty row from processing step.")

            if len(rows) >= flush_every:
                header_written = flush_rows(
                    rows=rows,
                    out_path=str(out_path),
                    header_written=header_written,
                    output_format=output_format,
                )
                total_written += len(rows)
                rows.clear() 

    except Exception as e:
        for task in tasks:
            task.cancel()
        write_run_error(run_dir, e)
        log("Stopping early due to error.", exc=e)
        print(f'Stopping early due to error: {e}')

    finally:
        if rows:
            header_written = flush_rows(
                rows=rows,
                out_path=str(out_path),
                header_written=header_written,
                output_format=output_format,
            )
            total_written += len(rows)
            log(f"Wrote final batch of {len(rows)} rows.")
        else:
            if total_written == 0:
                log("No rows written; output file may be missing.")
        if args.verbose:
            total_images = len(input_ids)
            covered_after = total_written
            try:
                _, final_files, _ = load_existing_dataset(out_path, output_format)
                final_ids = build_path_id_set(final_files, target_folder)
                covered_after = len(input_ids & final_ids)
                missing = len(input_ids - final_ids)
                if missing:
                    print(f"Missing {missing} images from dataset coverage.")
            except FileNotFoundError:
                covered_after = len(input_ids & normalized_existing) if args.continue_dataset else 0
            print(
                f"Dataset covers {covered_after}/{total_images} images after run."
            )


if __name__ == '__main__':
    asyncio.run(main())
