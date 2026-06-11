from __future__ import annotations

import argparse
import asyncio

from patientjournals.local.service import (
    LocalRunProgress,
    LocalRunRequest,
    resolve_data_folder,
    run_local_job,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process journal images and write a dataset."
    )
    parser.add_argument(
        "--data-folder",
        dest="data_folder",
        help=(
            "Folder to process as input data. Accepts an absolute path, a "
            "path relative to the current working directory, or a path "
            "relative to the configured target folder."
        ),
    )
    parser.add_argument(
        "--continue-dataset",
        dest="continue_dataset",
        help=(
            "Path to an existing dataset file to append to, or 'newest' to "
            "select the latest run's dataset. "
            "Rows already in the dataset (by image_name) will be skipped."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress details about dataset coverage.",
    )
    return parser.parse_args()


def _progress_printer(enabled: bool):
    def emit(progress: LocalRunProgress) -> None:
        if not enabled:
            return
        if progress.event == "started":
            print(progress.message)
            return
        if progress.event == "image_processed":
            print(
                "Processed "
                f"{progress.processed_images}/{progress.total_images} image(s)."
            )
            return
        if progress.event == "finished":
            print(
                "Dataset covers "
                f"{progress.processed_images}/{progress.total_images} processed image(s); "
                f"rows={progress.rows_written}."
            )

    return emit


async def main() -> None:
    args = parse_args()
    result = await run_local_job(
        LocalRunRequest(
            data_folder=args.data_folder,
            continue_dataset=args.continue_dataset,
            verbose=bool(args.verbose),
        ),
        progress_callback=_progress_printer(bool(args.verbose)),
    )
    if result.status == "failed":
        print(f"Stopping early due to error: {result.error}")
        return
    if args.verbose:
        missing = result.total_images - result.covered_after
        if missing:
            print(f"Missing {missing} images from dataset coverage.")
        print(
            f"Dataset covers {result.covered_after}/{result.total_images} images after run."
        )


if __name__ == "__main__":
    asyncio.run(main())
