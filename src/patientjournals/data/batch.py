from __future__ import annotations

import argparse
from datetime import datetime

from patientjournals.config import config
from patientjournals.data.bucket import summarize_bucket_data, validate_bucket_data
from patientjournals.data.inspection import (
    default_glob_pattern,
    default_recursive,
    summarize_batch_data,
    validate_batch_data,
    write_json_report,
    write_validation_csv,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and validate local or GCS batch image data."
    )
    parser.add_argument("--summary", action="store_true", help="Write a data summary report.")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate image readability and basic image health.",
    )
    parser.add_argument(
        "--root",
        help="Data root to inspect. Defaults to configured batch/local image folder.",
    )
    parser.add_argument(
        "--bucket",
        action="store_true",
        help="Inspect the configured GCS bucket instead of a local folder.",
    )
    parser.add_argument(
        "--bucket-name",
        help="GCS bucket name to inspect. Defaults to config.gcs_bucket_name.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional GCS object prefix to inspect when --bucket is set.",
    )
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        default=default_glob_pattern(),
        help="Image glob to inspect. Defaults to config.upload_images_glob.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only inspect files directly under the root.",
    )
    parser.add_argument(
        "--summaries-dir",
        default="summaries",
        help="Directory where summary JSON reports are written.",
    )
    parser.add_argument(
        "--validations-dir",
        default="validations",
        help="Directory where validation JSON/CSV reports are written.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Exit successfully even if validation finds corrupt or invalid images.",
    )
    return parser.parse_args()


def _print_summary(report: dict) -> None:
    size_stats = report["image_size_bytes"]
    print(f"Root: {report['root']}")
    print(f"Images: {report['image_files']} / files: {report['total_files']}")
    print(
        "Folders: "
        f"{report['folder_count']} "
        f"(with images={report['folders_with_images']}, empty={report['empty_folder_count']})"
    )
    print(f"Extensions: {report['files_by_extension']}")
    print(
        "Image bytes: "
        f"total={size_stats['total']} mean={size_stats['mean']} median={size_stats['median']}"
    )


def _print_validation(report: dict) -> None:
    print(f"Root: {report['root']}")
    print(
        "Validation: "
        f"status={report['status']} total={report['total_images']} "
        f"ok={report['ok_count']} warnings={report['warning_count']} errors={report['error_count']}"
    )
    if report["duplicate_basename_count"]:
        print(f"Duplicate basenames: {report['duplicate_basename_count']}")


def main() -> None:
    args = _parse_args()
    if not args.summary and not args.validate:
        raise SystemExit("Choose at least one operation: --summary or --validate.")
    if args.bucket and args.root:
        raise SystemExit("--root is only valid for local inspection; use --prefix for bucket inspection.")

    recursive = default_recursive() and not args.no_recursive
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bucket_name = args.bucket_name or config.gcs_bucket_name

    if args.summary:
        if args.bucket:
            report = summarize_bucket_data(
                bucket_name=bucket_name,
                prefix=args.prefix,
                glob_pattern=args.glob_pattern,
            )
            stem = f"data_bucket_summary_{timestamp}"
        else:
            report = summarize_batch_data(
                args.root,
                glob_pattern=args.glob_pattern,
                recursive=recursive,
            )
            stem = f"data_batch_summary_{timestamp}"
        path = write_json_report(report, args.summaries_dir, stem)
        _print_summary(report)
        print(f"Summary report: {path}")

    if args.validate:
        if args.bucket:
            report = validate_bucket_data(
                bucket_name=bucket_name,
                prefix=args.prefix,
                glob_pattern=args.glob_pattern,
            )
            stem = f"data_bucket_validation_{timestamp}"
        else:
            report = validate_batch_data(
                args.root,
                glob_pattern=args.glob_pattern,
                recursive=recursive,
            )
            stem = f"data_batch_validation_{timestamp}"
        json_path = write_json_report(report, args.validations_dir, stem)
        csv_path = write_validation_csv(report, args.validations_dir, stem)
        _print_validation(report)
        print(f"Validation report: {json_path}")
        print(f"Validation records: {csv_path}")
        if report["status"] == "error" and not args.allow_failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
