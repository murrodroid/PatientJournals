from __future__ import annotations

import argparse
import fnmatch
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

from tqdm import tqdm

from patientjournals.batch.output_records import (
    GeminiOutputParseResult,
    add_response_metadata_columns,
    iter_gemini_jsonl_results,
)
from patientjournals.config import config
from patientjournals.data.bucket import (
    build_storage_bucket,
    list_bucket_blobs,
    normalize_prefix,
    select_bucket_image_blobs,
)
from patientjournals.shared.output_handler import data_to_rows
from patientjournals.shared.dataset_coverage import (
    copy_dataset_rows_for_keys,
    load_dataset_key_coverage,
    resolve_continue_dataset_path,
)
from patientjournals.shared.tools import create_subfolder, flush_rows, get_run_logger


@dataclass
class CollectedGeminiOutputs:
    selected: dict[str, GeminiOutputParseResult]
    observed_keys: set[str]
    rejected_reasons_by_key: dict[str, Counter[str]]
    stats: Counter[str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect valid Gemini prediction rows from all batch output JSONL files "
            "and report coverage against GCS pages."
        )
    )
    parser.add_argument(
        "--bucket-name",
        help="GCS bucket name. Defaults to config.gcs_bucket_name.",
    )
    parser.add_argument(
        "--outputs-prefix",
        default=config.batch_outputs_gcs_prefix,
        help="GCS prefix containing batch output folders.",
    )
    parser.add_argument(
        "--output-glob",
        default="*predictions.jsonl",
        help="Filename glob used to select prediction JSONL files.",
    )
    parser.add_argument(
        "--pages-prefix",
        default=config.gcs_pages_prefix,
        help="GCS prefix containing source page images for coverage.",
    )
    parser.add_argument(
        "--pages-glob",
        default="*",
        help="Filename glob used to select page images for coverage.",
    )
    parser.add_argument(
        "--local-output",
        action="append",
        default=[],
        help=(
            "Local prediction JSONL file or directory to parse in addition to GCS "
            "outputs. Directories are searched recursively."
        ),
    )
    parser.add_argument(
        "--continue-dataset",
        dest="continue_dataset",
        help=(
            "Existing dataset to continue, or 'newest'. Existing rows are "
            "preloaded and only newly covered output keys are appended."
        ),
    )
    parser.add_argument(
        "--skip-gcs-outputs",
        action="store_true",
        help="Only parse --local-output paths; do not list GCS batch outputs.",
    )
    parser.add_argument(
        "--skip-pages",
        action="store_true",
        help="Skip GCS pages coverage counting.",
    )
    parser.add_argument(
        "--output-format",
        choices=("csv", "jsonl"),
        default=config.output_format,
        help="Dataset output format.",
    )
    parser.add_argument(
        "--run-root",
        default=config.output_root,
        help="Directory where the collection run folder is created.",
    )
    return parser.parse_args()


def _prediction_blob_name(blob: object) -> str:
    return str(getattr(blob, "name", "") or "")


def list_prediction_output_blobs(
    bucket,
    *,
    prefix: str,
    output_glob: str,
) -> list[object]:
    normalized_prefix = normalize_prefix(prefix)
    blobs = list_bucket_blobs(bucket, prefix=normalized_prefix)
    return [
        blob
        for blob in blobs
        if not _prediction_blob_name(blob).endswith("/")
        and fnmatch.fnmatch(Path(_prediction_blob_name(blob)).name, output_glob)
    ]


def _iter_blob_lines(blob: object) -> Iterator[str]:
    if hasattr(blob, "open"):
        try:
            with blob.open("r", encoding="utf-8") as handle:
                yield from handle
            return
        except TypeError:
            pass

    payload = blob.download_as_bytes()
    text = payload.decode("utf-8")
    yield from text.splitlines()


def _expand_local_output_paths(
    paths: Iterable[str],
    *,
    output_glob: str,
) -> list[Path]:
    selected: list[Path] = []
    for value in paths:
        path = Path(value).expanduser()
        if path.is_dir():
            selected.extend(
                item
                for item in path.rglob(output_glob)
                if item.is_file()
            )
        elif path.is_file():
            if fnmatch.fnmatch(path.name, output_glob):
                selected.append(path)
        else:
            raise FileNotFoundError(f"Local output path not found: {path}")
    return sorted(dict.fromkeys(selected))


def collect_valid_outputs_from_jsonl_sources(
    sources: Iterable[tuple[str, Iterable[str]]],
) -> CollectedGeminiOutputs:
    selected: dict[str, GeminiOutputParseResult] = {}
    observed_keys: set[str] = set()
    rejected_reasons_by_key: dict[str, Counter[str]] = {}
    stats: Counter[str] = Counter()

    for source, lines in sources:
        stats["output_files"] += 1
        for result in iter_gemini_jsonl_results(lines, source=source):
            stats["output_rows"] += 1
            if result.key:
                observed_keys.add(result.key)

            if result.is_valid and result.key:
                stats["valid_response_candidates"] += 1
                if result.key in selected:
                    stats["duplicate_valid_candidates"] += 1
                    continue
                selected[result.key] = result
                continue

            reason = result.reason or "unknown"
            stats[f"rejected:{reason}"] += 1
            if result.key:
                rejected_reasons_by_key.setdefault(result.key, Counter())[reason] += 1

    stats["unique_observed_keys"] = len(observed_keys)
    stats["unique_valid_keys"] = len(selected)
    stats["unique_rejected_only_keys"] = len(observed_keys - set(selected))
    return CollectedGeminiOutputs(
        selected=selected,
        observed_keys=observed_keys,
        rejected_reasons_by_key=rejected_reasons_by_key,
        stats=stats,
    )


def _flush_collected_rows(
    *,
    rows_to_flush: list[dict],
    out_path: Path,
    output_format: str,
    header_written: bool,
) -> tuple[bool, int]:
    if not rows_to_flush:
        return header_written, 0
    header_written = flush_rows(
        rows=rows_to_flush,
        out_path=str(out_path),
        header_written=header_written,
        output_format=output_format,
        sep=config.csv_sep,
    )
    count = len(rows_to_flush)
    rows_to_flush.clear()
    return header_written, count


def write_collected_dataset(
    collected: CollectedGeminiOutputs,
    *,
    out_path: Path,
    output_format: str,
    keys: set[str] | None = None,
    header_written: bool = False,
) -> tuple[bool, int]:
    rows_to_flush: list[dict] = []
    total_rows = 0
    flush_every = max(1, int(config.flush_every or config.batch_size))

    for key in sorted(collected.selected):
        if keys is not None and key not in keys:
            continue
        result = collected.selected[key]
        if result.parsed_model is None:
            continue
        rows = data_to_rows(
            result.parsed_model,
            file_name=key,
            field_confidence_by_pointer=result.metadata.get(
                "field_confidence_by_pointer"
            ),
        )
        add_response_metadata_columns(rows, result.metadata)
        rows_to_flush.extend(rows)

        if len(rows_to_flush) >= flush_every:
            header_written, wrote = _flush_collected_rows(
                rows_to_flush=rows_to_flush,
                out_path=out_path,
                output_format=output_format,
                header_written=header_written,
            )
            total_rows += wrote

    if rows_to_flush:
        header_written, wrote = _flush_collected_rows(
            rows_to_flush=rows_to_flush,
            out_path=out_path,
            output_format=output_format,
            header_written=header_written,
        )
        total_rows += wrote
    return header_written, total_rows


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str))
            handle.write("\n")
            count += 1
    return count


def _list_page_keys(
    bucket,
    *,
    pages_prefix: str,
    pages_glob: str,
) -> set[str]:
    blobs = list_bucket_blobs(bucket, prefix=normalize_prefix(pages_prefix))
    image_blobs = select_bucket_image_blobs(blobs, glob_pattern=pages_glob)
    return {str(getattr(blob, "name", "") or "") for blob in image_blobs}


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted((str(key), int(value)) for key, value in counter.items()))


def collect_outputs() -> Path:
    args = _parse_args()
    if args.skip_gcs_outputs and not args.local_output:
        raise ValueError(
            "--skip-gcs-outputs requires at least one --local-output path."
        )

    run_dir = create_subfolder(args.run_root, prefix="collect_outputs_")
    log = get_run_logger(run_dir)
    bucket = (
        None
        if args.skip_gcs_outputs and args.skip_pages
        else build_storage_bucket(args.bucket_name)
    )

    sources: list[tuple[str, Iterable[str]]] = []
    prediction_blobs: list[object] = []
    if not args.skip_gcs_outputs:
        if bucket is None:
            bucket = build_storage_bucket(args.bucket_name)
        prediction_blobs = list_prediction_output_blobs(
            bucket,
            prefix=args.outputs_prefix,
            output_glob=args.output_glob,
        )
        log(
            f"Found {len(prediction_blobs)} prediction JSONL file(s) under "
            f"gs://{bucket.name}/{normalize_prefix(args.outputs_prefix)}."
        )
        for blob in prediction_blobs:
            sources.append((_prediction_blob_name(blob), _iter_blob_lines(blob)))

    local_paths = _expand_local_output_paths(
        args.local_output,
        output_glob=args.output_glob,
    )
    for path in local_paths:
        sources.append((str(path), path.open("r", encoding="utf-8")))

    if not sources:
        raise RuntimeError("No prediction JSONL files found to collect.")

    collected = collect_valid_outputs_from_jsonl_sources(
        tqdm(sources, desc="Parsing prediction files", unit="file")
    )

    for _, lines in sources:
        close = getattr(lines, "close", None)
        if callable(close):
            close()

    selected_keys = set(collected.selected)
    pages_checked = not args.skip_pages
    page_keys: set[str] = set()
    if pages_checked:
        if bucket is None:
            bucket = build_storage_bucket(args.bucket_name)
        page_keys = _list_page_keys(
            bucket,
            pages_prefix=args.pages_prefix,
            pages_glob=args.pages_glob,
        )

    output_format = args.output_format.strip().lower().lstrip(".")
    continue_dataset_path: Path | None = None
    existing_dataset_keys: set[str] = set()
    existing_dataset_rows = 0
    kept_existing_rows = 0
    if args.continue_dataset:
        continue_dataset_path = resolve_continue_dataset_path(
            args.continue_dataset,
            run_root=args.run_root,
            dataset_name=config.dataset_file_name,
        )
        existing_format, existing_dataset_keys, existing_dataset_rows = (
            load_dataset_key_coverage(
                continue_dataset_path,
                csv_sep=config.csv_sep,
                bucket_name=getattr(bucket, "name", args.bucket_name or None),
            )
        )
        if output_format != existing_format:
            log(
                "Overriding output_format to match continued dataset "
                f"({existing_format})."
            )
            output_format = existing_format
        log(
            f"Continuing dataset {continue_dataset_path}: "
            f"rows={existing_dataset_rows}, unique_keys={len(existing_dataset_keys)}."
        )

    out_path = run_dir / f"{run_dir.name}_{config.dataset_file_name}.{output_format}"
    header_written = False
    dataset_rows = 0
    if continue_dataset_path is not None:
        kept_existing_rows = copy_dataset_rows_for_keys(
            continue_dataset_path,
            out_path,
            keys=page_keys if pages_checked else None,
            output_format=output_format,
            csv_sep=config.csv_sep,
            bucket_name=getattr(bucket, "name", args.bucket_name or None),
        )
        header_written = kept_existing_rows > 0 or (
            output_format == "csv" and out_path.exists()
        )
        dataset_rows += kept_existing_rows

    existing_covered_keys = (
        existing_dataset_keys & page_keys if pages_checked else existing_dataset_keys
    )
    keys_to_append = selected_keys - existing_covered_keys
    header_written, added_rows = write_collected_dataset(
        collected,
        out_path=out_path,
        output_format=output_format,
        keys=keys_to_append,
        header_written=header_written,
    )
    dataset_rows += added_rows

    final_dataset_keys = existing_covered_keys | selected_keys
    covered_page_keys = final_dataset_keys & page_keys if pages_checked else set()
    missing_page_keys = page_keys - final_dataset_keys if pages_checked else set()
    extra_output_keys = final_dataset_keys - page_keys if pages_checked else set()

    manifest_path = run_dir / "selected_outputs.jsonl"
    _write_jsonl(
        manifest_path,
        (
            {
                "key": key,
                "source": collected.selected[key].source,
                "line_number": collected.selected[key].line_number,
            }
            for key in sorted(selected_keys)
        ),
    )

    rejected_only_keys = collected.observed_keys - selected_keys
    rejected_path = run_dir / "rejected_output_keys.jsonl"
    _write_jsonl(
        rejected_path,
        (
            {
                "key": key,
                "reasons": _counter_to_dict(
                    collected.rejected_reasons_by_key.get(key, Counter())
                ),
            }
            for key in sorted(rejected_only_keys)
        ),
    )

    missing_path = run_dir / "missing_page_keys.jsonl"
    if pages_checked:
        _write_jsonl(
            missing_path,
            ({"key": key} for key in sorted(missing_page_keys)),
        )

    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "bucket": getattr(bucket, "name", args.bucket_name or config.gcs_bucket_name),
        "outputs_prefix": normalize_prefix(args.outputs_prefix),
        "output_glob": args.output_glob,
        "pages_prefix": normalize_prefix(args.pages_prefix),
        "pages_glob": args.pages_glob,
        "gcs_prediction_files": len(prediction_blobs),
        "local_prediction_files": len(local_paths),
        "stats": _counter_to_dict(collected.stats),
        "unique_valid_output_keys": len(selected_keys),
        "continue_dataset_path": (
            str(continue_dataset_path) if continue_dataset_path is not None else None
        ),
        "existing_dataset_rows": existing_dataset_rows,
        "existing_dataset_keys": len(existing_dataset_keys),
        "kept_existing_rows": kept_existing_rows,
        "new_output_keys_added": len(keys_to_append),
        "new_output_rows_added": added_rows,
        "final_dataset_keys": len(final_dataset_keys),
        "dataset_rows": dataset_rows,
        "pages_total": len(page_keys) if pages_checked else None,
        "pages_covered": len(covered_page_keys) if pages_checked else None,
        "coverage_ratio": (
            _ratio(len(covered_page_keys), len(page_keys)) if pages_checked else None
        ),
        "missing_pages": len(missing_page_keys) if pages_checked else None,
        "extra_output_keys": len(extra_output_keys) if pages_checked else None,
        "dataset_path": str(out_path),
        "selected_outputs_path": str(manifest_path),
        "rejected_output_keys_path": str(rejected_path),
        "missing_page_keys_path": str(missing_path) if pages_checked else None,
        "missing_page_key_samples": sorted(missing_page_keys)[:10],
        "extra_output_key_samples": sorted(extra_output_keys)[:10],
    }
    report_path = run_dir / "coverage_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    if pages_checked:
        ratio = report["coverage_ratio"]
        percentage = f"{ratio * 100:.2f}%" if isinstance(ratio, float) else "n/a"
        print(
            "Coverage: "
            f"{len(covered_page_keys)}/{len(page_keys)} pages ({percentage}); "
            f"missing={len(missing_page_keys)}."
        )
    print(
        f"Collected {len(selected_keys)} unique valid output key(s), "
        f"added {len(keys_to_append)} new key(s), "
        f"into {out_path} ({dataset_rows} dataset row(s))."
    )
    print(f"Coverage report: {report_path}")
    log(f"Collected outputs into {out_path}; report={report_path}.")
    return out_path


if __name__ == "__main__":
    collect_outputs()
