from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from google import genai

from batch_client import get_batch_client
from config import config


_TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check status of a Gemini batch job."
    )
    parser.add_argument(
        "--batch-name",
        dest="batch_name",
        help="Batch job name (overrides config.batch_job_name).",
    )
    parser.add_argument(
        "--run-dir",
        dest="run_dir",
        help="Run directory containing batch_job.json (overrides auto-discovery).",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Poll until the job reaches a terminal state.",
    )
    parser.add_argument(
        "--cancel",
        action="store_true",
        help="Cancel the batch job and exit.",
    )
    return parser.parse_args()


def _read_batch_name_from_job_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = payload.get("batch_job_name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _latest_batch_job_file(output_root: str) -> Path | None:
    root = Path(output_root).expanduser()
    if not root.exists() or not root.is_dir():
        return None
    run_dirs = sorted((item for item in root.iterdir() if item.is_dir()), reverse=True)
    for run_dir in run_dirs:
        candidate = run_dir / "batch_job.json"
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _resolve_batch_name(args: argparse.Namespace) -> str:
    if args.batch_name:
        return args.batch_name

    if args.run_dir:
        candidate = Path(args.run_dir).expanduser() / "batch_job.json"
        batch_name = _read_batch_name_from_job_file(candidate)
        if batch_name:
            return batch_name
        raise ValueError(f"No batch_job_name found in {candidate}.")

    latest_job_file = _latest_batch_job_file(config.output_root)
    if latest_job_file is not None:
        batch_name = _read_batch_name_from_job_file(latest_job_file)
        if batch_name:
            return batch_name

    if config.batch_job_name:
        return config.batch_job_name

    raise ValueError(
        "Batch job name not found. Use --batch-name, --run-dir, or set config.batch_job_name."
    )


def _batch_summary(batch_job: object) -> str:
    name = getattr(batch_job, "name", None)
    state = getattr(batch_job, "state", None)
    model = getattr(batch_job, "model", None)
    create_time = getattr(batch_job, "create_time", None)
    update_time = getattr(batch_job, "update_time", None)

    dest = getattr(batch_job, "dest", None)
    dest_file_name = getattr(dest, "file_name", None) if dest else None
    dest_gcs_uri = getattr(dest, "gcs_uri", None) if dest else None
    error = getattr(batch_job, "error", None)

    def _fmt_time(value: object) -> str:
        if isinstance(value, datetime):
            # API timestamps are typically timezone-aware (UTC). Keep raw tz info.
            return value.isoformat()
        return str(value)

    def _maybe_progress(batch: object) -> str | None:
        # The public Developer API generally only exposes coarse states.
        # If progress counters are present on the object, surface them.
        candidates: list[tuple[str, object]] = [
            ("request_count", getattr(batch, "request_count", None)),
            ("total_count", getattr(batch, "total_count", None)),
            ("completed_count", getattr(batch, "completed_count", None)),
            ("succeeded_count", getattr(batch, "succeeded_count", None)),
            ("failed_count", getattr(batch, "failed_count", None)),
            ("cancelled_count", getattr(batch, "cancelled_count", None)),
        ]

        stats = getattr(batch, "completion_stats", None) or getattr(batch, "stats", None)
        if isinstance(stats, dict):
            for k, v in stats.items():
                candidates.append((str(k), v))
        elif stats is not None:
            for key in ("total", "completed", "succeeded", "failed", "cancelled"):
                candidates.append((key, getattr(stats, key, None)))

        items: list[str] = []
        for key, value in candidates:
            if isinstance(value, int):
                items.append(f"{key}={value}")
        if not items:
            return None
        return "Progress: " + ", ".join(items)

    lines = [
        f"Batch: {name}",
        f"State: {state}",
        f"Model: {model}",
        f"Created: {_fmt_time(create_time)}",
        f"Updated: {_fmt_time(update_time)}",
    ]
    progress = _maybe_progress(batch_job)
    if progress:
        lines.append(progress)
    if error:
        lines.append(f"Error: {error}")
    if dest_file_name:
        lines.append(f"Output file: {dest_file_name}")
    if dest_gcs_uri:
        lines.append(f"Output GCS: {dest_gcs_uri}")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    batch_name = _resolve_batch_name(args)

    if args.cancel and args.watch:
        raise ValueError("--cancel and --watch cannot be used together.")

    client = get_batch_client()

    if args.cancel:
        batch_job = client.batches.get(name=batch_name)
        state = getattr(batch_job, "state", None)
        if state in _TERMINAL_STATES:
            print(f"Batch {batch_name} already terminal (state={state}); nothing to cancel.")
            return
        client.batches.cancel(name=batch_name)
        print(f"Cancel requested for batch {batch_name}.")
        return

    poll_interval = max(1, int(config.batch_poll_interval_seconds))

    last_state = None
    while True:
        batch_job = client.batches.get(name=batch_name)
        state = getattr(batch_job, "state", None)
        if state != last_state:
            print(_batch_summary(batch_job))
            print("")
            last_state = state
        if not args.watch or state in _TERMINAL_STATES:
            break
        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
