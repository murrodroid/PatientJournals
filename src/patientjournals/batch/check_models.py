from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

from google.genai.errors import ClientError

from patientjournals.batch.client import get_batch_client
from patientjournals.config import config


@dataclass
class _ConfigSnapshot:
    backend: str
    project: str
    location: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List available models for the active backend/region/project and "
            "optionally validate specific model IDs."
        )
    )
    parser.add_argument(
        "--backend",
        choices=("vertex", "mldev"),
        help="Temporarily override config.batch_backend for this run.",
    )
    parser.add_argument(
        "--project",
        help="Temporarily override config.gcp_project_id for this run.",
    )
    parser.add_argument(
        "--location",
        help="Temporarily override config.vertex_model_location for this run.",
    )
    parser.add_argument(
        "--contains",
        help="Case-insensitive substring filter on model name/display name.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of models to print (default: 200).",
    )
    parser.add_argument(
        "--show-actions",
        action="store_true",
        help="Show supported_actions for each listed model.",
    )
    parser.add_argument(
        "--check",
        action="append",
        default=[],
        help=(
            "Model ID to validate via models.get (can be repeated). "
            "Examples: gemini-3-pro, publishers/google/models/gemini-3-pro"
        ),
    )
    return parser.parse_args()


def _snapshot() -> _ConfigSnapshot:
    return _ConfigSnapshot(
        backend=(config.batch_backend or "").strip().lower(),
        project=(config.gcp_project_id or "").strip(),
        location=(
            (config.vertex_model_location or "").strip()
            or (config.gcp_location or "").strip()
        ),
    )


def _norm(text: object) -> str:
    if not isinstance(text, str):
        return ""
    return text.strip()


def _candidate_model_ids(model_id: str, backend: str) -> list[str]:
    cleaned = _norm(model_id)
    if not cleaned:
        return []

    short_id = cleaned.split("/")[-1]
    candidates: list[str] = [cleaned, short_id]

    if backend == "vertex":
        candidates.append(f"publishers/google/models/{short_id}")
    else:
        candidates.append(f"models/{short_id}")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if not item or item in seen:
            continue
        deduped.append(item)
        seen.add(item)
    return deduped


def _iter_models(*, contains: str | None) -> Iterable[object]:
    client = get_batch_client()
    needle = _norm(contains).lower()
    for model in client.models.list():
        name = _norm(getattr(model, "name", ""))
        display_name = _norm(getattr(model, "display_name", ""))
        if needle:
            haystack = f"{name} {display_name}".lower()
            if needle not in haystack:
                continue
        yield model


def _print_models(*, contains: str | None, limit: int, show_actions: bool) -> int:
    shown = 0
    for model in _iter_models(contains=contains):
        if shown >= limit:
            break
        name = _norm(getattr(model, "name", ""))
        short_name = name.split("/")[-1] if "/" in name else name
        display_name = _norm(getattr(model, "display_name", ""))
        line = f"- {short_name}"
        if display_name and display_name != short_name:
            line += f" ({display_name})"
        if name and name != short_name:
            line += f" [{name}]"
        if show_actions:
            actions = getattr(model, "supported_actions", None) or []
            line += f" actions={list(actions)}"
        print(line)
        shown += 1
    return shown


def _check_model_ids(model_ids: list[str], backend: str) -> int:
    if not model_ids:
        return 0

    client = get_batch_client()
    failures = 0

    for raw_model_id in model_ids:
        candidates = _candidate_model_ids(raw_model_id, backend)
        if not candidates:
            continue

        success_id: str | None = None
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                _ = client.models.get(model=candidate)
                success_id = candidate
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc

        if success_id is not None:
            print(f"CHECK OK: {raw_model_id} -> {success_id}")
            continue

        failures += 1
        print(f"CHECK FAIL: {raw_model_id}")
        if last_error is not None:
            print(f"  reason: {type(last_error).__name__}: {last_error}")
        print(f"  tried: {', '.join(candidates)}")

    return failures


def main() -> None:
    args = _parse_args()
    original = {
        "batch_backend": config.batch_backend,
        "gcp_project_id": config.gcp_project_id,
        "gcp_location": config.gcp_location,
        "vertex_model_location": config.vertex_model_location,
    }

    try:
        if args.backend:
            config.batch_backend = args.backend
        if args.project:
            config.gcp_project_id = args.project
        if args.location:
            config.vertex_model_location = args.location

        snap = _snapshot()
        print(
            "Using config: "
            f"backend={snap.backend} "
            f"project={snap.project or '<none>'} "
            f"location={snap.location or '<none>'}"
        )

        try:
            shown = _print_models(
                contains=args.contains,
                limit=max(1, int(args.limit)),
                show_actions=bool(args.show_actions),
            )
            print(f"Listed {shown} model(s).")
        except ClientError as exc:
            print(f"Model list failed: {exc}")
            raise

        failures = _check_model_ids(args.check, snap.backend)
        if failures:
            raise SystemExit(1)
    finally:
        config.batch_backend = original["batch_backend"]
        config.gcp_project_id = original["gcp_project_id"]
        config.gcp_location = original["gcp_location"]
        config.vertex_model_location = original["vertex_model_location"]


if __name__ == "__main__":
    main()
