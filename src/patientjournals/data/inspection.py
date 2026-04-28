from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from PIL import Image

from patientjournals.config import config


DEFAULT_IMAGE_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp",
    "tif",
    "tiff",
    "bmp",
}

FORMAT_EXTENSIONS = {
    "BMP": {"bmp"},
    "JPEG": {"jpg", "jpeg"},
    "PNG": {"png"},
    "TIFF": {"tif", "tiff"},
    "WEBP": {"webp"},
}


def default_batch_root() -> Path:
    upload_source = str(config.upload_source or "").strip().lower()
    if upload_source in {"images", "auto"} and str(config.upload_images_folder).strip():
        return Path(config.upload_images_folder).expanduser()
    return Path(config.target_folder).expanduser()


def default_glob_pattern() -> str:
    value = str(config.upload_images_glob or config.input_glob or "*").strip()
    return value or "*"


def default_recursive() -> bool:
    return bool(config.upload_images_recursive)


def configured_image_extensions() -> set[str]:
    configured = {
        str(ext).strip().lower().lstrip(".")
        for ext in (config.batch_input_extensions or ())
        if str(ext).strip()
    }
    return configured | DEFAULT_IMAGE_EXTENSIONS


def resolve_root(root: str | Path | None = None) -> Path:
    path = Path(root).expanduser() if root else default_batch_root()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Data root not found or not a directory: {path}")
    return path.resolve()


def collect_files(
    root: str | Path | None = None,
    *,
    glob_pattern: str | None = None,
    recursive: bool | None = None,
    allowed_extensions: set[str] | None = None,
) -> tuple[Path, list[Path], list[Path]]:
    root_path = resolve_root(root)
    pattern = glob_pattern or default_glob_pattern()
    use_recursive = default_recursive() if recursive is None else recursive
    extensions = allowed_extensions or configured_image_extensions()

    all_candidates = root_path.rglob("*") if use_recursive else root_path.glob("*")
    all_files = sorted(path for path in all_candidates if path.is_file())
    candidates = root_path.rglob(pattern) if use_recursive else root_path.glob(pattern)
    selected = sorted(
        path
        for path in candidates
        if path.is_file() and path.suffix.lower().lstrip(".") in extensions
    )
    return root_path, all_files, selected


def _relative_parent(path: Path, root: Path) -> str:
    try:
        parent = path.parent.relative_to(root)
    except ValueError:
        return str(path.parent)
    return "." if str(parent) == "." else str(parent)


def _file_depth(path: Path, root: Path) -> int:
    try:
        return max(0, len(path.relative_to(root).parts) - 1)
    except ValueError:
        return 0


def _numeric_stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "total": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    return {
        "count": len(values),
        "total": int(sum(values)),
        "min": int(min(values)),
        "max": int(max(values)),
        "mean": float(mean(values)),
        "median": float(median(values)),
    }


def summarize_batch_data(
    root: str | Path | None = None,
    *,
    glob_pattern: str | None = None,
    recursive: bool | None = None,
    allowed_extensions: set[str] | None = None,
) -> dict[str, Any]:
    root_path, all_files, image_files = collect_files(
        root,
        glob_pattern=glob_pattern,
        recursive=recursive,
        allowed_extensions=allowed_extensions,
    )
    use_recursive = default_recursive() if recursive is None else recursive
    folder_candidates = root_path.rglob("*") if use_recursive else root_path.glob("*")
    folders = sorted(path for path in folder_candidates if path.is_dir())
    child_folder_count = len(folders)
    all_folder_count = child_folder_count + 1
    folder_file_counts: dict[str, int] = defaultdict(int)
    extension_counts: Counter[str] = Counter()
    depth_counts: Counter[int] = Counter()
    sizes: list[int] = []

    for path in image_files:
        folder_file_counts[_relative_parent(path, root_path)] += 1
        extension_counts[path.suffix.lower().lstrip(".") or "<none>"] += 1
        depth_counts[_file_depth(path, root_path)] += 1
        try:
            sizes.append(path.stat().st_size)
        except OSError:
            pass

    empty_folder_count = 0
    for folder in [root_path, *folders]:
        try:
            if not any(folder.iterdir()):
                empty_folder_count += 1
        except OSError:
            pass

    non_image_files = [
        path
        for path in all_files
        if path.suffix.lower().lstrip(".")
        not in (allowed_extensions or configured_image_extensions())
    ]

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "root": str(root_path),
        "glob": glob_pattern or default_glob_pattern(),
        "recursive": default_recursive() if recursive is None else bool(recursive),
        "folder_count": all_folder_count,
        "child_folder_count": child_folder_count,
        "empty_folder_count": empty_folder_count,
        "folders_with_images": len(folder_file_counts),
        "total_files": len(all_files),
        "image_files": len(image_files),
        "non_image_files": len(non_image_files),
        "files_by_extension": dict(sorted(extension_counts.items())),
        "files_by_depth": {
            str(depth): count for depth, count in sorted(depth_counts.items())
        },
        "files_by_folder": dict(sorted(folder_file_counts.items())),
        "image_size_bytes": _numeric_stats(sizes),
    }


def _extension_format_issue(path: Path, image_format: str | None) -> str | None:
    if not image_format:
        return None
    expected = FORMAT_EXTENSIONS.get(image_format.upper())
    if not expected:
        return None
    suffix = path.suffix.lower().lstrip(".")
    if suffix not in expected:
        return f"extension_format_mismatch:{suffix}:{image_format}"
    return None


def validate_image(path: Path, root: Path, duplicate_basenames: set[str]) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    width: int | None = None
    height: int | None = None
    image_format: str | None = None
    mode: str | None = None

    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        size_bytes = None
        issues.append(f"stat_failed:{type(exc).__name__}")

    if size_bytes == 0:
        issues.append("empty_file")

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format
            mode = image.mode
            if width <= 0 or height <= 0:
                issues.append("invalid_dimensions")
            mismatch = _extension_format_issue(path, image_format)
            if mismatch:
                warnings.append(mismatch)
    except Exception as exc:
        issues.append(f"image_open_failed:{type(exc).__name__}")

    if path.name in duplicate_basenames:
        warnings.append("duplicate_basename")

    status = "ok"
    if issues:
        status = "error"
    elif warnings:
        status = "warning"

    return {
        "status": status,
        "issues": ";".join(issues),
        "warnings": ";".join(warnings),
        "file_name": path.name,
        "relative_path": str(path.relative_to(root)),
        "parent_folder": _relative_parent(path, root),
        "extension": path.suffix.lower().lstrip("."),
        "size_bytes": size_bytes,
        "width": width,
        "height": height,
        "format": image_format,
        "mode": mode,
    }


def validate_batch_data(
    root: str | Path | None = None,
    *,
    glob_pattern: str | None = None,
    recursive: bool | None = None,
    allowed_extensions: set[str] | None = None,
) -> dict[str, Any]:
    root_path, _all_files, image_files = collect_files(
        root,
        glob_pattern=glob_pattern,
        recursive=recursive,
        allowed_extensions=allowed_extensions,
    )
    basename_counts = Counter(path.name for path in image_files)
    duplicate_basenames = {name for name, count in basename_counts.items() if count > 1}
    records = [
        validate_image(path, root_path, duplicate_basenames)
        for path in image_files
    ]
    status_counts = Counter(record["status"] for record in records)
    report_status = "ok"
    if status_counts.get("error", 0) > 0 or not image_files:
        report_status = "error"
    elif status_counts.get("warning", 0) > 0:
        report_status = "warning"

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "root": str(root_path),
        "glob": glob_pattern or default_glob_pattern(),
        "recursive": default_recursive() if recursive is None else bool(recursive),
        "status": report_status,
        "total_images": len(image_files),
        "ok_count": status_counts.get("ok", 0),
        "warning_count": status_counts.get("warning", 0),
        "error_count": status_counts.get("error", 0),
        "duplicate_basename_count": len(duplicate_basenames),
        "duplicate_basenames": sorted(duplicate_basenames),
        "records": records,
    }


def write_json_report(payload: dict[str, Any], output_dir: str | Path, stem: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / f"{stem}.json"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def write_validation_csv(report: dict[str, Any], output_dir: str | Path, stem: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / f"{stem}.csv"
    records = list(report.get("records") or [])
    fieldnames = [
        "status",
        "issues",
        "warnings",
        "file_name",
        "relative_path",
        "parent_folder",
        "extension",
        "size_bytes",
        "width",
        "height",
        "format",
        "mode",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return path
