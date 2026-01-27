from __future__ import annotations

import pandas as pd
from datetime import datetime
from pathlib import Path
import json
import shutil
import types
import traceback
from dataclasses import asdict, is_dataclass

import config as config_module
from config import config


def data_to_row(data: Journal, file_name: str) -> dict:
    row = data.model_dump(mode="python")
    row["file_name"] = file_name
    return row

def _normalize_output_format(output_format: str) -> str:
    fmt = output_format.strip().lower().lstrip(".")
    if fmt not in {"csv", "jsonl"}:
        raise ValueError(f"Unsupported output_format: {output_format}")
    return fmt

def _rows_to_flat_dataframe(rows: list[dict]) -> pd.DataFrame:
    return pd.json_normalize(rows, sep=".")

def flush_rows(
    rows: list[dict],
    out_path: str,
    header_written: bool,
    output_format: str,
    sep: str = "$",
) -> bool:
    fmt = _normalize_output_format(output_format)
    if fmt == "csv":
        _rows_to_flat_dataframe(rows).to_csv(
            out_path,
            mode="a",
            index=False,
            header=not header_written,
            sep=sep,
        )
        return True

    with open(out_path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str))
            handle.write("\n")
    return header_written

def create_subfolder(root: str | Path = "runs") -> Path:
    root_path = Path(root)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root_path / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    raw_config = Path(config_module.__file__).read_text(encoding="utf-8")
    (run_dir / "config_snapshot.py").write_text(raw_config, encoding="utf-8")

    def serializable_config(module: types.ModuleType) -> dict:
        out: dict[str, object] = {}
        cfg = getattr(module, "config", None)
        if cfg is not None and is_dataclass(cfg):
            out["config"] = asdict(cfg)
        return out

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_file": "config_snapshot.py",
        "config_values": serializable_config(config_module),
        "output_schema": config.output_schema,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return run_dir

def _cfg_get(cfg_obj: object, key: str, default: object | None = None):
    if isinstance(cfg_obj, dict):
        return cfg_obj.get(key, default)
    if hasattr(cfg_obj, key):
        return getattr(cfg_obj, key)
    return default

def list_input_files(cfg_obj: object) -> list[str]:
    folder_value = _cfg_get(cfg_obj, "target_folder")
    if folder_value is None:
        raise KeyError("target_folder is missing from config")
    folder = Path(folder_value).expanduser()
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"target_folder not found or not a directory: {folder}")

    pattern = _cfg_get(cfg_obj, "input_glob", "*")
    recursive = bool(_cfg_get(cfg_obj, "recursive", False))

    paths = folder.rglob(pattern) if recursive else folder.glob(pattern)
    files = sorted(p for p in paths if p.is_file())
    if not files:
        raise FileNotFoundError(f"No files matched {pattern} in {folder} (recursive={recursive})")

    return [str(p) for p in files]

def normalize_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())

def _candidate_path_ids(path: str | Path, target_folder: str | Path | None) -> set[str]:
    p = Path(path)
    ids = {normalize_path(p)}
    if not p.is_absolute() and target_folder is not None:
        base = Path(target_folder)
        rel_parts = base.parts
        p_parts = p.parts
        if not (rel_parts and list(p_parts[:len(rel_parts)]) == list(rel_parts)):
            ids.add(normalize_path(base / p))
    return ids

def build_path_id_set(
    paths: list[str],
    target_folder: str | Path | None = None,
) -> set[str]:
    ids: set[str] = set()
    for p in paths:
        ids.update(_candidate_path_ids(p, target_folder))
    return ids

def filter_dataset_by_input_ids(
    src_path: str | Path,
    dest_path: str | Path,
    input_ids: set[str],
    output_format: str | None = None,
    csv_sep: str = "$",
    target_folder: str | Path | None = None,
) -> int:
    src = Path(src_path).expanduser()
    dest = Path(dest_path).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)

    fmt = _normalize_output_format(output_format or src.suffix.lstrip("."))
    kept = 0

    if fmt == "jsonl":
        with open(src, "r", encoding="utf-8") as src_handle, open(dest, "w", encoding="utf-8") as dst_handle:
            for line in src_handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                name = payload.get("file_name")
                if not isinstance(name, str) or not name:
                    continue
                candidate_ids = _candidate_path_ids(name, target_folder)
                if input_ids.intersection(candidate_ids):
                    dst_handle.write(raw)
                    dst_handle.write("\n")
                    kept += 1
        return kept

    df = pd.read_csv(src, sep=csv_sep)
    if "file_name" in df.columns:
        def _match(name: object) -> bool:
            if not isinstance(name, str) or not name:
                return False
            return bool(input_ids.intersection(_candidate_path_ids(name, target_folder)))
        mask = df["file_name"].apply(_match)
        df = df[mask]
    kept = len(df)
    df.to_csv(dest, index=False, sep=csv_sep)
    return kept

def find_newest_dataset(
    run_root: str | Path,
    dataset_name: str = "dataset",
) -> Path:
    root = Path(run_root).expanduser()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"runs folder not found: {root}")

    run_dirs = sorted(
        (p for p in root.iterdir() if p.is_dir()),
        reverse=True,
    )
    patterns = (
        f"*_{dataset_name}.jsonl",
        f"*_{dataset_name}.csv",
    )
    for run_dir in run_dirs:
        for pattern in patterns:
            matches = sorted(run_dir.glob(pattern))
            if matches:
                return matches[-1]

    raise FileNotFoundError(
        f"No dataset files found in {root} for name '{dataset_name}'."
    )

def load_existing_dataset(
    dataset_path: str | Path,
    output_format: str | None = None,
    csv_sep: str = "$",
) -> tuple[str, set[str], int]:
    path = Path(dataset_path).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"dataset not found or not a file: {path}")

    fmt = _normalize_output_format(
        output_format or path.suffix.lstrip(".")
    )

    file_names: set[str] = set()
    row_count = 0

    if fmt == "jsonl":
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    name = payload.get("file_name")
                    if isinstance(name, str) and name:
                        file_names.add(name)
                row_count += 1
    else:
        df = pd.read_csv(path, sep=csv_sep)
        row_count = len(df)
        if "file_name" in df.columns:
            file_names = set(df["file_name"].dropna().astype(str))

    return fmt, file_names, row_count

def copy_dataset(src_path: str | Path, dest_path: str | Path) -> None:
    src = Path(src_path).expanduser()
    dest = Path(dest_path).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)

def write_run_error(run_dir: str | Path, exc: BaseException) -> Path:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    err_path = run_path / f"error_{stamp}.txt"

    msg = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    err_path.write_text(msg, encoding="utf-8")
    return err_path

def get_run_logger(run_dir: str | Path, log_name: str = "run.log"):
    log_path = Path(run_dir) / log_name

    def log(message: str, exc: BaseException | None = None) -> None:
        stamp = datetime.now().isoformat(timespec="seconds")
        line = f"[{stamp}] {message}"
        if exc is not None:
            detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            line = f"{line}\n{detail}"
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line)
            if not line.endswith("\n"):
                handle.write("\n")

    return log
