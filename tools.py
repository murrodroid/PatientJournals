from __future__ import annotations

import pandas as pd
from datetime import datetime
from pathlib import Path
import json
import types
import traceback

import config as config_module
from config import cfg
from classes import Journal


def data_to_row(data: Journal, file_name: str) -> dict:
    row = data.model_dump(mode="python")
    row["file_name"] = file_name
    return row

def flush_csv(rows: list[dict], out_csv: str, header_written: bool, sep: str = '$') -> bool:
    pd.DataFrame.from_records(rows).to_csv(
        out_csv,
        mode="a",
        index=False,
        header=not header_written,
        sep=sep
    )
    return True

def create_subfolder(root: str | Path = "runs") -> Path:
    root_path = Path(root)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root_path / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    raw_config = Path(config_module.__file__).read_text(encoding="utf-8")
    (run_dir / "config_snapshot.py").write_text(raw_config, encoding="utf-8")

    def serializable_config(module: types.ModuleType) -> dict:
        out: dict[str, object] = {}
        for k, v in module.__dict__.items():
            if k.startswith("_"):
                continue
            if isinstance(v, types.ModuleType):
                continue
            if callable(v):
                continue
            out[k] = v
        return out

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_file": "config_snapshot.py",
        "config_values": serializable_config(config_module),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return run_dir

def list_input_files(cfg: dict) -> list[str]:
    folder = Path(cfg["target_folder"]).expanduser()
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"target_folder not found or not a directory: {folder}")

    pattern = cfg.get("input_glob", "*")
    recursive = bool(cfg.get("recursive", False))

    paths = folder.rglob(pattern) if recursive else folder.glob(pattern)
    files = sorted(p for p in paths if p.is_file())
    if not files:
        raise FileNotFoundError(f"No files matched {pattern} in {folder} (recursive={recursive})")

    return [str(p) for p in files]

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


