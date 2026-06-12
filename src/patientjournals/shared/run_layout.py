"""Central conventions for the runs/ output folder.

All job output lives under a single root (``runs/`` by default), organised into
one subfolder per job type::

    runs/
      submits/<timestamp>/
      retrieves/<timestamp>/
      collect_outputs/<timestamp>/
      local/<timestamp>/

Each run directory carries a ``metadata.json`` documenting its ``kind`` and the
config it ran with. Older flat run directories (``submit_<ts>``, ``retrieve_<ts>``,
``collect_outputs_<ts>`` and bare ``<ts>`` local runs) are still discovered for
backward compatibility, so the two layouts can coexist during migration.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

# Logical job category -> subfolder name under the runs root.
CATEGORY_DIRS: dict[str, str] = {
    "submit": "submits",
    "retrieve": "retrieves",
    "collect_outputs": "collect_outputs",
    "local": "local",
}

# Legacy flat-directory prefixes that used to encode the job type in the name.
LEGACY_PREFIXES: dict[str, str] = {
    "submit": "submit_",
    "retrieve": "retrieve_",
    "collect_outputs": "collect_outputs_",
}

_TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")
_SUBFOLDER_NAMES = set(CATEGORY_DIRS.values())


def category_root(root: str | Path, category: str) -> Path:
    """Return the subfolder that holds runs of ``category`` under ``root``."""
    if category not in CATEGORY_DIRS:
        raise ValueError(f"Unknown run category: {category!r}")
    return Path(root).expanduser() / CATEGORY_DIRS[category]


def _sorted_by_recency(dirs: list[Path]) -> list[Path]:
    def sort_key(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    return sorted(dirs, key=sort_key, reverse=True)


def iter_run_dirs(root: str | Path, category: str) -> list[Path]:
    """Run directories for a category: new subfolder plus legacy flat dirs."""
    root_path = Path(root).expanduser()
    dirs: list[Path] = []
    sub = root_path / CATEGORY_DIRS[category]
    if sub.is_dir():
        dirs.extend(item for item in sub.iterdir() if item.is_dir())
    prefix = LEGACY_PREFIXES.get(category)
    if prefix and root_path.is_dir():
        dirs.extend(
            item
            for item in root_path.iterdir()
            if item.is_dir() and item.name.startswith(prefix)
        )
    return _sorted_by_recency(dirs)


def iter_all_run_dirs(root: str | Path) -> list[Path]:
    """Every run directory across all categories (new layout and legacy)."""
    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        return []
    dirs: list[Path] = []
    for item in root_path.iterdir():
        if not item.is_dir():
            continue
        if item.name in _SUBFOLDER_NAMES:
            dirs.extend(child for child in item.iterdir() if child.is_dir())
        else:
            dirs.append(item)  # legacy flat run dir
    return _sorted_by_recency(dirs)


def classify_legacy_dir(name: str) -> str | None:
    """Return the category for a legacy flat run dir name, or None if unknown."""
    for category, prefix in LEGACY_PREFIXES.items():
        if name.startswith(prefix):
            return category
    if _TIMESTAMP_RE.match(name):
        return "local"  # bare-timestamp dirs were local-API runs
    return None


def _stripped_run_name(name: str, category: str) -> str:
    prefix = LEGACY_PREFIXES.get(category, "")
    return name[len(prefix):] if prefix and name.startswith(prefix) else name


README_NAME = "README.md"

_README_BODY = """# Runs

Job output is organised into one subfolder per job type. Each run directory is
named by its timestamp (`YYYYMMDD_HHMMSS`) and contains a `metadata.json` that
documents its `kind` and the configuration it ran with.

```
runs/
  submits/<timestamp>/          # batch submissions (batch_job.json, requests)
  retrieves/<timestamp>/        # batch retrievals (dataset, batch_results.json)
  collect_outputs/<timestamp>/  # output collection runs
  local/<timestamp>/            # local-API processing runs (dataset)
```

This file is generated automatically; edits may be overwritten.
"""


def write_runs_readme(root: str | Path) -> Path:
    """Write a README documenting the runs/ layout. Returns its path."""
    root_path = Path(root).expanduser()
    root_path.mkdir(parents=True, exist_ok=True)
    readme = root_path / README_NAME
    readme.write_text(_README_BODY, encoding="utf-8")
    return readme


def _created_at_from_name(name: str) -> str:
    match = re.match(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", name)
    if not match:
        return ""
    year, month, day, hour, minute, second = match.groups()
    return f"{year}-{month}-{day}T{hour}:{minute}:{second}"


def document_existing_runs(root: str | Path) -> int:
    """Ensure every run directory's metadata.json records its ``kind``.

    Returns the number of metadata files written. New runs get this for free at
    creation time; this backfills runs that predate the per-type layout.
    """
    root_path = Path(root).expanduser()
    documented = 0
    for category, subdir in CATEGORY_DIRS.items():
        base = root_path / subdir
        if not base.is_dir():
            continue
        for run_dir in base.iterdir():
            if not run_dir.is_dir():
                continue
            meta_path = run_dir / "metadata.json"
            meta: dict = {}
            if meta_path.exists():
                try:
                    loaded = json.loads(meta_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        meta = loaded
                except (OSError, json.JSONDecodeError):
                    meta = {}
            if meta.get("kind") == category:
                continue
            meta["kind"] = category
            meta.setdefault("created_at", _created_at_from_name(run_dir.name))
            meta_path.write_text(
                json.dumps(meta, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            documented += 1
    return documented


def reorganize_runs(root: str | Path, *, apply: bool = True) -> dict:
    """Move legacy flat run directories into the per-type subfolders.

    Returns a report describing what moved (or would move, when ``apply`` is
    False) and how many dataset references were rewritten. References inside
    ``batch_results.json`` files are updated so retrieval results keep resolving.
    """
    root_path = Path(root).expanduser()
    report: dict = {"moved": [], "skipped": [], "reference_fixes": 0, "applied": apply}
    if not root_path.is_dir():
        return report

    # old flat-dir name -> (category, new timestamp name)
    renames: dict[str, tuple[str, str]] = {}
    for item in sorted(root_path.iterdir()):
        if not item.is_dir() or item.name in _SUBFOLDER_NAMES:
            continue
        category = classify_legacy_dir(item.name)
        if category is None:
            report["skipped"].append({"name": item.name, "reason": "unrecognised"})
            continue
        new_name = _stripped_run_name(item.name, category)
        dest = root_path / CATEGORY_DIRS[category] / new_name
        if dest.exists():
            report["skipped"].append({"name": item.name, "reason": "destination exists"})
            continue
        renames[item.name] = (category, new_name)
        report["moved"].append(
            {
                "from": item.name,
                "to": f"{CATEGORY_DIRS[category]}/{new_name}",
                "category": category,
            }
        )
        if apply:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item), str(dest))

    if apply and renames:
        report["reference_fixes"] = _rewrite_dataset_references(root_path, renames)
    if apply:
        report["documented"] = document_existing_runs(root_path)
        write_runs_readme(root_path)
    return report


def _rewrite_dataset_references(
    root_path: Path,
    renames: dict[str, tuple[str, str]],
) -> int:
    """Rewrite batch_results.json dataset paths after legacy dirs were moved."""
    replacements = {
        old_name: f"{CATEGORY_DIRS[category]}/{new_name}"
        for old_name, (category, new_name) in renames.items()
    }
    fixes = 0
    for results_path in root_path.rglob("batch_results.json"):
        try:
            payload = json.loads(results_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        dataset_path = str(payload.get("dataset_path") or "")
        new_path = dataset_path
        for old_name, replacement in replacements.items():
            if f"{old_name}/" in new_path:
                new_path = new_path.replace(f"{old_name}/", f"{replacement}/")
        if new_path != dataset_path:
            payload["dataset_path"] = new_path
            results_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            fixes += 1
    return fixes
