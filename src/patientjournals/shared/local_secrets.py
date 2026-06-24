from __future__ import annotations

import json
import os
from pathlib import Path


def local_secrets_path(root: str | Path | None = None) -> Path:
    base = Path(root).expanduser() if root is not None else Path.home() / ".patientjournals"
    return base / "secrets.json"


def load_local_api_keys(path: str | Path | None = None) -> dict[str, str]:
    secrets_path = Path(path).expanduser() if path is not None else local_secrets_path()
    if not secrets_path.exists():
        return {}
    try:
        payload = json.loads(secrets_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    keys = payload.get("provider_api_keys") or {}
    if not isinstance(keys, dict):
        return {}
    return {
        str(provider).strip().lower(): str(value).strip()
        for provider, value in keys.items()
        if str(provider).strip() and str(value).strip()
    }


def save_local_api_key(
    provider: str,
    api_key: str,
    *,
    path: str | Path | None = None,
) -> Path:
    provider_name = str(provider or "").strip().lower()
    value = str(api_key or "").strip()
    if not provider_name:
        raise ValueError("Provider name is empty.")
    if not value:
        raise ValueError("API key is empty.")

    secrets_path = Path(path).expanduser() if path is not None else local_secrets_path()
    payload: dict[str, object] = {}
    if secrets_path.exists():
        try:
            current = json.loads(secrets_path.read_text(encoding="utf-8"))
            if isinstance(current, dict):
                payload = current
        except (OSError, json.JSONDecodeError):
            payload = {}

    keys = payload.get("provider_api_keys") or {}
    if not isinstance(keys, dict):
        keys = {}
    keys[provider_name] = value
    payload["provider_api_keys"] = keys

    secrets_path.parent.mkdir(parents=True, exist_ok=True)
    secrets_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    try:
        os.chmod(secrets_path, 0o600)
    except OSError:
        pass
    return secrets_path
