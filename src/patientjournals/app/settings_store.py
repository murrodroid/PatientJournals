from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from patientjournals.app.models import AppSettings, app_settings_path


def _coerce_settings(payload: dict[str, object]) -> AppSettings:
    defaults = AppSettings.from_runtime_config().to_json_dict()
    allowed = {field.name for field in fields(AppSettings)}
    values = {
        key: payload.get(key, defaults.get(key))
        for key in allowed
    }
    return AppSettings(**values)  # type: ignore[arg-type]


def load_app_settings(path: str | Path | None = None) -> AppSettings:
    config_path = Path(path).expanduser() if path else app_settings_path()
    if not config_path.exists():
        return AppSettings.from_runtime_config()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid app settings JSON: {config_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid app settings payload: {config_path}")
    return _coerce_settings(payload)


def save_app_settings(
    settings: AppSettings,
    path: str | Path | None = None,
) -> Path:
    config_path = Path(path).expanduser() if path else app_settings_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(settings.to_json_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return config_path


def command_override_payload(
    settings: AppSettings,
    *,
    model_name: str = "",
    schema_name: str = "",
    output_format: str = "",
    local_path: str = "",
    cloud_prefix: str = "",
    cloud_prefixes: tuple[str, ...] = (),
    duplicate_strategy: str = "",
) -> dict[str, object]:
    payload = settings.to_json_dict()
    if model_name:
        payload["model"] = model_name
    if schema_name:
        payload["schema_name"] = schema_name
    if output_format:
        payload["output_format"] = output_format
    if local_path:
        payload["target_folder"] = local_path
        payload["upload_images_folder"] = local_path
    selected_prefixes = tuple(prefix for prefix in cloud_prefixes if prefix)
    if selected_prefixes:
        payload["batch_input_prefixes"] = selected_prefixes
        payload["batch_input_prefix"] = selected_prefixes[0]
    elif cloud_prefix:
        payload["batch_input_prefix"] = cloud_prefix
        payload["batch_input_prefixes"] = (cloud_prefix,)
    if duplicate_strategy:
        payload["batch_duplicate_strategy"] = duplicate_strategy
    return payload


def write_command_overrides(
    payload: dict[str, object],
    *,
    root: str | Path | None = None,
    stem: str = "job_config",
) -> Path:
    base = app_settings_path(root).parent
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{stem}.json"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
