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
    thinking = str(values.get("thinking_level") or "high")
    values["thinking_level"] = (
        thinking if thinking in {"low", "medium", "high"} else "high"
    )
    verification_thinking = str(
        values.get("verification_thinking_level") or "high"
    )
    values["verification_thinking_level"] = (
        verification_thinking
        if verification_thinking in {"low", "medium", "high"}
        else "high"
    )
    scope = str(values.get("verification_scope") or "flagged")
    values["verification_scope"] = scope if scope in {"all", "flagged"} else "flagged"
    values["verification_control_sample_percent"] = max(
        0.0,
        min(100.0, float(values.get("verification_control_sample_percent") or 0.0)),
    )
    # Final-model corrections remain automatic after the immutable schema and
    # complete-population consolidation gates.
    values["verification_apply_mode"] = "apply_patches"
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
    thinking_level: str = "",
    schema_name: str = "",
    schema_version_id: str = "",
    schema_payload: dict[str, object] | None = None,
    output_format: str = "",
    local_path: str = "",
    cloud_prefix: str = "",
    cloud_prefixes: tuple[str, ...] = (),
    submission_type: str = "complete",
    sample_percent: float | None = None,
    sample_seed: str = "",
    duplicate_strategy: str = "",
    ocr_enabled: bool | None = None,
    subagents: bool | None = None,
    model_validation_enabled: bool | None = None,
    verification_model: str = "",
    verification_thinking_level: str = "",
    verification_scope: str = "",
    verification_control_sample_percent: float | None = None,
    verification_apply_mode: str = "",
    verification_max_output_tokens: int | None = None,
    verification_num_chunks: int | None = None,
) -> dict[str, object]:
    payload = settings.to_json_dict()
    if model_name:
        payload["model"] = model_name
    if thinking_level:
        payload["thinking_level"] = thinking_level
    if schema_name:
        payload["schema_name"] = schema_name
        payload["output_schema_name"] = schema_name
    if schema_version_id:
        payload["schema_version_id"] = schema_version_id
        payload["output_schema_version_id"] = schema_version_id
    if schema_payload:
        payload["schema_payload"] = schema_payload
        payload["output_schema_override"] = schema_payload
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
    payload["batch_submission_type"] = submission_type
    if submission_type == "sample" and sample_percent is not None:
        payload["batch_sample_percent"] = float(sample_percent)
        payload["batch_sample_seed"] = str(sample_seed)
    if duplicate_strategy:
        payload["batch_duplicate_strategy"] = duplicate_strategy
    if ocr_enabled is not None:
        payload["ocr_enabled"] = bool(ocr_enabled)
    if subagents is not None:
        payload["subagents"] = bool(subagents)
    if model_validation_enabled is not None:
        payload["model_validation_enabled"] = bool(model_validation_enabled)
    if verification_model:
        payload["verification_model"] = verification_model
    if verification_thinking_level:
        payload["verification_thinking_level"] = verification_thinking_level
    if verification_scope:
        payload["verification_scope"] = verification_scope
    if verification_control_sample_percent is not None:
        payload["verification_control_sample_percent"] = max(
            0.0, min(100.0, float(verification_control_sample_percent))
        )
    if verification_apply_mode:
        payload["verification_apply_mode"] = verification_apply_mode
    if verification_max_output_tokens is not None:
        payload["verification_max_output_tokens"] = max(
            1, int(verification_max_output_tokens)
        )
    if verification_num_chunks is not None:
        payload["verification_num_chunks"] = max(1, int(verification_num_chunks))
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
