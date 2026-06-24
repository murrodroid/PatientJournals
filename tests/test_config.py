import sys
import types

import pytest

from patientjournals.config import settings as settings_module
from patientjournals.config.settings import Config
from patientjournals.shared.local_secrets import load_local_api_keys, save_local_api_key


def test_config_normalizes_provider_keys() -> None:
    cfg = Config(
        provider_api_keys={" OpenAI ": "  openai-key  "},
        api_key="",
    )

    assert cfg.api_key_for_provider("openai") == "openai-key"


def test_config_rejects_missing_provider_key() -> None:
    cfg = Config(provider_api_keys={}, api_key="")

    with pytest.raises(ValueError, match="No API key configured"):
        cfg.api_key_for_provider("anthropic")


def test_api_recovery_model_is_separate_from_standard_model() -> None:
    cfg = Config()

    assert cfg.model == "gemini-3.1-pro"
    assert cfg.api_recovery_model == "gemini-3.1-pro-preview"


def test_local_api_secret_roundtrip(tmp_path) -> None:
    path = save_local_api_key("gemini", "local-key", path=tmp_path / "secrets.json")

    assert load_local_api_keys(path) == {"gemini": "local-key"}


def test_config_loader_reads_local_api_secret(tmp_path, monkeypatch) -> None:
    fake_api_keys = types.ModuleType("api_keys")
    monkeypatch.setitem(sys.modules, "api_keys", fake_api_keys)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    save_local_api_key("gemini", "local-key")

    assert settings_module._load_provider_api_keys()["gemini"] == "local-key"
