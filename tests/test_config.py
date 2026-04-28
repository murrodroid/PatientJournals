import pytest

from patientjournals.config.settings import Config


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

