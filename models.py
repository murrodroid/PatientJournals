from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ProviderName = Literal["gemini", "openai", "anthropic"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: ProviderName
    supports_batch: bool = False
    supports_confidence_scores: bool = False
    supports_thoughts: bool = False


_REGISTERED_MODELS: dict[str, ModelSpec] = {
    # Gemini
    "gemini-3.1-pro-preview": ModelSpec(
        name="gemini-3.1-pro-preview",
        provider="gemini",
        supports_batch=True,
        supports_confidence_scores=True,
        supports_thoughts=True,
    ),
    "gemini-2.5-pro": ModelSpec(
        name="gemini-2.5-pro",
        provider="gemini",
        supports_batch=True,
        supports_confidence_scores=True,
        supports_thoughts=True,
    ),
    "gemini-2.5-flash": ModelSpec(
        name="gemini-2.5-flash",
        provider="gemini",
        supports_batch=True,
        supports_confidence_scores=True,
        supports_thoughts=True,
    ),
    # OpenAI
    "gpt-5": ModelSpec(name="gpt-5", provider="openai"),
    "gpt-5-mini": ModelSpec(name="gpt-5-mini", provider="openai"),
    "gpt-4.1": ModelSpec(name="gpt-4.1", provider="openai"),
    "gpt-4.1-mini": ModelSpec(name="gpt-4.1-mini", provider="openai"),
    # Anthropic
    "claude-opus-4-6": ModelSpec(
        name="claude-opus-4-6",
        provider="anthropic",
        supports_batch=True,
    ),
    "claude-sonnet-4-5": ModelSpec(
        name="claude-sonnet-4-5",
        provider="anthropic",
        supports_batch=True,
    ),
    "claude-haiku-4-5": ModelSpec(
        name="claude-haiku-4-5",
        provider="anthropic",
        supports_batch=True,
    ),
}


def all_registered_models() -> tuple[ModelSpec, ...]:
    return tuple(sorted(_REGISTERED_MODELS.values(), key=lambda item: item.name))


def _infer_provider_from_model_name(model_name: str) -> ProviderName | None:
    value = model_name.strip().lower()
    if not value:
        return None

    if (
        value.startswith("gemini-")
        or value.startswith("models/gemini-")
        or value.startswith("publishers/google/models/gemini-")
    ):
        return "gemini"

    if value.startswith("claude-") or value.startswith("anthropic/claude-"):
        return "anthropic"

    if (
        value.startswith("openai/")
        or value.startswith("models/gpt-")
        or value.startswith("gpt-")
        or value.startswith("o1")
        or value.startswith("o3")
        or value.startswith("o4")
        or value.startswith("o5")
    ):
        return "openai"

    return None


def resolve_model_spec(model_name: str) -> ModelSpec:
    normalized = model_name.strip()
    if not normalized:
        raise ValueError("config.model is empty.")

    known = _REGISTERED_MODELS.get(normalized)
    if known is not None:
        return known

    provider = _infer_provider_from_model_name(normalized)
    if provider is None:
        available = ", ".join(sorted(_REGISTERED_MODELS.keys()))
        raise ValueError(
            f"Unable to infer provider for model '{model_name}'. "
            "Add it to models.py or use a provider-specific model prefix. "
            f"Registered models: {available}"
        )

    inferred_capabilities = {
        "gemini": dict(
            supports_batch=True,
            supports_confidence_scores=True,
            supports_thoughts=True,
        ),
        "openai": dict(
            supports_batch=False,
            supports_confidence_scores=False,
            supports_thoughts=False,
        ),
        "anthropic": dict(
            supports_batch=True,
            supports_confidence_scores=False,
            supports_thoughts=False,
        ),
    }[provider]
    return ModelSpec(
        name=normalized,
        provider=provider,
        supports_batch=inferred_capabilities["supports_batch"],
        supports_confidence_scores=inferred_capabilities["supports_confidence_scores"],
        supports_thoughts=inferred_capabilities["supports_thoughts"],
    )
