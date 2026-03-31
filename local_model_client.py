from __future__ import annotations

import base64
import inspect
from dataclasses import dataclass
from typing import Any

from config import config
from generation_spec import build_live_generation_config, build_live_request_contents
from models import ModelSpec, ProviderName, resolve_model_spec
from response_parsing import extract_response_metadata


def _pick_value(obj: object, *names: str) -> object | None:
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
        return None
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


@dataclass
class LocalGenerationResult:
    text: str
    thoughts: str | None = None
    field_confidence_by_pointer: dict[str, dict[str, float | None]] | None = None


def _import_openai_async_client():
    try:
        from openai import AsyncOpenAI
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "OpenAI provider requested but the 'openai' package is unavailable. "
            "Add it to dependencies and install."
        ) from exc
    return AsyncOpenAI


def _import_anthropic_async_client():
    try:
        from anthropic import AsyncAnthropic
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Anthropic provider requested but the 'anthropic' package is unavailable. "
            "Add it to dependencies and install."
        ) from exc
    return AsyncAnthropic


def _build_provider_client(provider: ProviderName) -> object:
    api_key = config.api_key_for_provider(provider)

    if provider == "gemini":
        from google import genai

        return genai.Client(api_key=api_key)

    if provider == "openai":
        AsyncOpenAI = _import_openai_async_client()
        return AsyncOpenAI(api_key=api_key)

    if provider == "anthropic":
        AsyncAnthropic = _import_anthropic_async_client()
        return AsyncAnthropic(api_key=api_key)

    raise ValueError(f"Unsupported provider '{provider}'.")


def _extract_openai_response_text(response: object) -> str | None:
    output_text = _pick_value(response, "output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    if isinstance(output_text, list):
        joined = "".join(item for item in output_text if isinstance(item, str)).strip()
        if joined:
            return joined

    output = _pick_value(response, "output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            content = _pick_value(item, "content")
            if not isinstance(content, list):
                continue
            for block in content:
                block_type = _pick_value(block, "type")
                if block_type not in {"text", "output_text"}:
                    continue
                text = _pick_value(block, "text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text)
        if chunks:
            return "".join(chunks).strip()
    return None


def _extract_anthropic_response_text(response: object) -> str | None:
    content = _pick_value(response, "content")
    if not isinstance(content, list):
        return None

    chunks: list[str] = []
    for block in content:
        block_type = _pick_value(block, "type")
        if block_type != "text":
            continue
        text = _pick_value(block, "text")
        if isinstance(text, str) and text.strip():
            chunks.append(text)

    if not chunks:
        return None
    return "".join(chunks).strip()


class LocalModelClient:
    def __init__(self, model_name: str):
        self.model_spec: ModelSpec = resolve_model_spec(model_name)
        self.model_name = model_name
        self.provider: ProviderName = self.model_spec.provider
        self.client = _build_provider_client(self.provider)

    def capability_warnings(self) -> list[str]:
        warnings: list[str] = []
        if config.include_confidence_scores and not self.model_spec.supports_confidence_scores:
            warnings.append(
                "include_confidence_scores=True is not supported by "
                f"provider '{self.provider}'. Field confidence output will be empty."
            )
        if config.include_thoughts and not self.model_spec.supports_thoughts:
            warnings.append(
                f"include_thoughts=True is not supported by provider '{self.provider}'."
            )
        return warnings

    async def generate_json(self, *, image_bytes: bytes, mime_type: str) -> LocalGenerationResult:
        if self.provider == "gemini":
            return await self._generate_with_gemini(image_bytes=image_bytes, mime_type=mime_type)
        if self.provider == "openai":
            return await self._generate_with_openai(image_bytes=image_bytes, mime_type=mime_type)
        if self.provider == "anthropic":
            return await self._generate_with_anthropic(image_bytes=image_bytes, mime_type=mime_type)
        raise ValueError(f"Unsupported provider '{self.provider}'.")

    async def _generate_with_gemini(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
    ) -> LocalGenerationResult:
        output = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=build_live_request_contents(image_bytes=image_bytes, mime_type=mime_type),
            config=build_live_generation_config(
                include_schema=True,
                include_temperature=True,
                include_thinking_level=True,
            ),
        )
        metadata = extract_response_metadata(output)
        text = metadata.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty response text from Gemini API.")
        return LocalGenerationResult(
            text=text,
            thoughts=metadata.get("thoughts"),
            field_confidence_by_pointer=metadata.get("field_confidence_by_pointer") or {},
        )

    async def _generate_with_openai(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
    ) -> LocalGenerationResult:
        image_data = base64.b64encode(image_bytes).decode("ascii")
        response = await self.client.responses.create(
            model=self.model_name,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": config.input_prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:{mime_type};base64,{image_data}",
                        },
                    ],
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "journal_output",
                    "schema": config.output_schema,
                    "strict": True,
                }
            },
            temperature=config.model_temperature,
            max_output_tokens=max(1, int(config.model_max_output_tokens)),
        )
        text = _extract_openai_response_text(response)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty response text from OpenAI API.")
        return LocalGenerationResult(
            text=text,
            thoughts=None,
            field_confidence_by_pointer={},
        )

    async def _generate_with_anthropic(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
    ) -> LocalGenerationResult:
        image_data = base64.b64encode(image_bytes).decode("ascii")
        response = await self.client.messages.create(
            model=self.model_name,
            max_tokens=max(1, int(config.model_max_output_tokens)),
            temperature=config.model_temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": config.input_prompt},
                    ],
                }
            ],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": config.output_schema,
                }
            },
        )
        text = _extract_anthropic_response_text(response)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty response text from Anthropic API.")
        return LocalGenerationResult(
            text=text,
            thoughts=None,
            field_confidence_by_pointer={},
        )

    async def aclose(self) -> None:
        close_candidates = ("aclose", "close")
        for method_name in close_candidates:
            method = getattr(self.client, method_name, None)
            if not callable(method):
                continue
            result = method()
            if inspect.isawaitable(result):
                await result
            return


def create_local_model_client(model_name: str) -> LocalModelClient:
    return LocalModelClient(model_name=model_name)
