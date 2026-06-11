from __future__ import annotations

from patientjournals.app.models import ModelOption, SchemaOption
from patientjournals.config.models import registered_google_models
from patientjournals.config.schemas import list_output_schemas, resolve_output_schema


def list_schema_options(*, include_nested: bool = False) -> list[SchemaOption]:
    schemas = list_output_schemas()
    options: list[SchemaOption] = []
    for name, cls in sorted(schemas.items()):
        options.append(
            SchemaOption(
                name=name,
                module=cls.__module__,
                field_count=len(cls.model_fields),
                is_top_level=True,
            )
        )
    return options


def resolve_schema_class(name: str):
    return resolve_output_schema(name)


def _model_option_from_name(name: str) -> ModelOption:
    return ModelOption(
        name=name,
        provider="gemini",
        supports_batch=True,
        supports_confidence_scores=True,
        supports_thoughts=True,
    )


def list_live_google_model_options(*, limit: int = 200) -> list[ModelOption]:
    from patientjournals.batch.client import get_batch_client

    client = get_batch_client()
    options: list[ModelOption] = []
    for index, model in enumerate(client.models.list(), start=1):
        if index > limit:
            break
        raw_name = str(getattr(model, "name", "") or "").strip()
        if not raw_name:
            continue
        short_name = raw_name.split("/")[-1]
        if not short_name.startswith("gemini-"):
            continue
        options.append(_model_option_from_name(short_name))

    deduped: dict[str, ModelOption] = {}
    for option in options:
        deduped.setdefault(option.name, option)
    return sorted(deduped.values(), key=lambda item: item.name)


def list_google_model_options(*, include_live: bool = False) -> list[ModelOption]:
    if include_live:
        try:
            live = list_live_google_model_options()
        except Exception:
            live = []
        if live:
            return live

    return [
        ModelOption(
            name=spec.name,
            provider=spec.provider,
            supports_batch=spec.supports_batch,
            supports_confidence_scores=spec.supports_confidence_scores,
            supports_thoughts=spec.supports_thoughts,
        )
        for spec in registered_google_models()
    ]
