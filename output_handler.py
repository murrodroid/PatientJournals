from __future__ import annotations

from typing import Callable

from pydantic import BaseModel

from schemas import FrontPage, TextPage


FieldConfidenceByPointer = dict[str, dict[str, float | None]]


def _escape_pointer_segment(segment: str) -> str:
    return segment.replace("~", "~0").replace("/", "~1")


def _pointer_from_path(path: tuple[str, ...]) -> str:
    if not path:
        return ""
    return "/" + "/".join(_escape_pointer_segment(segment) for segment in path)


def _leaf_confidence_payload(
    field_confidence_by_pointer: FieldConfidenceByPointer | None,
    pointer: str,
) -> dict[str, float | None]:
    metrics = (
        field_confidence_by_pointer.get(pointer)
        if isinstance(field_confidence_by_pointer, dict)
        else None
    )
    return {
        "field_confidence_logprobs": (
            metrics.get("field_confidence_logprobs")
            if isinstance(metrics, dict)
            else None
        ),
        "field_confidence_ratio": (
            metrics.get("field_confidence_ratio")
            if isinstance(metrics, dict)
            else None
        ),
    }


def _build_confidence_tree(
    value: object,
    field_confidence_by_pointer: FieldConfidenceByPointer | None,
    *,
    path: tuple[str, ...],
) -> object:
    if isinstance(value, dict):
        return {
            key: _build_confidence_tree(
                item,
                field_confidence_by_pointer,
                path=path + (str(key),),
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _build_confidence_tree(
                item,
                field_confidence_by_pointer,
                path=path + (str(index),),
            )
            for index, item in enumerate(value)
        ]
    pointer = _pointer_from_path(path)
    return _leaf_confidence_payload(field_confidence_by_pointer, pointer)


def _has_field_confidence(
    field_confidence_by_pointer: FieldConfidenceByPointer | None,
) -> bool:
    return bool(field_confidence_by_pointer)


def journal_rows(
    data: FrontPage,
    file_name: str,
    field_confidence_by_pointer: FieldConfidenceByPointer | None = None,
) -> list[dict]:
    row = data.model_dump(mode="python")
    if _has_field_confidence(field_confidence_by_pointer):
        row["field_confidence"] = _build_confidence_tree(
            row,
            field_confidence_by_pointer,
            path=(),
        )
    row["file_name"] = file_name
    return [row]


def text_page_rows(
    data: TextPage,
    file_name: str,
    field_confidence_by_pointer: FieldConfidenceByPointer | None = None,
) -> list[dict]:
    page_level = data.model_dump(mode="python", exclude={"page_lines"})
    rows: list[dict] = []
    for index, line in enumerate(data.page_lines):
        row = line.model_dump(mode="python")
        row.update(page_level)
        if _has_field_confidence(field_confidence_by_pointer):
            line_confidence = _build_confidence_tree(
                line.model_dump(mode="python"),
                field_confidence_by_pointer,
                path=("page_lines", str(index)),
            )
            page_level_confidence = _build_confidence_tree(
                page_level,
                field_confidence_by_pointer,
                path=(),
            )
            confidence_payload: dict[str, object] = {}
            if isinstance(line_confidence, dict):
                confidence_payload.update(line_confidence)
            if isinstance(page_level_confidence, dict):
                confidence_payload.update(page_level_confidence)
            row["field_confidence"] = confidence_payload
        row["file_name"] = file_name
        rows.append(row)
    return rows


def default_rows(
    data: BaseModel,
    file_name: str,
    field_confidence_by_pointer: FieldConfidenceByPointer | None = None,
) -> list[dict]:
    row = data.model_dump(mode="python")
    if _has_field_confidence(field_confidence_by_pointer):
        row["field_confidence"] = _build_confidence_tree(
            row,
            field_confidence_by_pointer,
            path=(),
        )
    row["file_name"] = file_name
    return [row]


_HANDLERS: dict[
    type[BaseModel],
    Callable[[BaseModel, str, FieldConfidenceByPointer | None], list[dict]],
] = {
    FrontPage: journal_rows,
    TextPage: text_page_rows,
}


def data_to_rows(
    data: BaseModel,
    file_name: str,
    field_confidence_by_pointer: FieldConfidenceByPointer | None = None,
) -> list[dict]:
    for model_type, handler in _HANDLERS.items():
        if isinstance(data, model_type):
            return handler(data, file_name, field_confidence_by_pointer)
    return default_rows(data, file_name, field_confidence_by_pointer)
