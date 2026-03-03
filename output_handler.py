from __future__ import annotations

from typing import Callable

from pydantic import BaseModel

from schemas import Journal, TextPage


def journal_rows(data: Journal, file_name: str) -> list[dict]:
    row = data.model_dump(mode="python")
    row["file_name"] = file_name
    return [row]


def text_page_rows(data: TextPage, file_name: str) -> list[dict]:
    page_level = data.model_dump(mode="python", exclude={"page_lines"})
    rows: list[dict] = []
    for line in data.page_lines:
        row = line.model_dump(mode="python")
        row.update(page_level)
        row["file_name"] = file_name
        rows.append(row)
    return rows


def default_rows(data: BaseModel, file_name: str) -> list[dict]:
    row = data.model_dump(mode="python")
    row["file_name"] = file_name
    return [row]


_HANDLERS: dict[type[BaseModel], Callable[[BaseModel, str], list[dict]]] = {
    Journal: journal_rows,
    TextPage: text_page_rows,
}


def data_to_rows(data: BaseModel, file_name: str) -> list[dict]:
    for model_type, handler in _HANDLERS.items():
        if isinstance(data, model_type):
            return handler(data, file_name)
    return default_rows(data, file_name)
