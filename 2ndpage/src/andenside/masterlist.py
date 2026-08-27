"""Læsning og opslag i Blegdam_master_list.csv."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from andenside.kilder import masterliste

MASTERLIST_PATH = masterliste()


@dataclass(frozen=True)
class Side:
    image_name: str
    folder_name: str
    page_type: str
    month: str
    year: str
    patient_page_counter: int | None
    group_id: str

    @property
    def recto_verso(self) -> str:
        """Recto/verso ud fra patient_page_counters paritet.

        Journalerne var loese blade, senere indbundet; indbinding af
        foldede blade starter altid paa en enkelt recto. Forside (0) er
        derfor altid recto, andenside (1) altid verso, tredjeside (2)
        altid recto, osv.
        """
        if self.patient_page_counter is None:
            return "ukendt"
        return "recto" if self.patient_page_counter % 2 == 0 else "verso"

    @property
    def rolle(self) -> str:
        match self.patient_page_counter:
            case 0:
                return "forside"
            case 1:
                return "andenside"
            case 2:
                return "tredjeside"
            case None:
                return "ukendt"
            case n:
                return f"side {n + 1}"


def _to_int_or_none(value: str) -> int | None:
    value = value.strip()
    if not value or value == "NA":
        return None
    return int(value)


def load_masterlist(path: Path = MASTERLIST_PATH) -> dict[str, Side]:
    """Indlæser hele masterlisten, nøglet på image_name."""
    sides: dict[str, Side] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        assert header[1] == "image_name", f"uventet header: {header}"
        for row in reader:
            image_name = row[1]
            sides[image_name] = Side(
                image_name=image_name,
                folder_name=row[2],
                page_type=row[5],
                month=row[6],
                year=row[7],
                patient_page_counter=_to_int_or_none(row[8]),
                group_id=row[9],
            )
    return sides


def lookup(image_name: str, index: dict[str, Side]) -> Side:
    try:
        return index[image_name]
    except KeyError as exc:
        raise KeyError(f"{image_name} findes ikke i masterlisten") from exc
