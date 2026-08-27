"""Hvor kildematerialet ligger paa den enkelte maskine.

Facit-RTF'erne og masterlisten ligger ikke i repoet -- de ligger i et delt
drev, og stien er forskellig fra maskine til maskine. Tidligere var den
hardkodet, hvilket gav to problemer paa én gang: koden virkede kun paa én
maskine, og stien roebede baade et brugernavn og et internt drev i et
offentligt repo.

Stien saettes derfor med miljoevariablen `ANDENSIDE_KILDER`, eller i filen
`kilder.local.json` ved siden af `pyproject.toml`. Den fil er git-ignoreret.

    { "rod": "<sti til mappen 'Patient journals'>" }

Findes ingen af delene, fejler opslaget med en besked, der siger hvad man
skal goere -- i stedet for at falde tilbage paa et gaet.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

PROJEKTROD = Path(__file__).resolve().parents[2]
LOKAL_FIL = PROJEKTROD / "kilder.local.json"
MILJOEVARIABEL = "ANDENSIDE_KILDER"


class KildeFejl(RuntimeError):
    """Rejses, naar kildematerialets placering ikke kan afgoeres."""


def _vejledning() -> str:
    return (
        f"kildematerialets placering er ukendt. Saet {MILJOEVARIABEL} til "
        f"mappen 'Patient journals', eller opret {LOKAL_FIL.name} ved siden af "
        f'pyproject.toml med indholdet {{"rod": "<sti>"}}.'
    )


@lru_cache(maxsize=1)
def kilde_rod() -> Path:
    """Roden af kildematerialet. Rejser `KildeFejl`, hvis den ikke er sat."""
    fra_miljoe = os.environ.get(MILJOEVARIABEL, "").strip()
    if fra_miljoe:
        return Path(fra_miljoe)
    if LOKAL_FIL.exists():
        try:
            rod = json.loads(LOKAL_FIL.read_text(encoding="utf-8")).get("rod", "")
        except json.JSONDecodeError as fejl:
            raise KildeFejl(f"{LOKAL_FIL.name} kunne ikke laeses: {fejl.msg}") from None
        if str(rod).strip():
            return Path(str(rod).strip())
    raise KildeFejl(_vejledning())


def _under(*dele: str) -> Path:
    """En sti under kilderoden, eller en sti der beviseligt ikke findes.

    Modulerne bruger disse som konstanter paa modulniveau, og en manglende
    opsaetning maa derfor ikke sprænge selve importen -- saa kunne hverken
    tests eller `--help` koere paa en maskine uden adgang til drevet. I
    stedet peges paa noget, `.exists()` siger nej til, og koden opdager det
    dér, hvor den faktisk skal bruge filerne.
    """
    try:
        return kilde_rod().joinpath(*dele)
    except KildeFejl:
        return PROJEKTROD / "_kilder_ikke_konfigureret" / Path(*dele)


def transskriptioner() -> Path:
    return _under("Manual transcriptions")


def masterliste() -> Path:
    return _under("Meta data", "Blegdam_master_list.csv")
