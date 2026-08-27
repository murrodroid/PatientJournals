"""Vaern om den laaste proevemaengde.

Stage 02 delte de 39 patienter i 26 oeve- og 13 proeve-patienter. Proeven er
laast til den ENDELIGE bedoemmelse: ser vi facit for de sider undervejs, er
sluttallet ikke laengere en uafhaengig maaling, og det kan ikke repareres
bagefter -- et modelsvar baerer ikke praeg af, at vi kiggede foerst.

Derfor gaar enhver side, stage 05 og senere roerer, igennem
`sikr_oevemaengde()`. Vejen udenom findes (`tillad_proeve=True`), fordi den
endelige bedoemmelse selv skal kunne koere, men den skal skrives med vilje.

Opslaget sker pr. SIDE via dens forside, fordi opdelingen blev truffet pr.
patient -- ikke pr. side.
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

ROD = Path(__file__).resolve().parents[2]
FACIT = ROD / "stages" / "02_facit" / "output" / "facit.jsonl"
OPDELING = ROD / "stages" / "02_facit" / "output" / "opdeling.csv"


class ProeveMaengdeFejl(RuntimeError):
    """Rejses, naar kode vil roere den laaste proevemaengde uden tilladelse."""


@lru_cache(maxsize=1)
def _maengde_pr_billede() -> dict[str, str]:
    maengde_for_forside = {
        raekke["forside"]: raekke["maengde"]
        for raekke in csv.DictReader(OPDELING.open(encoding="utf-8"))
    }
    ud: dict[str, str] = {}
    for linje in FACIT.read_text(encoding="utf-8").splitlines():
        post = json.loads(linje)
        ud[post["image_name"]] = maengde_for_forside[post["forside"]]
    return ud


def maengde_for(billede: str) -> str:
    """`"oeve"` eller `"proeve"` for et billed-id.

    Et ukendt id er en FEJL, ikke et gaet. Gik ukendte sider stiltiende for
    "oeve", ville en tastefejl i et proeve-id vaere nok til at aabne vaernet.
    """
    try:
        return _maengde_pr_billede()[billede]
    except KeyError:
        raise KeyError(
            f"{billede!r} findes ikke i facit -- kan ikke afgoere, om den "
            f"tilhoerer oeve- eller proevemaengden"
        ) from None


def sikr_oevemaengde(
    billeder: list[str], *, tillad_proeve: bool = False
) -> list[str]:
    """Slaar fejl, hvis `billeder` rummer sider fra den laaste proevemaengde.

    Hele kaldet fejler -- de ulovlige sider frasorteres IKKE i stilhed. Kaldte
    nogen med et saet, de troede var oevemaengden, skal de vide det, ikke faa
    en delvis koersel tilbage, der ligner en hel.

    `tillad_proeve` er keyword-only, saa den ikke kan rutsje ind som en
    tilfaeldig positionsparameter.
    """
    if not tillad_proeve:
        forbudte = [b for b in billeder if maengde_for(b) == "proeve"]
        if forbudte:
            raise ProeveMaengdeFejl(
                "Disse sider tilhoerer den laaste proevemaengde og maa ikke "
                "koeres paa uden udtrykkelig tilladelse "
                "(sikr_oevemaengde(..., tillad_proeve=True)): "
                + ", ".join(sorted(forbudte))
            )
    return billeder
