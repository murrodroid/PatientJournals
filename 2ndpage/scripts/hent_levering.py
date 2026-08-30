"""Henter kollegaens billedlevering ind i projektet.

Leveringen er svaret paa vores egen `billedanmodning/` fra 18. august 2026:
**307 PNG-filer** fra en ekstern harddisk, fladt navngivet `<bind>_<id>.png`,
1,4 GB i alt. Det er de samme `image_name`-vaerdier som i masterlisten.

## Hvad der hentes -- og hvad der IKKE goer

| Gruppe | Antal | Hentes |
|---|---:|---|
| Oevemaengde (patienter i `opdeling.csv` = oeve) | 173 | ja |
| Ekstra andensider uden facit (spredt 1889-1897) | 50 | ja |
| **Proevemaengde** (patienter = proeve) | 84 | **NEJ** |

Proevemaengden hentes bevidst ikke. Beslutningen fra stage 02 er, at de
sider foerst maa roeres ved den endelige bedoemmelse, og den letteste maade
at bryde den paa er at have filerne liggende, hvor et glob kan samle dem op.
De ligger paa harddisken og kan hentes den dag, bedoemmelsen skal koere.

## Om kvaliteten

PNG'erne har SAMME opløsning som de webp-filer, vi hentede selv fra
kbharkiv -- de er ikke skarpere, kun ukomprimerede. Maalt paa to sider:
middelafvigelse 1,3-1,6 graatoner, PSNR 41-42 dB, og kun 2-3 % af
billedpunkterne afviger mere end 5 niveauer. Gevinsten ved at skifte er
altsaa reel, men lille, og et skifte ville ugyldiggoere stage 04's snit.
Derfor lægges leveringen ved siden af i stedet for at erstatte noget.
"""

from __future__ import annotations

import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from andenside.masterlist import load_masterlist

KILDE = Path("D:/notatsider_til_jonas_august2026")
STAGE01 = Path(__file__).resolve().parents[1] / "stages" / "01_datagrundlag" / "output"
MAAL = STAGE01 / "levering_2026-08"
OPDELING = Path(__file__).resolve().parents[1] / "stages" / "02_facit" / "output" / "opdeling.csv"


def forside_pr_billede(index) -> dict[str, str]:
    """Knytter hvert billede til sin patients forside.

    Masterlisten grupperer paa (bindmappe, group_id); forsiden er den side i
    gruppen, hvor `patient_page_counter` er 0.
    """
    grupper = defaultdict(list)
    for side in index.values():
        grupper[(side.folder_name, side.group_id)].append(side)
    ud: dict[str, str] = {}
    for sider in grupper.values():
        forsider = [s for s in sider if s.patient_page_counter == 0]
        if forsider:
            for s in sider:
                ud[s.image_name] = forsider[0].image_name
    return ud


def inddel() -> dict[str, list[str]]:
    """Deler leveringens filer i oeve, proeve og ekstra."""
    if not KILDE.exists():
        raise SystemExit(f"harddisken er ikke tilsluttet: {KILDE}")
    leveret = sorted(p.stem for p in KILDE.glob("*.png"))
    forside_for = forside_pr_billede(load_masterlist())
    with OPDELING.open(encoding="utf-8") as f:
        maengde = {r["forside"]: r["maengde"] for r in csv.DictReader(f)}

    grupper: dict[str, list[str]] = {"oeve": [], "proeve": [], "ekstra_uden_facit": []}
    for navn in leveret:
        forside = forside_for.get(navn)
        gruppe = maengde.get(forside) if forside else None
        grupper[gruppe if gruppe in ("oeve", "proeve") else "ekstra_uden_facit"].append(navn)
    return grupper


def run(udfoer: bool) -> None:
    grupper = inddel()
    for navn, filer in grupper.items():
        print(f"{navn:20s} {len(filer):4d} billeder")

    hentes = {k: v for k, v in grupper.items() if k != "proeve"}
    i_alt = sum(len(v) for v in hentes.values())
    mb = sum((KILDE / f"{n}.png").stat().st_size for v in hentes.values() for n in v) / 1e6
    print(f"\nhentes: {i_alt} billeder, {mb:.0f} MB -> {MAAL}")
    print(f"hentes IKKE: {len(grupper['proeve'])} proeve-billeder "
          f"(beslutning fra stage 02 -- de roeres foerst ved den endelige bedoemmelse)")

    if not udfoer:
        print("\nTOERLOEB. Kør med --yes for at kopiere.")
        return

    for gruppe, filer in hentes.items():
        mappe = MAAL / gruppe
        mappe.mkdir(parents=True, exist_ok=True)
        for i, navn in enumerate(filer, 1):
            maal = mappe / f"{navn}.png"
            if not maal.exists():
                shutil.copy2(KILDE / f"{navn}.png", maal)
            if i % 25 == 0 or i == len(filer):
                print(f"  {gruppe}: {i}/{len(filer)}")
    print(f"\nfaerdig -> {MAAL}")


if __name__ == "__main__":
    run("--yes" in sys.argv)
