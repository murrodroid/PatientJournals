"""Maaler yderkant-detektionen paa hele oevemaengden og holder den op mod facit.

Kun MAALING: der skrives tal, ingen beskaarne billeder. Selve beskaeringen
kraever leads go (AGENTS.md: ingen fulde koersler uden go).

Forsoeg A: hvor lægger detektionen sidens egen yderkant?
Forsoeg B: siger den ja til et fremmed blad dér, hvor facit siger ja?

Et forsoeg paa at maale snittets rigtighed automatisk -- "staar der blaek
paa begge sider af linjen?" -- blev opgivet 2026-08-28. Det bestod ikke sin
egen proeve: `273108_001555`, hvor snittet BEVISLIGT gaar gennem teksten,
scorede lavere end sider, hvor snittet sidder rigtigt. Bogsnittets moerke
striber taeller som blaek, uanset taerskel. Snittene maa ses efter med
oejnene; kontaktarket er `scripts/yderkant_snit_ark.py`.

Facit er `yderkant_facit.csv` -- min egen visuelle gennemgang af de 118
yderkantsstrimler, endnu IKKE gennemgaaet af lead. Tallene er derfor
foreloebige.
"""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from andenside.masterlist import load_masterlist, lookup
from andenside.yderkant import baandkanter_ydre, har_fremmed_blad, ydre_graense

STAGE04 = Path(__file__).resolve().parents[1] / "stages" / "04_billedforberedelse" / "output"
BESKAARNE = STAGE04 / "beskaarne"
FACIT = STAGE04 / "yderkant_facit.csv"
UD_CSV = STAGE04 / "yderkant_maal.csv"


def maal_en(navn: str, side) -> dict:
    img = Image.open(BESKAARNE / f"{navn}.webp")
    kanter = baandkanter_ydre(img, side)
    fundne = [x for _, x in kanter if x is not None]
    graense = ydre_graense(img, side, kanter=kanter)
    blad = har_fremmed_blad(img, side)

    if graense:
        # hvor meget af bredden ville forsoeg A skaere vaek?
        yderst = max(graense) if side.recto_verso == "recto" else min(graense)
        fjernet = img.width - yderst - 1 if side.recto_verso == "recto" else yderst
    else:
        fjernet = 0

    return {
        "billede": navn,
        "recto_verso": side.recto_verso,
        "bredde": img.width,
        "baand_med_kant": len(fundne),
        "baand_i_alt": len(kanter),
        "sikker": "ja" if len(fundne) * 2 >= len(kanter) else "nej",
        "kant_median": int(sorted(fundne)[len(fundne) // 2]) if fundne else "",
        "haeldning_px": (max(fundne) - min(fundne)) if fundne else 0,
        "fjernet_px": fjernet,
        "fjernet_andel": round(fjernet / img.width, 4),
        "blad": "ja" if blad.er_blad else "nej",
        "baelte_bredde": blad.baelte_bredde,
        "baand_med_blad": blad.baand_med_blad,
    }


def run() -> None:
    index = load_masterlist()
    with FACIT.open(encoding="utf-8") as f:
        facit = {r["billede"]: r for r in csv.DictReader(f)}

    raekker = []
    for navn in sorted(facit):
        raekker.append(maal_en(navn, lookup(navn, index)) | {
            "facit_klasse": facit[navn]["klasse"],
            "facit_blad": facit[navn]["blad_synligt"],
        })

    with UD_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raekker[0].keys()))
        w.writeheader()
        w.writerows(raekker)

    # --- opgoerelse, skrevet til skaermen; fortolkningen hoerer i eval-notatet
    n = len(raekker)
    usikre = [r for r in raekker if r["sikker"] == "nej"]
    print(f"{n} sider maalt -> {UD_CSV}")
    print(f"usikre (under halvdelen af baandene fandt en kant): {len(usikre)}")
    for r in usikre:
        print(f"   {r['billede']}  {r['baand_med_kant']}/{r['baand_i_alt']} baand")

    fjernet = sorted(r["fjernet_andel"] for r in raekker)
    print(f"\nforsoeg A -- hvor meget af bredden snittet fjerner:")
    print(f"   median {fjernet[n // 2]:.1%}, spaend {fjernet[0]:.1%}-{fjernet[-1]:.1%}")
    vaerste = sorted(raekker, key=lambda r: -r["fjernet_andel"])[:5]
    for r in vaerste:
        print(f"   {r['billede']}: {r['fjernet_andel']:.1%} ({r['fjernet_px']} px)")

    print("\nforsoeg B -- blad-detektion mod facit (facit_blad):")
    for facit_vaerdi in ("ja", "nej", "usikker"):
        gruppe = [r for r in raekker if r["facit_blad"] == facit_vaerdi]
        ja = sum(1 for r in gruppe if r["blad"] == "ja")
        print(f"   facit '{facit_vaerdi}' ({len(gruppe)} sider): detektion siger ja paa {ja}")

    print("\nde 7 sider med FREMMED TEKST i facit -- dem der faktisk skal fanges:")
    for r in raekker:
        if r["facit_klasse"] == "fremmed_tekst":
            print(f"   {r['billede']}: blad={r['blad']} baelte={r['baelte_bredde']}px "
                  f"kant={r['kant_median']} fjernet={r['fjernet_andel']:.1%} "
                  f"sikker={r['sikker']}")


if __name__ == "__main__":
    run()
