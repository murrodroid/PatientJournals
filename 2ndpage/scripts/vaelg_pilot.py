"""Vaelger pilotens sider og skriver dem til stage 05's output.

Reglen, fast og uden lodtraekning (samme aand som stage 02's "hver tredje
patient"): ÉN side pr. bind, nemlig den med FLEST `[?]`-linjer; er der
uafgjort, vinder det laveste billed-id.

Hvorfor netop den regel:

*   **Én pr. bind** proever alle 15 bind praecis én gang. Stage 04's kendte
    aabne begraensning er, at bogryg-snittet kun er afproevet paa 2 bind fra
    samme fotograferingssession -- det her er den skarpeste proeve paa den.
*   **Flest `[?]`** maksimerer projektets bedste hallucinationsproeve: hvad
    skriver modellen dér, hvor transskribenten selv gav op.

Koerer man modulet direkte, skrives `output/pilotsider.csv`.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROD / "src"))

from andenside.vaern import sikr_oevemaengde  # noqa: E402

FACIT = ROD / "stages" / "02_facit" / "output" / "facit.jsonl"
BILLEDER = ROD / "stages" / "01_datagrundlag" / "output" / "oeve_billeder"
UD = ROD / "stages" / "05_foerste_transskription" / "output"

MAERKE = "[?]"


def pilotsider() -> list[dict]:
    """Én post pr. bind, valgt efter reglen ovenfor."""
    haves = {sti.stem for sti in BILLEDER.glob("*.webp")}
    bedste: dict[str, dict] = {}

    for linje in FACIT.read_text(encoding="utf-8").splitlines():
        post = json.loads(linje)
        billede = post["image_name"]
        if billede not in haves:
            continue
        bind = billede.split("_")[0]
        svaere = sum(1 for l in post["alt_linjer"] if MAERKE in l)
        kandidat = {
            "billede": billede,
            "bind": bind,
            "linjer": len(post["alt_linjer"]),
            "svaere_linjer": svaere,
            "forside": post["forside"],
        }
        staaende = bedste.get(bind)
        # Flest svaere linjer vinder; ved uafgjort det laveste billed-id.
        if staaende is None or (-svaere, billede) < (
            -staaende["svaere_linjer"], staaende["billede"]
        ):
            bedste[bind] = kandidat

    valgte = [bedste[bind] for bind in sorted(bedste)]
    # Vaernet er ikke pynt: piloten maa aldrig roere den laaste proevemaengde.
    sikr_oevemaengde([v["billede"] for v in valgte])
    return valgte


def main() -> None:
    valgte = pilotsider()
    UD.mkdir(parents=True, exist_ok=True)
    sti = UD / "pilotsider.csv"
    with sti.open("w", encoding="utf-8", newline="") as f:
        skriver = csv.DictWriter(f, fieldnames=list(valgte[0]))
        skriver.writeheader()
        skriver.writerows(valgte)

    print(f"{len(valgte)} sider valgt, én pr. bind -> {sti.relative_to(ROD)}\n")
    print(f"{'billede':<16}{'bind':<9}{'linjer':>7}{'heraf svære':>13}")
    for v in valgte:
        print(f"{v['billede']:<16}{v['bind']:<9}{v['linjer']:>7}{v['svaere_linjer']:>13}")
    print(f"\nI alt {sum(v['linjer'] for v in valgte)} linjer, "
          f"heraf {sum(v['svaere_linjer'] for v in valgte)} svære.")


if __name__ == "__main__":
    main()
