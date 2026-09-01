"""Gennemsynsside: sidder der et NABOBLAD inden for yderkant-snittet?

Falsen røres ikke her. Spørgsmålet er alene, om det færdige udsnit stadig
rummer en stump af bladet ved siden af -- typisk en strimmel med fremmed
håndskrift langs den ydre kant.

Der vises den samme slags tavle som i `beskaer_levering.py`: hele siden med
det bortskårne tonet -- falsen rød, yderkanten blå -- bare stor nok til at
kunne bedømmes. Tavlerne i `levering_beskaaret/*/tavler/` er kun 640 px
brede, og dér fylder en strimmel på 100 px ikke engang 40 px.

Siderne stilles i kø efter mistanke: et blad, der bliver stående, gør
udsnittet bredere end de andre sider i samme bind på samme side af opslaget.
Målet er ikke, at rækkefølgen er rigtig -- kun at de værste kommer først.

Kør:  uv run python scripts/nabo_blad_gennemsyn.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

ROD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROD / "scripts"))
sys.path.insert(0, str(ROD / "src"))

from beskaer_levering import _kildefil, _tavle  # noqa: E402
from andenside.bogryg import soegevindue  # noqa: E402
from andenside.masterlist import load_masterlist, lookup  # noqa: E402
from andenside.skraa import beskaer_langs_fals, fals_graense  # noqa: E402
from andenside.yderkant import ydre_graense  # noqa: E402

LEVERING = ROD / "stages" / "04_billedforberedelse" / "output" / "levering_beskaaret"
UD = ROD / "stages" / "04_billedforberedelse" / "output" / "nabo_blad_gennemsyn"
GRUPPER = ("oeve", "ekstra_uden_facit", "proeve_LAAST", "selvhentet")

VIST_BREDDE = 1100     # px paa skaermen -- stort nok til at se en smal strimmel
KVALITET = 72

_INDEX = None


def _laes_snit(gruppe: str) -> list[dict]:
    sti = LEVERING / gruppe / "snit.csv"
    if not sti.exists():
        return []
    with sti.open(encoding="utf-8") as f:
        return [{**r, "gruppe": gruppe} for r in csv.DictReader(f)]


def _mistanke(poster: list[dict]) -> None:
    """Hvor meget bredere er udsnittet end sine naboer i samme bind?

    Sammenligningen holdes inden for (bind, recto/verso): recto-sider er
    bredere end verso-sider fra starten, saa de to maa ikke blandes.
    """
    grupper: dict[tuple[str, str], list[int]] = {}
    for p in poster:
        bind = p["billede"].split("_")[0]
        grupper.setdefault((bind, p["recto_verso"]), []).append(int(p["bredde_efter_begge"]))
    for p in poster:
        bind = p["billede"].split("_")[0]
        median = statistics.median(grupper[(bind, p["recto_verso"])])
        p["afvigelse_px"] = int(p["bredde_efter_begge"]) - int(median)


def _tegn_én(opgave: tuple[str, str]) -> str:
    """Én tavle. Koeres i sin egen proces -- masterlisten laeses én gang pr. proces.

    Snittene regnes forfra, praecis som i `beskaer_levering.py`. Der findes
    ingen gemt kopi af graenserne raekke for raekke; kun bredderne staar i
    `snit.csv`, og de kan ikke tegnes med.
    """
    gruppe, billede = opgave
    global _INDEX
    if _INDEX is None:
        _INDEX = load_masterlist()

    side = lookup(billede, _INDEX)
    with Image.open(_kildefil(gruppe, billede)) as img:
        img.load()
        fals_g = fals_graense(img, side)
        efter_fals, _ = beskaer_langs_fals(img, side)
        ydre_g = ydre_graense(efter_fals, side)
        retning = soegevindue(side, img.width).retning
        forskydning = 0 if retning == "fra_hoejre" else (max(0, min(fals_g)) if fals_g else 0)
        tavle = _tavle(img, fals_g, ydre_g, forskydning, retning)

    h = round(tavle.height * VIST_BREDDE / tavle.width)
    tavle.resize((VIST_BREDDE, h)).save(UD / "billeder" / f"{billede}.jpg",
                                        quality=KVALITET, optimize=True)
    return billede


def main() -> None:
    poster = [p for g in GRUPPER for p in _laes_snit(g)]
    if not poster:
        raise SystemExit(f"ingen snit.csv fundet under {LEVERING}")
    _mistanke(poster)
    poster.sort(key=lambda p: -p["afvigelse_px"])

    (UD / "billeder").mkdir(parents=True, exist_ok=True)
    opgaver = [(p["gruppe"], p["billede"]) for p in poster]
    print(f"{len(opgaver)} tavler...", flush=True)
    with ProcessPoolExecutor() as pulje:
        for nr, _ in enumerate(pulje.map(_tegn_én, opgaver), 1):
            if nr % 25 == 0 or nr == len(opgaver):
                print(f"  [{nr}/{len(opgaver)}]", flush=True)

    koe = [{
        "billede": p["billede"],
        "gruppe": p["gruppe"],
        "recto_verso": p["recto_verso"],
        "bredde": int(p["bredde_efter_begge"]),
        "afvigelse_px": p["afvigelse_px"],
        "ydre_baand": p["ydre_baand"],
    } for p in poster]
    # Skrives som .js, ikke .json: en side aabnet fra harddisken maa ikke
    # hente filer med fetch(), men den maa indlaese et <script>.
    (UD / "koe.js").write_text(
        "const KOE = " + json.dumps(koe, ensure_ascii=False, indent=1) + ";\n",
        encoding="utf-8")
    print(f"\nAabn: {UD / 'gennemsyn.html'}")


if __name__ == "__main__":
    main()
