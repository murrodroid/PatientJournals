"""Maaler en gemt koersel mod facit og skriver rapporten ved siden af svarene.

Det er broen mellem stage 05 og stage 03: piloten gemmer modelsvar, dette
script foerer dem gennem maaleapparatet uden at roere hverken facit eller
svarene. Der kaldes INGEN model -- alt materiale ligger allerede paa disken,
saa en maaling kan koeres om og gentages uden at koste noget.

    .venv/Scripts/python.exe scripts/maal_koersel.py            # nyeste koersel
    .venv/Scripts/python.exe scripts/maal_koersel.py 20260830_141937_beskaaret
    .venv/Scripts/python.exe scripts/maal_koersel.py --alle     # alle koersler

## Laes tallene med dækningen i haanden

Maaleapparatet finder facits kendte tekststumper i modellens tekst. Hvad det
IKKE finder, kan det ikke maale paa -- og de linjer, det taber, er systematisk
de haardest ramte. Selvtesten opgjorde skævheden til, at maalingen finder ca.
93 % af de fejl, der faktisk er lagt ind (stage 03). En tegnfejl herfra er
derfor et GULV, ikke et facit.

Rapporten skriver dækningen ved hvert tal af netop den grund. Et pænt tal med
lav dækning betyder, at der er maalt paa lidt -- ikke at siden var let.

## Hvad der skrives

    <koersel>/rapport.md   maaletallene i den aftalte form
    <koersel>/gab.csv      hvad modellen skrev, hvor facit ikke kunne laeses
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROD / "src"))

from andenside.koersel import find_koersler, laes_koersel  # noqa: E402
from andenside.maal import maal_saet  # noqa: E402
from andenside.rapport import skriv_gab, skriv_rapport  # noqa: E402
from andenside.vaern import sikr_oevemaengde  # noqa: E402

FACIT = ROD / "stages" / "02_facit" / "output" / "facit.jsonl"
KOERSLER = ROD / "stages" / "05_foerste_transskription" / "output" / "koersler"


def _andel(m, slags: str) -> str:
    """Fejlandelen som decimaltal. Nul tegn giver 0, ikke en division med nul."""
    if slags == "tegn":
        return f"{m.tegnafstand / m.facit_tegn:.4f}" if m.facit_tegn else "0.0000"
    return f"{m.ordafstand / m.facit_ord:.4f}" if m.facit_ord else "0.0000"


def _facit_for(navne: set[str]) -> list[dict]:
    poster = [json.loads(l) for l in FACIT.read_text(encoding="utf-8").splitlines()]
    return sorted((p for p in poster if p["image_name"] in navne),
                  key=lambda p: p["image_name"])


def maal_en(mappe: Path) -> dict:
    opsaetning, svar = laes_koersel(mappe)
    # `laes_koersel` giver ikke datoen tilbage, men den staar i filen, og
    # rapporten skal baere den: uden dato kan et tal ikke stedfaestes.
    dato = json.loads((mappe / "opsaetning.json").read_text(encoding="utf-8"))["dato"]
    sikr_oevemaengde(sorted(svar))

    poster = _facit_for(set(svar))
    uden_facit = sorted(set(svar) - {p["image_name"] for p in poster})
    if uden_facit:
        # En side uden facit kan ikke maales. Den skal naevnes, ikke skjules:
        # ellers ser daekningen bedre ud, end den er.
        print(f"  {len(uden_facit)} side(r) uden facit, ikke maalt: "
              + ", ".join(uden_facit))
    if not poster:
        print("  ingen sider med facit -- intet at maale")
        return {}

    saet = maal_saet(poster, svar)
    (mappe / "rapport.md").write_text(
        skriv_rapport(saet, titel=f"Koersel {mappe.name}",
                      model=opsaetning.model,
                      promptversion=opsaetning.promptversion,
                      dato=dato, noter=opsaetning.noter),
        encoding="utf-8")
    (mappe / "gab.csv").write_text(skriv_gab(saet), encoding="utf-8")

    return {
        "koersel": mappe.name,
        "model": opsaetning.model,
        "variant": opsaetning.variant,
        "sider": len(saet.sider),
        # `raa` er den strengeste og den, der ikke kan pyntes; `arbejdstal`
        # er stage 03's aftalte hovedtal. Begge staar med, saa ingen af dem
        # kan vaelges bagefter, alt efter hvad der ser bedst ud.
        "cer_raa": _andel(saet.fladet["raa"], "tegn"),
        "cer_arbejdstal": _andel(saet.fladet["arbejdstal"], "tegn"),
        "wer_raa": _andel(saet.fladet["raa"], "ord"),
        "cer_raa_streng": _andel(saet.rene["raa"], "tegn"),
        "daekning": f"{saet.daekning:.4f}",
        "daekning_streng": f"{saet.rene_daekning:.4f}",
        "linjedaekning": f"{saet.linjedaekning:.4f}",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("koersel", nargs="?",
                   help="mappenavn under output/koersler/; standard er den nyeste")
    p.add_argument("--alle", action="store_true",
                   help="maal alle koersler og skriv en samlet oversigt")
    args = p.parse_args()

    alle = find_koersler(KOERSLER)
    if not alle:
        raise SystemExit(f"ingen koersler i {KOERSLER}")

    if args.alle:
        valgte = alle
    elif args.koersel:
        valgte = [m for m in alle if m.name == args.koersel]
        if not valgte:
            raise SystemExit(f"findes ikke: {args.koersel}\n"
                             + "\n".join(f"  {m.name}" for m in alle))
    else:
        valgte = [max(alle, key=lambda m: m.name)]

    raekker = []
    for mappe in sorted(valgte, key=lambda m: m.name):
        print(f"\n=== {mappe.name} ===")
        r = maal_en(mappe)
        if not r:
            continue
        raekker.append(r)
        print(f"  model {r['model']}, {r['sider']} sider")
        print(f"  tegnfejl raa        {float(r['cer_raa']):.2%} "
              f"(dækning {float(r['daekning']):.1%})")
        print(f"  tegnfejl arbejdstal {float(r['cer_arbejdstal']):.2%}")
        print(f"  tegnfejl streng     {float(r['cer_raa_streng']):.2%} "
              f"(dækning {float(r['daekning_streng']):.1%})")
        print(f"  ordfejl raa         {float(r['wer_raa']):.2%}")
        print(f"  linjedækning        {float(r['linjedaekning']):.1%}")
        print(f"  rapport  {(mappe / 'rapport.md').relative_to(ROD)}")

    if len(raekker) > 1:
        sti = KOERSLER / "oversigt.csv"
        with sti.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(raekker[0]))
            w.writeheader()
            w.writerows(raekker)
        print(f"\nSamlet oversigt: {sti.relative_to(ROD)}")


if __name__ == "__main__":
    main()
