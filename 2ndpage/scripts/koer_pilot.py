"""Kører pilotens transskriptioner. TØRLØB som standard.

Uden `--yes` foretages der NUL netværkskald: scriptet skriver, hvad det ville
gøre, og hvad det anslås at koste. Det er projektets regel, at alt med
eksterne bivirkninger er tørløb som standard -- og at ingen fuld kørsel sker
uden leads udtrykkelige go.

    .venv/Scripts/python.exe scripts/koer_pilot.py                # tørløb
    .venv/Scripts/python.exe scripts/koer_pilot.py --variant beskaaret --yes

De to varianter:

    helt_opslag   hele billedet, som det kom fra kildeviseren
    beskaaret     stage 04's snit, skrevet ud af scripts/beskaer_pilot.py

Kørslen gemmes med hele sin opsætning i `output/koersler/` (se
`andenside.koersel`), saa den kan genfindes og køres om.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROD / "src"))

from andenside.koersel import Opsaetning, gem_koersel  # noqa: E402
from andenside.model import transskriber  # noqa: E402
from andenside.vaern import sikr_oevemaengde  # noqa: E402

STAGE05 = ROD / "stages" / "05_foerste_transskription"
UD = STAGE05 / "output"
PROMPTFIL = STAGE05 / "prompts" / "textpage_uaendret.md"

KILDER = {
    "helt_opslag": ROD / "stages" / "01_datagrundlag" / "output" / "oeve_billeder",
    "beskaaret": UD / "beskaarne",
}

# Groft prisoverslag, gemini-3.1-pro: 2 USD pr. mio. ind, 12 USD pr. mio. ud.
# Et sidekald er skoennet til ca. 2.000 tokens ind og 1.500 ud. Tallet er et
# OVERSLAG og staar her, saa en koersel ikke sker i blinde -- ikke som en
# maaling.
PRIS_PR_KALD_USD = 2_000 / 1e6 * 2.0 + 1_500 / 1e6 * 12.0
USD_TIL_DKK = 6.9


def _prompt() -> str:
    """Selve promptteksten ud af den menneskelaesbare promptfil."""
    tekst = PROMPTFIL.read_text(encoding="utf-8")
    foer, _, rest = tekst.partition("```")
    krop, _, _ = rest.partition("```")
    if not krop.strip():
        raise ValueError(f"fandt ingen prompttekst i {PROMPTFIL}")
    return krop.strip()


def _sider(variant: str) -> list[tuple[str, Path]]:
    mappe = KILDER[variant]
    navne = [r["billede"] for r in csv.DictReader(
        (UD / "pilotsider.csv").open(encoding="utf-8"))]
    sikr_oevemaengde(navne)

    fundet, mangler = [], []
    for navn in navne:
        traef = list(mappe.glob(f"{navn}.*"))
        (fundet if traef else mangler).append((navn, traef[0] if traef else None))
    if mangler:
        print(f"  Bemærk: {len(mangler)} side(r) findes ikke som {variant}: "
              + ", ".join(n for n, _ in mangler))
    return [(n, s) for n, s in fundet if s is not None]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=sorted(KILDER), default="beskaaret")
    p.add_argument("--model", default="gemini-3.1-pro")
    p.add_argument("--temperatur", type=float, default=0.0)
    p.add_argument("--yes", action="store_true",
                   help="udfør kaldene. Uden dette flag sker der intet.")
    args = p.parse_args()

    prompt = _prompt()
    sider = _sider(args.variant)
    pris = len(sider) * PRIS_PR_KALD_USD

    print(f"\nVariant:      {args.variant}")
    print(f"Model:        {args.model} (temperatur {args.temperatur})")
    print(f"Prompt:       {PROMPTFIL.relative_to(ROD)} ({len(prompt)} tegn)")
    print(f"Sider:        {len(sider)}")
    print(f"Prisoverslag: ca. {pris:.2f} USD / {pris * USD_TIL_DKK:.0f} kr\n")
    for navn, sti in sider:
        print(f"   {navn}  <-  {sti.relative_to(ROD)}")

    if not args.yes:
        print(f"\nTØRLØB — der er ikke kaldt noget. Tilføj --yes for at køre.")
        return

    opsaetning = Opsaetning(
        model=args.model,
        promptversion="textpage-uaendret",
        prompt=prompt,
        variant=args.variant,
        temperatur=args.temperatur,
        noter=f"pilot, {len(sider)} sider, én pr. bind",
    )

    svar: dict[str, str] = {}
    raa: dict[str, dict] = {}
    fejlede: list[tuple[str, str]] = []
    print()
    for nummer, (navn, sti) in enumerate(sider, start=1):
        try:
            tekst, struktur = transskriber(
                sti, prompt, model=args.model, temperatur=args.temperatur
            )
        except Exception as fejl:                      # noqa: BLE001
            # En enkelt fejlet side maa ikke koste de foregaaende svar.
            fejlede.append((navn, f"{type(fejl).__name__}: {fejl}"))
            print(f"   [{nummer}/{len(sider)}] {navn}  FEJLEDE: {type(fejl).__name__}")
            continue
        svar[navn] = tekst
        raa[navn] = struktur
        print(f"   [{nummer}/{len(sider)}] {navn}  {len(tekst.splitlines())} linjer")

    if not svar:
        print("\nIngen sider lykkedes — intet gemt.")
        return

    mappe = gem_koersel(UD / "koersler", opsaetning, svar)
    (mappe / "raa_skemasvar.json").write_text(
        json.dumps(raa, ensure_ascii=False, indent=2), encoding="utf-8")
    if fejlede:
        (mappe / "fejlede.txt").write_text(
            "\n".join(f"{n}: {g}" for n, g in fejlede), encoding="utf-8")

    print(f"\n{len(svar)} af {len(sider)} sider gemt i {mappe.relative_to(ROD)}")
    if fejlede:
        print(f"{len(fejlede)} fejlede — se fejlede.txt")


if __name__ == "__main__":
    main()
