"""Kører pilotens transskriptioner. TØRLØB som standard.

Uden `--yes` foretages der NUL netværkskald: scriptet skriver, hvad det ville
gøre, og hvad det anslås at koste. Det er projektets regel, at alt med
eksterne bivirkninger er tørløb som standard -- og at ingen fuld kørsel sker
uden leads udtrykkelige go.

    .venv/Scripts/python.exe scripts/koer_pilot.py                    # tørløb, 8 sider
    .venv/Scripts/python.exe scripts/koer_pilot.py --antal 5           # færre
    .venv/Scripts/python.exe scripts/koer_pilot.py --sider 273105_001570
    .venv/Scripts/python.exe scripts/koer_pilot.py --yes               # udfør

De to varianter:

    helt_opslag   hele billedet, som det kom fra kildeviseren
    beskaaret     stage 04's færdige snit (begge kanter), lånt fra
                  stage 04's `levering_beskaaret/`. Stage 05 laver ikke
                  sine egne snit.

## Hvor mange sider

Piloten er dér, prompten formes -- ikke en måling af beskæringen
(beslutning 52). Standard er 8 af de 15 pilotsider, valgt så
sværhedsgraden spændes ud fra den letteste til den hårdeste; `--antal 0`
tager alle 15. Skal en enkelt side køres om, mens en prompt slibes til,
er `--sider` vejen.

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

# De faerdige snit ligger i stage 04, ikke her. Stage 05 laaner dem; den
# laver ikke sine egne. Gruppen `proeve_LAAST` er med vilje IKKE med -- de
# sider maa ikke maales paa foer den endelige bedoemmelse, og `sikr_oevemaengde`
# er andet lag under det samme vaern.
BESKAARET = (ROD / "stages" / "04_billedforberedelse" / "output"
             / "levering_beskaaret")
BESKAARET_GRUPPER = ("oeve", "selvhentet")

KILDER = {
    "helt_opslag": ROD / "stages" / "01_datagrundlag" / "output" / "oeve_billeder",
    "beskaaret": BESKAARET,
}

# Groft prisoverslag, gemini-3.1-pro-preview: 2 USD pr. mio. ind, 12 USD pr. mio. ud.
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


def _udsnit(raekker: list[dict], antal: int | None) -> list[dict]:
    """Vaelger `antal` sider spredt ud over svaerhedsgraden.

    Piloten er stedet, hvor prompten formes (beslutning 52), saa den maa
    hverken bestaa af de nemmeste sider -- der ville prompten se bedre ud,
    end den er -- eller kun af de haardeste, hvor alt fejler og intet kan
    skelnes. Maalestokken er facits egne `svaere_linjer`.

    Der spredes over sværhedsgradens VÆRDIER, ikke over sidernes raekkefoelge.
    Forskellen er ikke teoretisk: de 15 pilotsider har 1-6 svaere linjer paa
    de tretten og saa et spring til 9 og 10 paa de sidste to. Et lige spring
    hen over raekkefoelgen rammer de tretten igen og igen og springer den
    haarde hale over -- praecis den ende, en prompt skal proeves paa.

    Rækkefølgen i det returnerede er billed-id, ikke svaerhedsgrad, saa to
    koersler med samme `antal` giver samme liste i samme orden.
    """
    if antal is None or antal >= len(raekker):
        return raekker
    if antal < 1:
        raise ValueError("--antal skal vaere mindst 1")

    efter_svaerhed = sorted(raekker, key=lambda r: (int(r["svaere_linjer"]),
                                                    r["billede"]))
    laveste = int(efter_svaerhed[0]["svaere_linjer"])
    hoejeste = int(efter_svaerhed[-1]["svaere_linjer"])

    tilbage = list(efter_svaerhed)
    valgte = []
    for i in range(antal):
        maal = laveste if antal == 1 else (
            laveste + (hoejeste - laveste) * i / (antal - 1))
        # nærmeste ubrugte side; ved lige afstand vinder den laveste
        # svaerhedsgrad, saa valget ikke afhaenger af listens orden
        naermeste = min(tilbage, key=lambda r: (abs(int(r["svaere_linjer"]) - maal),
                                                int(r["svaere_linjer"]),
                                                r["billede"]))
        tilbage.remove(naermeste)
        valgte.append(naermeste)
    return sorted(valgte, key=lambda r: r["billede"])


def _find_billede(variant: str, navn: str) -> Path | None:
    """Filen for én side. `beskaaret` ligger fordelt paa stage 04's grupper."""
    if variant != "beskaaret":
        traef = sorted(KILDER[variant].glob(f"{navn}.*"))
        return traef[0] if traef else None
    for gruppe in BESKAARET_GRUPPER:
        traef = sorted((BESKAARET / gruppe / "beskaarne").glob(f"{navn}.*"))
        if traef:
            return traef[0]
    return None


def _sider(variant: str, antal: int | None,
           kun: list[str] | None) -> list[tuple[str, Path]]:
    raekker = list(csv.DictReader((UD / "pilotsider.csv").open(encoding="utf-8")))
    if kun:
        kendte = {r["billede"] for r in raekker}
        ukendte = [n for n in kun if n not in kendte]
        if ukendte:
            raise SystemExit("ikke i pilotsider.csv: " + ", ".join(ukendte))
        raekker = [r for r in raekker if r["billede"] in kun]
    else:
        raekker = _udsnit(raekker, antal)

    navne = [r["billede"] for r in raekker]
    sikr_oevemaengde(navne)

    fundet, mangler = [], []
    for navn in navne:
        sti = _find_billede(variant, navn)
        (mangler if sti is None else fundet).append((navn, sti))
    if mangler:
        print(f"  Bemærk: {len(mangler)} side(r) findes ikke som {variant}: "
              + ", ".join(n for n, _ in mangler))
    return fundet


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=sorted(KILDER), default="beskaaret")
    p.add_argument("--antal", type=int, default=8,
                   help="antal sider, spredt ud over svaerhedsgraden "
                        "(0 = alle 15)")
    p.add_argument("--sider", default="",
                   help="navngivne sider i stedet for udsnittet, adskilt af komma")
    p.add_argument("--model", default="gemini-3.1-pro-preview")
    p.add_argument("--temperatur", type=float, default=0.0)
    p.add_argument("--yes", action="store_true",
                   help="udfør kaldene. Uden dette flag sker der intet.")
    args = p.parse_args()

    prompt = _prompt()
    kun = [n.strip() for n in args.sider.split(",") if n.strip()]
    sider = _sider(args.variant, args.antal or None, kun or None)
    if not sider:
        raise SystemExit("ingen sider at koere paa")
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
        noter=f"pilot, {len(sider)} af 15 sider, én pr. bind",
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
