"""Bygger stage 02's fire outputfiler ud af de 39 haandlavede RTF-filer.

Koerer man modulet direkte, skrives alt i `stages/02_facit/output/`.
Selve tolkningen af opmaerkningen ligger i `facit.py` og er testet der; her
er kun samling, optaelling og skrivning.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from andenside.facit import (
    FACIT_ROOT,
    Sideblok,
    del_i_sideblokke,
    klassificer_klamme,
    laes_side,
    ren_laesetekst,
    rtf_til_tekst,
    saml_orddeling,
)

STAGE02_OUTPUT = Path(__file__).resolve().parents[2] / "stages" / "02_facit" / "output"

# Hver tredje patient, sorteret efter forsidens billed-id, laegges i
# proevemaengden. Id'erne loeber kronologisk gennem bindene, saa udvalget
# spreder sig af sig selv over hele perioden maj 1896 - august 1897 uden at
# vi skal traekke lod. Ingen tilfaeldighed betyder ingen frøkerne at glemme,
# og samme opdeling hver eneste gang.
PROEVE_HVER = 3


@dataclass(frozen=True)
class Opslag:
    """Én journalside med facit i to udgaver.

    `alt_*` er alt hvad der staar paa siden, ogsaa det overstregede. Det er
    den, maalingen bruger, fordi modellen bliver bedt om at laese hele siden.
    `rettet_*` er den rettede laesning, hvor det overstregede er fjernet og
    kun erstatningen staar tilbage -- det laegen endte med at mene. Den er
    den historisk rigtige tekst og den, et faerdigt datasaet skal rumme.
    """

    image_name: str
    forside: str
    kildefil: str
    raa: str
    alt_linjer: list[str]
    alt_fladet: str
    rettet_linjer: list[str]
    rettet_fladet: str
    understreget: list[dict[str, object]]
    noter: list[str]


def laes_alle_blokke(rod: Path = FACIT_ROOT) -> list[Sideblok]:
    """Laeser alle RTF-filer under `rod`, sorteret efter filnavn."""
    blokke: list[Sideblok] = []
    for fil in sorted(rod.rglob("*.rtf"), key=lambda p: p.name):
        tekst = rtf_til_tekst(fil.read_text(encoding="cp1252", errors="replace"))
        blokke.extend(del_i_sideblokke(tekst, kildefil=fil.name))
    return blokke


def byg_opslag(blok: Sideblok) -> Opslag:
    alt, noter, understreget = laes_side(blok.raa, behold_overstreget=True)
    rettet, _ = ren_laesetekst(blok.raa)
    return Opslag(
        image_name=blok.image_name,
        forside=blok.forside or "",
        kildefil=blok.kildefil,
        raa=blok.raa,
        alt_linjer=alt.split("\n"),
        alt_fladet=saml_orddeling(alt),
        rettet_linjer=rettet.split("\n"),
        rettet_fladet=saml_orddeling(rettet),
        understreget=understreget,
        noter=noter,
    )


def opdel_patienter(forsider: list[str]) -> dict[str, str]:
    """Fordeler patienter paa oevemaengde og laast proevemaengde.

    Opdelingen sker pr. patient, aldrig pr. side: to sider fra samme
    indlaeggelse ligner hinanden i haandskrift, blaek og ordforraad, og
    ville laekke fra oeve- til proevemaengde, hvis de blev skilt ad.
    """
    return {
        forside: ("proeve" if n % PROEVE_HVER == 0 else "oeve")
        for n, forside in enumerate(sorted(set(forsider)))
    }


# ---------------------------------------------------------------------------
# Optaelling af klammeformer
# ---------------------------------------------------------------------------

_KLAMME = re.compile(r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]")


def tael_klammeformer(blokke: list[Sideblok]) -> list[tuple[str, str, int, str]]:
    """Optaeller hver klammeform i materialet.

    Returnerer (type, form, antal, eksempel) sorteret efter type og faldende
    antal. Formen er skrevet med smaa bogstaver og tal erstattet af N, saa fx
    alle sidemaerker samles i én raekke.
    """
    antal: Counter[tuple[str, str]] = Counter()
    eksempel: dict[tuple[str, str], str] = {}
    for blok in blokke:
        tekst = blok.raa
        for fund in _KLAMME.finditer(tekst):
            indre = fund.group(0)[1:-1]
            typ, _ = klassificer_klamme(indre)
            form = re.sub(r"\d+", "N", " ".join(indre.split()).lower())
            noegle = (typ, form)
            antal[noegle] += 1
            eksempel.setdefault(
                noegle,
                " ".join(tekst[max(0, fund.start() - 40) : fund.end() + 25].split()),
            )
    return sorted(
        ((typ, form, n, eksempel[(typ, form)]) for (typ, form), n in antal.items()),
        key=lambda r: (r[0], -r[2], r[1]),
    )


# ---------------------------------------------------------------------------
# Skrivning
# ---------------------------------------------------------------------------


def _skriv_facit(opslag: list[Opslag], sti: Path) -> None:
    with sti.open("w", encoding="utf-8", newline="\n") as f:
        for o in opslag:
            f.write(
                json.dumps(
                    {
                        "image_name": o.image_name,
                        "forside": o.forside,
                        "kildefil": o.kildefil,
                        "raa": o.raa,
                        "alt_linjer": o.alt_linjer,
                        "alt_fladet": o.alt_fladet,
                        "rettet_linjer": o.rettet_linjer,
                        "rettet_fladet": o.rettet_fladet,
                        "understreget": o.understreget,
                        "noter": o.noter,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _skriv_klammekonventioner(rader: list[tuple[str, str, int, str]], sti: Path) -> None:
    forklaring = {
        "ulaeselig": (
            "Ulaeseligt sted. Bevares som `[?]` i facit og skaeres ud af begge "
            "tekster, foer der maales. Ogsaa klammer med prikker eller ellipse "
            "som pladsholder havner her: `[..rede?]` betyder \"et ord der ender "
            "paa -rede\", ikke laesningen \"..rede\", saa bogstaverne er ukendte."
        ),
        "gaet": (
            "Forslag til laesning, med eller uden spoergsmaalstegn. Klammen "
            "falder vaek, ordet bliver staaende."
        ),
        "understregning": (
            "Note om at noget er understreget. Falder helt vaek af laeseteksten, "
            "men HVAD der var understreget gemmes i feltet `understreget`."
        ),
        "overstreget": (
            "Start paa overstreget tekst. I `rettet_*` falder teksten vaek; i "
            "`alt_*` -- den der maales paa -- bliver den staaende, fordi "
            "modellen bedes laese hele siden."
        ),
        "erstatning": "Det der blev skrevet i stedet. Bliver staaende.",
        "fortsaet": "Tilbage til hovedlinjen. Falder vaek som maerke.",
        "position": "Hvor paa siden teksten staar. Maerket falder vaek, teksten bliver.",
        "indskud": "Tekst skudt ind over eller under linjen. Teksten bliver.",
        "ukendt": (
            "Flere ord, vi ikke genkendte som maerke. Indholdet bliver staaende "
            "som tekst, og stedet er flaget i `udeladte.md`."
        ),
    }
    linjer = [
        "# Klammekonventioner i facit",
        "",
        "Udtoemmende optaelling af hver klammeform i alle 39 RTF-filer, lavet af",
        "`andenside.facit_bygger.tael_klammeformer`. Tal er erstattet af `N`, og",
        "store bogstaver er slaaet ned, saa ens former samles i én raekke.",
        "",
        "Under hver overskrift staar **Tolkning** -- den regel, laeseren foelger",
        "for netop den slags maerke. Det er reglerne, der skal bekraeftes af et",
        "menneske: giver de den tekst, du ville regne for en korrekt laesning?",
        "",
    ]
    for typ in sorted({r[0] for r in rader}):
        i_typen = [r for r in rader if r[0] == typ]
        linjer += [
            f"## {typ} ({sum(r[2] for r in i_typen)} forekomster, {len(i_typen)} former)",
            "",
            "**Tolkning:** " + forklaring.get(typ, "(ingen regel nedskrevet)"),
            "",
            "| Antal | Form | Eksempel i sammenhaeng |",
            "|---:|---|---|",
        ]
        for _, form, n, eks in i_typen:
            linjer.append(f"| {n} | `{form}` | {eks.replace('|', '/')} |")
        linjer.append("")
    sti.write_text("\n".join(linjer), encoding="utf-8", newline="\n")


def _skriv_opdeling(opslag: list[Opslag], hold: dict[str, str], sti: Path) -> None:
    pr_patient: Counter[str] = Counter(o.forside for o in opslag)
    with sti.open("w", encoding="utf-8", newline="") as f:
        skriver = csv.writer(f)
        skriver.writerow(["forside", "maengde", "antal_sider", "kildefil"])
        kilde = {o.forside: o.kildefil for o in opslag}
        for forside in sorted(pr_patient):
            skriver.writerow([forside, hold[forside], pr_patient[forside], kilde[forside]])


def _skriv_udeladte(tomme: list[Sideblok], opslag: list[Opslag], sti: Path) -> None:
    linjer = [
        "# Udeladte blokke og flagede steder",
        "",
        "## Sidemaerker uden tekst",
        "",
        "Siderne ER beskrevet i journalen -- de er bare ikke transskriberet.",
        "Kontrolleret 2026-08-20 paa ti af dem, spredt over alle syv patienter:",
        "alle ti har blaek i samme maengde som sider, vi ved har tekst, og to",
        "blev set efter med oejnene (den ene ender med 'doede Kl. 8 3/4').",
        "De maa derfor ikke bruges som facit -- en model, der laeser siden",
        "rigtigt, ville se ud til at digte det hele.",
        "",
        "Bemaerk moenstret: de fyrre ligger i kun syv patienter, altid som en",
        "sammenhaengende hale sidst i forloebet. Transskriptionen stopper",
        "tidligere end indlaeggelsen goer.",
        "",
        "| Billed-id | Kildefil |",
        "|---|---|",
    ]
    for b in tomme:
        linjer.append(f"| {b.image_name} | {b.kildefil} |")
    flaget = [(o, n) for o in opslag for n in o.noter]
    linjer += [
        "",
        f"## Steder hvor opmaerkningen ikke kunne tolkes ({len(flaget)})",
        "",
        "Blokken er stadig med i facit -- teksten er bevaret -- men stedet skal",
        "ses efter med oejnene, foer facit bruges til at maale paa.",
        "",
        "| Billed-id | Note |",
        "|---|---|",
    ]
    for o, note in flaget:
        linjer.append(f"| {o.image_name} | {note.replace('|', '/')} |")
    linjer.append("")
    sti.write_text("\n".join(linjer), encoding="utf-8", newline="\n")


def byg(ud: Path = STAGE02_OUTPUT) -> dict[str, int]:
    ud.mkdir(parents=True, exist_ok=True)
    blokke = laes_alle_blokke()
    tomme = [b for b in blokke if b.tom]
    opslag = [byg_opslag(b) for b in blokke if not b.tom]
    hold = opdel_patienter([o.forside for o in opslag])

    _skriv_facit(opslag, ud / "facit.jsonl")
    _skriv_klammekonventioner(tael_klammeformer(blokke), ud / "klammekonventioner.md")
    _skriv_opdeling(opslag, hold, ud / "opdeling.csv")
    _skriv_udeladte(tomme, opslag, ud / "udeladte.md")

    return {
        "filer": len({b.kildefil for b in blokke}),
        "blokke": len(blokke),
        "opslag": len(opslag),
        "tomme": len(tomme),
        "patienter": len(hold),
        "proeve": sum(1 for v in hold.values() if v == "proeve"),
        "flagede": sum(len(o.noter) for o in opslag),
    }


if __name__ == "__main__":
    for navn, tal in byg().items():
        print(f"{navn:12s} {tal}")
