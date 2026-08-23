"""Selvtest af maaleapparatet -- stage 03.

Koerer maalingen mod facit selv og mod ti konstruerede forvanskninger, hvor
det rigtige svar er kendt paa forhaand. Ingen modelkald: der er ikke koert et
eneste endnu, og formatet skal aftales FOER, saa tallene ikke bliver formet
efter, hvad der ser godt ud.

Det vigtigste, testen leverer, er ikke at tallene er "rigtige" -- det er
**forskellen mellem den forvanskning, vi selv lagde ind, og den, apparatet
maaler**. Naar vi selv har byttet 1.000 bogstaver og maaleren finder 940, er
de 60 skaevheden, og den skal staa skrevet ved siden af hovedtallet i stedet
for at vaere et skjult fradrag.

Kun **oevemaengden** bruges. Proevemaengden er laast til den endelige
bedoemmelse, og selvom en selvtest uden modelkald ikke kan afsloere noget om
den, holdes vanen: proevesiderne roeres ikke, foer der skal doemmes.

Skriver:
    stages/03_maaleapparat/output/selvtest.md
    stages/03_maaleapparat/output/rapportformat.md
    stages/03_maaleapparat/output/gab_eksempel.csv

Koerer ca. 9 minutter: de forvanskede udgaver rammer sjaeldent et ordret traef,
saa hver stump skal findes ved naermeste-udsnit-soegning. Det er ikke haengt.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROD / "src"))

from andenside import cer  # noqa: E402
from andenside.maal import MAKS_AFVIGELSE, maal_saet  # noqa: E402
from andenside.rapport import skriv_gab, skriv_rapport  # noqa: E402

FACIT = ROD / "stages" / "02_facit" / "output" / "facit.jsonl"
OPDELING = ROD / "stages" / "02_facit" / "output" / "opdeling.csv"
UD = ROD / "stages" / "03_maaleapparat" / "output"

MAERKE = "[?]"
FYLD = "utydeligt"          # det en model kunne finde paa at skrive paa et [?]
FROE = 20260822             # fast, saa to koersler giver samme forvanskning
BOGSTAVER = "abcdefghijklmnopqrstuvwxyzæøå"


def oevemaengden() -> list[dict]:
    forsider = {
        r["forside"] for r in csv.DictReader(OPDELING.open(encoding="utf-8"))
        if r["maengde"] == "oeve"
    }
    poster = [json.loads(l) for l in FACIT.read_text(encoding="utf-8").splitlines()]
    return sorted(
        (p for p in poster if p["forside"] in forsider), key=lambda p: p["image_name"]
    )


# --------------------------------------------------------------------------
# Forvanskninger
#
# Hver returnerer (modeltekst, antal_indlagte_tegnfejl). Tallet er nul for de
# forvanskninger, der ikke aendrer et bogstav -- de flytter kun linjeskift
# eller lignende, og der ER derfor ingen fejl at finde.
# --------------------------------------------------------------------------

def _uden_maerker(linjer: list[str]) -> list[str]:
    """Modellen ser ikke facits [?] -- den skriver sit eget bud paa stedet."""
    return [l.replace(MAERKE, FYLD) for l in linjer]


def perfekt(linjer, rng):
    return "\n".join(_uden_maerker(linjer)), 0


def omlyd(linjer, rng):
    tekst = perfekt(linjer, rng)[0]
    ny = tekst.replace("ø", "ö").replace("Ø", "Ö")
    return ny, sum(1 for a, b in zip(tekst, ny) if a != b)


def smaat(linjer, rng):
    tekst = perfekt(linjer, rng)[0]
    ny = tekst.lower()
    return ny, sum(1 for a, b in zip(tekst, ny) if a != b)


def uden_tegnsaetning(linjer, rng):
    tekst = perfekt(linjer, rng)[0]
    # Bindestregen bliver staaende: fjernes den, forsvinder orddelingen med
    # den, og saa maaler forvanskningen to ting paa én gang.
    ny = "".join(c for c in tekst if c not in cer.PUNCTUATION or c == "-")
    return ny, len(tekst) - len(ny)


def et_afsnit(linjer, rng):
    return " ".join(_uden_maerker(linjer)), 0


def forskudte_brud(linjer, rng):
    """Hvert linjebrud flyttet ét ord til hoejre. Ingen bogstaver aendres."""
    rene = _uden_maerker(linjer)
    ord_ = " ".join(rene).split()
    ud, i = [], 0
    for n in (len(l.split()) for l in rene):
        ud.append(" ".join(ord_[i : i + n + 1]))
        i += n + 1
    return "\n".join(l for l in ud if l), 0


def opdigtet(linjer, rng):
    tekst, _ = perfekt(linjer, rng)
    return tekst + "\nPatienten blev udskrevet rask og velbefindende.", 0


def _forvansk_tegn(tekst: str, rng: random.Random, andel: float) -> tuple[str, int]:
    """Bytter bogstaver ud. Det nye bogstav er altid et ANDET -- ellers ville
    en del af de talte fejl ikke vaere fejl, og sammenligningen med det maalte
    tal ville skride."""
    ud, aendret = [], 0
    for c in tekst:
        if c.isalpha() and rng.random() < andel:
            nyt = rng.choice([b for b in BOGSTAVER if b != c.lower()])
            ud.append(nyt.upper() if c.isupper() else nyt)
            aendret += 1
        else:
            ud.append(c)
    return "".join(ud), aendret


def to_procent(linjer, rng):
    return _forvansk_tegn(perfekt(linjer, rng)[0], rng, 0.02)


def ti_procent(linjer, rng):
    return _forvansk_tegn(perfekt(linjer, rng)[0], rng, 0.10)


def halv_side(linjer, rng):
    """Modellen springer den midterste tredjedel af siden over."""
    rene = _uden_maerker(linjer)
    n = len(rene)
    return "\n".join(rene[: n // 3] + rene[2 * n // 3 :]), 0


FORVANSKNINGER = [
    ("facit mod sig selv", perfekt,
     "Nul fejl i alle varianter. Sætter samtidig **gulvet** for kolonnen "
     "\"modeltekst uden modstykke\": den er ikke nul her, selvom intet er "
     "digtet. Det, der står, er ordet `utydeligt` dér hvor facit har `[?]` "
     "plus teksten på de linjer, der ikke kunne forankres. Ved en rigtig "
     "måling skal tallet læses som et tillæg til dette gulv, ikke som et "
     "absolut mål for opdigtning."),
    ("alle ø skrevet som ö", omlyd,
     "`raa` får fejl; `uden_diakritika` og `lempeligst` skal være nul. Det er "
     "den hyppigste enkeltforveksling i materialet, og den er ortografisk "
     "støj, ikke en læsefejl."),
    ("alt med små bogstaver", smaat,
     "`raa` får fejl; `uden_versaler` og `arbejdstal` skal være nul."),
    ("al tegnsætning fjernet", uden_tegnsaetning,
     "`raa` får fejl; `uden_tegnsaetning` og `arbejdstal` skal være tæt på "
     "nul. Bindestregen er bevidst ladt stå — fjernes den, forsvinder "
     "orddelingen med den, og så måler prøven to ting på én gang."),
    ("hele siden som ét afsnit", et_afsnit,
     "Samme tal som facit mod sig selv. Målingen må ikke afhænge af, om "
     "modellen laver sine egne linjeskift (beslutning 35)."),
    ("hvert linjebrud flyttet ét ord", forskudte_brud,
     "Samme tal som facit mod sig selv. Uden forankringen ville alt efter "
     "det første brud være forkert — det er hele grunden til, at linjerne "
     "parres på indhold og ikke på linjenummer."),
    ("et opdigtet afsnit tilføjet", opdigtet,
     "Tegnfejlen ser det ikke. Kun \"modeltekst uden modstykke\" gør, og den "
     "springer fra gulvet på ~2.500 tegn til ~7.500. Det er derfor det tal "
     "skal stå ved siden af hovedtallet i enhver rapport."),
    ("2 % af bogstaverne byttet", to_procent,
     "Målt tegnafstand skal ligge tæt på antallet af indlagte fejl — se "
     "næste tabel for hvor tæt."),
    ("10 % af bogstaverne byttet", ti_procent,
     "Samme, men her begynder dækningen at falde: de hårdest forvanskede "
     "linjer kan ikke forankres."),
    ("den midterste tredjedel sprunget over", halv_side,
     "Dækningen skal falde til omkring to tredjedele. **Tegnfejlen bliver "
     "IKKE nul**, og det er et målt fund, ikke en forventning — se afsnittet "
     "\"Falske forankringer\" nedenfor for hvad der faktisk sker."),
]


def koer(poster: list[dict], forvansk, *, maks_afvigelse: float = MAKS_AFVIGELSE):
    """Returnerer (maaling, antal indlagte tegnfejl)."""
    rng = random.Random(FROE)
    modeller, sande = {}, 0
    for post in poster:
        tekst, antal = forvansk(post["alt_linjer"], rng)
        modeller[post["image_name"]] = tekst
        sande += antal
    return maal_saet(poster, modeller, maks_afvigelse=maks_afvigelse), sande


def _pct(x: float) -> str:
    return f"{x * 100:.2f}".replace(".", ",") + " %"


# --------------------------------------------------------------------------
# Udskrivning
# --------------------------------------------------------------------------

def selvtest(poster: list[dict]) -> str:
    ud = [
        "# Selvtest af måleapparatet",
        "",
        f"Kørt på **øvemængdens {len(poster)} sider**. Ingen modelkald — hver",
        '"modeltekst" er facit selv, forvansket på en måde hvor det rigtige svar',
        "er kendt på forhånd. Kør igen med `scripts/selvtest_maaleapparat.py`.",
        "",
        "Forvanskningerne er konstruerede, ikke repræsentative. Det er meningen:",
        "data der fremkalder en bestemt fejl er sjældent typiske.",
        "",
        "## Tallene",
        "",
        "| Forvanskning | raa | uden_versaler | uden_diakritika | uden_tegnsætn. | arbejdstal | arbejdstal, strengt | Dækning | Modeltekst uden modstykke |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    resultater = []
    for navn, funktion, forventet in FORVANSKNINGER:
        saet, sande = koer(poster, funktion)
        resultater.append((navn, saet, sande, forventet))
        f = saet.fladet
        uforankret = sum(s.model_tegn_uforankret for s in saet.sider)
        ud.append(
            f"| {navn} | {_pct(f['raa'].cer)} | {_pct(f['uden_versaler'].cer)} | "
            f"{_pct(f['uden_diakritika'].cer)} | {_pct(f['uden_tegnsaetning'].cer)} | "
            f"{_pct(f['arbejdstal'].cer)} | {_pct(saet.rene['arbejdstal'].cer)} | "
            f"{_pct(saet.daekning)} | {uforankret} tegn |"
        )

    ud += [
        "",
        "Kolonnen **arbejdstal, strengt** er den samme måling med linjer, der",
        "rummer et `[?]`, helt ude (beslutning 44). Her i selvtesten forvanskes",
        "alle bogstaver med samme sandsynlighed, så de svære linjer er IKKE",
        "sværere end de andre — de to tal bør derfor ligge tæt. Gør de det,",
        "ved vi, at selve maskineriet ikke skaber en forskel, og at en forskel",
        "på rigtige data kommer fra materialet, ikke fra måden at måle på.",
        "",
        "### Hvad hver linje skal vise",
        "",
    ]
    for navn, _, _, forventet in resultater:
        ud.append(f"- **{navn}** — {forventet}")

    # Kernen: hvor meget af det, vi selv lagde ind, finder apparatet igen?
    ud += [
        "",
        "## Hvor meget apparatet finder af det, vi selv lagde ind",
        "",
        "Den vigtigste tabel i hele selvtesten. Venstre kolonne er bogstaver, vi",
        "selv byttede om; midterkolonnen er den tegnafstand, målingen fandt. Er de",
        "ikke ens, er forskellen **skævheden i tallet** — og den peger altid samme",
        "vej: målingen finder mindre, end der er, fordi de linjer den ikke kan",
        "forankre, er de hårdest ramte.",
        "",
        "| Forvanskning | Fejl vi lagde ind | Fejl målingen fandt | Fundet |",
        "|---|---:|---:|---:|",
    ]
    for navn, saet, sande, _ in resultater:
        if not sande:
            continue
        fundet = saet.fladet["raa"].tegnafstand
        ud.append(
            f"| {navn} | {sande} | {fundet} | "
            f"{_pct(fundet / sande) if sande else '—'} |"
        )
    ud += [
        "",
        "Tallet kan ikke nå 100 %. Tre grunde, alle kendte:",
        "",
        "1. **Uforankrede linjer falder ud** — de hårdest forvanskede først.",
        "2. **Stumper under fem tegn bruges ikke** til forankring, så teksten",
        "   omkring et `[?]` er ikke altid med.",
        "3. **Levenshtein kan være billigere end vores ombytninger** — to fejl",
        "   ved siden af hinanden kan af og til rettes med ét greb.",
        "",
        "Det er derfor, dækningen skal stå ved hvert tal. Et tal på 5 % tegnfejl",
        "målt på 88 % af teksten er ikke det samme som 5 % på det hele.",
    ]

    # Falske forankringer -- efterprøvet, ikke formodet.
    ud += [
        "",
        "## Falske forankringer",
        "",
        "Springer modellen en del af siden over, bliver tegnfejlen ikke nul,",
        "selvom hvert eneste ord, den faktisk skrev, er rigtigt. Første forklaring var",
        "en formodning; her er hvad der faktisk sker, efterprøvet linje for linje",
        "på forvanskningen \"den midterste tredjedel sprunget over\":",
        "",
        "**1. En manglende linje forankrer sig i en linje, der ligner.** Facits",
        "`Hendes tilstand er i løbet af natten bleven` findes ikke i modellen, men",
        "`I løbet af natten` gør — og stumpen lander dér. `Tungen` lander i",
        "`Lunge`. `ingen Appetit, ligget hen og døset,` lander i `Det ligger hen",
        "og døser,`.",
        "",
        "**2. Og det skader de EFTERFØLGENDE linjer.** Det var ikke med i den",
        "første forklaring, og det er den vigtigere halvdel. Forankringen går fra",
        "venstre mod højre, så et falsk træf flytter søgepunktet frem forbi det",
        "sted, hvor de næste linjer i virkeligheden står. De finder så kun en",
        "afskåret rest af sig selv: `begge Lunger overalt en Mængde fugtige` blev",
        "målt mod `r overalt en Mængde fugtige`, selvom modellen havde skrevet",
        "hele linjen rigtigt.",
        "",
        "Prisen er lille på dette materiale — 181 tegn fordelt på 27 af de 118",
        "sider — men den vokser med, hvor meget modellen springer over. Derfor:",
        "**en side med lav dækning skal ses efter med øjnene**, ikke bare tros.",
        "Rapporten har sin egen liste over de tyndest målte sider netop derfor.",
    ]

    # Knappen.
    ud += [
        "",
        "## Knappen `MAKS_AFVIGELSE`",
        "",
        "Hvor meget en stump må afvige og stadig regnes for fundet. Tabellen står",
        "her, fordi knappen kan bruges til at pynte: sættes den lavere, falder",
        "dækningen, og de linjer der bliver tilbage, er de letteste. Tegnfejlen ser",
        "bedre ud og måler mindre og mindre repræsentativt materiale.",
        "",
        'Målt på forvanskningen "10 % af bogstaverne byttet".',
        "",
        "| MAKS_AFVIGELSE | raa | Dækning | Linjer målt | Fundet af de indlagte fejl |",
        "|---:|---:|---:|---:|---:|",
    ]
    for graense in (0.2, 0.4, 0.6):
        saet, sande = koer(poster, ti_procent, maks_afvigelse=graense)
        maalt = sum(s.linjer_maalt for s in saet.sider)
        i_alt = sum(s.linjer_i_alt for s in saet.sider)
        fundet = saet.fladet["raa"].tegnafstand
        ud.append(
            f"| {str(graense).replace('.', ',')} | {_pct(saet.fladet['raa'].cer)} | "
            f"{_pct(saet.daekning)} | {maalt} af {i_alt} | {_pct(fundet / sande)} |"
        )
    ud += [
        "",
        f"Projektets værdi er **{str(MAKS_AFVIGELSE).replace('.', ',')}**. Den er sat",
        "rundhåndet med vilje. Læg mærke til, at den strengeste indstilling giver den",
        "*laveste* tegnfejl — den ser bedst ud og er mest misvisende.",
    ]

    # Hvad forankringen henter hjem.
    saet, _ = koer(poster, perfekt)
    svaere = sum(s.svaere_linjer for s in saet.sider)
    reddet = sum(s.svaere_linjer_reddet for s in saet.sider)
    linjer = sum(s.linjer_i_alt for s in saet.sider)
    ud += [
        "",
        "## Hvad forankringen henter hjem",
        "",
        "Beslutning 38 skærer hele linjen fra, når den rummer et `[?]`.",
        "Forankringen henter de kendte stumper på linjen tilbage i målingen.",
        "",
        "| Mål | Værdi |",
        "|---|---:|",
        f"| Linjer i øvemængden | {linjer} |",
        f"| Heraf med mindst ét `[?]` | {svaere} = {_pct(svaere / linjer)} |",
        f"| Svære linjer forankringen redder | {reddet} = {_pct(reddet / svaere)} af dem |",
        f"| Dækning med forankring | {_pct(saet.daekning)} |",
        f"| Gab fundet (modellens bud på et `[?]`) | {len(saet.gab)} |",
        "",
        "**Bemærk at dette er en øvre grænse.** Her er \"modellen\" facit selv, så",
        "hver stump findes ordret. En rigtig model læser dårligere, og færre",
        "stumper vil kunne forankres. Det rigtige tal kommer først i stage 05.",
    ]
    return "\n".join(ud) + "\n"


def _konstrueret_modelsvar(poster: list[dict]):
    """Facit med 5 % af bogstaverne byttet + et opdigtet afsnit pr. side.

    Ligger for sig, fordi baade rapportformatet og gab-filen skal vise DEN
    SAMME maaling. Blev de regnet hver for sig, kunne de vise hver sit.
    """
    rng = random.Random(FROE)
    modeller = {}
    for post in poster:
        tekst, _ = _forvansk_tegn(perfekt(post["alt_linjer"], rng)[0], rng, 0.05)
        modeller[post["image_name"]] = tekst + "\nPatienten udskrevet rask."
    return maal_saet(poster, modeller)


def rapportformat(saet) -> str:
    """Rapporten, som den vil se ud efter stage 05's første kørsel.

    Bygget på en konstrueret forvanskning, ikke på et modelsvar. Formålet er
    at aftale formatet, før der er tal at blive glad eller skuffet over.
    """
    return skriv_rapport(
        saet,
        titel="Rapportformat — eksempel på en færdig måling",
        model="INGEN — konstrueret prøve, ikke et modelsvar",
        promptversion="—",
        dato="2026-08-22",
        noter=(
            "**Dette er ikke en måling af en model.** \"Modelteksten\" er facit selv "
            "med 5 % af bogstaverne byttet tilfældigt og et opdigtet afsnit sat til "
            "sidst på hver side, så alle rapportens felter har noget at vise. "
            "Formatet er aftalt her, før første modelkald, så tallene ikke bliver "
            "formet efter, hvad der ser godt ud.\n\n"
            "Tallene selv betyder derfor ingenting. Det, der skal tages stilling til, "
            "er om det er DE FELTER, der skal træffes valg ud fra.\n\n"
            "To ting i tabellerne er artefakter af den konstruerede prøve og vil se "
            "anderledes ud ved et rigtigt modelsvar: `raa`, `uden_versaler` og "
            "`uden_diakritika` er ens, fordi forvanskningen hverken ændrer store "
            "bogstaver eller omlyde — og linjetrofastheden er 100 %, fordi "
            '"modellen" her per konstruktion skriver facits egne linjeskift.\n\n'
            "Kørt på øvemængden. Prøvemængden røres først ved den endelige "
            "bedømmelse."
        ),
    )


def main() -> None:
    poster = oevemaengden()
    UD.mkdir(parents=True, exist_ok=True)
    (UD / "selvtest.md").write_text(selvtest(poster), encoding="utf-8")
    saet = _konstrueret_modelsvar(poster)
    (UD / "rapportformat.md").write_text(rapportformat(saet), encoding="utf-8")
    (UD / "gab_eksempel.csv").write_text(skriv_gab(saet), encoding="utf-8")
    print(f"{len(poster)} sider. Skrevet til {UD}")


if __name__ == "__main__":
    main()
