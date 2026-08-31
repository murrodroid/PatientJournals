"""Selvtest af maaleapparatet -- stage 03.

Koerer maalingen mod facit selv og mod elleve konstruerede forvanskninger, hvor
det rigtige svar er kendt paa forhaand. Ingen modelkald: der er ikke koert et
eneste endnu, og formatet skal aftales FOER, saa tallene ikke bliver formet
efter, hvad der ser godt ud.

Det vigtigste, testen leverer, er ikke at tallene er "rigtige" -- det er
**forskellen mellem den forvanskning, vi selv lagde ind, og den, apparatet
maaler**. Naar vi selv har byttet 1.000 bogstaver og maaleren finder 940, er
de 60 skaevheden, og den skal staa skrevet ved siden af hovedtallet i stedet
for at vaere et skjult fradrag.

Skrevet om 2026-08-31, da forankringen blev fjernet (se `maal.py`s docstring).
Maalingen er nu én redigeringsafstand over hele siden, i raekkefoelge, uden
soegning. Alt hvad selvtesten sagde om daekning, uforankrede linjer og knappen
`MAKS_AFVIGELSE` er faldet vaek med mekanismen. Til gengaeld er den fejltype,
der vaeltede den gamle maaling, kommet ind som sin egen forvanskning:
`gentaget_ord`.

Kun **oevemaengden** bruges. Proevemaengden er laast til den endelige
bedoemmelse, og selvom en selvtest uden modelkald ikke kan afsloere noget om
den, holdes vanen: proevesiderne roeres ikke, foer der skal doemmes.

Skriver:
    stages/03_maaleapparat/output/selvtest.md
    stages/03_maaleapparat/output/rapportformat.md
    stages/03_maaleapparat/output/gab_eksempel.csv

Koerer nogle minutter: hver forvanskning er en fuld tabel-DP pr. side, og
tabellen er sidens tegn gange modelsvarets tegn. Det er ikke haengt.
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
from andenside.maal import maal_saet  # noqa: E402
from andenside.rapport import skriv_gab, skriv_rapport  # noqa: E402

FACIT = ROD / "stages" / "02_facit" / "output" / "facit.jsonl"
OPDELING = ROD / "stages" / "02_facit" / "output" / "opdeling.csv"
UD = ROD / "stages" / "03_maaleapparat" / "output"

MAERKE = "[?]"
FYLD = "utydeligt"          # det en model kunne finde paa at skrive paa et [?]
FROE = 20260822             # fast, saa to koersler giver samme forvanskning
BOGSTAVER = "abcdefghijklmnopqrstuvwxyzæøå"

# Mindste laengde for et ord, der taeller som "gentaget" i `gentaget_ord`.
# Kortere ord er funktionsord ("ikke", "den"), og de staar paa naesten hver
# eneste side flere gange -- de ville goere proeven til en proeve paa noget
# andet end det, der faktisk skete paa 273107_001864.
GENTAGET_MINDSTE_LAENGDE = 5


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


# --------------------------------------------------------------------------
# Gentaget ord -- den fejltype, der vaeltede den gamle maaling
# --------------------------------------------------------------------------

def _ordkerne(raa: str) -> str:
    """Ordet uden omgivende tegnsaetning og uden hensyn til store bogstaver."""
    return raa.strip("".join(cer.PUNCTUATION)).lower()


def _gentaget_ord(linjer: list[str]) -> str | None:
    """Det laengste ord paa mindst fem tegn, der staar paa mindst to linjer.

    Ord med et `[?]` i springes over: de er ikke ord, og modelteksten skriver
    noget andet paa stedet, saa de kunne alligevel ikke findes igen dér.

    Ved lige lange kandidater vaelges den, der optraeder foerst paa siden --
    `dict` bevarer indsaettelsesordenen, og `max` tager den foerste af de
    stoerste. Valget skal vaere fast, ellers giver to koersler hvert sit tal.
    """
    linjer_med: dict[str, set[int]] = {}
    for nr, linje in enumerate(linjer):
        for raa in linje.split():
            if MAERKE in raa:
                continue
            kerne = _ordkerne(raa)
            if len(kerne) >= GENTAGET_MINDSTE_LAENGDE:
                linjer_med.setdefault(kerne, set()).add(nr)
    kandidater = [k for k, hvor in linjer_med.items() if len(hvor) >= 2]
    return max(kandidater, key=len) if kandidater else None


def gentaget_ord(linjer, rng):
    """Almindelig lille laesefejl paa FOERSTE forekomst af et gentaget ord.

    Det er praecis mekanismen fra `273107_001864`: "ingen Snue" staar to gange
    i facit selv. Skrev modellen den foerste forekomst en anelse forkert, fandt
    den gamle soegning i stedet det ordrette traef nede i den ANDEN forekomst,
    flyttede soegepunktet dertil og tabte alt derimellem -- 26 af 29 linjer.
    Uden soegning findes der kun én vej gennem siden, og begge de to indlagte
    bogstavfejl skal derfor vaere at finde igen.

    Senere forekomster af ordet staar uroert. Findes der ikke et gentaget ord
    paa siden, er der intet at proeve, og siden leveres som `perfekt`.
    """
    kerne = _gentaget_ord(linjer)
    if kerne is None:
        return perfekt(linjer, rng)

    rene = _uden_maerker(linjer)
    for nr, linje in enumerate(rene):
        ord_ = linje.split()
        for i, raa in enumerate(ord_):
            if _ordkerne(raa) != kerne:
                continue
            pladser = [j for j, c in enumerate(raa) if c.isalpha()]
            if len(pladser) < 2:
                # Kan ikke bytte to bogstaver, saa der er ingen proeve at
                # lave. Siden leveres urort frem for med ét enkelt bytte, saa
                # det talte antal indlagte fejl altid passer.
                return perfekt(linjer, rng)
            nyt = list(raa)
            for j in rng.sample(pladser, 2):
                c = raa[j]
                b = rng.choice([x for x in BOGSTAVER if x != c.lower()])
                nyt[j] = b.upper() if c.isupper() else b
            ord_[i] = "".join(nyt)
            rene[nr] = " ".join(ord_)
            return "\n".join(rene), 2
    # Ordet fandtes i facit, men ikke som eget ord i modelteksten (et `[?]`
    # kan have klaebet det sammen med naboen). Ingen proeve.
    return perfekt(linjer, rng)


FORVANSKNINGER = [
    ("facit mod sig selv", perfekt,
     "Nul fejl i alle varianter. Det ord, \"modellen\" skriver dér hvor facit "
     "har `[?]`, er kortere end jokerfeltets loft og skal derfor slippe helt "
     "gratis igennem. Er tallet ikke nul her, er alt andet i tabellen "
     "ligegyldigt."),
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
     "Tæt på facit mod sig selv, men **ikke helt nul**, og resten er et målt "
     "fund: uden linjeskift kan et ord, facit har delt hen over to linjer, "
     "ikke samles igen, så `Infektions- sygdomme` bliver stående som to ord "
     "(efterprøvet linje for linje). Målingen må ellers ikke afhænge af, om "
     "modellen laver sine egne linjeskift (beslutning 35)."),
    ("hvert linjebrud flyttet ét ord", forskudte_brud,
     "Samme lille rest og samme årsag: bindestregen står nu midt på en linje "
     "i stedet for sidst, og så samles det delte ord ikke. Linjeskiftene er "
     "taget ud på begge sider før målingen, så hvor de sad, må ellers ikke "
     "kunne ses i tallet."),
    ("et opdigtet afsnit tilføjet", opdigtet,
     "Her ses forskellen fra de gamle rapporter tydeligst: det opdigtede "
     "afsnit koster nu ét point pr. indsat tegn i selve tegnfejlen. Under "
     "forankringen var afsnittet gratis, fordi det ikke havde nogen "
     "facit-linje at blive parret med."),
    ("2 % af bogstaverne byttet", to_procent,
     "Målt tegnafstand skal ligge tæt på antallet af indlagte fejl — se "
     "næste tabel for hvor tæt."),
    ("10 % af bogstaverne byttet", ti_procent,
     "Samme prøve, ti gange så hårdt. Med så mange fejl tæt på hinanden "
     "begynder redigeringsafstanden at kunne finde en billigere vej end vores "
     "egne ombytninger, og det skal kunne ses i næste tabel."),
    ("den midterste tredjedel sprunget over", halv_side,
     "Den sprungne tredjedel koster nu direkte: hvert tegn, modellen ikke "
     "skrev, er en sletning. Tegnfejlen skal derfor ligge omkring en "
     "tredjedel. Under forankringen faldt de manglende linjer helt ud af "
     "regnestykket og kostede næsten ingenting."),
    ("et gentaget ord læst en anelse forkert", gentaget_ord,
     "Prøven på netop dét, der væltede den gamle måling. To bogstaver byttet "
     "i **første** forekomst af et ord, der står på mindst to linjer i facit; "
     "de senere forekomster står urørt. Begge fejl skal findes igen. Antallet "
     "af sider, der overhovedet har sådan et ord, står under tabellen — er "
     "det lavt, er prøven svag."),
]


def koer(poster: list[dict], forvansk):
    """Returnerer (maaling, antal indlagte tegnfejl)."""
    rng = random.Random(FROE)
    modeller, sande = {}, 0
    for post in poster:
        tekst, antal = forvansk(post["alt_linjer"], rng)
        modeller[post["image_name"]] = tekst
        sande += antal
    return maal_saet(poster, modeller), sande


def _pct(x: float) -> str:
    return f"{x * 100:.2f}".replace(".", ",") + " %"


# --------------------------------------------------------------------------
# Jokerfeltets egen skaevhed: fejl, der lander paa et `[?]`, slipper gratis
# --------------------------------------------------------------------------

def _perfekt_med_joker(linjer: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Samme tekst som `perfekt`, plus hvor i den jokerfelterne ligger.

    Intervallerne er de tegn, "modellen" skrev dér hvor facit har `[?]`. En
    indlagt fejl inde i et af dem koster ingenting -- den slippes gratis
    igennem, fordi der ikke findes nogen sandhed at maale den imod.
    """
    dele: list[str] = []
    omraader: list[tuple[int, int]] = []
    pos = 0
    for nr, linje in enumerate(linjer):
        if nr:
            dele.append("\n")
            pos += 1
        for i, stykke in enumerate(linje.split(MAERKE)):
            if i:
                omraader.append((pos, pos + len(FYLD)))
                dele.append(FYLD)
                pos += len(FYLD)
            dele.append(stykke)
            pos += len(stykke)
    return "".join(dele), omraader


def fejl_i_joker(poster: list[dict], andel: float) -> tuple[int, int]:
    """(indlagte fejl i alt, heraf inde i et jokerfelt) for en tegnforvanskning.

    Regnestykket gentager `_forvansk_tegn`s traek af tilfaeldighedsgeneratoren
    tegn for tegn med samme froe og samme raekkefoelge af sider. Derfor giver
    det NOEJAGTIG de samme ombytninger som `to_procent`/`ti_procent`, og
    totalen skal stemme med deres. `selvtest()` tjekker det og siger fra, hvis
    de to skrider fra hinanden.
    """
    rng = random.Random(FROE)
    i_alt = i_joker = 0
    for post in poster:
        tekst, omraader = _perfekt_med_joker(post["alt_linjer"])
        for i, c in enumerate(tekst):
            if c.isalpha() and rng.random() < andel:
                rng.choice([b for b in BOGSTAVER if b != c.lower()])
                i_alt += 1
                if any(a <= i < b for a, b in omraader):
                    i_joker += 1
    return i_alt, i_joker


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
        "Målingen er én redigeringsafstand over hele siden, i rækkefølge, uden",
        "søgning. Der findes derfor ikke længere nogen dækning: hele facit er",
        "altid i nævneren, og en linje kan ikke falde ud af regnestykket.",
        "",
        "## Tallene",
        "",
        "| Forvanskning | raa | uden_versaler | uden_diakritika | uden_tegnsætn. | arbejdstal | arbejdstal, strengt | Model-tegn af facit-tegn | Omrokerede linjer |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    resultater = []
    for navn, funktion, forventet in FORVANSKNINGER:
        saet, sande = koer(poster, funktion)
        resultater.append((navn, saet, sande, forventet))
        f = saet.fladet
        model_tegn = sum(s.model_tegn_i_alt for s in saet.sider)
        facit_tegn = sum(s.facit_tegn_i_alt for s in saet.sider)
        ud.append(
            f"| {navn} | {_pct(f['raa'].cer)} | {_pct(f['uden_versaler'].cer)} | "
            f"{_pct(f['uden_diakritika'].cer)} | {_pct(f['uden_tegnsaetning'].cer)} | "
            f"{_pct(f['arbejdstal'].cer)} | {_pct(saet.rene['arbejdstal'].cer)} | "
            f"{model_tegn} af {facit_tegn} | "
            f"{saet.linjer_omrokeret} af {saet.linjer_i_alt} |"
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
        "**Model-tegn af facit-tegn** er, om modellen overhovedet skrev lige så",
        "meget tekst, som der stod på siden — tegn uden mellemrum. Den erstatter",
        'den gamle kolonne "modeltekst uden modstykke", som kun gav mening,',
        "mens der blev forankret. **Omrokerede linjer** er linjer, modellen",
        "skrev i en anden orden end facit; den måles for sig af `orden.py`,",
        "fordi hovedtallet er strengt om rækkefølge og ellers ville skjule,",
        "hvor stor en del af fejlen der bare er ombytning.",
        "",
        "### Hvad hver linje skal vise",
        "",
    ]
    for navn, _, _, forventet in resultater:
        ud.append(f"- **{navn}** — {forventet}")

    med_gentaget = sum(1 for p in poster if _gentaget_ord(p["alt_linjer"]))
    ud += [
        "",
        f"Af øvemængdens {len(poster)} sider har **{med_gentaget}** et ord på "
        f"mindst {GENTAGET_MINDSTE_LAENGDE} tegn, der står på to forskellige",
        "linjer. Kun de sider bidrager med indlagte fejl i prøven",
        '"et gentaget ord læst en anelse forkert" — resten leveres urørt. Er',
        "tallet lavt, er prøven tilsvarende svag, og det står her frem for at",
        "blive gemt bag procenten.",
    ]

    # Kernen: hvor meget af det, vi selv lagde ind, finder apparatet igen?
    ud += [
        "",
        "## Hvor meget apparatet finder af det, vi selv lagde ind",
        "",
        "Den vigtigste tabel i hele selvtesten. Venstre kolonne er tegn, vi selv",
        "byttede eller fjernede; midterkolonnen er den tegnafstand, målingen",
        "fandt. Er de ikke ens, er forskellen **skævheden i tallet**, og den skal",
        "stå her frem for at være et skjult fradrag.",
        "",
        "Tallet kan ligge på begge sider af 100 %. Under 100 %: redigerings-",
        "afstanden fandt en billigere vej end vores egne ombytninger — to fejl",
        "ved siden af hinanden kan af og til rettes med ét greb — eller fejlen",
        "landede inde i et jokerfelt og slap gratis igennem (se næste afsnit).",
        "Over 100 %: en ombytning kan have gjort teksten dyrere at rette end de",
        "enkelttegn, vi ændrede.",
        "",
        "| Forvanskning | Fejl vi lagde ind | Fejl målingen fandt | Fundet |",
        "|---|---:|---:|---:|",
    ]
    for navn, saet, sande, _ in resultater:
        if not sande:
            continue
        fundet = saet.fladet["raa"].tegnafstand
        ud.append(f"| {navn} | {sande} | {fundet} | {_pct(fundet / sande)} |")

    # Jokerfeltets egen skaevhed -- maalt, ikke formodet.
    ud += [
        "",
        "## Fejl, der forsvinder ned i et jokerfelt",
        "",
        "Hvor facit siger `[?]`, må modellen skrive hvad som helst op til",
        "jokerfeltets loft, uden at det koster. Det er en aftalt fribillet — der",
        "findes ingen sandhed at måle stedet imod — men den er samtidig",
        "målingens egen skævhed: en indlagt fejl, der tilfældigvis rammer inde i",
        "det ord, \"modellen\" skrev på et `[?]`, kan ikke findes igen.",
        "",
        "Her er den talt op i stedet for antaget. Optællingen gentager de samme",
        "ombytninger tegn for tegn og ser efter, hvor de landede.",
        "",
        "| Forvanskning | Indlagte fejl | Heraf inde i et jokerfelt | Andel |",
        "|---|---:|---:|---:|",
    ]
    joker_tal = {}
    for navn, andel in (("2 % af bogstaverne byttet", 0.02),
                        ("10 % af bogstaverne byttet", 0.10)):
        i_alt, i_joker = fejl_i_joker(poster, andel)
        joker_tal[navn] = (i_alt, i_joker)
        ud.append(
            f"| {navn} | {i_alt} | {i_joker} | "
            f"{_pct(i_joker / i_alt) if i_alt else '—'} |"
        )

    # Optaellingen skal stemme med selve forvanskningen, ellers maaler den
    # noget andet end den giver sig ud for.
    for navn, saet, sande, _ in resultater:
        if navn in joker_tal and joker_tal[navn][0] != sande:
            raise AssertionError(
                f"{navn}: optaellingen fandt {joker_tal[navn][0]} indlagte fejl, "
                f"forvanskningen lagde {sande} ind"
            )

    ud += [
        "",
        "De øvrige forvanskninger kan ikke ramme et jokerfelt. Omlyd, små",
        "bogstaver og fjernet tegnsætning rører ikke det ord, der står på et",
        "`[?]` — det har hverken ø, versaler eller tegnsætning — og det",
        "gentagne ord vælges udtrykkeligt blandt ord uden `[?]` i.",
        "",
        "Tallet er et loft for, hvad fribilletten koster i selvtesten, ikke et",
        "skøn over rigtige data. En rigtig model skriver noget andet og længere",
        "på et ulæseligt sted, og hvad den så gør, kan kun ses i gab-filen.",
    ]

    # Jokerfelterne, naar modellen skriver præcis det rigtige overalt ellers.
    saet, _ = koer(poster, perfekt)
    svaere = sum(s.svaere_linjer for s in saet.sider)
    linjer = sum(s.linjer_i_alt for s in saet.sider)
    ud += [
        "",
        "## De ulæselige steder i øvemængden",
        "",
        "| Mål | Værdi |",
        "|---|---:|",
        f"| Linjer i øvemængden | {linjer} |",
        f"| Heraf med mindst ét `[?]` | {svaere} = {_pct(svaere / linjer)} |",
        f"| Jokerfelter i alt | {len(saet.gab)} |",
        f"| Tegn \"modellen\" lagde i dem | {saet.joker_tegn_i_alt} |",
        f"| Tegn ud over loftet (det der kostede) | {saet.joker_overskud} |",
        "",
        "Målt på \"facit mod sig selv\", altså med det korte ord `utydeligt` på",
        "hvert `[?]`. Det ligger under loftet og koster derfor ingenting. En",
        "rigtig model kan skrive mere, og så begynder overskuddet at tælle —",
        "det tal er derfor ikke en forudsigelse, men et udgangspunkt at måle",
        "de rigtige kørsler op imod.",
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
