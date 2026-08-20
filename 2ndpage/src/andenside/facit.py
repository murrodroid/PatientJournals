r"""Laeser de haandlavede facit-filer: RTF ind, ren tekst ud.

Filerne er skrevet i Apple TextEdit paa en Mac og gemt som RTF med
cp1252-escapes (`\'e6` for ae). De ser ud som almindelig tekst, men et
enkelt kontrolord kan aede bogstavet efter sig, hvis man bare fjerner
backslash-sekvenser med et soeg-og-erstat. Derfor afkoder vi RTF'en
ordentligt frem for at strippe den.

Vi bruger ikke et faerdigt RTF-bibliotek: filerne er meget ensartede
(TextEdit, ingen tabeller, ingen billeder), og vi har brug for at
kontrollere praecis, hvad der bliver til linjeskift -- linjeskiftene ER
data her, fordi de svarer til linjerne paa den haandskrevne side.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

BS = chr(92)  # backslash, skrevet saadan for at holde kildekoden laesbar

FACIT_ROOT = Path(
    r"<kilderod>"
    r"\PID-scapes and Blegdam Patient journals\Patient journals\Manual transcriptions"
)

# Grupper hvis indhold er formatering/metadata, ikke tekst. Naar en gruppe
# aabner med et af disse kontrolord, springes hele gruppen over.
SKJULTE_GRUPPER = frozenset(
    {
        "fonttbl",
        "colortbl",
        "expandedcolortbl",
        "stylesheet",
        "listtable",
        "listoverridetable",
        "info",
        "pgdsctbl",
        "themedata",
        "colorschememapping",
        "latentstyles",
        "datastore",
        "generator",
    }
)

# Kontrolord der giver et linjeskift i den udlaeste tekst.
LINJESKIFT = frozenset({"par", "line", "sect"})

# Kontrolord der giver et konkret tegn.
TEGNORD = {"tab": "\t", "emdash": "—", "endash": "–", "bullet": "•",
           "lquote": "‘", "rquote": "’", "ldblquote": "“", "rdblquote": "”"}


def _laes_kontrolord(raw: str, i: int) -> tuple[str, int | None, int]:
    """Laeser ét kontrolord fra og med bogstavet paa plads `i`.

    Returnerer (navn, talparameter eller None, ny position). Et enkelt
    mellemrum efter kontrolordet er selve afgraensningen og hoerer ikke
    med til teksten -- det er den fejl, en naiv stripper laver.
    """
    start = i
    while i < len(raw) and raw[i].isalpha():
        i += 1
    navn = raw[start:i]
    tal_start = i
    if i < len(raw) and raw[i] == "-":
        i += 1
    while i < len(raw) and raw[i].isdigit():
        i += 1
    tal = int(raw[tal_start:i]) if raw[tal_start:i] not in ("", "-") else None
    if i < len(raw) and raw[i] == " ":
        i += 1
    return navn, tal, i


def rtf_til_tekst(raw: str) -> str:
    """Afkoder en TextEdit-RTF til ren tekst med linjeskift bevaret."""
    ud: list[str] = []
    # Én post pr. aaben tuborgklamme: skal indholdet skrives ud eller ej.
    stak: list[bool] = [True]
    # Antal tegn der skal springes over efter en \uN-escape (\ucN styrer det).
    unicode_spring = 1
    spring_tilbage = 0
    i = 0
    n = len(raw)
    while i < n:
        c = raw[i]
        if c == "{":
            stak.append(stak[-1])
            i += 1
            continue
        if c == "}":
            if len(stak) > 1:
                stak.pop()
            i += 1
            continue
        if c == "\\":
            i += 1
            if i >= n:
                break
            k = raw[i]
            if k == "'":
                byte = int(raw[i + 1 : i + 3], 16)
                i += 3
                if spring_tilbage > 0:
                    spring_tilbage -= 1
                elif stak[-1]:
                    ud.append(bytes([byte]).decode("cp1252", errors="replace"))
                continue
            if k in (BS, "{", "}"):
                i += 1
                if stak[-1]:
                    ud.append(k)
                continue
            if k in "\r\n":
                # TextEdits afsnitsskift: backslash umiddelbart foer nylinje.
                i += 1
                if k == "\r" and i < n and raw[i] == "\n":
                    i += 1
                if stak[-1]:
                    ud.append("\n")
                continue
            if k == "*":
                # `{\*\...}` er en gruppe, laeseren skal ignorere helt.
                stak[-1] = False
                i += 1
                continue
            if not k.isalpha():
                i += 1
                if stak[-1]:
                    ud.append({"~": " ", "_": "‑", "-": ""}.get(k, k))
                continue
            navn, tal, i = _laes_kontrolord(raw, i)
            if navn in SKJULTE_GRUPPER:
                stak[-1] = False
                continue
            if navn == "uc":
                unicode_spring = tal if tal is not None else 1
                continue
            if navn == "u" and tal is not None:
                if stak[-1] and spring_tilbage == 0:
                    ud.append(chr(tal + 65536 if tal < 0 else tal))
                spring_tilbage = unicode_spring
                continue
            if spring_tilbage > 0:
                continue
            if navn in LINJESKIFT and stak[-1]:
                ud.append("\n")
            elif navn in TEGNORD and stak[-1]:
                ud.append(TEGNORD[navn])
            continue
        if c in "\r\n":
            # Raa nylinje i kildefilen er kun linjeombrydning, ikke afsnit.
            i += 1
            continue
        i += 1
        if spring_tilbage > 0:
            spring_tilbage -= 1
            continue
        if stak[-1]:
            ud.append(c)
    return "".join(ud).strip("\n")


# ---------------------------------------------------------------------------
# Opdeling i sideblokke
# ---------------------------------------------------------------------------

# Bind-id'et er ikke altid seks cifre: bind 37554 har fem. En regex paa
# seks cifre taber to hele filer uden at sige noget.
_ID = r"\d+_\d+"
FORSIDEMARKOER = re.compile(r"\[transcription of frontpage\s+(" + _ID + r")[^\]]*\]", re.I)
SIDEMARKOER = re.compile(r"\[(?:page\s+)?(" + _ID + r")\]", re.I)


@dataclass(frozen=True)
class Sideblok:
    """Én transskriberet journalside, som den staar i facit-filen."""

    image_name: str
    forside: str | None
    kildefil: str
    raa: str

    @property
    def tom(self) -> bool:
        """Sandt naar sidemaerket staar uden tekst efter sig.

        Siden ER beskrevet i journalen -- den er bare ikke transskriberet.
        Kontrolleret 2026-08-20 paa ti af de fyrre: alle ti har blaek i samme
        maengde som sider, vi ved har tekst. De maa derfor ikke bruges som
        facit: en model, der laeser siden rigtigt, ville se ud til at digte
        det hele.
        """
        return not self.raa.strip()


def del_i_sideblokke(tekst: str, kildefil: str) -> list[Sideblok]:
    """Deler en hel journalfil op i én blok pr. fortsaettelsesside.

    Teksten foer det foerste sidemaerke hoerer til forsiden. Forsiderne er
    feltopdelte, transskriberet andetsteds og ikke vores maalside -- de
    kommer ikke med.
    """
    fund = FORSIDEMARKOER.search(tekst)
    forside = fund.group(1) if fund else None
    dele = SIDEMARKOER.split(tekst)
    ider, kroppe = dele[1::2], dele[2::2]
    return [
        Sideblok(image_name=i, forside=forside, kildefil=kildefil, raa=krop)
        for i, krop in zip(ider, kroppe)
    ]


# ---------------------------------------------------------------------------
# Klammeopmaerkning
# ---------------------------------------------------------------------------

# Maerkerne er skrevet i haanden over flere aar og har tastefejl. Vi matcher
# derfor paa moenster, ikke paa faste strenge. Optaellingen af alle
# forekomster staar i stage 02's `output/klammekonventioner.md`.
_ULAESELIG = re.compile(r"^\?+$")
_UNDERSTREGNING = re.compile(r"underlin", re.I)
_OVERSTREGET = re.compile(r"^crossed\s*out$", re.I)
_ERSTATNING = re.compile(r"^written\s+instead$", re.I)
_FORTSAET = re.compile(r"^contin\w*\s+(?:on|under)\s*line$", re.I)
# Positionsmaerker er fritekst, men naevner altid side, hjoerne eller page.
_POSITION = re.compile(r"\b(?:side|corner)\b|page", re.I)
_INDSKUD = re.compile(r"^(?:note\s+)?add(?:ed|et)\b", re.I)
_GAET = re.compile(r"^(.+)\?$", re.S)


def klassificer_klamme(indre: str) -> tuple[str, str]:
    """Afgoer hvad et klammemaerke betyder. Returnerer (type, nyttetekst).

    Raekkefoelgen er ikke ligegyldig: `[note added right side page]` er en
    margennote, ikke et indskud over linjen, saa position skal proeves foer
    indskud.
    """
    kerne = " ".join(indre.split())
    if _ULAESELIG.match(kerne):
        return "ulaeselig", "[?]"
    if _UNDERSTREGNING.search(kerne):
        return "understregning", ""
    if _OVERSTREGET.match(kerne):
        return "overstreget", ""
    if _ERSTATNING.match(kerne):
        return "erstatning", ""
    if _FORTSAET.match(kerne):
        return "fortsaet", ""
    if _POSITION.search(kerne):
        return "position", ""
    if _INDSKUD.match(kerne):
        return "indskud", ""
    gaet = _GAET.match(kerne)
    if gaet:
        return "gaet", gaet.group(1)
    return "ukendt", kerne


def _tokens(tekst: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Deler teksten i almindelig tekst og klammemaerker.

    Klammer kan vaere indlejrede -- en understregningsnote kan rumme et
    ulaeselighedsmaerke -- saa vi taeller dybde frem for at bruge en regex.
    Ubalanceret opmaerkning er en tastefejl i kilden (fire filer har den) og
    bliver rapporteret, ikke gaettet paa plads.
    """
    ud: list[tuple[str, str]] = []
    noter: list[str] = []
    buffer: list[str] = []
    dybde = 0
    start = 0
    for i, c in enumerate(tekst):
        if c == "[":
            if dybde == 0:
                buffer.append(tekst[start:i])
                start = i + 1
            dybde += 1
        elif c == "]":
            if dybde == 0:
                noter.append("overskydende slutklamme ved " + repr(tekst[max(0, i - 30) : i]))
                buffer.append(tekst[start:i])
                start = i + 1
                continue
            dybde -= 1
            if dybde == 0:
                ud.append(("tekst", "".join(buffer)))
                buffer = []
                ud.append(("klamme", tekst[start:i]))
                start = i + 1
    rest = tekst[start:]
    if dybde > 0:
        noter.append("uafsluttet klamme ved " + repr(rest[:40]))
    buffer.append(rest)
    ud.append(("tekst", "".join(buffer)))
    return ud, noter


def ren_laesetekst(raa: str) -> tuple[str, list[str]]:
    """Udleder den tekst, vi regner som korrekt laesning af siden.

    Overstreget tekst falder ud, erstatningen bliver staaende, indskud og
    margentekst bliver staaende (ordene ER skrevet paa siden), mens noter om
    understregning og placering falder ud (de omtaler tekst, der allerede er
    der). Ulaeselighedsmaerket bevares.

    Returnerer (tekst, noter), hvor noter er de steder, opmaerkningen ikke
    kunne tolkes -- de skal ses efter med oejnene, ikke gaettes paa plads.
    """
    # Transskribenten skriver et bogstaveligt backslash-n for et haandskrevet
    # linjeskift, saerlig i receptblokke i margenen.
    tekst = raa.replace(BS + "n", "\n")
    tokens, noter = _tokens(tekst)

    ud: list[str] = []

    def slutter_linje() -> bool:
        return not ud or "".join(ud).endswith("\n")

    overstreget = False
    for slags, vaerdi in tokens:
        if slags == "tekst":
            if not overstreget:
                ud.append(vaerdi)
                continue
            # Overstregningen loeber til linjeskiftet, ikke laengere.
            brud = vaerdi.find("\n")
            if brud >= 0:
                overstreget = False
                if not slutter_linje():
                    ud.append("\n")
                ud.append(vaerdi[brud + 1 :])
            continue

        typ, nytte = klassificer_klamme(vaerdi)
        if overstreget:
            # Klammer inde i det overstregede falder med, undtagen de to
            # maerker der afslutter overstregningen.
            if typ in ("erstatning", "fortsaet"):
                overstreget = False
            continue
        if typ == "overstreget":
            overstreget = True
        elif typ in ("ulaeselig", "gaet"):
            ud.append(nytte)
        elif typ == "position":
            # Margentekst skal ikke klistre sig til journallinjen.
            if not slutter_linje():
                ud.append("\n")
        elif typ == "ukendt":
            noter.append("ukendt klammemaerke [" + vaerdi + "]")
            ud.append(nytte)
        # understregning, erstatning, fortsaet og indskud giver ingen tekst

    samlet = "".join(ud).replace("\r", "")
    # Fjernede maerker efterlader mellemrum i hver ende af linjen. Indrykning
    # baerer ingen betydning i disse filer, saa den kan trimmes vaek.
    renset = "\n".join(linje.strip() for linje in samlet.split("\n"))
    # Flere blanke linjer i traek er ogsaa et affald af fjernede maerker;
    # ÉN blank linje adskiller de daglige notater og skal blive staaende.
    renset = re.sub(r"\n{3,}", "\n\n", renset)
    return renset.strip("\n"), noter


def saml_orddeling(tekst: str) -> str:
    """Flader teksten ud: orddeling samles, linjeskift bliver mellemrum.

    Det er den udgave, tegnfejlene maales paa. StadsCER mangler netop dette
    trin, og dagbogen dér udpeger orddeling som det dominerende fejlmoenster.
    """
    linjer = tekst.split("\n")
    dele: list[str] = []
    sammenhaeng = False
    for i, linje in enumerate(linjer):
        s = linje.strip()
        if not s and not sammenhaeng:
            continue
        naeste = linjer[i + 1].strip() if i + 1 < len(linjer) else ""
        # Bindestreg sidst paa linjen deler kun et ord, naar naeste linje
        # fortsaetter med lille bogstav. Materialet bruger nemlig ogsaa
        # bindestreg som punktum -- "enkelte Rhonchi-" efterfulgt af en ny
        # saetning -- og de to er ikke ét ord.
        deler = s.endswith("-") and naeste[:1].islower()
        if deler:
            kerne = s[:-1]
            # Selve bindestregen falder kun vaek, naar der staar et bogstav
            # foran den. Staar den efter fx et ulaeselighedsmaerke, ved vi
            # ikke hvad der blev delt -- saa bliver den staaende.
            stykke = kerne if (kerne and kerne[-1].isalpha()) else s
        else:
            stykke = s
        if sammenhaeng and dele:
            dele[-1] += stykke
        else:
            dele.append(stykke)
        sammenhaeng = deler
    return " ".join(d for d in dele if d)
