"""Finder snitpunktet, der adskiller en sides hovedindhold fra naboopslagets
strimmel -- afgraenset til den kant, recto/verso-reglen allerede har afgjort.

Se stages/04_billedforberedelse/CONTEXT.md for den fulde begrundelse.
Sidevalget er IKKE en gaetteopgave (det foelger recto/verso-paritet); det
der findes her, er selve snitpunktet inden for den kendte kant.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from andenside.masterlist import Side

STRIMMEL_ANDEL = 0.40  # hvor stor en del af bredden strimlen forventes at fylde
MIN_STIGNING = 0.01  # mindste troevaerdige spring i blaekprofilen over én kolonne


@dataclass(frozen=True)
class SoegeVindue:
    start: int
    slut: int
    retning: str  # "fra_venstre" eller "fra_hoejre"


def soegevindue(side: Side, bredde: int, *, strimmel_andel: float = STRIMMEL_ANDEL) -> SoegeVindue:
    """Afgraenser hvor i billedet strimlen forventes, ud fra recto/verso.

    Andenside (verso): hovedindhold venstre, strimmel hoejre -- soeg i
    hoejre kant, fra kanten og indad.
    Tredjeside (recto): hovedindhold hoejre, strimmel venstre -- soeg i
    venstre kant, fra kanten og indad.
    """
    vindue_bredde = int(bredde * strimmel_andel)
    if side.recto_verso == "verso":
        return SoegeVindue(start=bredde - vindue_bredde, slut=bredde, retning="fra_hoejre")
    if side.recto_verso == "recto":
        return SoegeVindue(start=0, slut=vindue_bredde, retning="fra_venstre")
    raise ValueError(f"{side.image_name}: recto/verso er '{side.recto_verso}', kan ikke afgraense soegevindue")


def column_ink_profile(
    img: Image.Image,
    *,
    dark_threshold: int = 180,
    step: int = 2,
    top_bottom_margin: float = 0.05,
) -> list[float]:
    """Blaekmaengde pr. kolonne, malt indenfor et lodret midterbaand.

    De yderste top_bottom_margin (standard 5%) af hoejden udelades bevidst.
    Affotograferingens baggrund/skygge fylder disse raekker paa en maade,
    der ikke haenger sammen med selve sideindholdet, og traekker snittet
    skaevt mod en bred, tom margen paa vores egen side i stedet for den
    fysiske rille. Fundet 2026-08-18 ved at lead paapegede fejlen paa
    273098_001496/1497 og 273099_001360/1361.
    """
    gray = img.convert("L")
    width, height = gray.size
    pixels = gray.load()
    y_start = int(height * top_bottom_margin)
    y_end = height - y_start
    profile = []
    for x in range(width):
        dark = 0
        total = 0
        for y in range(y_start, y_end, step):
            total += 1
            if pixels[x, y] < dark_threshold:
                dark += 1
        profile.append(dark / total)
    return profile


def smooth(values: list[float], window: int = 9) -> list[float]:
    half = window // 2
    out = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


@dataclass(frozen=True)
class SnitResultat:
    x: int
    styrke: float  # stoerrelsen af springet i blaekprofilen; 0.0 = intet troevaerdigt snit
    vindue: SoegeVindue


def find_snitpunkt(
    img: Image.Image,
    side: Side,
    *,
    min_stigning: float = MIN_STIGNING,
    buffer_andel: float = 0.01,
) -> SnitResultat:
    """Finder graensen mellem hovedsidens tekst og naboopslagets strimmel.

    Tre tidligere forsoeg fejlede. Et globalt lyseste-punkt kunne lande
    langt inde i naboens egen margen (forbi ryggen). En "foerste
    sammenhaengende blanke plet"-soegning fandt intet, fordi naboens tekst
    ofte begynder for taet paa ryggen til at give en blank periode. Og en
    fast NIVEAU-taerskel (blaekmaengden skal krydse fx 0,30) kan slet ikke
    virke generelt: det, profilen faktisk faar oeje paa, er SKYGGEN nede i
    falsen, og den er kraftig i de fleste bind, men naesten fravaerende i
    velfotograferede bind. Maalt paa oevemaengden topper falsen paa
    273104_001639/1640 ved 0,291/0,293, mens den vaerste haandskrifts-stoej
    paa vejen frem naar 0,300 paa andre sider -- intervallerne OVERLAPPER,
    saa ingen absolut taerskel kan skille fals fra haandskrift.

    Vi soeger derfor paa AENDRINGSHASTIGHEDEN i stedet for paa niveauet:
    foerste differens af den udglattede profil, taget langs soegeretningen
    (fra vores egen, betroede side og ind mod naboopslaget). Sidekanten er
    et braat spring, uanset hvor moerk skyggen er, mens haandskrift stiger
    og falder jaevnt. Snittet laegges ved den stoerste positive stigning i
    soegevinduet.

    Et lille buffer (standard 1% af billedbredden) flyttes derefter VAEK
    fra vores egen tekst og ind mod ryggen/naboen. Uden buffer risikerer et
    snit lige paa graensen at skaere brodstykker af hoeje eller dybe
    bogstavtraek, der straekker sig en anelse laengere end den udglattede
    profil viser. Lead paapegede behovet 2026-08-18.

    Er den stoerste stigning mindre end min_stigning, er profilen for flad
    til at rumme en kant overhovedet: der returneres styrke 0,0, saa
    kaldere (fx beskaer.py) kan lade siden vaere i stedet for at skaere paa
    et gaet. Gulvet ligger under det halve af den svageste bekraeftede fals
    i oevemaengden (0,021), saa en svag men rigtig fals ikke kasseres.
    """
    profile = smooth(column_ink_profile(img))
    vindue = soegevindue(side, img.width)
    buffer_px = max(1, int(img.width * buffer_andel))

    if vindue.retning == "fra_hoejre":
        # vores side er til venstre for vinduet; gaa fra vindue.start og udad
        raekkefoelge = range(vindue.start + 1, vindue.slut)
        forrige = -1
    else:
        # vores side er til hoejre for vinduet; gaa fra vindue.slut og indad
        raekkefoelge = range(vindue.slut - 2, vindue.start - 1, -1)
        forrige = 1

    bedste_x: int | None = None
    bedste_stigning = 0.0
    for x in raekkefoelge:
        stigning = profile[x] - profile[x + forrige]
        if stigning > bedste_stigning:
            bedste_stigning = stigning
            bedste_x = x

    if bedste_x is None or bedste_stigning < min_stigning:
        # ingen kant i vinduet -- usikkert, marker med styrke 0.0 saa det
        # kan filtreres fra i stedet for at blive brugt uden videre
        return SnitResultat(x=vindue.start, styrke=0.0, vindue=vindue)

    if vindue.retning == "fra_hoejre":
        x_med_buffer = min(bedste_x + buffer_px, vindue.slut - 1)
    else:
        x_med_buffer = max(bedste_x - buffer_px, vindue.start)
    return SnitResultat(x=x_med_buffer, styrke=bedste_stigning, vindue=vindue)
