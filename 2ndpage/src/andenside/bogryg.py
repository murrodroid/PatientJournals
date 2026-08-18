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

STRIMMEL_ANDEL = 0.30  # hvor stor en del af bredden strimlen forventes at fylde


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
    styrke: float  # blaekmaengde ved snittet (lavere = mere sikker dal)
    vindue: SoegeVindue


def find_snitpunkt(
    img: Image.Image,
    side: Side,
    *,
    ryg_taerskel: float = 0.30,
) -> SnitResultat:
    """Finder graensen mellem hovedsidens tekst og naboopslagets strimmel.

    To tidligere forsoeg fejlede: et globalt lyseste-punkt kunne lande
    langt inde i naboens egen margen (forbi rillen), og en "foerste
    sammenhaengende blanke plet"-soegning fandt intet, fordi naboens
    tekst ofte begynder for taet paa rillen til at give en blank periode.

    Den fysiske bogryg viser sig i praksis som en KRAFTIG TOP i
    blaekprofilen (0,5-1,0 -- langt over almindelig haandskrifts 0,05-0,15),
    ikke en dal -- bekraeftet visuelt 2026-08-18 paa flere billeder efter
    at soegningen blev afgraenset korrekt til kant-vinduet. Vi gaar fra
    vores egen, betroede side og ind mod naboopslaget, og snitter ved
    foerste kolonne hvor blaekmaengden krydser ryg_taerskel -- det er
    rygningens naere kant, saa hele vores egen side bevares, og baade
    ryggen og naboens strimmel skaeres fra.
    """
    profile = smooth(column_ink_profile(img))
    vindue = soegevindue(side, img.width)

    if vindue.retning == "fra_hoejre":
        # vores side er til venstre for vinduet; gaa fra vindue.start og udad
        raekkefoelge = range(vindue.start, vindue.slut)
    else:
        # vores side er til hoejre for vinduet; gaa fra vindue.slut og indad
        raekkefoelge = range(vindue.slut - 1, vindue.start - 1, -1)

    for x in raekkefoelge:
        if profile[x] >= ryg_taerskel:
            return SnitResultat(x=x, styrke=profile[x], vindue=vindue)

    # ingen ryg fundet i vinduet -- usikkert, marker med styrke 0.0 saa det
    # kan filtreres fra i stedet for at blive brugt uden videre
    return SnitResultat(x=vindue.start, styrke=0.0, vindue=vindue)
