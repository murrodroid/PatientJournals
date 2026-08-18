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


def column_ink_profile(img: Image.Image, *, dark_threshold: int = 180, step: int = 2) -> list[float]:
    gray = img.convert("L")
    width, height = gray.size
    pixels = gray.load()
    profile = []
    for x in range(width):
        dark = 0
        total = 0
        for y in range(0, height, step):
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


def find_snitpunkt(img: Image.Image, side: Side) -> SnitResultat:
    """Finder den lyseste (mindst blaekfyldte) kolonne inden for det
    forventede strimmel-vindue -- den lokale dal er graensen mellem
    hovedsidens tekst og naboopslagets strimmel."""
    profile = smooth(column_ink_profile(img))
    vindue = soegevindue(side, img.width)
    baand = profile[vindue.start : vindue.slut]
    if not baand:
        raise ValueError(f"{side.image_name}: tomt soegevindue {vindue}")
    valley_offset = min(range(len(baand)), key=lambda i: baand[i])
    x = vindue.start + valley_offset
    return SnitResultat(x=x, styrke=profile[x], vindue=vindue)
