"""Kontaktark KUN over de beskaarne siders YDRE kant.

Falsbeskaeringen (skraa.py) renser den ene kant. Paa den modsatte kant --
sidens yderkant -- ligger der stadig enten bogsnittet (bogblokkens
sammenpressede sidekanter, harmloest) eller et blad, der er faldet fladt ud
og fotograferet med, saa der staar FREMMED haandskrift langs kanten.

Dette ark er til at SE forskellen med oejnene, ikke til at maale den.
Strimlerne spejles, saa yderkanten altid vender samme vej i arket -- ellers
skal oejet skifte retning for hver anden side.

Kaldes uden argumenter; skriver til stagens output.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw

STAGE04 = Path(__file__).resolve().parents[1] / "stages" / "04_billedforberedelse" / "output"
BESKAARNE = STAGE04 / "beskaarne"
SNIT_CSV = STAGE04 / "snit_alle.csv"
ARK_DIR = STAGE04 / "yderkant_ark"

ANDEL = 0.28    # hvor stor en del af bredden strimlen viser
HOEJDE = 820    # hver strimmels hoejde i arket
PR_ARK = 7      # strimler pr. ark
# 7, ikke 14: ved 14 skaleres arket ned til under strimlernes egen
# oploesning, og netop det, der skal bedoemmes -- om de faa tegn ude ved
# kanten er fremmed haandskrift -- forsvinder i nedskaleringen.


def laes_recto_verso(snit_csv: Path) -> dict[str, str]:
    with snit_csv.open(encoding="utf-8") as handle:
        return {row["billede"]: row["recto_verso"] for row in csv.DictReader(handle)}


def ydre_strimmel(img: Image.Image, recto_verso: str, *, andel: float = ANDEL) -> Image.Image:
    """Klipper den ydre kant ud og vender den, saa kanten altid er til hoejre.

    Recto har falsen til venstre, altsaa yderkanten til hoejre; verso omvendt.
    Verso-strimlen spejles, saa alle strimler kan laeses ens.
    """
    n = max(1, int(img.width * andel))
    if recto_verso == "recto":
        return img.crop((img.width - n, 0, img.width, img.height))
    if recto_verso == "verso":
        return img.crop((0, 0, n, img.height)).transpose(Image.FLIP_LEFT_RIGHT)
    raise ValueError(f"recto_verso er '{recto_verso}', kan ikke afgoere yderkanten")


def _ton_det_bortskaarne(img: Image.Image, graense: list[int], recto_verso: str) -> Image.Image:
    """Toner det, snittet fjerner, roedt -- i stedet for at tegne en streg.

    En streg oven paa billedet daekker de bogstaver, der staar taettest paa
    snittet, og efter nedskaleringen til kontaktarket ser det ud, som om de
    er klippet af. Lead blev ført bag lyset af netop det 2026-08-28. Med en
    tonet flade males intet over: det, der beholdes, staar uroert, og det,
    der ryger, kan stadig laeses igennem tonen.
    """
    import numpy as np

    data = np.asarray(img.convert("RGB")).astype(float)
    kol = np.arange(img.width)[None, :]
    g = np.asarray(graense)[:, None]
    ude = (kol > g) if recto_verso == "recto" else (kol < g)
    ude = ude[:, :, None]
    roed = np.array([255.0, 60.0, 60.0])[None, None, :]
    blandet = np.where(ude, data * 0.55 + roed * 0.45, data)
    return Image.fromarray(blandet.astype(np.uint8), mode="RGB")


def byg(ark_dir: Path = ARK_DIR, *, med_snit: bool = False) -> list[Path]:
    """Bygger arkene. `med_snit` tegner den fundne yderkant ind med roedt.

    Uden snit er arket til at klassificere efter (hvad LIGGER der ude ved
    kanten?); med snit er det til at bedoemme detektionen efter (rammer
    den?). Samme strimler, saa de to kan holdes op mod hinanden.
    """
    recto_verso = laes_recto_verso(SNIT_CSV)
    ark_dir.mkdir(parents=True, exist_ok=True)
    if med_snit:
        from andenside.masterlist import load_masterlist, lookup
        from andenside.yderkant import ydre_graense

        index = load_masterlist()

    strimler: list[tuple[str, Image.Image]] = []
    for navn in sorted(recto_verso):
        img = Image.open(BESKAARNE / f"{navn}.webp").convert("RGB")
        if med_snit:
            side = lookup(navn, index)
            graense = ydre_graense(img, side)
            if graense:
                img = _ton_det_bortskaarne(img, graense, side.recto_verso)
        s = ydre_strimmel(img, recto_verso[navn])
        skala = HOEJDE / s.height
        strimler.append((navn, s.resize((max(1, round(s.width * skala)), HOEJDE))))

    skrevne = []
    for start in range(0, len(strimler), PR_ARK):
        parti = strimler[start : start + PR_ARK]
        ark = Image.new("RGB", (sum(s.width + 8 for _, s in parti), HOEJDE + 26), (255, 255, 255))
        tegn = ImageDraw.Draw(ark)
        x = 0
        for navn, s in parti:
            ark.paste(s, (x, 26))
            tegn.text((x + 3, 8), navn, fill=(0, 0, 0))
            # roed skillelinje: uden den flyder to lyse strimler sammen
            tegn.line([(x + s.width + 4, 0), (x + s.width + 4, ark.height)], fill=(200, 0, 0), width=2)
            x += s.width + 8
        sti = ark_dir / f"yderkant_{start // PR_ARK + 1:02d}.png"
        ark.save(sti)
        skrevne.append(sti)
        print(f"{sti}  ({len(parti)} strimler)")
    return skrevne


if __name__ == "__main__":
    med_snit = "--snit" in sys.argv
    mappe = (STAGE04 / "yderkant_snit_ark") if med_snit else ARK_DIR
    sys.exit(0 if byg(mappe, med_snit=med_snit) else 1)
