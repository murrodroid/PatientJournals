"""Prototype: find bogryggen i et dobbeltopslag ved en kolonnevis blækprofil.

Midlertidigt udforskningsscript til stage 04 (billedforberedelse), IKKE en
del af den endelige billedforberedelses-kode -- det er en første afprøvning
af metoden beskrevet i stage 04's CONTEXT.md, bygget paa en observation fra
magresprot_xmltools' separator-research (se references/icm_metodik.md):

    En bogryg viser sig som en moerk TOP i en kolonnevis blaekmaengde-profil
    (modsat en blaekstreg, der giver en dal), og ryggens position er
    tilnaermelsesvis konstant hele siden igennem, hvor haandskrift flytter
    sig fra linje til linje.

Metode her: konverter til graatoner, beregn for hver kolonne andelen af
moerke pixels, glat profilen, og find den hoejeste top inden for det
midterste baand af billedet (bogryggen ligger pr. definition mellem de to
sider, ikke ude i margenerne).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def load_grayscale(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def column_ink_profile(img: Image.Image, *, dark_threshold: int = 180) -> list[float]:
    """Andel moerke pixels pr. kolonne (0..1)."""
    width, height = img.size
    pixels = img.load()
    profile = []
    for x in range(width):
        dark = 0
        for y in range(0, height, 2):  # hvert andet punkt er nok, hurtigere
            if pixels[x, y] < dark_threshold:
                dark += 1
        profile.append(dark / (height / 2))
    return profile


def column_longest_dark_run(img: Image.Image, *, dark_threshold: int = 200) -> list[float]:
    """Laengste sammenhaengende moerke lodrette straek pr. kolonne (0..1 af hoejden).

    Haandskrift er lokal -- bogstaver har huller mellem sig og linjer imellem.
    Bogryggens skygge/fold er derimod naesten sammenhaengende hele siden
    igennem. Denne profil skelner de to, hvor ren blaekmaengde ikke kan.
    """
    width, height = img.size
    pixels = img.load()
    profile = []
    for x in range(width):
        longest = 0
        current = 0
        for y in range(0, height, 2):
            if pixels[x, y] < dark_threshold:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        profile.append(longest / (height / 2))
    return profile


def smooth(values: list[float], window: int = 9) -> list[float]:
    half = window // 2
    out = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def find_spine_candidate(profile: list[float], *, band_fraction: float = 0.5) -> tuple[int, float]:
    """Find den hoejeste top inden for det midterste baand af billedet."""
    width = len(profile)
    margin = int(width * (1 - band_fraction) / 2)
    band = profile[margin : width - margin]
    peak_offset = max(range(len(band)), key=lambda i: band[i])
    peak_x = margin + peak_offset
    return peak_x, profile[peak_x]


def find_spine_valley(profile: list[float], *, band_fraction: float = 0.4) -> tuple[int, float]:
    """Find det lyseste (mindst blaekfyldte) sted i et smalt midterbaand.

    Modsat find_spine_candidate: bogryggen kan ogsaa vaere et lyst mellemrum
    mellem to tekstblokke snarere end en moerk fold -- afhaenger af
    fotograferingen. Baandet holdes smalt (40% af bredden) for at undgaa at
    ramme sidernes egne, ogsaa lyse yderkanter.
    """
    width = len(profile)
    margin = int(width * (1 - band_fraction) / 2)
    band = profile[margin : width - margin]
    valley_offset = min(range(len(band)), key=lambda i: band[i])
    valley_x = margin + valley_offset
    return valley_x, profile[valley_x]


def annotate(path: Path, spine_x: int, out_path: Path) -> None:
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.line([(spine_x, 0), (spine_x, img.height)], fill=(255, 0, 0), width=4)
    img.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="sti til ét opslag (webp/jpg)")
    parser.add_argument("--out", type=Path, help="hvor kontrolbilledet gemmes")
    args = parser.parse_args()

    gray = load_grayscale(args.image)
    profile = smooth(column_ink_profile(gray))
    spine_x, strength = find_spine_valley(profile)
    print(f"{args.image.name}: kandidat-rygposition (dal) x={spine_x} (bredde {gray.width}), styrke={strength:.3f}")

    out = args.out or args.image.with_name(args.image.stem + "_bogryg_kontrol.png")
    annotate(args.image, spine_x, out)
    print(f"Kontrolbillede gemt: {out}")


if __name__ == "__main__":
    main()
