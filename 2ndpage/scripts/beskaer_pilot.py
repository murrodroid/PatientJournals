"""Beskaerer pilotens sider og tegner et gennemsynsark pr. side.

Dette er stage 05's foerste gennemgangspunkt: der kaldes INGEN model, foer
Lead har set snittene efter. Et daarligt snit paa et ukendt bind er langt
billigere at opdage her end i et maaletal.

Arket viser to ting ved siden af hinanden:

  VENSTRE  hele billedet med snitlinjen tegnet ind (roed) og soegevinduet
           markeret (blaa) -- er snittet lagt det rigtige sted?
  HOEJRE   det faktisk beskaarne billede -- er der fremmed tekst tilbage?

Den anden halvdel er ikke pynt. Et snit kan ligge helt korrekt ved bogryggen
og ALLIGEVEL efterlade fremmed tekst: blade fra andre steder i bindet kan rage
ud i indbindingen og blive fotograferet med (lead 2026-08-27). Det ses kun
paa resultatet, ikke paa snitlinjen.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROD / "src"))

from andenside.bogryg import find_snitpunkt  # noqa: E402
from andenside.skraa import beskaer_langs_fals, fals_graense  # noqa: E402
from andenside.masterlist import load_masterlist, lookup  # noqa: E402
from andenside.vaern import sikr_oevemaengde  # noqa: E402

KILDER = ROD / "stages" / "01_datagrundlag" / "output" / "oeve_billeder"
UD = ROD / "stages" / "05_foerste_transskription" / "output"
BESKAARNE = UD / "beskaarne"
ARK = UD / "gennemsyn_snit"

ROED = (220, 30, 30)
BLAA = (0, 128, 255)
GRAA = (140, 140, 140)


def _gennemsynsark(img: Image.Image, snit, graense: list[int],
                   beskaaret: Image.Image | None) -> Image.Image:
    """Original med den baandvise snitgraense til venstre, resultatet til hoejre.

    Graensen tegnes raekke for raekke, saa dens haeldning kan ses -- en enkelt
    lodret streg ville skjule netop det, arket er til for at vise.
    """
    venstre = img.convert("RGB")
    tegn = ImageDraw.Draw(venstre)
    tegn.rectangle(
        [(snit.vindue.start, 0), (snit.vindue.slut - 1, venstre.height - 1)],
        outline=BLAA, width=3,
    )
    for y, x in enumerate(graense):
        if 0 <= x < venstre.width:
            tegn.rectangle([(x - 2, y), (x + 2, y)], fill=ROED)

    hoejre = beskaaret.convert("RGB") if beskaaret is not None else None
    mellemrum = 24
    bredde = venstre.width + mellemrum + (hoejre.width if hoejre else 0)
    hoejde = max(venstre.height, hoejre.height if hoejre else 0)

    ark = Image.new("RGB", (bredde, hoejde), (255, 255, 255))
    ark.paste(venstre, (0, 0))
    if hoejre is not None:
        ark.paste(hoejre, (venstre.width + mellemrum, 0))
        # tynd ramme om resultatet, saa dets kanter kan ses mod hvid bund
        ImageDraw.Draw(ark).rectangle(
            [(venstre.width + mellemrum, 0),
             (venstre.width + mellemrum + hoejre.width - 1, hoejre.height - 1)],
            outline=GRAA, width=2,
        )
    return ark


def main() -> None:
    sider = [r["billede"] for r in csv.DictReader(
        (UD / "pilotsider.csv").open(encoding="utf-8"))]
    sikr_oevemaengde(sider)

    index = load_masterlist()
    BESKAARNE.mkdir(parents=True, exist_ok=True)
    ARK.mkdir(parents=True, exist_ok=True)

    raekker = []
    usikre = []
    for billede in sider:
        side = lookup(billede, index)
        kilde = KILDER / f"{billede}.webp"
        with Image.open(kilde) as img:
            img.load()
            snit = find_snitpunkt(img, side)
            graense = fals_graense(img, side)
            beskaaret, maaling = beskaer_langs_fals(img, side)
            _gennemsynsark(img, snit, graense, beskaaret).save(
                ARK / f"{billede}_gennemsyn.png")

        ud_fil = BESKAARNE / f"{billede}.webp"
        beskaaret.save(ud_fil)
        if not maaling.sikker:
            usikre.append(billede)
        raekker.append({
            "billede": maaling.billede,
            "recto_verso": maaling.recto_verso,
            "bredde_foer": maaling.bredde_foer,
            "bredde_efter": maaling.bredde_efter,
            "haeldning_px": maaling.haeldning_px,
            "baand_med_kant": f"{maaling.baand_med_kant}/{maaling.baand_i_alt}",
            "sikker": "ja" if maaling.sikker else "nej",
            "fil": ud_fil.name,
        })

        fjernet = 1 - maaling.bredde_efter / maaling.bredde_foer
        print(f"{billede}  {side.recto_verso:<6} fjernet {fjernet:>5.1%}"
              f"   haeldning {maaling.haeldning_px:>3} px"
              f"   baand med fals {maaling.baand_med_kant}/{maaling.baand_i_alt}"
              + ("   USIKKER" if not maaling.sikker else ""))

    sti = UD / "snit_pilot.csv"
    with sti.open("w", encoding="utf-8", newline="") as f:
        skriver = csv.DictWriter(f, fieldnames=list(raekker[0]))
        skriver.writeheader()
        skriver.writerows(raekker)

    print(f"\n{len(raekker) - len(usikre)} af {len(raekker)} sider beskaaret.")
    if usikre:
        print(f"USIKRE, ikke beskaaret: {', '.join(usikre)}")
    print(f"Gennemsynsark: {ARK.relative_to(ROD)}")
    print(f"Snit-register: {sti.relative_to(ROD)}")


if __name__ == "__main__":
    main()
