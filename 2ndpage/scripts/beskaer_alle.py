"""Beskaerer ALLE oevebilleder langs falsen og laegger dem til gennemsyn.

Piloten (15 sider) viste, at det baandvise snit i `andenside.skraa` holder.
Dette script koerer den samme beskaering paa hele oevemaengden, saa stage 04
kan bedoemmes paa bredden i stedet for paa et haandplukket udsnit. Der kaldes
INGEN model her -- det er ren lokal billedbehandling.

Der skrives tre ting:

  beskaarne/            de beskaarne sider, én webp pr. billede
  snit_alle.csv         ét maaletal pr. side (hvor meget forsvandt, hvor
                        skaev falsen var, og om graensen overhovedet blev
                        fundet i nok baand)
  kontaktark_beskaarne/ miniaturer 12 ad gangen, saa resultatet kan skimmes
                        med oejnene i stedet for laeses i en tabel

Kontaktarkene viser de BESKAARNE sider -- ikke snitlinjen. Et snit kan ligge
korrekt ved bogryggen og alligevel efterlade fremmed tekst, og omvendt kan en
skaev fals se dramatisk ud i tallene uden at koste noget. Begge dele ses kun
paa resultatet.

Proevemaengden roeres ikke: listen gaar gennem `sikr_oevemaengde()`, foer
noget som helst aabnes.
"""

from __future__ import annotations

import csv
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROD / "src"))

from andenside.skraa import beskaer_langs_fals  # noqa: E402
from andenside.masterlist import load_masterlist, lookup  # noqa: E402
from andenside.vaern import sikr_oevemaengde  # noqa: E402

KILDER = ROD / "stages" / "01_datagrundlag" / "output" / "oeve_billeder"
UD = ROD / "stages" / "04_billedforberedelse" / "output"
BESKAARNE = UD / "beskaarne"
KONTAKTARK = UD / "kontaktark_beskaarne"

MINIATURE_BREDDE = 320
PR_ARK = 12
SOEJLER = 4
TEKSTHOEJDE = 34
MELLEMRUM = 12
BAGGRUND = (250, 250, 250)
RAMME = (170, 170, 170)
ADVARSEL = (200, 40, 40)


def _skrifttype() -> ImageFont.ImageFont:
    for navn in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(navn, 15)
        except OSError:
            continue
    return ImageFont.load_default()


def _kontaktark(poster: list[dict], sti: Path, skrift) -> None:
    """Et ark med op til 12 miniaturer, hver med sin egen billedtekst.

    Miniaturerne skaleres til samme bredde, ikke samme hoejde: sidernes
    hoejde varierer kun lidt, mens bredden er netop det, beskaeringen har
    aendret, og en faelles bredde ville skjule forskellen.
    """
    celler = []
    for post in poster:
        with Image.open(post["fil"]) as billede:
            billede.load()
            hoejde = max(1, round(billede.height * MINIATURE_BREDDE / billede.width))
            celler.append((post, billede.convert("RGB").resize(
                (MINIATURE_BREDDE, hoejde), Image.LANCZOS)))

    celle_hoejde = max(m.height for _, m in celler) + TEKSTHOEJDE
    raekker = (len(celler) + SOEJLER - 1) // SOEJLER
    ark = Image.new(
        "RGB",
        (SOEJLER * (MINIATURE_BREDDE + MELLEMRUM) + MELLEMRUM,
         raekker * (celle_hoejde + MELLEMRUM) + MELLEMRUM),
        BAGGRUND,
    )
    tegn = ImageDraw.Draw(ark)

    for i, (post, miniature) in enumerate(celler):
        x = MELLEMRUM + (i % SOEJLER) * (MINIATURE_BREDDE + MELLEMRUM)
        y = MELLEMRUM + (i // SOEJLER) * (celle_hoejde + MELLEMRUM)
        ark.paste(miniature, (x, y))
        tegn.rectangle(
            [(x, y), (x + miniature.width - 1, y + miniature.height - 1)],
            outline=RAMME, width=1)
        tekst = (f"{post['billede']}  {post['recto_verso']}  "
                 f"haeldning {post['haeldning_px']} px")
        if post["sikker"] == "nej":
            tekst += "  USIKKER"
        tegn.text((x, y + miniature.height + 6), tekst, font=skrift,
                  fill=ADVARSEL if post["sikker"] == "nej" else (20, 20, 20))

    ark.save(sti)


def _beskaer_én(billede: str) -> dict:
    """Beskaerer ét billede og gemmer det. Koeres i sin egen proces.

    Ligger paa modulniveau, fordi Windows starter delprocesser ved at
    importere modulet -- en indlejret funktion kan ikke sendes derover.
    Masterlisten laeses én gang pr. proces og genbruges.
    """
    global _INDEX
    try:
        index = _INDEX
    except NameError:
        index = _INDEX = load_masterlist()

    side = lookup(billede, index)
    with Image.open(KILDER / f"{billede}.webp") as img:
        img.load()
        beskaaret, maaling = beskaer_langs_fals(img, side)

    ud_fil = BESKAARNE / f"{billede}.webp"
    beskaaret.save(ud_fil)
    fjernet = 1 - maaling.bredde_efter / maaling.bredde_foer
    return {
        "billede": maaling.billede,
        "recto_verso": maaling.recto_verso,
        "bredde_foer": maaling.bredde_foer,
        "bredde_efter": maaling.bredde_efter,
        "fjernet_andel": f"{fjernet:.4f}",
        "haeldning_px": maaling.haeldning_px,
        "baand_med_kant": f"{maaling.baand_med_kant}/{maaling.baand_i_alt}",
        "sikker": "ja" if maaling.sikker else "nej",
        "fil": str(ud_fil),
    }


def main() -> None:
    billeder = sorted(p.stem for p in KILDER.glob("*.webp"))
    sikr_oevemaengde(billeder)

    index = load_masterlist()
    BESKAARNE.mkdir(parents=True, exist_ok=True)
    KONTAKTARK.mkdir(parents=True, exist_ok=True)

    # Beskaeringen er ren regnekraft pr. side og deler intet mellem sider,
    # saa den fordeles over kernerne. Processer og ikke traade: arbejdet er
    # Python-side, og traade ville blive serialiseret af GIL'en.
    kerner = max(1, min(len(billeder), (os.cpu_count() or 2) - 1))
    print(f"beskaerer {len(billeder)} sider paa {kerner} kerner", flush=True)

    raekker = []
    with ProcessPoolExecutor(max_workers=kerner) as pulje:
        for nr, raekke in enumerate(pulje.map(_beskaer_én, billeder), 1):
            raekker.append(raekke)
            fjernet = float(raekke["fjernet_andel"])
            print(f"[{nr:>3}/{len(billeder)}] {raekke['billede']}"
                  f"  {raekke['recto_verso']:<6}"
                  f" fjernet {fjernet:>5.1%}"
                  f"   haeldning {raekke['haeldning_px']:>4} px"
                  f"   fals i {raekke['baand_med_kant']} vinduer"
                  + ("   USIKKER" if raekke["sikker"] != "ja" else ""), flush=True)

    kolonner = ["billede", "recto_verso", "bredde_foer", "bredde_efter",
                "fjernet_andel", "haeldning_px", "baand_med_kant", "sikker"]
    sti = UD / "snit_alle.csv"
    with sti.open("w", encoding="utf-8", newline="") as f:
        skriver = csv.DictWriter(f, fieldnames=kolonner, extrasaction="ignore")
        skriver.writeheader()
        skriver.writerows(raekker)

    skrift = _skrifttype()
    for i in range(0, len(raekker), PR_ARK):
        nummer = i // PR_ARK + 1
        _kontaktark(raekker[i:i + PR_ARK],
                    KONTAKTARK / f"ark_{nummer:02d}.png", skrift)
        print(f"kontaktark {nummer} skrevet", flush=True)

    usikre = [r["billede"] for r in raekker if r["sikker"] == "nej"]
    haeldninger = [r["haeldning_px"] for r in raekker]
    fjernede = [float(r["fjernet_andel"]) for r in raekker]
    print(f"\n{len(raekker)} sider beskaaret, heraf {len(usikre)} usikre.")
    if usikre:
        print("USIKRE: " + ", ".join(usikre))
    print(f"haeldning: median {statistics.median(haeldninger):.0f} px, "
          f"mindst {min(haeldninger)}, stoerst {max(haeldninger)}")
    print(f"fjernet:  median {statistics.median(fjernede):.1%}, "
          f"mindst {min(fjernede):.1%}, stoerst {max(fjernede):.1%}")
    print(f"Snit-register: {sti.relative_to(ROD)}")
    print(f"Kontaktark:    {KONTAKTARK.relative_to(ROD)}")


if __name__ == "__main__":
    main()
