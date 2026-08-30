"""Koerer BEGGE snit paa kollegaens levering, saa de kan bedoemmes med oejnene.

`beskaer_alle.py` koerer kun falssnittet og kun paa oevemaengdens webp-filer.
Dette script koerer hele kaeden paa leveringens PNG-filer:

  1. `skraa.beskaer_langs_fals`  -- falsen, baand for baand
  2. `yderkant.beskaer_ydre`     -- den modsatte kant, langs sidens egen soem

Der kaldes INGEN model. Det er ren lokal billedbehandling.

## Om proevemaengden

`vaern.sikr_oevemaengde` findes, fordi vi ikke maa SE FACIT for
proevesiderne, foer den endelige bedoemmelse koerer -- det staar i vaernets
egen begrundelse. En beskaering roerer ikke facit; den ser kun paa
billedpunkter. Derfor maa proevesiderne godt beskaeres, og lead bad
udtrykkeligt om det 2026-08-30 for at kunne se snittene efter.

Resultatet lægges alligevel i sin EGEN mappe, og maalefilen holdes adskilt,
saa ingen senere kørsel kan komme til at blande de to maengder sammen.

## Hvad der skrives

  <ud>/<gruppe>/beskaarne/      de faerdige sider, én PNG pr. billede
  <ud>/<gruppe>/snit.csv        ét maaletal pr. side for begge snit
  <ud>/<gruppe>/kontaktark/     miniaturer 12 ad gangen til gennemsyn
"""

from __future__ import annotations

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROD / "src"))

from andenside.masterlist import load_masterlist, lookup  # noqa: E402
from andenside.skraa import beskaer_langs_fals  # noqa: E402
from andenside.yderkant import beskaer_ydre  # noqa: E402

LEVERING = ROD / "stages" / "01_datagrundlag" / "output" / "levering_2026-08"
UD = ROD / "stages" / "04_billedforberedelse" / "output" / "levering_beskaaret"

GRUPPER = ("oeve", "ekstra_uden_facit", "proeve_LAAST")
MINIATURE_BREDDE = 300
PR_ARK = 12
SOEJLER = 4
MELLEMRUM = 12
TEKSTHOEJDE = 34
BAGGRUND = (250, 250, 250)
RAMME = (170, 170, 170)
ADVARSEL = (200, 40, 40)


def _skrifttype():
    for navn in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(navn, 14)
        except OSError:
            continue
    return ImageFont.load_default()


def _beskaer_én(opgave: tuple[str, str]) -> dict:
    """Beskaerer ét billede i begge kanter. Koeres i sin egen proces.

    Ligger paa modulniveau, fordi Windows starter delprocesser ved at
    importere modulet. Masterlisten laeses én gang pr. proces.
    """
    gruppe, billede = opgave
    global _INDEX
    try:
        index = _INDEX
    except NameError:
        index = _INDEX = load_masterlist()

    side = lookup(billede, index)
    with Image.open(LEVERING / gruppe / f"{billede}.png") as img:
        img.load()
        foer_bredde = img.width
        efter_fals, m_fals = beskaer_langs_fals(img, side)
        efter_begge, m_ydre = beskaer_ydre(efter_fals, side)

    mappe = UD / gruppe / "beskaarne"
    mappe.mkdir(parents=True, exist_ok=True)
    efter_begge.save(mappe / f"{billede}.png", optimize=True)

    return {
        "billede": billede,
        "recto_verso": side.recto_verso,
        "bredde_foer": foer_bredde,
        "bredde_efter_fals": m_fals.bredde_efter,
        "bredde_efter_begge": m_ydre.bredde_efter,
        "fjernet_i_alt": f"{1 - m_ydre.bredde_efter / foer_bredde:.4f}",
        "fals_haeldning_px": m_fals.haeldning_px,
        "fals_sikker": "ja" if m_fals.sikker else "nej",
        "ydre_haeldning_px": m_ydre.haeldning_px,
        "ydre_baand": f"{m_ydre.baand_med_kant}/{m_ydre.baand_i_alt}",
        "ydre_sikker": "ja" if m_ydre.sikker else "nej",
    }


def _kontaktark(poster: list[dict], gruppe: str, sti: Path, skrift) -> None:
    miniaturer = []
    for post in poster:
        with Image.open(UD / gruppe / "beskaarne" / f"{post['billede']}.png") as img:
            img.load()
            h = round(img.height * MINIATURE_BREDDE / img.width)
            miniaturer.append(img.convert("RGB").resize((MINIATURE_BREDDE, h)))
    celle = max(m.height for m in miniaturer) + TEKSTHOEJDE
    raekker = (len(miniaturer) + SOEJLER - 1) // SOEJLER
    ark = Image.new("RGB",
                    (SOEJLER * (MINIATURE_BREDDE + MELLEMRUM) + MELLEMRUM,
                     raekker * (celle + MELLEMRUM) + MELLEMRUM), BAGGRUND)
    tegn = ImageDraw.Draw(ark)
    for i, (post, mini) in enumerate(zip(poster, miniaturer)):
        x = MELLEMRUM + (i % SOEJLER) * (MINIATURE_BREDDE + MELLEMRUM)
        y = MELLEMRUM + (i // SOEJLER) * (celle + MELLEMRUM)
        ark.paste(mini, (x, y))
        tegn.rectangle([(x, y), (x + mini.width - 1, y + mini.height - 1)],
                       outline=RAMME, width=1)
        usikker = post["fals_sikker"] == "nej" or post["ydre_sikker"] == "nej"
        tekst = (f"{post['billede']}  {post['recto_verso']}  "
                 f"fjernet {float(post['fjernet_i_alt']):.0%}  "
                 f"ydre {post['ydre_baand']}")
        if usikker:
            tekst += "  USIKKER"
        tegn.text((x, y + mini.height + 6), tekst, font=skrift,
                  fill=ADVARSEL if usikker else (20, 20, 20))
    ark.save(sti)


def koer_gruppe(gruppe: str, kerner: int, skrift) -> list[dict]:
    kilde = LEVERING / gruppe
    if not kilde.exists():
        print(f"  {gruppe}: mappen findes ikke, springes over")
        return []
    billeder = sorted(p.stem for p in kilde.glob("*.png"))
    print(f"\n=== {gruppe}: {len(billeder)} sider paa {kerner} kerner ===", flush=True)

    raekker: list[dict] = []
    opgaver = [(gruppe, b) for b in billeder]
    with ProcessPoolExecutor(max_workers=kerner) as pulje:
        for nr, r in enumerate(pulje.map(_beskaer_én, opgaver), 1):
            raekker.append(r)
            if nr % 20 == 0 or nr == len(opgaver):
                print(f"  [{nr}/{len(opgaver)}]", flush=True)

    (UD / gruppe).mkdir(parents=True, exist_ok=True)
    sti = UD / gruppe / "snit.csv"
    with sti.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raekker[0].keys()))
        w.writeheader()
        w.writerows(raekker)

    ark_mappe = UD / gruppe / "kontaktark"
    ark_mappe.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(raekker), PR_ARK):
        _kontaktark(raekker[start:start + PR_ARK], gruppe,
                    ark_mappe / f"ark_{start // PR_ARK + 1:02d}.png", skrift)

    usikre = [r for r in raekker
              if r["fals_sikker"] == "nej" or r["ydre_sikker"] == "nej"]
    fjernet = sorted(float(r["fjernet_i_alt"]) for r in raekker)
    print(f"  fjernet i alt: median {fjernet[len(fjernet)//2]:.1%} "
          f"({fjernet[0]:.1%}-{fjernet[-1]:.1%})")
    print(f"  usikre: {len(usikre)} af {len(raekker)}")
    for r in usikre[:10]:
        print(f"     {r['billede']}  fals={r['fals_sikker']} ydre={r['ydre_baand']}")
    return raekker


def main() -> None:
    valgte = [g for g in GRUPPER if not sys.argv[1:] or g in sys.argv[1:]]
    kerner = max(1, (os.cpu_count() or 2) - 1)
    skrift = _skrifttype()
    for gruppe in valgte:
        koer_gruppe(gruppe, kerner, skrift)
    print(f"\nfaerdig -> {UD}")


if __name__ == "__main__":
    main()
