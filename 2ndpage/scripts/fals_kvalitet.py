"""Maaler hvor TAKKET falssnittet er -- og finder de sider, hvor det er gaaet galt.

Baggrund: `snit_alle.csv`s `sikker`-kolonne taeller kun, om baandene fandt
NOGET, ikke om de er enige. Paa leveringens bredere materiale (1889-1897)
fandt alle 24 baand en kant paa sider, hvor snittet alligevel skar tvaers
gennem siden, og de stod derfor som `sikker=ja`.

Lead pegede 2026-08-30 paa fire sider som gaaet galt, fire som lidt galt og
resten som gode. Maalt paa dem skiller ét tal grupperne rent:

    dom            max spring   max afvigelse fra ret linje
    gaaet galt      204-467 px   245-314 px
    lidt galt        18-96 px     11-72 px
    god               3-6 px       3-7 px

`afvigelse` er den stoerste afstand fra et baands fundne kant til den bedste
rette linje gennem alle baandene. Den er skarpest af de to og bruges som
hovedtal; `spring` (stoerste forskel mellem to nabobaand) staar ved siden af,
fordi den fanger en anden slags fejl: en enkelt udskridende maaling mellem to
rigtige.

**Bemaerk hvad de gode tal betyder:** falsen afviger hoejst 7 px fra en ret
linje paa de sider, hvor snittet sidder rigtigt. Den frie interpolation
mellem baandene, `skraa.fals_graense` laver, er altsaa ikke noedvendig for
at foelge falsen -- men den er nok til at lade en enkelt gal maaling trække
snittet langt ind over siden.
"""

from __future__ import annotations

import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROD / "src"))

from andenside.masterlist import load_masterlist, lookup  # noqa: E402
from andenside.skraa import baandkanter  # noqa: E402

LEVERING = ROD / "stages" / "01_datagrundlag" / "output" / "levering_2026-08"
UD = ROD / "stages" / "04_billedforberedelse" / "output" / "fals_kvalitet.csv"
GRUPPER = ("oeve", "ekstra_uden_facit", "proeve_LAAST")

# Graenser aflaest af leads egne domme, ikke valgt paa faelt:
GALT = 100.0        # over dette laa alle fire, han kaldte gaaet galt
TVIVLSOM = 10.0     # under dette laa alle, han kaldte gode


def _maal_én(opgave: tuple[str, str]) -> dict | None:
    gruppe, billede = opgave
    global _INDEX
    try:
        index = _INDEX
    except NameError:
        index = _INDEX = load_masterlist()

    side = lookup(billede, index)
    with Image.open(LEVERING / gruppe / f"{billede}.png") as img:
        img.load()
        kanter = baandkanter(img, side)
        bredde = img.width

    fundne = [(y, x) for y, x in kanter if x is not None]
    if len(fundne) < 3:
        return {"gruppe": gruppe, "billede": billede, "bredde": bredde,
                "baand": f"{len(fundne)}/{len(kanter)}", "spaend": "",
                "max_spring": "", "afvigelse": "", "dom": "for faa baand"}

    ys = np.array([y for y, _ in fundne], dtype=float)
    xs = np.array([x for _, x in fundne], dtype=float)
    haeldning, skaering = np.polyfit(ys, xs, 1)
    afvigelse = float(np.abs(xs - (haeldning * ys + skaering)).max())
    spring = float(max(abs(b - a) for a, b in zip(xs, xs[1:])))

    dom = "galt" if afvigelse >= GALT else ("tvivlsom" if afvigelse >= TVIVLSOM else "god")
    return {"gruppe": gruppe, "billede": billede, "bredde": bredde,
            "baand": f"{len(fundne)}/{len(kanter)}",
            "spaend": int(xs.max() - xs.min()),
            "max_spring": round(spring), "afvigelse": round(afvigelse), "dom": dom}


def main() -> None:
    opgaver = [(g, p.stem) for g in GRUPPER
               for p in sorted((LEVERING / g).glob("*.png"))]
    kerner = max(1, (os.cpu_count() or 2) - 1)
    print(f"maaler {len(opgaver)} sider paa {kerner} kerner", flush=True)

    raekker = []
    with ProcessPoolExecutor(max_workers=kerner) as pulje:
        for nr, r in enumerate(pulje.map(_maal_én, opgaver), 1):
            raekker.append(r)
            if nr % 50 == 0 or nr == len(opgaver):
                print(f"  [{nr}/{len(opgaver)}]", flush=True)

    raekker.sort(key=lambda r: -(r["afvigelse"] or 0))
    with UD.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raekker[0].keys()))
        w.writeheader()
        w.writerows(raekker)

    from collections import Counter
    print(f"\n-> {UD}")
    print("domme:", dict(Counter(r["dom"] for r in raekker)))
    tal = sorted(r["afvigelse"] for r in raekker if isinstance(r["afvigelse"], int))
    print(f"afvigelse fra ret linje: median {tal[len(tal)//2]} px, "
          f"90-percentil {tal[int(len(tal)*0.9)]} px, max {tal[-1]} px")
    print("\nvaerste 20:")
    for r in raekker[:20]:
        print(f"   {r['billede']:16s} {r['gruppe']:18s} afvigelse {r['afvigelse']:>4} px"
              f"  spring {r['max_spring']:>4}  baand {r['baand']}")


if __name__ == "__main__":
    main()
