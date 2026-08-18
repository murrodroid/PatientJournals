"""Kører snitpunkt-detektion paa hele opslagsregisteret og tegner kontaktark."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw

from andenside.bogryg import find_snitpunkt
from andenside.masterlist import load_masterlist, lookup
from andenside.opslagsregister import PROEVE_OPSLAG, STAGE01_OUTPUT

STAGE04_OUTPUT = Path(__file__).resolve().parents[2] / "stages" / "04_billedforberedelse" / "output"


def run(register_csv: Path) -> None:
    index = load_masterlist()
    kontaktark_dir = STAGE04_OUTPUT / "kontaktark"
    kontaktark_dir.mkdir(parents=True, exist_ok=True)

    rows_out = []
    with register_csv.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            billede = row["billede"]
            side = lookup(billede, index)
            img_path = PROEVE_OPSLAG / f"{billede}.webp"
            img = Image.open(img_path)
            resultat = find_snitpunkt(img, side)

            annotated = img.convert("RGB")
            draw = ImageDraw.Draw(annotated)
            draw.line([(resultat.x, 0), (resultat.x, annotated.height)], fill=(255, 0, 0), width=4)
            draw.rectangle(
                [(resultat.vindue.start, 0), (resultat.vindue.slut, 20)],
                outline=(0, 128, 255),
                width=3,
            )
            out_path = kontaktark_dir / f"{billede}_snit.png"
            annotated.save(out_path)

            rows_out.append(
                {
                    "billede": billede,
                    "rolle": side.rolle,
                    "recto_verso": side.recto_verso,
                    "bredde_px": img.width,
                    "snit_x": resultat.x,
                    "snit_andel_af_bredde": round(resultat.x / img.width, 3),
                    "styrke": round(resultat.styrke, 4),
                    "soegevindue": f"{resultat.vindue.start}-{resultat.vindue.slut} ({resultat.vindue.retning})",
                }
            )
            print(f"{billede}: snit x={resultat.x} ({resultat.x / img.width:.1%} af bredden)")

    out_csv = STAGE04_OUTPUT / "snit.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nSkrevet {len(rows_out)} rækker til {out_csv}")
    print(f"Kontaktark gemt i {kontaktark_dir}")


if __name__ == "__main__":
    run(STAGE01_OUTPUT / "opslagsregister.csv")
