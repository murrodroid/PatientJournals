"""Bygger opslagsregisteret: de anden-/tredjesider vi faktisk har billeder af."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from andenside.kilder import transskriptioner
from andenside.masterlist import Side, load_masterlist, lookup

STAGE01_OUTPUT = Path(__file__).resolve().parents[2] / "stages" / "01_datagrundlag" / "output"
PROEVE_OPSLAG = STAGE01_OUTPUT / "proeve_opslag"

FACIT_ROOT = transskriptioner()


def frontpage_image_name(side: Side) -> str:
    """Regner sig frem til forsidens billed-id ud fra en sides egen counter.

    Facit-filen er navngivet efter FORSIDENS billed-id, ikke den enkelte
    sides eget. "273098_001472" med patient_page_counter=1 (andenside)
    hoerer fx til forsiden "273098_001471".
    """
    if side.patient_page_counter is None:
        raise ValueError(f"{side.image_name}: ukendt patient_page_counter, kan ikke udlede forside")
    bind, sep, counter_str = side.image_name.partition("_")
    assert sep, side.image_name
    frontpage_counter = int(counter_str) - side.patient_page_counter
    return f"{bind}_{frontpage_counter:06d}"


def find_facit_file(side: Side) -> Path | None:
    """Finder "full journal"-RTF'en for netop DENNE sides patient.

    Matcher praecist paa forsidens udledte billed-id -- IKKE kun paa
    bind-praefikset, som ville give falske match naar et bind rummer flere
    patienter (hvilket det altid goer).
    """
    if not FACIT_ROOT.exists():
        return None
    frontpage = frontpage_image_name(side)
    matches = list(FACIT_ROOT.rglob(f"{frontpage}*full_journal*.rtf"))
    if len(matches) > 1:
        raise ValueError(f"{frontpage}: flere facit-filer matcher, forventede noejagtig én: {matches}")
    return matches[0] if matches else None


def build_register(image_dir: Path = PROEVE_OPSLAG) -> list[dict]:
    index = load_masterlist()
    rows = []
    for path in sorted(image_dir.glob("*.webp")):
        image_name = path.stem
        side: Side = lookup(image_name, index)
        if side.patient_page_counter not in (1, 2):
            # kun anden- og tredjesider er i scope for dette register
            continue
        with Image.open(path) as img:
            width, height = img.size
        facit = find_facit_file(side)
        rows.append(
            {
                "billede": image_name,
                "sti": str(path.relative_to(STAGE01_OUTPUT.parents[2])),
                "rolle": side.rolle,
                "recto_verso": side.recto_verso,
                "aar": side.year,
                "maaned": side.month,
                "bind": side.folder_name,
                "bredde_px": width,
                "hoejde_px": height,
                "facit_fil": facit.name if facit else "",
            }
        )
    return rows


def write_register(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "billede",
        "sti",
        "rolle",
        "recto_verso",
        "aar",
        "maaned",
        "bind",
        "bredde_px",
        "hoejde_px",
        "facit_fil",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    rows = build_register()
    write_register(rows, STAGE01_OUTPUT / "opslagsregister.csv")
    print(f"Skrevet {len(rows)} rækker til opslagsregister.csv")
    manglende_facit = [r for r in rows if not r["facit_fil"]]
    if manglende_facit:
        print(f"OBS: {len(manglende_facit)} rækker uden fundet facit-fil")
