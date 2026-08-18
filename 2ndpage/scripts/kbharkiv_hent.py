"""Midlertidigt hente-script mod kbharkiv.dk's kildeviser-API.

Bruges til at hente prøvebilleder, mens den formelle billedanmodning til
kollegaen (billedanmodning/) endnu ikke er besvaret. Ikke en del af
ICM-stagestrukturen -- et engangsværktøj, der kan blive erstattet af
kollegaens Google Cloud-spand eller en rigtig stage senere.

Fundet API (ingen dokumentation, reverse-engineered 2026-08-18 fra
kildeviser.kbharkiv.dk's SvelteKit-bundle):

    GET https://api.kbharkiv.dk/pages?unit_id=<bind-id>
        -> liste af {id, unit_id, page_number, image_url, ...}
        page_number er kildeviserens EGET scan-sekvensnummer, IKKE
        masterlistens page_counter. page_number=0 er typisk et omslagsfoto.

    GET https://api.kbharkiv.dk/file/<id>
        -> selve billedet (WebP)

Empirisk fundet og verificeret mod TO forskellige facit-forsider
(273098_001471 = Christiane Marie Andersen, 273099_001359 = Esther
Engstrøm): masterlistens billed-id "<bind>_<counter>" svarer til
kildeviserens page_number = counter - 1. Dette er en antagelse, ikke en
dokumenteret kontrakt -- forskydningen boer efterproeves for hvert nyt bind,
foer den bruges ukritisk i stor skala.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API_BASE = "https://api.kbharkiv.dk"
USER_AGENT = "Mozilla/5.0 (academic research, 2ndpage transskriptionsprojekt)"
PAGE_NUMBER_OFFSET = 1  # masterliste-counter - kbharkiv page_number
REQUEST_DELAY_SECONDS = 0.3
MAX_RETRIES = 3


class KbharkivError(RuntimeError):
    pass


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(REQUEST_DELAY_SECONDS * attempt)
    raise KbharkivError(f"kunne ikke hente {url}: {last_error}")


def fetch_unit_pages(unit_id: str) -> dict[int, dict]:
    """Henter og cacher siderne for ét bind, nøglet på kbharkiv page_number."""
    raw = _get(f"{API_BASE}/pages?unit_id={unit_id}")
    pages = json.loads(raw)
    return {int(page["page_number"]): page for page in pages}


def parse_image_name(image_name: str) -> tuple[str, int]:
    """"273098_001471" -> ("273098", 1471)."""
    unit_id, _, counter_str = image_name.partition("_")
    if not unit_id or not counter_str:
        raise ValueError(f"uventet billed-id-format: {image_name!r}")
    return unit_id, int(counter_str)


def resolve_page(image_name: str, pages_cache: dict[str, dict[int, dict]]) -> dict:
    unit_id, counter = parse_image_name(image_name)
    if unit_id not in pages_cache:
        pages_cache[unit_id] = fetch_unit_pages(unit_id)
    page_number = counter - PAGE_NUMBER_OFFSET
    pages = pages_cache[unit_id]
    if page_number not in pages:
        raise KbharkivError(
            f"{image_name}: page_number {page_number} findes ikke i bind {unit_id} "
            f"(bindet har {len(pages)} sider)"
        )
    return pages[page_number]


def download_image_names(
    image_names: list[str], dest_dir: Path, *, dry_run: bool = False
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    pages_cache: dict[str, dict[int, dict]] = {}
    for i, image_name in enumerate(image_names, start=1):
        dest = dest_dir / f"{image_name}.webp"
        if dest.exists():
            print(f"[{i}/{len(image_names)}] {image_name}: findes allerede, springer over")
            continue
        try:
            page = resolve_page(image_name, pages_cache)
        except (ValueError, KbharkivError) as exc:
            print(f"[{i}/{len(image_names)}] {image_name}: FEJL -- {exc}")
            continue
        if dry_run:
            print(f"[{i}/{len(image_names)}] {image_name} -> {page['image_url']} (dry-run, henter ikke)")
            continue
        try:
            data = _get(page["image_url"])
        except KbharkivError as exc:
            print(f"[{i}/{len(image_names)}] {image_name}: FEJL ved download -- {exc}")
            continue
        dest.write_bytes(data)
        print(f"[{i}/{len(image_names)}] {image_name}: hentet ({len(data) / 1024:.0f} KB)")
        time.sleep(REQUEST_DELAY_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image_names_file",
        type=Path,
        help="tekstfil med ét billed-id pr. linje, fx 273098_001472",
    )
    parser.add_argument("--dest", type=Path, required=True, help="mappe billederne gemmes i")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="maks antal billeder pr. kørsel (sikkerhedsgrænse, standard 20)",
    )
    parser.add_argument("--dry-run", action="store_true", help="vis kun hvad der ville blive hentet")
    args = parser.parse_args()

    image_names = [
        line.strip()
        for line in args.image_names_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(image_names) > args.limit:
        print(
            f"FEJL: {len(image_names)} billeder bedt om, men grænsen er {args.limit}. "
            "Brug --limit for at hæve den bevidst.",
            file=sys.stderr,
        )
        sys.exit(1)

    download_image_names(image_names, args.dest, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
