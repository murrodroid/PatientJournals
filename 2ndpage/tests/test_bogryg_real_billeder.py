"""Regressionstest af snitpunkt-detektion mod rigtige, øjenbekræftede billeder.

Otte billeder er gennemset visuelt (2026-08-18) og alle otte snit landede
præcist i den fysiske rygning -- se
stages/04_billedforberedelse/output/kontaktark/. Denne test låser den
adfærd, så en fremtidig ændring i algoritmen bliver opdaget, hvis den
flytter snittet væk fra det bekræftede bånd.

OBS: en tidligere version af denne test låste en dal-baseret algoritme,
som lead fangede som forkert ved selv at se kontaktarkene igennem --
den ramte langt inde i naboopslagets tekst i 4 ud af 8 tilfælde. Ny
algoritme (2026-08-18, senere): find rygningens KANT som en top i
blækprofilen, ikke det lyseste punkt i vinduet. De fire tidligere
fejlende billeder er gennemset igen og bekræftet korrekte.

Billederne ligger uden for git (.webp er gitignored) -- testen springes
over på en maskine uden dem, i stedet for at fejle.
"""

from pathlib import Path

import pytest
from PIL import Image

from andenside.bogryg import find_snitpunkt
from andenside.masterlist import load_masterlist, lookup
from andenside.opslagsregister import PROEVE_OPSLAG

pytestmark = pytest.mark.skipif(
    not PROEVE_OPSLAG.exists() or not any(PROEVE_OPSLAG.glob("*.webp")),
    reason="prøveopslag mangler lokalt (hentes med scripts/kbharkiv_hent.py)",
)


@pytest.mark.parametrize(
    "billede,forventet_x,tolerance",
    [
        # Øjenbekræftede snit -- se kontaktark/*_snit.png for de faktiske billeder.
        ("273098_001496", 1226, 30),  # andenside/verso
        ("273098_001497", 269, 30),  # tredjeside/recto
        ("273099_001360", 1342, 30),  # andenside/verso
        ("273099_001361", 412, 30),  # tredjeside/recto
    ],
)
def test_snitpunkt_matcher_oejenbekraeftet_position(billede, forventet_x, tolerance):
    index = load_masterlist()
    side = lookup(billede, index)
    img = Image.open(PROEVE_OPSLAG / f"{billede}.webp")

    resultat = find_snitpunkt(img, side)

    assert abs(resultat.x - forventet_x) <= tolerance, (
        f"{billede}: snit x={resultat.x}, forventet {forventet_x}±{tolerance} "
        "-- se det annoterede kontaktark for at afgøre om ALGORITMEN er "
        "flyttet sig, eller om den nye position rent faktisk er den rigtige."
    )
