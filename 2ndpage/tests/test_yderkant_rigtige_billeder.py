"""Regressionsprøve for yderkant-snittet paa de RIGTIGE billeder.

Hvorfor ikke syntetiske data her: den fejl, proeven skal vogte imod, er en
svag, ret skygge INDE paa papiret, som baandvist ligner en kant. Jeg kunne
ikke konstruere et syntetisk billede, hvor den skelnen opfoerer sig som paa
de virkelige sider -- enhver skygge, jeg tegnede, fik ogsaa en rigtig soem.
En proeve, der bestaar af de forkerte grunde, vogter ingenting, saa den er
her bundet til virkelige sider og til leads egne domme i stedet.

Lead gennemgik disse seks sider i fuld oploesning 2026-08-29. Paa de to
foerste laa snittet forkert; paa de fire sidste laa det rigtigt. Den
egenskab, der skiller dem, er SOEM-DYBDEN paa den valgte linje: de forkerte
laa i 3,0 og 5,0, de rigtige i 12-25.

Proeven springes over, hvis de beskaarne billeder ikke er hentet.
"""

import pytest

pytest.importorskip("numpy")

import numpy as np
from PIL import Image

from andenside.masterlist import load_masterlist, lookup
from andenside.opslagsregister import STAGE01_OUTPUT  # noqa: F401  (sti-kontrakt)
from andenside.yderkant import SOEM_GULV, soem_dybde, ydre_graense

from pathlib import Path

BESKAARNE = Path(__file__).resolve().parents[1] / "stages" / "04_billedforberedelse" / "output" / "beskaarne"

# Leads domme 2026-08-29. Vaerdien er den soem-dybde, snittet laa i FOER
# soem-kravet blev indfoert -- den er kun med som forklaring.
LEADS_DOMME = {
    "273105_001569": ("gik galt", 3.0),
    "273103_001437": ("gik lidt galt", 5.0),
    "273108_001555": ("god", 15.0),
    "273111_001376": ("god", 18.0),
    "37554_001492": ("god", 25.0),
    "37554_001494": ("god", 12.0),
}

MINDSTE_SOEM = 10.0   # alle seks skal ligge klart over gulvet, ikke lige paa det


def _graense_og_soem(navn: str) -> tuple[list[int], float]:
    side = lookup(navn, load_masterlist())
    img = Image.open(BESKAARNE / f"{navn}.webp")
    graense = ydre_graense(img, side)
    if not graense:
        return [], -1.0
    graa = np.asarray(img.convert("L"), dtype=float)
    # skaering og haeldning laeses tilbage af den faerdige graense
    skaering = float(graense[0])
    haeldning = float(graense[-1] - graense[0])
    return graense, soem_dybde(graa, skaering, haeldning)


@pytest.mark.skipif(not BESKAARNE.exists(), reason="beskaarne billeder ikke hentet")
@pytest.mark.parametrize("navn", sorted(LEADS_DOMME))
def test_snittet_ligger_paa_en_rigtig_soem(navn):
    """Alle seks sider skal nu ende paa en linje med en tydelig soem.

    Det er den egenskab, der skiller leads to forkerte snit fra hans fire
    rigtige -- og den skal gaelde for dem alle sammen efter rettelsen, ogsaa
    de to der foer laa i 3,0 og 5,0.
    """
    dom, foer = LEADS_DOMME[navn]
    graense, soem = _graense_og_soem(navn)
    assert graense, f"{navn}: ingen graense fundet (lead kaldte snittet '{dom}')"
    assert soem >= MINDSTE_SOEM, (
        f"{navn}: soem-dybde {soem:.1f} -- laa foer i {foer} og blev doemt '{dom}'"
    )


@pytest.mark.skipif(not BESKAARNE.exists(), reason="beskaarne billeder ikke hentet")
def test_soem_gulvet_ligger_under_alle_godkendte_snit():
    """Gulvet maa ikke ligge oven i de rigtige snit.

    Marginen er projektets svageste led: de forkerte linjer maalte 3,0 og
    5,0, og 273107_001866s RIGTIGE kant maaler kun 5-7. Gaar gulvet op over
    det, tabes den side. Proeven fastholder, at gulvet bliver liggende under
    alle de snit, lead har godkendt.
    """
    for navn in LEADS_DOMME:
        _, soem = _graense_og_soem(navn)
        assert soem > SOEM_GULV, f"{navn}: soem {soem:.1f} ligger paa eller under gulvet {SOEM_GULV}"
