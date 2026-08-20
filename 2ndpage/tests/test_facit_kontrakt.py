"""Kontrakttests for stage 02: passer facit med resten af projektet?

Disse tests roerer rigtige filer -- facit-filerne paa OneDrive og stage 02's
egne outputs. De springes over, hvis filerne ikke er der, saa en klon uden
OneDrive-adgang stadig kan koere testene.
"""

import csv
import json
from pathlib import Path

import pytest

from andenside.facit import FACIT_ROOT
from andenside.facit_bygger import STAGE02_OUTPUT, byg_opslag, laes_alle_blokke, opdel_patienter

REGISTER = Path(__file__).resolve().parents[1] / "stages" / "01_datagrundlag" / "output" / "opslagsregister.csv"
FACIT_JSONL = STAGE02_OUTPUT / "facit.jsonl"

kraever_onedrive = pytest.mark.skipif(not FACIT_ROOT.exists(), reason="facit-filerne ligger paa OneDrive")
kraever_output = pytest.mark.skipif(not FACIT_JSONL.exists(), reason="stage 02 er ikke bygget endnu")


@pytest.fixture(scope="module")
def facit() -> dict[str, dict]:
    with FACIT_JSONL.open(encoding="utf-8") as f:
        return {json.loads(linje)["image_name"]: json.loads(linje) for linje in f}


@kraever_output
def test_hver_side_vi_har_billede_af_har_ogsaa_facit(facit):
    """Bindeleddet mellem stage 01 og 02 er billed-id'et. Holder det ikke,
    maaler stage 05 en side mod en anden sides tekst uden at sige noget."""
    with REGISTER.open(encoding="utf-8") as f:
        billeder = [r["billede"] for r in csv.DictReader(f) if r["facit_fil"]]
    assert billeder, "opslagsregistret er tomt -- saa tester det her ingenting"
    manglende = [b for b in billeder if b not in facit]
    assert not manglende, f"billeder uden facit: {manglende}"


@kraever_output
def test_ingen_opslag_i_facit_er_uden_tekst(facit):
    """Tomme blokke hoerer i `udeladte.md`, ikke i facit -- ellers ville en
    model, der laeser en side med tekst paa, se ud til at digte det hele."""
    tomme = [navn for navn, o in facit.items() if not o["alt_fladet"].strip()]
    assert not tomme, tomme


@kraever_output
def test_de_to_facit_udgaver_er_ens_paa_sider_uden_overstregning(facit):
    """De 33 overstregninger ligger paa faa sider. Alle andre sider skal have
    to identiske udgaver -- er de forskellige, roerer overstregningsreglen
    noget, den ikke skal."""
    med_overstregning = [
        navn for navn, o in facit.items() if "crossed" in o["raa"].lower()
    ]
    assert med_overstregning, "ingen overstregninger fundet -- testen tester intet"
    afvigende = [
        navn
        for navn, o in facit.items()
        if navn not in med_overstregning and o["alt_fladet"] != o["rettet_fladet"]
    ]
    assert not afvigende, afvigende


@kraever_output
def test_hvert_opslag_hoerer_til_praecis_én_patient(facit):
    """Samme billed-id maa ikke optraede i to journalfiler. Sker det, kan en
    side ende i baade oeve- og proevemaengden."""
    with FACIT_JSONL.open(encoding="utf-8") as f:
        ider = [json.loads(linje)["image_name"] for linje in f]
    assert len(ider) == len(set(ider)), "samme billed-id optraeder flere gange"


@kraever_onedrive
def test_alle_journalfiler_bliver_laest_og_delt_op():
    """Pinner tallene fra kortlaegningen. Falder ét af dem, har en aendring i
    laeseren tabt materiale -- fx da en regex paa seks cifre tabte bind 37554.
    """
    blokke = laes_alle_blokke()
    assert len({b.kildefil for b in blokke}) == 39
    assert len(blokke) == 208
    assert sum(1 for b in blokke if b.tom) == 40
    assert all(b.forside for b in blokke), "en fil manglede forsidemaerke"


@kraever_onedrive
def test_oeve_og_proevemaengde_deler_ingen_patient():
    blokke = [b for b in laes_alle_blokke() if not b.tom]
    hold = opdel_patienter([b.forside for b in blokke if b.forside])
    oeve = {p for p, h in hold.items() if h == "oeve"}
    proeve = {p for p, h in hold.items() if h == "proeve"}
    assert oeve and proeve
    assert not (oeve & proeve)


@kraever_onedrive
def test_opdelingen_er_den_samme_hver_gang():
    """Ingen loddraekning, ingen froekerne at glemme. Iteration over maengder
    har givet ikke-reproducerbare resultater i andre af projekterne her."""
    forsider = [b.forside for b in laes_alle_blokke() if b.forside]
    assert opdel_patienter(forsider) == opdel_patienter(list(reversed(forsider)))


@kraever_onedrive
def test_facit_bygges_ens_to_gange_i_traek():
    blokke = [b for b in laes_alle_blokke() if not b.tom]
    foerste = [byg_opslag(b) for b in blokke]
    anden = [byg_opslag(b) for b in blokke]
    assert foerste == anden
