"""Vaernet om den laaste proevemaengde.

De 13 proeve-patienter er laast til den ENDELIGE bedoemmelse. Roerer vi dem
undervejs, er tallet til sidst ikke laengere en uafhaengig maaling -- vi har
set facit for de sider, vi bedoemmer paa. Vaernet er der, fordi den fejl ikke
kan opdages bagefter: et modelsvar baerer ikke praeg af, at vi kiggede.

Testene her holder to ting oppe:

  1. En proeve-side skal STOPPE koden, ikke bare give en advarsel.
  2. Der skal findes en udtrykkelig vej udenom -- ellers kan den endelige
     bedoemmelse selv ikke koere.
"""

import csv
import json
from pathlib import Path

import pytest

ROD = Path(__file__).resolve().parents[1]
FACIT = ROD / "stages" / "02_facit" / "output" / "facit.jsonl"
OPDELING = ROD / "stages" / "02_facit" / "output" / "opdeling.csv"

from andenside.vaern import ProeveMaengdeFejl, maengde_for, sikr_oevemaengde


def _billeder_pr_maengde() -> dict[str, list[str]]:
    """Facit og opdeling laest raat -- uafhaengigt af vaernets egen kode.

    Testen maa ikke bruge modulet, den tester, til at finde sine egne
    forventninger; saa ville en fejl i opslaget bekraefte sig selv.
    """
    maengde_for_forside = {
        r["forside"]: r["maengde"]
        for r in csv.DictReader(OPDELING.open(encoding="utf-8"))
    }
    ud: dict[str, list[str]] = {"oeve": [], "proeve": []}
    for linje in FACIT.read_text(encoding="utf-8").splitlines():
        post = json.loads(linje)
        ud[maengde_for_forside[post["forside"]]].append(post["image_name"])
    return ud


BILLEDER = _billeder_pr_maengde()


def test_opdelingen_har_begge_maengder():
    # Sikrer at de oevrige tests herunder faktisk proever noget.
    assert len(BILLEDER["oeve"]) > 0
    assert len(BILLEDER["proeve"]) > 0


def test_maengde_for_kender_begge_sider():
    assert maengde_for(BILLEDER["oeve"][0]) == "oeve"
    assert maengde_for(BILLEDER["proeve"][0]) == "proeve"


def test_ukendt_billede_er_en_fejl_ikke_et_gaet():
    # Et billed-id, vi ikke kan placere, maa ikke stiltiende gaa for "oeve".
    # Det er praecis den vej, en proeve-side kunne slippe igennem.
    with pytest.raises(KeyError):
        maengde_for("000000_999999")


def test_oevemaengden_slipper_igennem():
    billeder = BILLEDER["oeve"][:5]
    assert sikr_oevemaengde(billeder) == billeder


def test_en_enkelt_proeveside_stopper_hele_saettet():
    # Blandet saet: 5 lovlige og 1 ulovlig. Hele kaldet skal fejle -- ikke
    # frasortere den ene i stilhed og koere videre paa resten.
    billeder = BILLEDER["oeve"][:5] + [BILLEDER["proeve"][0]]
    with pytest.raises(ProeveMaengdeFejl) as fejl:
        sikr_oevemaengde(billeder)
    # Beskeden skal navngive synderen, ellers kan den ikke handles paa.
    assert BILLEDER["proeve"][0] in str(fejl.value)


def test_fejlen_naevner_alle_proevesider_ikke_kun_den_foerste():
    billeder = BILLEDER["proeve"][:3]
    with pytest.raises(ProeveMaengdeFejl) as fejl:
        sikr_oevemaengde(billeder)
    for billede in billeder:
        assert billede in str(fejl.value)


def test_udtrykkelig_tilladelse_aabner_vejen():
    # Den endelige bedoemmelse skal kunne koere. Vejen udenom findes, men
    # den skal skrives med vilje.
    billeder = BILLEDER["proeve"][:3]
    assert sikr_oevemaengde(billeder, tillad_proeve=True) == billeder


def test_tilladelsen_kan_ikke_gives_ved_et_uheld():
    # `tillad_proeve` er keyword-only. Ellers kunne en positionsparameter
    # rutsje ind som en tilfaeldig sandhedsvaerdi og aabne vaernet.
    with pytest.raises(TypeError):
        sikr_oevemaengde(BILLEDER["proeve"][:1], True)  # type: ignore[misc]
