"""Tests for pilotens sideudvalg og kildeopslag i scripts/koer_pilot.py.

Udsnittet afgoer, hvilke sider prompten formes paa. Vaelger det skaevt, ser
prompten bedre ud end den er -- og det opdages foerst langt senere, naar der
maales paa hele oevemaengden.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROD = Path(__file__).resolve().parents[1]
BESKAARET = (ROD / "stages" / "04_billedforberedelse" / "output"
             / "levering_beskaaret")

# Én side fra den laaste proevemaengde. Kun billed-id'et staar her; facit for
# siden roeres ikke, og maa ikke roeres foer den endelige bedoemmelse.
EN_PROEVESIDE = "273098_001472"


def _indlaes_script():
    sti = ROD / "scripts" / "koer_pilot.py"
    spec = importlib.util.spec_from_file_location("koer_pilot", sti)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


@pytest.fixture(scope="module")
def script():
    return _indlaes_script()


def _konstrueret() -> list[dict]:
    """Femten sider med SKAEV svaerhedsfordeling, som pilotsiderne har den.

    Tretten klumper sig i 1-6, og saa springer de to sidste til 9 og 10.
    Det er praecis den form, der afgoer, om et udsnit rammer den haarde hale.
    """
    svaerhed = [1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 9, 10]
    return [{"billede": f"bind{i:02d}_000{i:03d}", "svaere_linjer": str(v)}
            for i, v in enumerate(svaerhed)]


def test_baade_letteste_og_haardeste_er_med(script):
    """Uden den letteste kan en fejl ikke skilles fra svaer skrift; uden den
    haardeste proeves prompten aldrig, hvor den betyder noget."""
    valgt = script._udsnit(_konstrueret(), 8)
    grader = [int(r["svaere_linjer"]) for r in valgt]
    assert min(grader) == 1
    assert max(grader) == 10


def test_den_haarde_hale_er_ikke_sprunget_over(script):
    """Den egentlige fælde.

    Spredes der lige hen over sidernes RAEKKEFOELGE i stedet for over
    svaerhedsgradens vaerdier, rammer udsnittet klumpen 1-6 igen og igen og
    faar kun ét af de to haarde ekstremer med. Begge skal med.
    """
    valgt = script._udsnit(_konstrueret(), 8)
    haarde = [r for r in valgt if int(r["svaere_linjer"]) >= 9]
    assert len(haarde) == 2, [r["svaere_linjer"] for r in valgt]


def test_antal_overholdes_og_ingen_side_gaar_igen(script):
    for antal in (1, 2, 5, 8, 14):
        valgt = script._udsnit(_konstrueret(), antal)
        navne = [r["billede"] for r in valgt]
        assert len(valgt) == antal
        assert len(set(navne)) == antal
        assert navne == sorted(navne), "raekkefoelgen skal vaere billed-id"


def test_udsnittet_er_det_samme_i_TO_processer(script):
    """To koersler med samme tal skal give samme sider -- ogsaa i hver sin proces.

    At kalde funktionen to gange i SAMME proces beviser ingenting: den
    sorterer selv sit input, saa selv en maengde-baseret udgave ville se
    stabil ud. Fælden er hash-randomiseringen, som foerst slaar igennem paa
    tvaers af processer. Den har kostet reproducerbarhed i et tidligere
    projekt, saa den proeves her, som den faktisk optraeder.
    """
    import json
    import os
    import subprocess

    kode = (
        "import importlib.util, json, sys;"
        "spec = importlib.util.spec_from_file_location('kp', sys.argv[1]);"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
        "raekker = [{'billede': b, 'svaere_linjer': str(v)} "
        "for b, v in json.loads(sys.argv[2])];"
        "print(json.dumps([r['billede'] for r in m._udsnit(raekker, 8)]))"
    )
    data = json.dumps([(r["billede"], int(r["svaere_linjer"]))
                       for r in _konstrueret()])
    sti = str(ROD / "scripts" / "koer_pilot.py")

    svar = []
    for froe in ("0", "12345"):
        miljoe = dict(os.environ, PYTHONHASHSEED=froe)
        ud = subprocess.run([sys.executable, "-c", kode, sti, data],
                            capture_output=True, text=True, check=True,
                            env=miljoe)
        svar.append(json.loads(ud.stdout))
    assert svar[0] == svar[1]
    assert len(svar[0]) == 8


def test_for_stort_antal_giver_alle_uden_at_sprænge(script):
    raekker = _konstrueret()
    assert script._udsnit(raekker, 99) == raekker
    assert script._udsnit(raekker, None) == raekker


def test_nul_eller_negativt_antal_afvises(script):
    with pytest.raises(ValueError):
        script._udsnit(_konstrueret(), 0)
    with pytest.raises(ValueError):
        script._udsnit(_konstrueret(), -3)


@pytest.mark.skipif(not BESKAARET.exists(), reason="snittene findes ikke lokalt")
def test_selvhentet_side_kan_findes(script):
    """273104_001639 ligger i gruppen `selvhentet`, ikke i `oeve`.

    Tabes den gruppe af BESKAARET_GRUPPER, forsvinder ti sider lydloest ud af
    piloten -- og de er blandt de haardeste, vi har.
    """
    sti = script._find_billede("beskaaret", "273104_001639")
    assert sti is not None and sti.exists()
    assert sti.parent.parent.name == "selvhentet"


@pytest.mark.skipif(not BESKAARET.exists(), reason="snittene findes ikke lokalt")
def test_proevemaengden_kan_ikke_naas_gennem_kildeopslaget(script):
    """Andet lag under vaernet: selv med et proeve-id i haanden maa opslaget
    ikke finde filen. Proevesiderne er beskaaret og ligger paa disken."""
    assert (BESKAARET / "proeve_LAAST" / "beskaarne"
            / f"{EN_PROEVESIDE}.png").exists(), "testens forudsaetning holder ikke"
    assert script._find_billede("beskaaret", EN_PROEVESIDE) is None
