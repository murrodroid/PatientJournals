"""Tests for gruppeinddelingen i scripts/beskaer_levering.py.

Gruppen `selvhentet` er de ti oevesider, vi hentede selv via kildeviseren, og
som aldrig kom med i kollegaens levering. De findes kun som webp og skal
beskaeres med samme kode som resten.

Den farlige fejl er ikke, at gruppen bliver tom -- det ses med det samme.
Den farlige er, at den bliver for STOR: globber man bare alle webp i
oevemaengden, kommer 118 sider med, og 108 af dem beskaeres saa ANDEN gang,
fra en daarligere kilde end leveringens PNG, og overskriver leveringens
resultat. Testene her holder de to mængder fra hinanden.
"""

import importlib.util
from pathlib import Path

import pytest

ROD = Path(__file__).resolve().parents[1]
LEVERING = ROD / "stages" / "01_datagrundlag" / "output" / "levering_2026-08"
OEVE_BILLEDER = ROD / "stages" / "01_datagrundlag" / "output" / "oeve_billeder"


def _indlaes_script():
    sti = ROD / "scripts" / "beskaer_levering.py"
    spec = importlib.util.spec_from_file_location("beskaer_levering", sti)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


@pytest.fixture(scope="module")
def script():
    return _indlaes_script()


def _leverede_id(script):
    return {p.stem for g in script.LEVEREDE_GRUPPER
            for p in (LEVERING / g).glob("*.png")}


@pytest.mark.skipif(not LEVERING.exists(), reason="leveringen findes ikke lokalt")
def test_selvhentet_og_levering_overlapper_ikke(script):
    """Ingen side maa beskaeres to gange fra to forskellige kilder."""
    selvhentet = set(script._billeder_i(script.SELVHENTET))
    assert selvhentet, "gruppen er tom -- kilden er formentlig flyttet"
    assert not (selvhentet & _leverede_id(script))


@pytest.mark.skipif(not LEVERING.exists(), reason="leveringen findes ikke lokalt")
def test_selvhentet_daekker_praecis_hullet(script):
    """Gruppen skal vaere HELE differencen, hverken mere eller mindre.

    Bliver den mindre, er der oevesider, der aldrig bliver beskaaret. Bliver
    den stoerre, beskaeres leverede sider fra den daarligere webp-kilde.
    """
    alle_webp = {p.stem for p in OEVE_BILLEDER.glob("*.webp")}
    forventet = alle_webp - _leverede_id(script)
    assert set(script._billeder_i(script.SELVHENTET)) == forventet


@pytest.mark.skipif(not LEVERING.exists(), reason="leveringen findes ikke lokalt")
def test_kildefil_peger_paa_det_format_gruppen_faktisk_har(script):
    """webp for de selvhentede, PNG for leveringen -- og filen skal findes.

    En forkert endelse giver en FileNotFoundError foerst midt i en lang
    parallel koersel, efter mange minutters arbejde.
    """
    for gruppe in script.GRUPPER:
        billeder = script._billeder_i(gruppe)
        if not billeder:
            continue
        sti = script._kildefil(gruppe, billeder[0])
        assert sti.exists(), f"{gruppe}: {sti} findes ikke"
        ventet = ".webp" if gruppe == script.SELVHENTET else ".png"
        assert sti.suffix == ventet


def test_selvhentet_er_ikke_en_liste_i_koden(script):
    """Gruppen skal udregnes, saa den toemmer sig selv ved en senere levering.

    Skrives de ti id'er ind som en konstant, staar de der stadig den dag,
    siderne bliver leveret som PNG -- og saa beskaeres de fra webp for evigt.
    """
    kilde = (ROD / "scripts" / "beskaer_levering.py").read_text(encoding="utf-8")
    krop = kilde.split("def _billeder_i")[1].split("\ndef ")[0]
    assert "273104" not in krop
