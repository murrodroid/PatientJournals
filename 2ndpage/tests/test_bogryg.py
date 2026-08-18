"""Tests for soegevindue-udledning -- ren regnekode, ingen billeder involveret."""

import pytest

from andenside.bogryg import soegevindue
from andenside.masterlist import Side


def _side(recto_verso_counter: int) -> Side:
    return Side(
        image_name="273098_001472",
        folder_name="273098",
        page_type="journal page",
        month="05",
        year="1896",
        patient_page_counter=recto_verso_counter,
        group_id="1",
    )


def test_andenside_soeger_i_hoejre_kant():
    vindue = soegevindue(_side(1), bredde=1000, strimmel_andel=0.3)
    assert vindue.retning == "fra_hoejre"
    assert vindue.start == 700
    assert vindue.slut == 1000


def test_tredjeside_soeger_i_venstre_kant():
    vindue = soegevindue(_side(2), bredde=1000, strimmel_andel=0.3)
    assert vindue.retning == "fra_venstre"
    assert vindue.start == 0
    assert vindue.slut == 300


def test_ukendt_recto_verso_fejler_tydeligt():
    with pytest.raises(ValueError, match="recto/verso"):
        soegevindue(_side(None), bredde=1000)
