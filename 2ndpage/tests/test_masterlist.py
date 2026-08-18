"""Tests for recto/verso-udledning og rolle-navngivning."""

import pytest

from andenside.masterlist import Side


def _side(counter: int | None) -> Side:
    return Side(
        image_name="273098_001471",
        folder_name="273098",
        page_type="front page",
        month="05",
        year="1896",
        patient_page_counter=counter,
        group_id="1",
    )


@pytest.mark.parametrize(
    "counter,expected_recto_verso,expected_rolle",
    [
        (0, "recto", "forside"),
        (1, "verso", "andenside"),
        (2, "recto", "tredjeside"),
        (3, "verso", "side 4"),
        (4, "recto", "side 5"),
        (None, "ukendt", "ukendt"),
    ],
)
def test_recto_verso_and_rolle(counter, expected_recto_verso, expected_rolle):
    side = _side(counter)
    assert side.recto_verso == expected_recto_verso
    assert side.rolle == expected_rolle


def test_recto_verso_alternates_strictly():
    """Regression: en fejl her forveksler andenside og tredjeside i stage 04."""
    results = [_side(n).recto_verso for n in range(6)]
    assert results == ["recto", "verso", "recto", "verso", "recto", "verso"]
