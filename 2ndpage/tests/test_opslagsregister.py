"""Tests for facit-matching -- skal skelne mellem flere patienter i samme bind."""

from andenside.masterlist import Side
from andenside.opslagsregister import find_facit_file, frontpage_image_name


def _side(image_name: str, counter: int) -> Side:
    bind = image_name.split("_")[0]
    return Side(
        image_name=image_name,
        folder_name=bind,
        page_type="journal page",
        month="05",
        year="1896",
        patient_page_counter=counter,
        group_id="1",
    )


def test_frontpage_image_name_andenside():
    side = _side("273098_001472", counter=1)
    assert frontpage_image_name(side) == "273098_001471"


def test_frontpage_image_name_tredjeside():
    side = _side("273098_001497", counter=2)
    assert frontpage_image_name(side) == "273098_001495"


def test_find_facit_file_matches_correct_patient_not_first_in_bind(tmp_path, monkeypatch):
    """Regression: et bind har flere patienter/facit-filer. Skal ramme den
    RIGTIGE, ikke bare den foerste alfabetisk i mappen."""
    facit_root = tmp_path / "Manual transcriptions" / "Deaths 1896-97"
    facit_root.mkdir(parents=True)
    (facit_root / "273098_001471_full_journal.rtf").write_text("patient A")
    (facit_root / "273098_001495_full_journal.rtf").write_text("patient B")
    (facit_root / "273098_001507_full_journal.rtf").write_text("patient C")

    monkeypatch.setattr("andenside.opslagsregister.FACIT_ROOT", tmp_path)

    side_b = _side("273098_001497", counter=2)  # tredjeside for patient B (forside 001495)
    found = find_facit_file(side_b)

    assert found is not None
    assert found.name == "273098_001495_full_journal.rtf"
    assert found.read_text() == "patient B"
