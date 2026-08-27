"""Tests for bogholderiet omkring en modelkoersel.

Stage 05's Test Contract siger det praecist: modelsvar kan ikke testes fast,
men bogholderiet kan. To ting skal holde:

  1. En koersel gemmes ALTID med den opsaetning, den blev koert med.
  2. To koersler med samme opsaetning kan skelnes fra hinanden paa dato.

Hvorfor det er vaerd at teste: et raat modelsvar uden opsaetning er
vaerdiloest. Vi kan ikke vide, om det daarlige tal skyldtes prompten,
billedvarianten eller modellen -- og vi kan ikke koere det om. Det er den
slags tab, man foerst opdager, naar det er for sent.
"""

import json
import time
from pathlib import Path

import pytest

from andenside.koersel import Opsaetning, gem_koersel, laes_koersel, find_koersler

PROMPT = "Transskriber siden linje for linje.\nIgnorer modstaaende side."


def _opsaetning(**ret) -> Opsaetning:
    grund = dict(
        model="gemini-3.1-pro",
        promptversion="textpage-uaendret",
        prompt=PROMPT,
        variant="beskaaret",
        temperatur=0.0,
    )
    return Opsaetning(**{**grund, **ret})


SVAR = {
    "273098_001503": "Rask indtil for 8 Dage siden\nda hun fik Brystkatarrh.",
    "273099_001445": "Barnet er meget medtaget.",
}


def test_koersel_kan_laeses_tilbage_praecis_som_den_blev_gemt(tmp_path):
    mappe = gem_koersel(tmp_path, _opsaetning(), SVAR)
    opsaetning, svar = laes_koersel(mappe)
    assert opsaetning == _opsaetning()
    assert svar == SVAR


def test_prompten_gemmes_ordret_ikke_kun_som_fingeraftryk(tmp_path):
    """Et hash alene kan ikke koere en koersel om -- teksten skal vaere der."""
    mappe = gem_koersel(tmp_path, _opsaetning(), SVAR)
    opsaetning, _ = laes_koersel(mappe)
    assert opsaetning.prompt == PROMPT


def test_prompt_fingeraftrykket_skifter_naar_prompten_skifter(tmp_path):
    # Samme promptversion-navn, ANDEN tekst: det skal kunne ses.
    en = gem_koersel(tmp_path, _opsaetning(), SVAR)
    to = gem_koersel(tmp_path, _opsaetning(prompt=PROMPT + " Skriv intet andet."), SVAR)
    assert laes_koersel(en)[0].prompt_aftryk != laes_koersel(to)[0].prompt_aftryk


def test_to_koersler_med_SAMME_opsaetning_kan_skelnes(tmp_path):
    """Kontraktens andet krav, ordret."""
    en = gem_koersel(tmp_path, _opsaetning(), SVAR)
    time.sleep(1.1)  # datoen har sekund-oploesning
    to = gem_koersel(tmp_path, _opsaetning(), SVAR)

    assert en != to, "to koersler maa ikke skrive oven i hinanden"
    assert laes_koersel(en)[0].prompt_aftryk == laes_koersel(to)[0].prompt_aftryk
    # ... og de skal kunne skelnes paa netop datoen.
    assert _dato(en) != _dato(to)
    assert len(find_koersler(tmp_path)) == 2


def _dato(mappe: Path) -> str:
    return json.loads((mappe / "opsaetning.json").read_text(encoding="utf-8"))["dato"]


def test_to_koersler_i_samme_sekund_skriver_ikke_oven_i_hinanden(tmp_path):
    """Sekund-oploesning er ikke nok i sig selv -- gem maa aldrig tabe data."""
    en = gem_koersel(tmp_path, _opsaetning(), SVAR)
    to = gem_koersel(tmp_path, _opsaetning(), {"273098_001503": "et andet svar"})
    assert en != to
    assert laes_koersel(en)[1] == SVAR
    assert laes_koersel(to)[1] == {"273098_001503": "et andet svar"}


def test_svaret_gemmes_raat_uden_oprydning(tmp_path):
    """Renser vi svaret foer det gemmes, kan vi aldrig se, hvad modellen skrev.

    Stage 03 advarer om, at en indledning som "Her er transskriptionen:"
    taelles som digtning. Beslutningen om at rense hoerer til MAALINGEN --
    bogholderiet skal bevare originalen.
    """
    beskidt = {"273098_001503": "```\nHer er transskriptionen:\nRask indtil...\n```\n"}
    mappe = gem_koersel(tmp_path, _opsaetning(), beskidt)
    assert laes_koersel(mappe)[1] == beskidt


def test_en_koersel_uden_opsaetning_kan_ikke_gemmes(tmp_path):
    with pytest.raises(TypeError):
        gem_koersel(tmp_path, SVAR)  # type: ignore[arg-type]


def test_ufuldstaendig_opsaetning_afvises(tmp_path):
    # Et tomt modelnavn goer koerslen umulig at genfinde bagefter.
    with pytest.raises(ValueError, match="model"):
        gem_koersel(tmp_path, _opsaetning(model=""), SVAR)


def test_bogholderiet_rummer_aldrig_en_api_noegle(tmp_path):
    """Noeglen maa ikke kunne havne i et bogholderi, vi deler eller committer."""
    mappe = gem_koersel(tmp_path, _opsaetning(), SVAR)
    for sti in mappe.rglob("*"):
        if sti.is_file():
            tekst = sti.read_text(encoding="utf-8").lower()
            for forbudt in ("api_key", "apikey", "api-noegle", "secret", "token"):
                assert forbudt not in tekst, f"{sti.name} rummer {forbudt!r}"


def test_find_koersler_giver_nyeste_foerst(tmp_path):
    en = gem_koersel(tmp_path, _opsaetning(), SVAR)
    time.sleep(1.1)
    to = gem_koersel(tmp_path, _opsaetning(variant="helt_opslag"), SVAR)
    assert find_koersler(tmp_path)[0] == to
    assert find_koersler(tmp_path)[1] == en


def test_varianten_staar_i_bogholderiet(tmp_path):
    """Hele stage 05 handler om at sammenligne de to billedvarianter.

    Kan en koersel ikke fortaelle, hvilken variant den brugte, er
    sammenligningen ikke til at stole paa.
    """
    mappe = gem_koersel(tmp_path, _opsaetning(variant="helt_opslag"), SVAR)
    assert laes_koersel(mappe)[0].variant == "helt_opslag"


def test_ukendt_variant_afvises(tmp_path):
    with pytest.raises(ValueError, match="variant"):
        gem_koersel(tmp_path, _opsaetning(variant="halvt_opslag"), SVAR)
