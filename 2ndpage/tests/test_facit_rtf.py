"""Tests for RTF-afkodningen i facit-laeseren.

Hvert tilfaelde her er hentet fra et virkeligt moenster i de 39
transskriptionsfiler -- ikke opfundet. Filerne er Apple TextEdit-RTF med
cp1252-escapes, og den slags ligner nem tekst lige indtil et enkelt
kontrolord aeder bogstavet efter sig.
"""

import pytest

BS = chr(92)

from andenside.facit import rtf_til_tekst

HOVED = r"{\rtf1\ansi\ansicpg1252\cocoartf2867" "\n" r"\pard\tx566\pardirnatural\partightenfactor0" "\n\n" r"\f0\fs24 \cf0 "


def _dok(krop: str) -> str:
    return HOVED + krop + "}"


def test_danske_tegn_afkodes_fra_cp1252_escapes():
    assert rtf_til_tekst(_dok(r"L\'e6gen s\'e5 Bel\'e6gn. i Halsen")) == "Lægen så Belægn. i Halsen"


def test_typografiske_anfoerselstegn_afkodes():
    assert rtf_til_tekst(_dok(r"\'93Ansigtet\'94.")) == "“Ansigtet”."


def test_backslash_nylinje_er_linjeskift_ikke_tegn():
    krop = "Hun har hostet en Del " + BS + "\nmen f" + BS + "'f8rst til Morgen"
    assert rtf_til_tekst(_dok(krop)) == "Hun har hostet en Del \nmen først til Morgen"


def test_raa_nylinje_i_kilden_er_ikke_linjeskift():
    r"""RTF bryder linjer for laesbarhed; kun \ + nylinje betyder afsnit."""
    krop = "Der er ikke iagttaget Exan\n" + r"\f1\b " + "them."
    assert rtf_til_tekst(_dok(krop)) == "Der er ikke iagttaget Exanthem."


def test_kontrolord_med_efterfoelgende_mellemrum_spiser_kun_mellemrummet():
    r"""`\b Rask` er fed-Rask, ikke ' Rask' -- mellemrummet afslutter kontrolordet."""
    assert rtf_til_tekst(_dok(r"\b Rask\b0  indtil")) == "Rask indtil"


def test_kontrolord_med_taltilbehoer_afgraenses_korrekt():
    r"""`\fs24` maa ikke laese `24` som tekst, og `\f0 0` skal give '0'."""
    assert rtf_til_tekst(_dok(r"\fs24 39,5-38,2")) == "39,5-38,2"


def test_fonttabel_og_farvetabel_udelades_helt():
    raw = (
        r"{\rtf1\ansi\ansicpg1252"
        r"{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fswiss\fcharset0 Helvetica-Bold;}"
        r"{\colortbl;\red255\green255\blue255;}"
        r"{\*\expandedcolortbl;;}"
        r"\f0\fs24 \cf0 Rask" "}"
    )
    assert rtf_til_tekst(raw) == "Rask"


def test_stjernegruppe_udelades_med_sit_fulde_indhold():
    r"""`{\*\...}` er metadata, ogsaa naar den rummer almindelig tekst."""
    assert rtf_til_tekst(_dok(r"f\'f8r{\*\bkmkstart skjult}efter")) == "førefter"


def test_klammer_i_teksten_overlever_som_almindelige_tegn():
    """Opmaerkningen `[page ...]`/`[?]` er tekst -- RTF-grupper bruger tuborg."""
    assert rtf_til_tekst(_dok(r"og Canylen [?][?] i Trachea.")) == "og Canylen [?][?] i Trachea."


def test_escapede_tuborgklammer_bliver_til_tegn():
    assert rtf_til_tekst(_dok(r"a\{b\}c" + BS + BS + "d")) == "a{b}c" + BS + "d"


def test_unicode_escape_laeses_og_erstatningstegnet_springes_over():
    r"""`\uN?` baerer tegnet i N; `?` bagefter er en fallback, ikke tekst."""
    assert rtf_til_tekst(_dok(r"pris 20\u8211 ?30")) == "pris 20–30"


def test_negativ_unicode_escape_tolkes_som_usigneret():
    """RTF skriver tegn over 32767 som negative tal."""
    assert rtf_til_tekst(_dok(r"\u-3600 ?")) == chr(65536 - 3600)


def test_afsnitskommando_par_giver_linjeskift():
    assert rtf_til_tekst(_dok(r"linje et\par linje to")) == "linje et\nlinje to"


def test_tabulatorstop_i_pard_giver_ikke_tekst():
    r"""`\pard\tx566\tx1133...` gentages pr. afsnit fra juni 1896 og frem."""
    krop = "Rask" + BS + "\n" + r"\pard\tx566\tx1133\tx1700\pardirnatural\partightenfactor0" + "\n" + "indtil"
    assert rtf_til_tekst(_dok(krop)) == "Rask\nindtil"


def test_efterstillet_mellemrum_bevares_men_afsluttende_tomme_linjer_trimmes():
    assert rtf_til_tekst(_dok("Rask " + (BS + "\n") * 3)) == "Rask "


@pytest.mark.parametrize("tegn,forventet", [(r"\'f8", "ø"), (r"\'e5", "å"), (r"\'c6", "Æ"), (r"\'a7", "§")])
def test_alle_brugte_cp1252_escapes(tegn, forventet):
    assert rtf_til_tekst(_dok(tegn)) == forventet
