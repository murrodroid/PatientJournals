"""Tests for skemavarianterne og udfoldningen af et svar til tekst.

Udfoldningen er det farligste sted i forsoeget. Maaleapparatet vil have én
sammenhaengende tekst, og hvordan et skemasvar bliver til den tekst er et
VALG. Vaelges det forskelligt for to varianter, vinder den ene paa
udfoldningen i stedet for paa skemaet -- og forsoeget maaler saa noget andet,
end det tror.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from andenside.skemaer import (
    DELE_I_LAESEORDEN,
    SKEMAER,
    saml_linje,
    tekst_af_svar,
)

ROD = Path(__file__).resolve().parents[1]


def test_feltbeskrivelserne_naar_faktisk_ud_i_skemaet():
    """Hele V1's eksistensberettigelse.

    Gemini sender `Field(description=...)` med som en del af skemaet, saa en
    beskrivelse er prompttekst. Falder den ud af den genererede JSON, er
    `beskrevet` bare `bar` med et andet navn, og forsoeget maaler nul --
    uden at nogen opdager det.
    """
    bar = SKEMAER["bar"].model_json_schema()
    beskrevet = SKEMAER["beskrevet"].model_json_schema()

    def beskrivelser(skema):
        linje = next(iter(skema["$defs"].values()))
        return {f: d.get("description") for f, d in linje["properties"].items()}

    assert all(v is None for v in beskrivelser(bar).values())
    assert all(v for v in beskrivelser(beskrevet).values())


def test_beskrevet_er_ordret_kollegaens_tekst():
    """Beskrivelserne er maalestokken, ikke vores bidrag.

    Omskrives de, maaler vi vores egen formulering i stedet for hans, og
    tallet kan ikke laengere sammenlignes med det, hans app producerer.
    """
    hans = ROD.parent / "src" / "patientjournals" / "config" / "schemas.py"
    if not hans.exists():
        pytest.skip("kollegaens kode ligger ikke ved siden af")
    kilde = hans.read_text(encoding="utf-8")
    linje = next(iter(SKEMAER["beskrevet"].model_json_schema()["$defs"].values()))
    for felt, d in linje["properties"].items():
        beskrivelse = d["description"]
        # Hans kilde bryder linjerne anderledes; sammenlign paa ordene.
        assert " ".join(beskrivelse.split()) in " ".join(kilde.split()), felt


def test_metadata_holdes_ude_som_standard_men_kan_foldes_ind():
    """Kollegaens app laegger margendatoen i `metadata`; facit har den inline.

    Begge udfoldninger skal vaere mulige paa SAMME svar, uden et nyt kald --
    ellers kan `linjefelter`, der samler sine dele, ikke sammenlignes retvist
    med `beskrevet`, der taber datoen.
    """
    svar = {"page_lines": [{"text": "Har doeset meget", "metadata": "19/12"}]}
    assert tekst_af_svar(svar) == "Har doeset meget"
    assert tekst_af_svar(svar, med_metadata=True) == "19/12 Har doeset meget"


def test_linjens_dele_samles_i_sidens_egen_raekkefoelge():
    """Facit skriver linjen som `19/12 39.5/39.4 Har doeset`.

    Samles delene i en anden orden, taeller hvert ombyttet felt som fejl paa
    en linje, modellen laeste rigtigt.
    """
    post = {"text": "Har doeset", "maalinger": "39.5/39.4", "dato": "19/12"}
    assert saml_linje(post, med_metadata=False) == "19/12 39.5/39.4 Har doeset"


def test_tomme_og_manglende_felter_giver_ikke_dobbelte_mellemrum():
    """Et `None`-felt maa ikke efterlade et mellemrum -- det er en tegnfejl."""
    assert saml_linje({"dato": None, "maalinger": "", "text": "Kun tekst"},
                      med_metadata=False) == "Kun tekst"
    assert saml_linje({"dato": "  ", "text": "Kun tekst"},
                      med_metadata=False) == "Kun tekst"


def test_manglende_tekstfelt_er_en_fejl_ikke_en_tom_linje():
    """Et svar uden `text` betyder, at skemaet ikke blev fulgt.

    Gaar det stille igennem som en tom linje, ser siden ud til at vaere
    daarligt laest i stedet for at vaere en fejlet koersel.
    """
    with pytest.raises(ValueError, match="text"):
        saml_linje({"dato": "19/12"}, med_metadata=False)


def test_et_svar_uden_linjer_er_en_fejl():
    with pytest.raises(ValueError, match="ingen linjer"):
        tekst_af_svar({"page_lines": []})


def test_ren_tekst_gaar_uaendret_igennem():
    assert tekst_af_svar({"ren_tekst": "linje 1\nlinje 2"}) == "linje 1\nlinje 2"


def test_alle_varianter_kan_udfoldes_uden_at_kende_deres_navn():
    """Udfoldningen maa ikke kende varianterne ved navn.

    Goer den det, skal den rettes hver gang en variant kommer til -- og den,
    der glemmer det, faar en tom tekst i stedet for en fejl.
    """
    for navn, model in SKEMAER.items():
        if model is None:
            continue
        felter = next(iter(model.model_json_schema()["$defs"].values()))["properties"]
        post = {f: "x" for f in felter}
        tekst = saml_linje(post, med_metadata=True)
        # Hvert felt, der hoerer til linjens tekst, skal vaere med.
        ventet = sum(1 for f in felter if f in DELE_I_LAESEORDEN)
        assert len(tekst.split()) == ventet, navn


def _model_modul():
    return importlib.import_module("andenside.model")


def test_markdownhegn_om_hele_svaret_fjernes():
    """Prompten beder om ren tekst uden hegn, men modeller foelger ikke altid.

    Bliver hegnet staaende, taeller ```-linjerne som tekst, modellen digtede,
    og `ren_tekst`-varianten straffes for noget, der ikke er en laesefejl.
    """
    m = _model_modul()
    assert m._rens_fri_tekst("```\nlinje 1\nlinje 2\n```") == "linje 1\nlinje 2"
    assert m._rens_fri_tekst("```text\nlinje 1\n```") == "linje 1"


def test_hegn_midt_i_teksten_roeres_ikke():
    """Et hegn inde i svaret er noget, modellen SKREV, og skal maales som det."""
    m = _model_modul()
    tekst = "linje 1\n```\nlinje 2"
    assert m._rens_fri_tekst(tekst) == tekst
