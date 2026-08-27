"""Tests for modellaget -- kun de deterministiske dele.

Selve modelkaldet testes ikke (svar kan ikke laases fast; de logges i stedet
som resultater i stagens output). Det der KAN gaa galt uden at nogen opdager
det, er de to stykker plumbing omkring kaldet:

  1. **Noegleopslaget.** Noeglefilen laeses af koden, aldrig af os. Vi kender
     ikke feltnavnet indeni, saa opslaget maa finde det selv -- og fejle
     tydeligt frem for at gaette, hvis der ikke er et.
  2. **Sammenfoejningen af svaret.** Modellen svarer linje for linje efter et
     skema. Maaleapparatet vil have én tekst. Gaar noget tabt i den
     oversaettelse, ser det ud som om modellen laeste daarligt.
"""

import pytest

from andenside.model import (
    NoegleFejl,
    find_noegle,
    tekst_af_sider,
)


# ---------------------------------------------------------------- noeglen

def test_noeglen_findes_uanset_hvad_feltet_hedder():
    # Vi kender ikke feltnavnet i brugerens fil, saa opslaget skal taale
    # de almindelige stavemaader.
    for navn in ("gemini", "GEMINI_API_KEY", "google", "genai_key", "Gemini Api Key"):
        assert find_noegle({navn: "xxx", "openai": "yyy"}) == "xxx"


def test_den_rigtige_udbyder_vaelges_naar_der_er_flere():
    data = {"openai": "aaa", "anthropic": "bbb", "gemini": "ccc"}
    assert find_noegle(data) == "ccc"


def test_manglende_noegle_fejler_tydeligt_og_naevner_feltnavnene():
    """Fejlen skal kunne handles paa uden at nogen aabner filen.

    Beskeden maa naevne, hvilke felter der ER i filen -- ellers er man
    henvist til at kigge i en fil, vi netop ikke vil kigge i.
    """
    with pytest.raises(NoegleFejl) as fejl:
        find_noegle({"openai": "aaa", "anthropic": "bbb"})
    assert "openai" in str(fejl.value)
    assert "anthropic" in str(fejl.value)


def test_selve_noeglen_lækker_aldrig_ud_i_fejlbeskeden():
    with pytest.raises(NoegleFejl) as fejl:
        find_noegle({"openai": "hemmelig-vaerdi-1234"})
    assert "hemmelig-vaerdi-1234" not in str(fejl.value)


def test_tom_noegle_taeller_ikke_som_fundet():
    # Et tomt felt er vaerre end intet felt: kaldet ville fejle langt senere.
    with pytest.raises(NoegleFejl):
        find_noegle({"gemini": "   "})


def test_indlejrede_noegler_findes_ogsaa():
    # Nogle noeglefiler grupperer pr. udbyder.
    assert find_noegle({"providers": {"gemini": {"api_key": "xxx"}}}) == "xxx"


# ------------------------------------------------------- svaret til tekst

def test_linjerne_bliver_til_én_tekst_i_raekkefoelge():
    sider = [{"text": "Rask indtil for 8 Dage siden"}, {"text": "da hun fik Brystkatarrh."}]
    assert tekst_af_sider(sider) == "Rask indtil for 8 Dage siden\nda hun fik Brystkatarrh."


def test_tomme_linjer_bevares_som_linjer():
    """En tom linje i svaret er stadig et linjeskift.

    Smider vi den vaek, skrider linjeparringen i maaleapparatet.
    """
    sider = [{"text": "en"}, {"text": ""}, {"text": "to"}]
    assert tekst_af_sider(sider) == "en\n\nto"


def test_margin_metadata_kommer_ikke_med_i_teksten():
    """Margendatoer staar i `metadata`, ikke i `text`.

    De TAELLER med i facit som tekst (lead 2026-08-20), men de staar dér,
    hvor de staar paa siden -- de maa ikke limes ind et vilkaarligt sted af
    os. Kommer de ikke med, er det en promptsag, ikke en oprydningssag.
    """
    sider = [{"text": "Rask indtil", "metadata": "18/12"}]
    assert tekst_af_sider(sider) == "Rask indtil"


def test_manglende_tekstfelt_er_en_fejl_ikke_en_tom_linje():
    # En linje uden `text` betyder, at skemaet ikke blev fulgt. Det skal
    # opdages, ikke maskeres som en tom linje.
    with pytest.raises(ValueError, match="text"):
        tekst_af_sider([{"metadata": "18/12"}])


def test_et_tomt_svar_er_en_fejl():
    """Nul linjer betyder, at modellen ikke svarede -- ikke at siden er tom.

    Gik det igennem som "", ville maalingen se det som en side, modellen
    laeste perfekt daarligt, i stedet for som en fejlet koersel.
    """
    with pytest.raises(ValueError, match="tomt"):
        tekst_af_sider([])
