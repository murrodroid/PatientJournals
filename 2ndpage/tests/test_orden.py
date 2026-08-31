"""Tests for `andenside.orden`: omrokerings-tallet ved siden af den strenge
raekkefoelges-maaling i `maal.py`.

Se `orden.py`s docstring for hvorfor tallet er LIS-baseret og ikke en
inversionstaelling -- de to testcases der eksplicit skelner mellem dem
(naboombytning og top-til-bund-flytning) er den vigtigste garanti i denne fil.
"""
from __future__ import annotations

from andenside.orden import maal_omrokering


def test_identiske_linjer_giver_nul_omrokerede():
    linjer = [
        "Patienten indlagt med feber.",
        "Puls 88, temperatur 39,2.",
        "Ordineret sengeleje og vaeske.",
    ]
    resultat = maal_omrokering(linjer, list(linjer))
    assert resultat.antal_flyttede == 0
    assert resultat.linjer_uparret == 0
    assert resultat.linjer_parret == 3


def test_to_ombyttede_nabolinjer_giver_en_ikke_to():
    # Den klassiske faelde: en naiv inversionstaelling ville sige 1 inversion
    # her ogsaa (kun ét par krydser), saa denne test alene skelner ikke de to
    # metoder -- den goer det sammen med top-til-bund-testen nedenfor, hvor
    # inversionstal og LIS-tal rykker fra hinanden.
    facit = ["Puls 88.", "Temperatur 39,2."]
    model = ["Temperatur 39,2.", "Puls 88."]
    resultat = maal_omrokering(facit, model)
    assert resultat.antal_flyttede == 1


def test_linje_flyttet_fra_top_til_bund_giver_en():
    # Her skiller LIS-maalet sig fra inversionstaelling: A krydser B, C OG D,
    # dvs. 3 inversioner -- men der skal kun flyttes ÉN linje (A) for at rette
    # raekken. Fejler denne test, er koden faldet tilbage til at taelle
    # inversioner i stedet for LIS.
    facit = ["A: forste observation.", "B: anden observation.", "C: tredje observation.", "D: fjerde observation."]
    model = ["B: anden observation.", "C: tredje observation.", "D: fjerde observation.", "A: forste observation."]
    resultat = maal_omrokering(facit, model)
    assert resultat.antal_flyttede == 1


def test_smaa_laesefejl_taeller_ikke_som_omrokering():
    # Modellen laeser "88" som "8B" og mangler et komma -- reelle laesefejl,
    # ikke omrokering. Parringen skal taale det og stadig se raekkefoelgen
    # som uaendret.
    facit = ["Puls 88, uregelmaessig.", "Respiration rolig."]
    model = ["Puls 8B uregelmaessig.", "Respiration rolig."]
    resultat = maal_omrokering(facit, model)
    assert resultat.antal_flyttede == 0
    assert resultat.linjer_uparret == 0


def test_model_med_faerre_linjer_end_facit_springer_ikke():
    # Modellen har droppet en linje. De to andre skal stadig parres korrekt,
    # og den manglende skal taelle som uparret -- ikke krasje og ikke tvinges
    # ind i omrokeringstallet.
    facit = ["Forste linje her.", "Anden linje mangler i model.", "Tredje linje her."]
    model = ["Forste linje her.", "Tredje linje her."]
    resultat = maal_omrokering(facit, model)
    assert resultat.linjer_uparret == 1
    assert resultat.linjer_parret == 2
    assert resultat.antal_flyttede == 0


def test_model_med_flere_linjer_end_facit_springer_ikke():
    # Modellen har digtet en ekstra linje ind midt i teksten. Facits tre
    # linjer skal stadig findes og parres i rigtig raekkefoelge.
    facit = ["Indlagt i gaar.", "Feber og hoste.", "Udskrevet i dag."]
    model = ["Indlagt i gaar.", "En linje modellen har fundet paa.", "Feber og hoste.", "Udskrevet i dag."]
    resultat = maal_omrokering(facit, model)
    assert resultat.linjer_parret == 3
    assert resultat.linjer_uparret == 0
    assert resultat.antal_flyttede == 0


def test_gentagne_naesten_ens_linjer_parres_hver_for_sig_ikke_paa_samme():
    # Journalmateriale gentager sig (vitalvaerdier). To facit-linjer der begge
    # ligner model-linjen "Puls 80." maa IKKE begge tildeles samme model-
    # linje -- parringen er et-til-et. Uden det ville begge facit-linjer faa
    # samme model-position, og "hvilken raekkefoelge" ville vaere meningsloest.
    facit = ["Puls 80.", "Puls 80.", "Temperatur 37,0."]
    model = ["Puls 80.", "Temperatur 37,0.", "Puls 80."]
    resultat = maal_omrokering(facit, model)
    assert resultat.linjer_parret == 3
    assert resultat.linjer_uparret == 0
    # De to "Puls 80."-linjer MAA have forskellige model-positioner.
    puls_positioner = (resultat.model_positioner[0], resultat.model_positioner[1])
    assert puls_positioner[0] != puls_positioner[1]


def test_determinisme_ved_uafgjort_parring():
    # Flere identiske linjer i baade facit og model giver flere lige gode
    # match. Resultatet skal vaere det samme, hver gang funktionen koeres --
    # ingen mængde- eller ordbogsiteration maa paavirke det.
    facit = ["Samme linje.", "Samme linje.", "Samme linje."]
    model = ["Samme linje.", "Samme linje.", "Samme linje."]
    foerste = maal_omrokering(facit, model)
    anden = maal_omrokering(facit, model)
    assert foerste == anden
    assert foerste.antal_flyttede == 0
    assert foerste.model_positioner == (0, 1, 2)


# --------------------------------------------------------------------------
# Helt korrekte linjer
# --------------------------------------------------------------------------
#
# Under forankringen kom "andel linjer der er noejagtig rigtige" fra
# `Maaltal.andel_identiske`, hvor ét stykke var én parret linje. Sidemaalingen
# maaler hele siden i ét straek, saa dér er ét stykke én SIDE -- og andelen
# ville blive "andel perfekte sider", et andet og naesten altid nul-tal.
# Linjeparringen her er det eneste sted, tallet stadig kan komme fra.


def test_identiske_linjer_taelles_kun_naar_teksten_er_ens():
    facit = ["Ingen Snue.", "Temperatur 39,5.", "Patienten har sovet."]
    model = ["Ingen Snue.", "Temperatur 38,5.", "Patienten har sovet."]

    resultat = maal_omrokering(facit, model)

    assert resultat.linjer_parret == 3
    assert resultat.linjer_identiske == 2


def test_identiske_linjer_ser_bort_fra_versaler_og_tegnsaetning():
    """Parringen sammenligner normaliseret tekst, og tallet skal foelge
    parringen -- ellers ville en linje kunne vaere parret og 'ikke identisk'
    paa en forskel, parringen selv har set bort fra."""
    resultat = maal_omrokering(["Ingen Snue."], ["ingen snue"])

    assert resultat.linjer_identiske == 1


def test_uparret_linje_er_ikke_identisk():
    resultat = maal_omrokering(["Ingen Snue."], ["Aldeles andet indhold her."])

    assert resultat.linjer_parret == 0
    assert resultat.linjer_identiske == 0
