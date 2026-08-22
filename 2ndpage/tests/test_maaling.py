"""Testkontrakt for maalingen af en hel side og et helt saet.

Stagens kontrakt kraever: nul fejl naar facit sammenlignes med sig selv, et
kendt forud udregnet tal paa konstruerede forvanskninger, en test pr. variant,
en test af orddelingssamlingen, og en test af at to koersler paa samme data
giver noejagtig samme rapport.

Forvanskningerne er konstrueret, ikke repraesentative. Det er meningen -- data
der fremkalder en bestemt fejl er sjaeldent typiske.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from andenside import cer
from andenside.facit import saml_orddeling
from andenside.maal import SaetMaaling, deler_ord, flad, maal_side
from andenside.rapport import skriv_rapport

FACIT = Path(__file__).resolve().parents[1] / "stages" / "02_facit" / "output" / "facit.jsonl"

SIDE = [
    "Rask indtil for 8 Dage siden",
    "da hun fik Brystkatarrh.",
    "Ingen tidligere Infektions-",
    "sygdomme.",
    "Barnet er kraftigt, velnæret.",
]


def som_model(linjer: list[str]) -> str:
    return "\n".join(linjer)


# --------------------------------------------------------------------------
# Facit mod sig selv
# --------------------------------------------------------------------------

def test_facit_mod_sig_selv_giver_nul_fejl_i_alle_varianter():
    m = maal_side("prøve", SIDE, som_model(SIDE))
    for navn in cer.VARIANTER:
        assert m.fladet[navn].tegnafstand == 0, navn
        assert m.fladet[navn].ordafstand == 0, navn
        assert m.pr_linje[navn].tegnafstand == 0, navn
    assert m.daekning == 1.0
    assert m.linjer_maalt == m.linjer_i_alt


def test_ulaeseligt_sted_koster_ikke_naar_stumperne_paa_begge_sider_findes():
    """Modellen skriver noget, hvor facit siger [?]. Det maa hverken taelle
    for eller imod -- kun stumperne omkring maales."""
    facit = ["Der er rigelig [?] Udflod af egen art", "Barnet er kraftigt."]
    model = som_model(["Der er rigelig blodig Udflod af egen art", "Barnet er kraftigt."])
    m = maal_side("prøve", facit, model)

    assert m.fladet["raa"].tegnafstand == 0
    assert len(m.gab) == 1
    assert m.gab[0].model_tekst.strip() == "blodig"
    assert m.svaere_linjer_reddet == 1


# --------------------------------------------------------------------------
# Beslutning 38: uforankret linje ud af maalingen -- paa BEGGE sider
# --------------------------------------------------------------------------

def test_uforankret_linje_falder_ud_af_baade_facit_og_model():
    """Grundreglen. Faldt linjen kun ud af facit, ville modellens tekst paa
    stedet blive talt som indsat og straffe den for noget, vi ikke maaler."""
    facit = list(SIDE)
    model = som_model([l for l in SIDE if l != "da hun fik Brystkatarrh."])
    m = maal_side("prøve", facit, model)

    assert m.linjer_maalt == len(SIDE) - 1
    assert m.fladet["raa"].tegnafstand == 0
    assert m.daekning < 1.0
    assert "Brystkatarrh" not in m.linjer[1].model_maalt


def test_daekningen_falder_med_de_linjer_der_ikke_kunne_forankres():
    facit = list(SIDE)
    model = som_model(SIDE[:2])
    m = maal_side("prøve", facit, model)
    assert 0 < m.daekning < 0.5
    assert m.linjer_maalt == 2


# --------------------------------------------------------------------------
# Kendte, forud udregnede tal paa konstruerede forvanskninger
# --------------------------------------------------------------------------

def test_tysk_omlyd_koster_praecis_et_tegn_raat_og_nul_uden_diakritika():
    """ø/ö er den hyppigste enkeltforveksling i materialet og er ortografisk
    stoej, ikke en laesefejl."""
    facit = ["Barnet er kraftigt, velnæret", "men Brystet er ømt"]
    model = som_model(["Barnet er kraftigt, velnæret", "men Brystet er ömt"])
    m = maal_side("prøve", facit, model)

    assert m.fladet["raa"].tegnafstand == 1
    assert m.fladet["uden_diakritika"].tegnafstand == 0
    assert m.fladet["lempeligst"].tegnafstand == 0


def test_versaler_koster_raat_men_ikke_uden_versaler():
    facit = ["Ingen Snue, Tungen belagt"]
    model = som_model(["ingen snue, tungen belagt"])
    m = maal_side("prøve", facit, model)

    assert m.fladet["raa"].tegnafstand == 3  # I, S, T
    assert m.fladet["uden_versaler"].tegnafstand == 0
    assert m.fladet["arbejdstal"].tegnafstand == 0


def test_tegnsaetning_koster_raat_men_ikke_uden_tegnsaetning():
    facit = ["Ingen Snue, Tungen belagt."]
    model = som_model(["Ingen Snue Tungen belagt"])
    m = maal_side("prøve", facit, model)

    assert m.fladet["raa"].tegnafstand == 2  # kommaet og punktummet
    assert m.fladet["uden_tegnsaetning"].tegnafstand == 0


# --------------------------------------------------------------------------
# Orddeling hen over linjeskift -- StadsCERs kendte mangel
# --------------------------------------------------------------------------

def test_model_der_skriver_delt_ord_i_et_stykke_straffes_ikke():
    """Facit deler "Infektions-" / "sygdomme."; modellen skriver ordet samlet.
    Den har laest rigtigt og maa ikke koste."""
    facit = ["Ingen tidligere Infektions-", "sygdomme. Barnet er kraftigt"]
    model = som_model(["Ingen tidligere Infektionssygdomme.", "Barnet er kraftigt"])
    m = maal_side("prøve", facit, model)

    assert m.fladet["arbejdstal"].tegnafstand == 0


def test_bindestreg_som_punktum_samles_ikke():
    """Materialet bruger ogsaa bindestreg som punktum. "Rhonchi-" efterfulgt
    af stort bogstav er to ord, ikke ét."""
    linjer = ["enkelte Rhonchi-", "Ingen Snue, Tungen"]
    assert deler_ord(linjer) == [False]
    assert flad(linjer, deler_ord(linjer)) == "enkelte Rhonchi- Ingen Snue, Tungen"


def test_fladning_er_enig_med_facits_egen_over_hele_facit():
    """`flad` og `facit.saml_orddeling` er to steder, samme regel. Skrider de
    fra hinanden, maales facit paa én maade og modellen paa en anden."""
    poster = [json.loads(l) for l in FACIT.read_text(encoding="utf-8").splitlines()]
    for post in poster:
        linjer = post["alt_linjer"]
        assert flad(linjer, deler_ord(linjer)) == saml_orddeling("\n".join(linjer)), (
            post["image_name"]
        )


# --------------------------------------------------------------------------
# Maalingen maa ikke afhaenge af, om modellen foelger sidens linjeskift
# --------------------------------------------------------------------------

def test_model_uden_linjeskift_giver_samme_fladede_tal():
    """Beslutning 35: vi VED ikke, om modellen laver sine egne linjeskift.
    Skriver den hele siden som ét afsnit, skal tallet vaere det samme."""
    med = maal_side("prøve", SIDE, som_model(SIDE))
    uden = maal_side("prøve", SIDE, " ".join(SIDE))

    assert uden.fladet["arbejdstal"].tegnafstand == med.fladet["arbejdstal"].tegnafstand
    assert uden.linjer_maalt == med.linjer_maalt


def test_forskudte_linjebrud_skrider_ikke():
    """Flytter modellen hvert linjebrud ét ord, skal parringen holde. Uden
    forankring ville alt efter det foerste brud vaere forkert."""
    forskudt = [
        "Rask indtil for 8 Dage siden da",
        "hun fik Brystkatarrh. Ingen",
        "tidligere Infektions-",
        "sygdomme. Barnet er",
        "kraftigt, velnæret.",
    ]
    m = maal_side("prøve", SIDE, som_model(forskudt))
    assert m.fladet["arbejdstal"].tegnafstand == 0
    assert m.linjer_maalt == len(SIDE)


def test_linjetrofasthed_maales_i_stedet_for_at_antages():
    """Svaret paa beslutning 35 skal vaere et tal, ikke en formodning."""
    tro = maal_side("prøve", SIDE, som_model(SIDE))
    assert tro.uden_linjeskift_indeni == len(SIDE)
    assert tro.egen_modellinje == len(SIDE)

    ét_afsnit = maal_side("prøve", SIDE, " ".join(SIDE))
    assert ét_afsnit.egen_modellinje == 1


# --------------------------------------------------------------------------
# Opdigtning
# --------------------------------------------------------------------------

def test_opdigtet_afsnit_dukker_op_som_uforankret_modeltekst():
    """Tegnfejlen ser den ikke -- den maaler kun det forankrede. Derfor SKAL
    dette tal staa ved siden af i rapporten."""
    model = som_model(SIDE + ["Patienten blev udskrevet rask den tolvte juni."])
    m = maal_side("prøve", SIDE, model)

    assert m.fladet["raa"].tegnafstand == 0
    assert m.model_tegn_uforankret > 30


def test_fuldsidekontrol_koeres_kun_paa_sider_uden_maerker():
    uden = maal_side("prøve", SIDE, som_model(SIDE))
    assert uden.fuldside is not None
    assert uden.fuldside["raa"].tegnafstand == 0

    med = maal_side("prøve", ["Der er rigelig [?] Udflod"], "Der er rigelig blodig Udflod")
    assert med.fuldside is None


def test_fuldsidekontrollen_ser_det_forankringen_ikke_ser():
    """Kontrollens hele formaal: paa en side uden maerker taeller opdigtet
    tekst med som indsaettelser, saa forankringen ikke kan pynte ubemaerket."""
    model = som_model(SIDE + ["Patienten blev udskrevet rask den tolvte juni."])
    m = maal_side("prøve", SIDE, model)

    assert m.fladet["raa"].tegnafstand == 0
    assert m.fuldside["raa"].tegnafstand > 30


# --------------------------------------------------------------------------
# Determinisme
# --------------------------------------------------------------------------

def test_to_koersler_giver_noejagtig_samme_rapport():
    """Maengde- og ordbogs-iteration har foer givet ikke-reproducerbare
    resultater i andre projekter. Rapporten skal vaere tegn for tegn ens."""
    poster = [
        {"image_name": "b", "alt_linjer": SIDE},
        {"image_name": "a", "alt_linjer": ["Der er rigelig [?] Udflod af egen art"]},
    ]
    modeller = {"b": som_model(SIDE), "a": "Der er rigelig blodig Udflod af egen art"}

    from andenside.maal import maal_saet

    def koer() -> str:
        return skriv_rapport(
            maal_saet(poster, modeller),
            titel="Prøve",
            model="ingen",
            promptversion="0",
            dato="2026-08-22",
        )

    assert koer() == koer()


def test_sider_kommer_i_sorteret_raekkefoelge():
    from andenside.maal import maal_saet

    poster = [
        {"image_name": "c", "alt_linjer": SIDE},
        {"image_name": "a", "alt_linjer": SIDE},
        {"image_name": "b", "alt_linjer": SIDE},
    ]
    modeller = {n: som_model(SIDE) for n in "abc"}
    saet = maal_saet(poster, modeller)
    assert [s.image_name for s in saet.sider] == ["a", "b", "c"]


def test_side_uden_modelsvar_springes_over_i_stedet_for_at_taelle_som_nul():
    """En side uden svar er en manglende maaling, ikke en perfekt eller en
    elendig. Kom den med som nul tegn, ville tallet blive meningsloest."""
    from andenside.maal import maal_saet

    poster = [{"image_name": "a", "alt_linjer": SIDE}, {"image_name": "b", "alt_linjer": SIDE}]
    saet = maal_saet(poster, {"a": som_model(SIDE)})
    assert [s.image_name for s in saet.sider] == ["a"]


# --------------------------------------------------------------------------
# Rapporten
# --------------------------------------------------------------------------

def test_rapporten_naevner_daekning_og_at_facit_rummer_fejl():
    """De to forbehold er obligatoriske. Uden dem er tallet misvisende,
    uanset hvor korrekt det er regnet ud."""
    from andenside.maal import maal_saet

    saet = maal_saet([{"image_name": "a", "alt_linjer": SIDE}], {"a": som_model(SIDE)})
    tekst = skriv_rapport(saet, titel="Prøve", model="m", promptversion="1", dato="2026-08-22")

    assert "Dækningen står ved hvert tal" in tekst
    assert "Facit rummer selv fejl" in tekst
    assert "37554_001491" in tekst
    for navn in cer.VARIANTER:
        assert f"`{navn}`" in tekst


def test_tomt_saet_giver_en_rapport_i_stedet_for_at_gaa_i_stykker():
    tekst = skriv_rapport(
        SaetMaaling(sider=()), titel="Tom", model="m", promptversion="1", dato="2026-08-22"
    )
    assert "Sider målt | 0" in tekst


@pytest.mark.parametrize("navn", list(cer.VARIANTER))
def test_hver_variant_har_sin_egen_raekke_i_tabellen(navn: str):
    from andenside.maal import maal_saet

    saet = maal_saet([{"image_name": "a", "alt_linjer": SIDE}], {"a": som_model(SIDE)})
    tekst = skriv_rapport(saet, titel="Prøve", model="m", promptversion="1", dato="2026-08-22")
    assert tekst.count(f"| `{navn}` |") >= 2  # fladet OG pr. linje


def test_side_med_lav_daekning_udpeges_selvom_dens_tegnfejl_er_flot():
    """Faelden i enhver kvalitetsrapport: en side hvor naesten intet kunne
    maales, faar et pænt tal og lander i bunden af 'de vaerste'. Den skal
    staa oeverst i sin EGEN liste."""
    from andenside.maal import maal_saet

    poster = [
        {"image_name": "tynd", "alt_linjer": SIDE},
        {"image_name": "fyldig", "alt_linjer": SIDE},
    ]
    modeller = {
        # Kun foerste linje genkendelig -- resten sprunget over. Nul tegnfejl
        # paa det maalte, men saa godt som intet er maalt.
        "tynd": som_model(SIDE[:1]),
        # Hele siden med, men med fejl i.
        "fyldig": som_model([l.replace("e", "o") for l in SIDE]),
    }
    tekst = skriv_rapport(
        maal_saet(poster, modeller), titel="Prøve", model="m",
        promptversion="1", dato="2026-08-22",
    )

    tyndest_afsnit = tekst.split("tyndest målte sider")[1]
    assert tyndest_afsnit.index("`tynd`") < tyndest_afsnit.index("`fyldig`")

    vaerste_afsnit = tekst.split("værste sider")[1].split("tyndest målte")[0]
    assert vaerste_afsnit.index("`fyldig`") < vaerste_afsnit.index("`tynd`")


def test_falsk_forankring_skader_ogsaa_den_naeste_linje():
    """Kendt begraensning, pinnet med vilje (CONTEXT.md 2026-08-22, senere).

    Forankringen gaar fra venstre mod hoejre. Et falsk traef flytter derfor
    soegepunktet frem forbi det sted, hvor NAESTE linje i virkeligheden staar,
    saa den kun finder en afskaaret rest af sig selv -- selvom modellen skrev
    den helt rigtigt. Her: den korte facit-linje `Lunge` findes ikke som en
    linje i modellen, men bogstaverne staar inde i `Lunger` paa naeste linje,
    og saa aeder forankringen dét ord op.

    Raekkefoelgen fjernes IKKE for at undgaa det: uden den kunne en gentaget
    vending forankre bagud og give et gab med negativ laengde. Prisen for at
    fjerne fejlen ville vaere en stoerre fejl. Testen staar her, saa
    begraensningen ikke kan aendre sig ubemaerket -- og saa den, der en dag
    laver den om, kan se hvad den kostede.
    """
    facit = ["Lunge", "begge Lunger overalt en Mængde fugtige"]
    model = som_model(["Tungen er tør og belagt", "begge Lunger overalt en Mængde fugtige"])
    m = maal_side("prøve", facit, model)

    assert m.linjer[0].forankret          # falsk traef paa "Lunge" inde i "Lunger"
    assert m.linjer[1].model_maalt == "r overalt en Mængde fugtige"
    assert m.fladet["raa"].tegnafstand == 11

    # Uden den korte, vildledende linje er der ingen fejl at finde.
    uden = maal_side("prøve", facit[1:], model)
    assert uden.fladet["raa"].tegnafstand == 0


def test_rapporten_forklarer_forankring_foer_den_bruger_ordet():
    """Rapporten skal kunne staa alene. Ordet 'forankring' baerer hele
    maalingen, og en laeser, der ikke har CONTEXT.md ved haanden, skal kunne
    forstaa tallene alligevel."""
    from andenside.maal import maal_saet

    saet = maal_saet([{"image_name": "a", "alt_linjer": SIDE}], {"a": som_model(SIDE)})
    tekst = skriv_rapport(saet, titel="Prøve", model="m", promptversion="1", dato="2026-08-22")

    forklaring = tekst.index("Sådan er der målt")
    assert forklaring < tekst.index("## Hovedtal")
    for ord in ("tegnafstand", "CER", "WER", "Fladet tekst"):
        assert ord in tekst[forklaring:], ord
