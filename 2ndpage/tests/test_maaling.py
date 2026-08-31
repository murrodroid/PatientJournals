"""Testkontrakt for maaleapparatet efter forankringen blev fjernet.

Hele siden maales nu i ét straek, i raekkefoelge, uden soegning. Det aendrer
hvad der overhovedet KAN gaa galt, og testene her er skrevet efter de nye
faldgruber -- ikke oversat fra de gamle.

De to vigtigste staar i afsnittet "Ingen rabat". Det gamle apparat maalte kun
de linjer, det kunne finde, og en variant der fik modellen til at afvige mest
tabte netop de svaere linjer ud af maalingen og saa dermed BEDRE ud. Den faelde
vendte en rangorden to gange paa én dag. Den maa ikke kunne komme igen, og
testene her siger hvorfor med tal.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from andenside import cer
from andenside.facit import saml_orddeling
from andenside.maal import _tegn, deler_ord, flad, maal_saet, maal_side, streng_facit

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
        assert m.rene[navn].tegnafstand == 0, navn


def test_ulaeseligt_sted_koster_ikke_naar_modellen_gaetter_kort():
    """Modellen skriver noget, hvor facit siger [?]. Det maa hverken taelle
    for eller imod, saa laenge gaettet holder sig inden for loftet."""
    facit = ["Der er rigelig [?] Udflod af egen art", "Barnet er kraftigt."]
    model = som_model(["Der er rigelig blodig Udflod af egen art", "Barnet er kraftigt."])
    m = maal_side("prøve", facit, model)

    assert m.fladet["raa"].tegnafstand == 0
    assert len(m.gab) == 1
    assert m.gab[0].model_tekst.strip() == "blodig"


def test_gabet_baerer_facits_egne_ord_omkring_stedet():
    """Modelteksten alene er tit ét ord og kan staa hvor som helst paa siden.
    Uden facits naboord kan et menneske ikke finde stedet igen."""
    facit = ["Der er rigelig [?] Udflod af egen art"]
    m = maal_side("prøve", facit, "Der er rigelig blodig Udflod af egen art")

    assert m.gab[0].facit_foer == "Der er rigelig"
    assert m.gab[0].facit_efter == "Udflod af egen art"


# --------------------------------------------------------------------------
# Ingen rabat -- de to vigtigste tests i filen
# --------------------------------------------------------------------------

def test_udeladt_linje_koster_hele_linjen_i_stedet_for_at_falde_ud():
    """DEN afgoerende forskel fra forankringen.

    Modellen springer en hel, fuldt laesbar linje over. Forankringen kunne
    ikke finde den, og en linje der ikke blev fundet, gik HELT ud af
    maalingen -- altsaa gratis. Her skal den koste sine egne tegn.
    """
    facit = ["Rask indtil for 8 Dage siden", "da hun fik Brystkatarrh.", "Ingen Snue."]
    uden_midten = som_model(["Rask indtil for 8 Dage siden", "Ingen Snue."])

    m = maal_side("prøve", facit, uden_midten)

    # Den udeladte linje er 24 tegn plus det mellemrum, den var adskilt med.
    assert m.fladet["raa"].tegnafstand == len("da hun fik Brystkatarrh.") + 1
    # ... og hele facit staar stadig i naevneren.
    assert m.fladet["raa"].facit_tegn == len(flad(facit, deler_ord(facit)))


def test_hvor_meget_der_maales_paa_afhaenger_ikke_af_modellens_svar():
    """Naevneren skal vaere en egenskab ved FACIT alene.

    Det var praecis her, det gamle apparat svigtede: jo mere en variant fik
    modellen til at afvige, jo mere tekst faldt ud af maalingen, og jo bedre
    saa varianten ud. To vidt forskellige svar paa samme side skal give
    naevneren uaendret -- baade i hovedtallet og i den strenge maaling.
    """
    facit = ["Rask indtil for 8 Dage siden", "da hun fik [?] Brystkatarrh.", "Ingen Snue."]
    god = som_model(facit)
    elendig = "Aldeles andet indhold, som intet har med siden at goere."

    a = maal_side("prøve", facit, god)
    b = maal_side("prøve", facit, elendig)

    for navn in cer.VARIANTER:
        assert a.fladet[navn].facit_tegn == b.fladet[navn].facit_tegn, navn
        assert a.rene[navn].facit_tegn == b.rene[navn].facit_tegn, navn
    assert a.rene_tegn_i_alt == b.rene_tegn_i_alt
    # Og det daarlige svar skal saa OGSAA koste mere, ikke mindre.
    assert b.fladet["raa"].tegnafstand > a.fladet["raa"].tegnafstand


def test_gentaget_vending_paa_siden_vaelter_ikke_maalingen():
    """Regressionen fra `273107_001864`, maalt gennem hele apparatet.

    "ingen Snue" staar to gange i FACIT selv. Forankringen soegte linje 1 frem,
    fandt det ordrette traef nede i den sidste linje, flyttede soegepunktet
    dertil, og alle mellemliggende linjer var derefter uden for raekkevidde --
    26 af 29 tabt. Her er der kun én vej gennem siden.
    """
    mellem = [f"Linje {nr} med almindelig journaltekst." for nr in range(2, 12)]
    facit = ["Ingen Snue."] + mellem + ["Tg. ikke suspect ingen Snue."]
    model = som_model(["Ingen Hoste."] + mellem + ["G. ikke suspect ingen Snue."])

    m = maal_side("prøve", facit, model)

    # Kun de to faktiske laesefejl: "Snue"->"Hoste" (5 tegn) og "Tg."->"G." (1).
    assert m.fladet["raa"].tegnafstand == 6


# --------------------------------------------------------------------------
# De seks varianters filtre
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


def test_modellens_egne_linjeskift_er_uden_betydning_for_hovedtallet():
    """Modellen maa gerne bryde linjerne anderledes end siden. Hovedtallet
    maales paa den fladede tekst og skal vaere det samme."""
    med_skift = maal_side("prøve", SIDE, som_model(SIDE))
    uden_skift = maal_side("prøve", SIDE, flad(SIDE, deler_ord(SIDE)))

    assert med_skift.fladet["raa"].tegnafstand == uden_skift.fladet["raa"].tegnafstand


# --------------------------------------------------------------------------
# Den strenge maaling (beslutning 44): linjer med [?] slet ikke med
# --------------------------------------------------------------------------

def test_streng_maaling_udelader_hele_linjen_med_ulaeseligt_sted():
    facit = ["Ingen Snue.", "Der er rigelig [?] Udflod.", "Barnet er kraftigt."]
    rene = ["Ingen Snue.", "Barnet er kraftigt."]

    m = maal_side("prøve", facit, som_model(facit))

    # Ét tegn mindre end de rene linjer hver for sig: maerket tager sine egne
    # afgraensende mellemrum med, naar det bliver til et jokerfelt. Det er
    # tilsigtet og dokumenteret i `sidemaaling._forbered_facit` -- jokeren
    # daekker stedet MED mellemrummene, saa modellen kan skrive dem igen uden
    # at betale. Prisen er ét tegn i naevneren pr. maerke.
    assert m.rene["raa"].facit_tegn == len(flad(rene, deler_ord(rene))) - 1
    assert m.rene["raa"].facit_tegn < m.fladet["raa"].facit_tegn


def test_streng_maaling_ser_stadig_fejl_paa_de_rene_linjer():
    """Udeladelsen maa ramme de svaere linjer -- ikke fejlene paa de nemme.

    Modellens tekst paa den udeladte linje holder sig praecis inden for dens
    loft (linjens eget indhold, 15 tegn), saa den slipper gratis. Tilbage staar
    kun fejlen paa en ren linje, og den SKAL koste. Kunne jokeren sluge videre
    ind i naboteksten, ville "Snue" forsvinde gratis -- og det er praecis den
    slags mildhed, tallet ikke maa have.
    """
    facit = [
        "Barnet er kraftigt og velnaeret ved indlaeggelsen.",
        "Der er [?] Udflod.",
        "Ingen Snue, Tungen belagt, Halsen uden belaegninger.",
    ]
    model = som_model([
        "Barnet er kraftigt og velnaeret ved indlaeggelsen.",
        "Der er tyk Udflod.",
        "Ingen Hoste, Tungen belagt, Halsen uden belaegninger.",
    ])

    m = maal_side("prøve", facit, model)

    # "Snue" -> "Hoste" er fire redigeringer, ikke fem: S->H, n->o, indsaet s,
    # u->t, og e staar. Efterregnet med `cer.levenshtein`, ikke talt i hovedet.
    assert m.rene["raa"].tegnafstand == 4


def test_streng_maaling_lader_den_udeladte_linjes_modeltekst_slippe_gratis():
    """Linjen er ude af naevneren, saa modellens modstykke til den skal ogsaa
    vaere ude af taelleren. Ellers ville hele den udeladte linjes tekst staa
    som indsaettelser og goere den strenge maaling meningsloes."""
    facit = ["Ingen Snue.", "Der er rigelig [?] Udflod af egen art.", "Barnet er kraftigt."]

    m = maal_side("prøve", facit, som_model(facit))

    assert m.rene["raa"].tegnafstand == 0


def test_side_uden_rene_linjer_giver_en_tom_streng_maaling_ikke_et_krak():
    m = maal_side("prøve", ["Alt er [?] her."], "Alt er ulaeseligt her.")

    assert m.rene["raa"].facit_tegn == 0
    assert m.rene["raa"].cer == 0.0


def test_streng_facit_giver_et_loft_pr_udeladt_linje():
    facit = ["Ingen Snue.", "Der er [?] Udflod.", "Barnet [?] kraftigt."]
    tekst, lofter = streng_facit(facit)

    assert tekst.count("[?]") == 2
    assert len(lofter) == 2
    # Loftet er linjens EGET indhold -- hverken mere eller mindre. De 15 tegn
    # fra `JOKER_LOFT` maa ikke laegges oveni: de er udledt af ordlaengden i
    # materialet og er maalet for et `[?]` inde i en linje, mens linjens egen
    # laengde er den tilsvarende udledning for en hel linje. Laegges de sammen,
    # taelles samme begrundelse to gange, og den strenge maaling bliver
    # maerkbart mildere, end den giver sig ud for.
    assert lofter == [_tegn("Der er [?] Udflod."), _tegn("Barnet [?] kraftigt.")]


# --------------------------------------------------------------------------
# Raekkefoelgen som sit eget tal
# --------------------------------------------------------------------------

def test_ombyttede_linjer_taelles_som_omrokering_og_koster_i_hovedtallet():
    """Begge dele er meningen. Maalingen er streng om raekkefoelgen, fordi
    journalen laeses kronologisk -- og omrokeringen opgoeres ved siden af, saa
    det kan SES, at fejlen var orden og ikke laesning."""
    facit = ["Foerste linje her.", "Anden linje her.", "Tredje linje her."]
    byttet = som_model(["Tredje linje her.", "Anden linje her.", "Foerste linje her."])

    m = maal_side("prøve", facit, byttet)

    assert m.omrokering.antal_flyttede > 0
    assert m.omrokering.linjer_identiske == 3  # alle tre er laest rigtigt
    assert m.fladet["raa"].tegnafstand > 0     # men de staar forkert


# --------------------------------------------------------------------------
# Determinisme og saet-haandtering
# --------------------------------------------------------------------------

def test_sider_kommer_i_sorteret_raekkefoelge():
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
    poster = [{"image_name": "a", "alt_linjer": SIDE}, {"image_name": "b", "alt_linjer": SIDE}]
    saet = maal_saet(poster, {"a": som_model(SIDE)})
    assert [s.image_name for s in saet.sider] == ["a"]


def test_to_koersler_giver_noejagtig_samme_tal():
    """Maengde- og ordbogs-iteration har foer givet ikke-reproducerbare
    resultater i andre projekter."""
    poster = [
        {"image_name": "b", "alt_linjer": SIDE},
        {"image_name": "a", "alt_linjer": ["Der er rigelig [?] Udflod af egen art"]},
    ]
    modeller = {"b": som_model(SIDE), "a": "Der er rigelig blodig Udflod af egen art"}

    def koer():
        s = maal_saet(poster, modeller)
        return (
            [(n, m.tegnafstand, m.facit_tegn) for n, m in sorted(s.fladet.items())],
            [(navn, g.model_tekst) for navn, g in s.gab],
        )

    assert koer() == koer()


@pytest.mark.parametrize("navn", sorted(cer.VARIANTER))
def test_hver_variant_maales_paa_baade_hovedtal_og_streng(navn: str):
    """Beslutning 26: alle seks staar side om side, og ingen af dem maa
    vaelges efter, hvilken der klaeder resultatet bedst."""
    saet = maal_saet([{"image_name": "a", "alt_linjer": SIDE}], {"a": som_model(SIDE)})

    assert navn in saet.fladet
    assert navn in saet.rene


def test_den_faste_udeladelse_er_den_samme_uanset_modelsvar():
    """`andel_af_facit_i_rene` er den strenge maalings udeladelse. Den maa kun
    afhaenge af facit -- er den variantafhaengig, er den glidende rabat
    tilbage under et nyt navn."""
    poster = [{"image_name": "a", "alt_linjer": ["Ingen Snue.", "Der er [?] Udflod."]}]

    god = maal_saet(poster, {"a": som_model(poster[0]["alt_linjer"])})
    daarlig = maal_saet(poster, {"a": "Noget helt andet."})

    assert god.andel_af_facit_i_rene == daarlig.andel_af_facit_i_rene


# --------------------------------------------------------------------------
# Rapporten
# --------------------------------------------------------------------------

def _proeverapport() -> str:
    from andenside.rapport import skriv_rapport

    poster = [
        {"image_name": "a", "alt_linjer": SIDE},
        {"image_name": "b", "alt_linjer": ["Der er rigelig [?] Udflod af egen art"]},
    ]
    modeller = {"a": som_model(SIDE), "b": "Der er rigelig blodig Udflod af egen art"}
    return skriv_rapport(
        maal_saet(poster, modeller),
        titel="Prøve",
        model="ingen",
        promptversion="0",
        dato="2026-08-31",
    )


def test_rapporten_bruger_ikke_forankringens_begreber_om_hovedtallet():
    """Ordene forsvandt sammen med mekanismen. Dukker de op igen som
    beskrivelse af hovedtallet, er den glidende rabat tilbage under et nyt
    navn -- og det var den, der vendte en rangorden to gange paa én dag.

    "Dækning" og "rabat" maa kun staa i den strenge maalings afsnit, hvor de
    udtrykkeligt forklares som noget, der IKKE laengere findes.
    """
    tekst = _proeverapport()

    assert "forankr" not in tekst.lower()
    assert "uforankret" not in tekst.lower()

    afsnit = tekst.split("## ")
    for a in afsnit:
        if a.startswith("Hovedtal"):
            assert "dækning" not in a.lower()
            assert "rabat" not in a.lower()
            break
    else:
        raise AssertionError("rapporten har intet hovedtals-afsnit")


def test_rapporten_forklarer_maalingen_foer_den_viser_det_foerste_tal():
    """Kravet gjaldt ogsaa den gamle rapport: den skal kunne laeses uden
    CONTEXT.md ved haanden."""
    tekst = _proeverapport()

    forklaring = tekst.index("Sådan er der målt")
    hovedtal = tekst.index("## Hovedtal")
    assert forklaring < hovedtal
    # De tre ting en laeser skal have for at forstaa tallet.
    assert "søg" in tekst[forklaring:hovedtal].lower()
    assert "rækkefølge" in tekst[forklaring:hovedtal].lower()
    assert "[?]" in tekst[forklaring:hovedtal]


@pytest.mark.parametrize("navn", sorted(cer.VARIANTER))
def test_hver_variant_har_sin_egen_raekke_i_tabellen(navn: str):
    assert f"`{navn}`" in _proeverapport()


def test_raekkefoelge_afsnittet_baerer_sit_forbehold():
    """Tallene kommer fra en linjeparring med kendt svaghed, ikke fra
    hovedmaalingen. Staar de uden forbehold, bliver de laest som
    beslutningstal."""
    tekst = _proeverapport()

    afsnit = [a for a in tekst.split("## ") if a.startswith("Rækkefølge")]
    assert afsnit, "rapporten har intet afsnit om rækkefølge"
    assert "vejledende" in afsnit[0].lower()


def test_tomt_saet_giver_en_rapport_i_stedet_for_at_gaa_i_stykker():
    from andenside.rapport import skriv_rapport

    tekst = skriv_rapport(
        maal_saet([], {}), titel="Tom", model="ingen", promptversion="0",
        dato="2026-08-31",
    )
    assert "Tom" in tekst


def test_to_koersler_giver_noejagtig_samme_rapport():
    """Maengde- og ordbogs-iteration har foer givet ikke-reproducerbare
    resultater i andre projekter. Rapporten skal vaere tegn for tegn ens."""
    assert _proeverapport() == _proeverapport()
