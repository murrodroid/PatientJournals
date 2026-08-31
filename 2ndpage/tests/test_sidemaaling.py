"""Testkontrakt for sidemaalingen (stage 03, trin 1).

Den vigtigste test staar foerst: det gentagne ord fra `273107_001864`, som
vaeltede forankringen. Alle tal i testene er eksakte, ikke oevre graenser --
en maaling, hvor et fremspring er sluppet ind, giver et MINDRE tal, ikke et
stoerre, og en "hoejst saa meget"-graense ville derfor ikke fange den.

Testdata er konstrueret. Det er meningen: data, der fremkalder en bestemt fejl,
er sjaeldent typiske.
"""
from __future__ import annotations

import pytest

from andenside import cer
from andenside.sidemaaling import JOKER_LOFT, MAERKE, maal_side


# --------------------------------------------------------------------------
# Regressionen: det gentagne ord
# --------------------------------------------------------------------------

# Facits linje 1 og linje 26 fra `273107_001864`. "ingen Snue" staar to gange i
# FACIT selv -- det er ikke noget, modellen fandt paa. Forankringen soegte linje
# 1 frem, fandt det ordrette traef nede i linje 26, flyttede soegepunktet dertil
# og kunne derefter ikke naa linje 2-25. 26 af 29 linjer tabt.
MELLEMLINJER = [f"Linje {nr} med almindelig journaltekst." for nr in range(2, 26)]

FACIT_GENTAGET = "\n".join(
    ["Ingen Snue."] + MELLEMLINJER + ["Tg. ikke [?] suspect ingen Snue."]
)

# Modellens svar: to rimelige laesefejl og et gaet paa det ulaeselige sted.
# "Snue" -> "Hoste" i linje 1 (5 tegn), "Tg." -> "G." i linje 26 (1 tegn).
MODEL_GENTAGET = "\n".join(
    ["Ingen Hoste."] + MELLEMLINJER + ["G. ikke Roedt. suspect ingen Snue."]
)


def _facit_tegn_i_alt(facit: str) -> int:
    """Hvad naevneren SKAL vaere: hele siden uden maerkerne, normaliseret
    stykke for stykke praecis som maalingen selv goer det."""
    return sum(len(cer.normalize(s)) for s in facit.split(MAERKE))


def test_gentaget_ord_koster_kun_de_faktiske_laesefejl():
    """Hele siden maales, og de to laesefejl koster praecis, hvad de vejer.

    Forankringen gav her 26 tabte linjer. Tallene er eksakte: en maaling, der
    tillader et fremspring i modelteksten, kan kun goere afstanden mindre.
    """
    r = maal_side(FACIT_GENTAGET, MODEL_GENTAGET)

    assert r.facit_tegn == _facit_tegn_i_alt(FACIT_GENTAGET)
    assert r.tegnafstand == 6  # 5 for Snue->Hoste, 1 for det tabte T i "Tg."
    assert r.joker_tegn == (8,)  # " Roedt. " -- under loftet, altsaa gratis
    assert r.cer < 0.01


def test_gentaget_ord_daekker_ogsaa_de_mellemliggende_linjer():
    """Et svar, der KUN indeholder de to linjer med det gentagne ord, skal
    straffes for alt det, det ikke skrev.

    Det er den anden halvdel af den gamle fejl: forankringen smed de tabte
    linjer ud af naevneren OGSAA, saa et svar som dette fik et paent tal.
    """
    kun_de_to = "Ingen Hoste.\nG. ikke Roedt. suspect ingen Snue."
    r = maal_side(FACIT_GENTAGET, kun_de_to)

    assert r.facit_tegn == _facit_tegn_i_alt(FACIT_GENTAGET)
    # De 24 mellemlinjer er ren mangel og koster deres fulde laengde. Tallet er
    # eksakt: 899 af sidens 943 tegn, altsaa alt paa naer de to linjer, modellen
    # faktisk skrev.
    assert sum(len(cer.normalize(linje)) for linje in MELLEMLINJER) == 880
    assert r.tegnafstand == 899
    assert r.cer > 0.95


def test_jokeren_flytter_ikke_teksten_omkring_sig():
    """Jokeren maa ikke bruges som springbraet.

    Modellen skriver ordet "suspect" tidligt OG sent. Kan jokeren springe frem
    til det sene, bliver alt derimellem gratis. Afstanden er derfor eksakt.
    """
    facit = "alfa [?] suspect beta gamma"
    model = "alfa gaet suspect beta gamma suspect"
    r = maal_side(facit, model)

    assert r.joker_tegn == (6,)  # " gaet " -- ikke helt frem til det sene ord
    assert r.tegnafstand == 8  # " suspect" haengt paa til sidst


# --------------------------------------------------------------------------
# Jokerfeltets loft
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "slugt, forventet_pris",
    [
        (JOKER_LOFT - 1, 0),   # under loftet: gratis
        (JOKER_LOFT, 0),       # praecis paa loftet: stadig gratis
        (JOKER_LOFT + 1, 1),   # ét for meget: ét tegn i straf
        (JOKER_LOFT + 5, 5),
        (JOKER_LOFT * 3, JOKER_LOFT * 2),
    ],
)
def test_loftet_gaelder_fra_og_med_det_foerste_tegn_derover(slugt, forventet_pris):
    """De foerste `JOKER_LOFT` tegn er gratis, hvert tegn derover koster 1.

    Modelteksten klaebes til nabotegnene uden mellemrum, saa det slugte stykke
    er noejagtig `slugt` tegn og intet andet.
    """
    facit = "abc[?]def"
    model = "abc" + "x" * slugt + "def"
    r = maal_side(facit, model)

    assert r.joker_tegn == (slugt,)
    assert r.tegnafstand == forventet_pris
    assert r.joker_overskud == forventet_pris


def test_gabet_opgoer_alt_modellen_lagde_paa_stedet():
    """Naar overskuddet kan betales enten i jokeren eller som indsaettelser i
    nabo-teksten, skal gab-tallet vise HELE det, modellen skrev.

    Gabet er arbejdsredskabet til at finde steder, der er vaerd at laese efter i
    haanden. Et for lavt tal ville skjule netop de vaerste af dem.
    """
    r = maal_side("abc [?] def", "abc " + "x" * 30 + " def")

    assert r.joker_tegn == (32,)      # gabet: mellemrummene paa begge sider med
    assert r.joker_indhold == (30,)   # loftet: de 30 x'er alene
    assert r.tegnafstand == 30 - JOKER_LOFT


@pytest.mark.parametrize(
    "facit",
    [
        "foer [?] efter",   # det typiske: maerket har mellemrum paa begge sider
        "foer[?]efter",     # klaebet til nabo-ordene
        "foer [?]efter",    # mellemrum kun paa den ene side
    ],
)
@pytest.mark.parametrize(
    "indhold, forventet_pris",
    [
        (JOKER_LOFT - 1, 0),
        (JOKER_LOFT, 0),      # praecis paa loftet: gratis, uanset typografi
        (JOKER_LOFT + 1, 1),
        (JOKER_LOFT + 4, 4),
    ],
)
def test_loftet_taeller_indhold_ikke_mellemrum(facit, indhold, forventet_pris):
    """Loftet skal vaere de 15 tegn INDHOLD, aftalen lyder paa.

    `cer.normalize()` aeder blanktegnene i enderne af facits stykker, saa
    jokeren ogsaa maa sluge modellens mellemrum omkring gaettet. Taeller de med
    mod loftet, er det effektive loft 13 ved det typiske "foer [?] efter" og 15
    ved "foer[?]efter" -- altsaa afhaengigt af, om skriveren satte mellemrum
    omkring et ulaeseligt ord. Det er samme slags vilkaarlighed som den, der
    lige har kostet forankringen livet, blot i lille format.

    Modellen bygges af facit selv, saa den arver praecis facits egen typografi
    og skriver noejagtig `indhold` ikke-blanktegn paa det ulaeselige sted.
    """
    model = facit.replace(MAERKE, "x" * indhold)
    r = maal_side(facit, model)

    assert r.tegnafstand == forventet_pris
    assert r.joker_overskud == forventet_pris


def test_gabet_og_loftet_er_to_forskellige_regnskaber():
    """Gabet viser ALT, modellen skrev paa stedet -- mellemrum med. Loftet
    regner kun paa indholdet. De to tal maa derfor ikke vaere det samme.

    Gabet er arbejdsredskabet, der skal foere en laeser hen til stedet; loftet
    er straffen. At blande dem er netop det, der gjorde loftet typografi-
    afhaengigt.
    """
    r = maal_side("foer [?] efter", "foer " + "x" * JOKER_LOFT + " efter")

    assert r.joker_tegn == (JOKER_LOFT + 2,)     # med begge mellemrum
    assert r.joker_indhold == (JOKER_LOFT,)      # uden dem
    assert r.tegnafstand == 0
    assert r.joker_overskud == 0


# --------------------------------------------------------------------------
# Naevneren
# --------------------------------------------------------------------------

def test_maerket_taeller_ikke_med_i_naevneren():
    """`[?]` er ikke kendt sandhed. Hverken maerket eller det, modellen skrev
    paa stedet, maa vokse facits tegn- eller ordantal."""
    uden = maal_side("abc def", "abc def")
    med = maal_side("abc [?] def", "abc noget helt andet def")

    assert med.facit_tegn == 6  # kun "abc" + "def"
    assert med.facit_ord == 2
    assert med.facit_tegn < uden.facit_tegn  # mellemrummet foelger med maerket


def test_alle_seks_varianter_bevarer_jokeren():
    """Antagelsen der baerer hele modulet: `normalize()` med `ignore_punctuation`
    goer `[?]` til den tomme streng.

    Deles facit derfor foerst EFTER normalisering, staar tre af de seks varianter
    uden jokere overhovedet, og de samme sider ville faa vidt forskellige tal alt
    efter variant. Testen fejler, hvis delingen nogensinde flyttes bagom.
    """
    facit = "Tg. ikke [?] suspect."
    model = "Tg. ikke Roedt suspect."

    for navn in sorted(cer.VARIANTER):
        r = maal_side(facit, model, **cer.VARIANTER[navn])
        assert r.jokere == 1, navn
        assert r.tegnafstand == 0, navn
        assert r.joker_tegn == (7,), navn  # " Roedt " -- samme gab i alle seks


# --------------------------------------------------------------------------
# Ordtallet
# --------------------------------------------------------------------------

def test_ordmaalingen_har_sin_egen_joker():
    """Ordafstanden maales med samme loft, opgjort i tegn.

    Uden en joker i ord-DP'en ville modellens gaet paa det ulaeselige sted vaere
    et forkert ord, og et facit med mange `[?]` ville have kunstigt hoej WER.
    """
    facit = "en [?] mand"
    med_gaet = maal_side(facit, "en gammel mand")
    tomt = maal_side(facit, "en mand")

    assert med_gaet.facit_ord == 2
    assert med_gaet.ordafstand == 0
    assert tomt.ordafstand == 0

    # Et gaet paa mere end loftet i tegn koster derimod ord.
    langt = maal_side(facit, "en aldeles overordentlig gammel mand")
    assert langt.ordafstand > 0


def test_ordene_smelter_ikke_sammen_over_maerket():
    """Stykkerne klaebes uden mellemrum i tegnmaalingen, men ordlisten skal
    stadig have to ord -- ellers taeller "ikke" og "suspect" som ét."""
    r = maal_side("ikke [?] suspect", "ikke suspect")
    assert r.facit_ord == 2
    assert r.ordafstand == 0


# --------------------------------------------------------------------------
# Tomme sider
# --------------------------------------------------------------------------

def test_tom_facit_giver_ingen_division_med_nul():
    r = maal_side("", "modellen skrev noget alligevel")
    assert r.facit_tegn == 0
    assert r.cer == 0.0
    assert r.wer == 0.0
    assert r.tegnafstand == len("modellen skrev noget alligevel")


def test_tom_modeltekst_koster_hele_facit():
    facit = "en hel side [?] med tekst"
    r = maal_side(facit, "")
    assert r.tegnafstand == r.facit_tegn
    assert r.cer == 1.0
    assert r.joker_tegn == (0,)


def test_tom_modeltekst_mod_facit_der_kun_er_et_maerke():
    """Et facit, der udelukkende er ulaeseligt, har ingen kendt tekst. Baade
    taeller og naevner er nul, og raten skal vaere nul frem for at sprænge."""
    r = maal_side(MAERKE, "")
    assert r.facit_tegn == 0
    assert r.tegnafstand == 0
    assert r.cer == 0.0
    assert r.joker_tegn == (0,)


# --------------------------------------------------------------------------
# Trin 3 kraever to ting mere af sidemaalingen
# --------------------------------------------------------------------------
#
# 1) Gabet skal kunne SES, ikke kun taelles. Rod-CONTEXT 2026-08-21 binder os
#    til at skrive gabene i en fil, og en fil med tal i stedet for tekst kan
#    ikke foere en laeser hen til stedet. Det er samtidig arbejdslisten over
#    steder, facit maaske kan rettes (planen, 2026-08-30).
#
# 2) Ét maerke skal kunne have sit eget loft. Den strenge maaling (beslutning
#    44) udelader hele linjer med et `[?]`. Uden forankringen er den eneste vej
#    til det at saette en joker ind, hvor linjen stod -- og den joker skal
#    kunne sluge en hel linje, ikke 15 tegn. Faar den standardloftet, koster
#    hver udeladt linje ~25 tegn, og den strenge maaling ville se vaerre ud end
#    hovedtallet af rene bogholderi-grunde.


def test_gabet_giver_teksten_modellen_skrev_ikke_kun_dens_laengde():
    """Uden selve teksten er gab-filen ubrugelig som arbejdsliste."""
    maal = maal_side("Patienten har [?] i dag.", "Patienten har hoved pine i dag.")

    assert maal.joker_tekst == (" hoved pine ",)
    # De tre regnskaber skal stemme indbyrdes: alle tegn, indholdstegn, tekst.
    assert maal.joker_tegn == (len(maal.joker_tekst[0]),)
    assert maal.joker_indhold == (len("hovedpine"),)


def test_gabteksten_er_tom_naar_modellen_sprang_stedet_over():
    """En model, der intet skrev paa det ulaeselige sted, skal give en tom
    streng -- ikke mangle en post, saa gabene ikke laengere staar i samme
    raekkefoelge som maerkerne."""
    maal = maal_side("Patienten har [?] i dag.", "Patienten hari dag.")

    assert maal.joker_tekst == ("",)


def test_hvert_maerke_kan_faa_sit_eget_loft():
    """Den strenge maalings udeladte linje skal kunne sluges hel."""
    facit = "Foerste linje. [?] Sidste linje."
    # 30 indholdstegn paa det ulaeselige sted -- det dobbelte af standardloftet.
    model = "Foerste linje. " + "x" * 30 + " Sidste linje."

    med_standardloft = maal_side(facit, model)
    assert med_standardloft.tegnafstand == 30 - JOKER_LOFT

    med_eget_loft = maal_side(facit, model, lofter=[40])
    assert med_eget_loft.tegnafstand == 0


def test_lofter_skal_passe_til_antallet_af_maerker():
    """En forkert lang liste er en programmeringsfejl, ikke noget der stille
    skal fyldes op med standardloftet -- saa ville et maerke lydloest faa et
    andet loft end tiltaenkt."""
    with pytest.raises(ValueError):
        maal_side("En [?] to [?] tre", "En x to y tre", lofter=[40])
