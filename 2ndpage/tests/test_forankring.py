"""Testkontrakt for `forankr()` -- stage 03's ene funktion.

Reglerne, der testes her, er laast i rod-CONTEXT.md 2026-08-21 ("De fire
forslag skaeres ned til ÉN funktion"):

- Stumper under 5 tegn bruges ikke til forankring.
- Et gab taelles kun, naar stumperne paa BEGGE sider er fundet.
- En stump, der ikke findes, er ikke en fejl -- linjen falder tilbage til
  beslutning 38 og gaar helt ud af maalingen.

Dertil den afgoerelse, der ikke stod skrevet, men foelger af, at stumperne
skal MAALES: soegningen maa taale laesefejl inde i stumpen. Kraevede den
ordret traef, ville hver forankret stump per definition have nul fejl, og
maalingen ville vaere selvbekraeftende. Se `maal.py`.
"""
from __future__ import annotations

import pytest

from andenside.maal import forankr


def test_ren_linje_uden_maerke_forankres_som_en_stump():
    """En linje uden [?] er ikke et saertilfaelde -- den er én stump."""
    fund = forankr("Rask indtil for 8 Dage siden", "Rask indtil for 8 Dage siden da hun")
    assert len(fund.stumper) == 1
    assert fund.stumper[0].fundet
    assert fund.stumper[0].model_tekst == "Rask indtil for 8 Dage siden"
    assert fund.gab == ()
    assert fund.forankret


def test_gab_mellem_to_fundne_stumper_er_modellens_bud():
    """Det, modellen skrev, hvor facit siger [?], er baade hallucinations-
    signal og modellens laesning af stedet. Det er samme gab."""
    facit = "væg, men denne var [?], og Canylen"
    model = "væg, men denne var tynd, og Canylen blev"
    fund = forankr(facit, model)

    assert [s.tekst for s in fund.stumper] == ["væg, men denne var", ", og Canylen"]
    assert all(s.fundet for s in fund.stumper)
    assert len(fund.gab) == 1
    assert fund.gab[0].model_tekst.strip() == "tynd"
    assert fund.gab[0].facit_mellem == "[?]"


def test_stump_findes_selvom_modellen_laeser_et_tegn_forkert():
    """Forankringen skal taale fejl -- ellers maaler den kun der, hvor
    modellen allerede var perfekt, og tallet bliver kunstigt pænt."""
    fund = forankr("Ingen tidligere Infektions-", "Ingen tidligere Infoktions- sygdomme")
    assert fund.stumper[0].fundet
    assert fund.stumper[0].model_tekst == "Ingen tidligere Infoktions-"


def test_stump_der_slet_ikke_findes_er_ikke_fundet():
    """En stump, der ikke findes, er ikke en fejl -- den er uforankret."""
    fund = forankr("Angularglandlerne middelstore", "Patienten kom ind om aftenen")
    assert not fund.stumper[0].fundet
    assert not fund.forankret
    assert fund.gab == ()


def test_stump_under_mindstelaengden_bruges_ikke():
    """122 af de 647 stumper er under 5 tegn. De kan forankre hvor som helst
    og ville give falsk tryghed."""
    facit = "[?] og [?] Belægning i Halsen"
    fund = forankr(facit, "der var noget og lidt Belægning i Halsen")

    korte = [s for s in fund.stumper if s.for_kort]
    assert [s.tekst for s in korte] == ["og"]
    assert all(not s.fundet for s in korte)


def test_gab_taelles_kun_naar_begge_sider_er_fundet():
    """Er hoejre stump ikke fundet, ved vi ikke hvor gabet slutter."""
    facit = "Der er rigelig [?] Udflod af en helt egen art"
    model = "Der er rigelig blodig"          # hoejre stump mangler helt
    fund = forankr(facit, model)

    assert fund.stumper[0].fundet
    assert not fund.stumper[1].fundet
    assert fund.gab == ()


def test_maerke_forrest_paa_linjen_giver_intet_gab():
    """Uden en stump til venstre er der ingen kendt begyndelse paa gabet."""
    fund = forankr("[?] i Trachea og videre ned", "noget uklart i Trachea og videre ned")
    assert fund.gab == ()
    assert fund.stumper[-1].fundet


def test_stumper_forankres_i_raekkefoelge():
    """Samme ord to gange i modelteksten maa ikke faa anden stump til at
    forankre foer den foerste -- saa ville gabet blive negativt."""
    facit = "Hun har hostet [?] Hun har hostet en Del"
    model = "Hun har hostet meget Hun har hostet en Del Slim"
    fund = forankr(facit, model)

    assert fund.stumper[0].fundet and fund.stumper[1].fundet
    assert fund.stumper[0].model_slut <= fund.stumper[1].model_start


def test_tom_modeltekst_forankrer_intet():
    fund = forankr("Barnet er kraftigt, velnæret", "")
    assert not fund.forankret
    assert fund.gab == ()


def test_facit_linje_uden_kendt_tekst_giver_ingen_stumper():
    """En linje der kun er ulaeselighedsmaerker har intet at maale paa."""
    fund = forankr("[?][?]", "hvad som helst")
    assert fund.stumper == ()
    assert not fund.forankret


@pytest.mark.parametrize("fra", [0, 5, 40])
def test_soegning_starter_hvor_kalderen_siger(fra: int):
    """Sidens linjer forankres fortloebende; en linje maa ikke kunne finde
    sit traef foer den forrige linjes."""
    model = "Belægn. i Halsen. Hun har hostet en Del men først til Morgen"
    fund = forankr("Belægn. i Halsen.", model, fra=fra)
    if fra == 0:
        assert fund.stumper[0].model_start == 0
    else:
        assert fund.stumper[0].model_start >= fra or not fund.stumper[0].fundet


# --------------------------------------------------------------------------
# Selve soegningen, holdt op mod en raa gennemsoegning af ALLE udsnit
# --------------------------------------------------------------------------

def test_naermeste_udsnit_finder_den_samme_afstand_som_en_raa_gennemsoegning():
    """`_naermeste_udsnit` er en dynamisk programmering med fri begyndelse og
    slutning -- den slags er let at faa halvvejs rigtig. Her proeves den mod
    den dumme, aabenlyst korrekte metode: proev hvert eneste udsnit.

    Koeres paa smaa, konstruerede strenge, fordi den raa metode er kubisk.
    """
    import random

    from andenside.cer import levenshtein
    from andenside.maal import _naermeste_udsnit

    rng = random.Random(4711)
    bogstaver = "abcde "
    for _ in range(120):
        naal = "".join(rng.choice(bogstaver) for _ in range(rng.randint(1, 6)))
        hoestak = "".join(rng.choice(bogstaver) for _ in range(rng.randint(0, 14)))

        _, _, afstand = _naermeste_udsnit(naal, hoestak)
        raa = min(
            levenshtein(naal, hoestak[i:j])
            for i in range(len(hoestak) + 1)
            for j in range(i, len(hoestak) + 1)
        )
        assert afstand == raa, (naal, hoestak, afstand, raa)


def test_naermeste_udsnit_peger_paa_et_udsnit_der_faktisk_har_den_afstand():
    """Afstanden kan vaere rigtig, mens start og slut peger et forkert sted
    hen -- og saa maales der paa den forkerte tekst."""
    import random

    from andenside.cer import levenshtein
    from andenside.maal import _naermeste_udsnit

    rng = random.Random(1848)
    bogstaver = "abcde "
    for _ in range(120):
        naal = "".join(rng.choice(bogstaver) for _ in range(rng.randint(1, 6)))
        hoestak = "".join(rng.choice(bogstaver) for _ in range(rng.randint(0, 14)))

        start, slut, afstand = _naermeste_udsnit(naal, hoestak)
        assert 0 <= start <= slut <= len(hoestak)
        assert levenshtein(naal, hoestak[start:slut]) == afstand, (naal, hoestak)


def test_samme_input_giver_samme_udsnit_hver_gang():
    """Ved lige billige veje skal valget vaere fast. Ellers kan to koersler
    af uaendret kode give hver sin rapport."""
    from andenside.maal import _naermeste_udsnit

    naal, hoestak = "abab", "xababababy"
    foerste = _naermeste_udsnit(naal, hoestak)
    for _ in range(5):
        assert _naermeste_udsnit(naal, hoestak) == foerste


def test_ved_flere_lige_gode_traef_vaelges_det_foerste():
    """Ikke pynt. Forankringen gaar fra venstre mod hoejre, og et gab er
    teksten MELLEM to traef -- vaelges et senere, lige saa godt traef, aeder
    stumpen ind i gabet ved siden af, og modellens bud paa et ulaeseligt sted
    bliver maalt som kendt tekst."""
    from andenside.maal import _naermeste_udsnit

    assert _naermeste_udsnit("kat", "en kat kat her") == (3, 6, 0)
    assert _naermeste_udsnit("hund", "hundhund") == (0, 4, 0)
    assert _naermeste_udsnit("abab", "xababababy") == (1, 5, 0)
