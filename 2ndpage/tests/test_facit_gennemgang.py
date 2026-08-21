"""Tests for de fem fejl, gennemgangen 2026-08-21 fandt i facit-laeseren.

De staar for sig, saa det er tydeligt hvad der blev fanget hvornaar. Hver af
dem er set fejle, foer rettelsen blev skrevet.
"""

from andenside.facit import klassificer_klamme, ren_laesetekst

NUL = chr(0)


def test_uafsluttet_klamme_slaar_ikke_resten_af_siden_ud():
    """Den uafsluttede klamme holdt dybden over nul resten af blokken, saa ALLE
    senere maerker paa siden blev til raa tekst.

    Set i det leverede facit: `[added over line](Fibiger)` stod bogstaveligt i
    teksten paa 273104_001643, fordi det kom efter materialets ene aegte
    uafsluttede klamme.
    """
    raa = (
        "Naar der [ophentes? sekret\n"
        "ved Inspiration [added over line]under affekt [continued on line]middel"
    )
    tekst, noter = ren_laesetekst(raa)
    assert "[added over line]" not in tekst
    assert tekst.endswith("under affekt middel")
    assert any("uafsluttet" in n.lower() for n in noter)


def test_overstregning_efter_en_uafsluttet_klamme_virker_stadig():
    """Vaerst af alt: en aegte overstregning EFTER det uafsluttede sted blev
    aldrig genkendt, saa den overstregede tekst laekkede uaendret ind i den
    rettede udgave af facit -- netop den udgave, der skal have den fjernet."""
    raa = (
        "Der [ophentes? noget\n"
        "hostet en Del. [crossed out]en fejlnotering [written instead]Lidt at Drikke"
    )
    rettet, _ = ren_laesetekst(raa)
    assert "fejlnotering" not in rettet
    assert rettet.endswith("hostet en Del. Lidt at Drikke")


def test_flere_uafsluttede_klammer_repareres_hver_for_sig():
    raa = "Der [ophentes? noget og [flere? ting [this line is underlined]"
    tekst, noter = ren_laesetekst(raa)
    assert tekst == "Der ophentes noget og flere ting"
    assert len([n for n in noter if "uafsluttet" in n.lower()]) == 2


def test_nultegn_i_kilden_vaelter_ikke_koerslen():
    """Laeseren bruger selv et nultegn som usynligt maerke, naar den holder styr
    paa understregningers linjenummer. Staar der ét i kilden, maa det ikke give
    et raat nedbrud midt i en koersel over 39 filer."""
    tekst, _ = ren_laesetekst("noget" + NUL + "andet [this line is underlined]")
    assert NUL not in tekst
    assert tekst == "nogetandet"


def test_tomme_linjer_slaas_sammen_ogsaa_omkring_en_understregningsnote():
    """Oprydningen af tomme linjer koerte FOER det usynlige maerke blev fjernet,
    saa maerket forhindrede den i at se hele loebet af nylinjer under ét."""
    tekst, _ = ren_laesetekst("foerste\n\n\n[this line is underlined]\n\n\nanden")
    assert "\n\n\n" not in tekst


def test_ord_der_blot_indeholder_page_er_ikke_et_positionsmaerke():
    r"""Ordgraensen manglede om "page" i positionsmoenstret, saa et laeseforslag
    med den bogstavfoelge forsvandt sporloest i stedet for at blive til tekst."""
    assert klassificer_klamme("spaget?")[0] == "gaet"
    tekst, _ = ren_laesetekst("Fra [spaget?] Saar")
    assert tekst == "Fra spaget Saar"


def test_positionsmaerker_med_page_som_helt_ord_genkendes_stadig():
    """Ordgraensen maa ikke koste os de rigtige maerker."""
    for form in ("right side of page", "mid page", "top page left", "midpage"):
        assert klassificer_klamme(form)[0] == "position", form
