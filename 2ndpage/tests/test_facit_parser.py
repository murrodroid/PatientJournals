"""Tests for opmaerkningen i facit: sideblokke, klammer og ren laesetekst.

Hvert tilfaelde svarer til en konvention, der faktisk optraeder i de 39
transskriptionsfiler -- ogsaa tastefejlene. De er talt op i
`stages/02_facit/output/klammekonventioner.md`.
"""

import pytest

from andenside.facit import (
    del_i_sideblokke,
    klassificer_klamme,
    understregninger,
    ren_laesetekst,
    saml_orddeling,
)

BS = chr(92)


# --- opdeling i sideblokke -------------------------------------------------


def test_sidemarkoer_med_og_uden_ordet_page_genkendes_ens():
    """Filen 273104_001636 skriver `[273104_001637]` uden ordet 'page'."""
    tekst = (
        "[transcription of frontpage 273104_001636 - full journal]\n"
        "[273104_001637]\nBarnet indlaegges\n"
        "[page 273104_001638]\n20/10 Har sovet\n"
    )
    blokke = del_i_sideblokke(tekst, kildefil="x.rtf")
    assert [b.image_name for b in blokke] == ["273104_001637", "273104_001638"]
    assert blokke[0].forside == "273104_001636"


def test_forsidens_egen_tekst_er_ikke_en_sideblok():
    """Alt foer foerste sidemarkoer hoerer til forsiden, som er transskriberet
    andetsteds og ikke er vores maalside."""
    tekst = "[transcription of frontpage 273098_001471 - full journal]\nnoget\n[page 273098_001472]\nandet\n"
    blokke = del_i_sideblokke(tekst, kildefil="x.rtf")
    assert len(blokke) == 1
    assert blokke[0].raa.strip() == "andet"


def test_femcifret_bind_id_genkendes():
    """Bind 37554 har fem cifre, ikke seks -- en snaever regex taber to filer."""
    tekst = "[transcription of frontpage 37554_001412 - full journal]\n[page 37554_001413]\nHar haft Morbilli\n"
    blokke = del_i_sideblokke(tekst, kildefil="x.rtf")
    assert [b.image_name for b in blokke] == ["37554_001413"]


def test_blok_uden_tekst_markeres_som_tom():
    tekst = "[transcription of frontpage 1_1 - full journal]\n[page 1_2]\nnoget\n[page 1_3]\n\n"
    blokke = del_i_sideblokke(tekst, kildefil="x.rtf")
    assert [b.tom for b in blokke] == [False, True]


# --- ulaeselighed og gaet --------------------------------------------------


def test_ulaeselighedsmaerker_bevares_ogsaa_naar_de_er_stablet():
    tekst, _ = ren_laesetekst("og Canylen [?][?] i Trachea.")
    assert tekst == "og Canylen [?][?] i Trachea."


def test_gaet_reduceres_til_selve_ordet():
    tekst, _ = ren_laesetekst("Stemmen [dygtig?] haes.")
    assert tekst == "Stemmen dygtig haes."


def test_gaet_uden_spoergsmaalstegn_regnes_ogsaa_som_gaet():
    tekst, _ = ren_laesetekst("Fra [hudsaaret] Saar")
    assert tekst == "Fra hudsaaret Saar"


def test_gaet_med_punktum_pladsholder_bliver_et_ulaeselighedsmaerke():
    tekst, _ = ren_laesetekst("Syg i 6 Dage, [..rede?] meget")
    assert tekst == "Syg i 6 Dage, [?] meget"


def test_klamme_med_ellipse_kan_heller_ikke_maales_paa():
    tekst, _ = ren_laesetekst("Tg. naturlig, paa [Oere...] og i Ansigtet")
    assert tekst == "Tg. naturlig, paa [?] og i Ansigtet"


def test_flerords_klamme_vi_ikke_kender_flages_stadig():
    tekst, noter = ren_laesetekst("noget [scrawled sideways in margin] mere")
    assert "scrawled sideways in margin" in tekst
    assert any("ukendt" in n.lower() for n in noter)


# --- understregning --------------------------------------------------------


@pytest.mark.parametrize(
    "note",
    [
        "[this line is underlined]",
        "[this lines is underlined]",
        "[»gullig Belægning« is underlined]",
        "[»ning« is underline]",
        "[»Velnært. Barn« and »spredte« is underlined]",
    ],
)
def test_understregningsnoter_fjernes_helt(note):
    """Noten omtaler tekst, der allerede staar paa linjen -- den er ikke tekst."""
    tekst, _ = ren_laesetekst(f"Rask indtil for 8 Dage siden {note}\nda hun fik Brystkatarrh.")
    assert tekst == "Rask indtil for 8 Dage siden\nda hun fik Brystkatarrh."


def test_understregningsnote_med_ulaeselighed_indeni_fjernes_helt():
    """Indlejrede klammer: `[»[?] Membraner.« is underlined]` er ÉN note."""
    tekst, _ = ren_laesetekst("[?] Membraner. Der er [»[?] Membraner.« is underlined]\nrigelig Snue")
    assert tekst == "[?] Membraner. Der er\nrigelig Snue"


# --- overstregning ---------------------------------------------------------


def test_overstreget_med_erstatning_beholder_kun_erstatningen():
    tekst, _ = ren_laesetekst("hostet en Del. [crossed out]En Del [written instead]Lidt at Drikke")
    assert tekst == "hostet en Del. Lidt at Drikke"


def test_overstreget_uden_mellemrum_i_maerket_virker_ogsaa():
    """Tastefejlen `[crossedout]` optraeder to gange i materialet."""
    tekst, _ = ren_laesetekst("synes at være [crossedout]på [continued on line]af ældre Dato")
    assert tekst == "synes at være af ældre Dato"


def test_overstregning_uden_afslutning_loeber_til_linjeskift_og_ikke_laengere():
    """`[crossed out]Resp c.` staar sidst paa linjen; naeste linje er ikke overstreget."""
    tekst, _ = ren_laesetekst("Puls c. 140, lille. [crossed out]Resp c.\nI fauces middelstørke")
    assert tekst == "Puls c. 140, lille.\nI fauces middelstørke"


def test_overstregning_stopper_ved_haandskrevet_linjeskift():
    r"""Recepter bruger et skrevet `\n` som linjeskift; kun ét led er streget ud."""
    kilde = "Sol. chlor. kalici." + BS + "n[crossed out]mixt. hydrarg." + BS + "nmixt. camphorat."
    tekst, _ = ren_laesetekst(kilde)
    assert tekst == "Sol. chlor. kalici.\nmixt. camphorat."


def test_to_overstregninger_paa_hver_sin_linje_holdes_adskilt():
    kilde = "I Fauces ses [crossed out]på h.\n[crossed out]Side [continued on line]misfarvede Belægn."
    tekst, _ = ren_laesetekst(kilde)
    assert tekst == "I Fauces ses\nmisfarvede Belægn."


def test_overstreget_ulaeseligt_forsvinder_ogsaa():
    """`[crossed out][?]` maa ikke efterlade et ulaeselighedsmaerke i facit."""
    tekst, _ = ren_laesetekst("Ret stærkt [crossed out][?] [continued on line]Tp. 38.3")
    assert tekst == "Ret stærkt Tp. 38.3"


# --- indskud og margen -----------------------------------------------------


def test_indskud_over_linjen_beholdes_som_tekst_uden_maerker():
    """Ordene ER skrevet paa siden -- en model, der laeser siden, ser dem."""
    kilde = "ved Inspiration [added over line]under affekt [continued on line]middelstærke Ind-"
    tekst, _ = ren_laesetekst(kilde)
    assert tekst == "ved Inspiration under affekt middelstærke Ind-"


@pytest.mark.parametrize(
    "maerke",
    ["[added under line]", "[added on top of line]", "[addet on top of line]", "[added overline]", "[added between lines]"],
)
def test_alle_indskudsvarianter_fjernes_som_maerke(maerke):
    tekst, _ = ren_laesetekst(f"Mixt camphorata {maerke}hvorfra\nder ord. i aftes")
    assert tekst == "Mixt camphorata hvorfra\nder ord. i aftes"


def test_margentekst_faar_sin_egen_linje():
    """Uden linjeskift ville margennoten klistre sig til journalteksten."""
    tekst, _ = ren_laesetekst("Ingen Opkastninger. [right side of page]croupal Hoste")
    assert tekst == "Ingen Opkastninger.\ncroupal Hoste"


@pytest.mark.parametrize(
    "maerke",
    ["[right side og page]", "[midpage]", "[bottom right corner]", "[top page left]", "[note added right side page]"],
)
def test_positionsmaerker_i_alle_stavemaader_fjernes(maerke):
    tekst, _ = ren_laesetekst(f"noget\n{maerke}margentekst")
    assert tekst == "noget\nmargentekst"


def test_haandskrevet_linjeskift_bliver_et_rigtigt_linjeskift():
    kilde = "[right side of page]Rp. Damp" + BS + "nLincet. expect."
    tekst, _ = ren_laesetekst("Journaltekst " + kilde)
    assert tekst == "Journaltekst\nRp. Damp\nLincet. expect."


# --- ukendt og defekt opmaerkning ------------------------------------------


def test_uafsluttet_klamme_lukkes_ved_naeste_mellemrum():
    """Tastefejlen `[ophentes? sekret ...` mangler kun sin slutklamme. Vi
    reparerer mindst muligt: klammen lukkes ved foerste mellemrum, og
    indholdet behandles som ethvert andet laeseforslag. En rigtig klamme kan
    godt spaende over et linjeskift, saa vi roerer kun dem, der ALDRIG lukkes."""
    tekst, noter = ren_laesetekst("Naar der\n[ophentes? sekret af caviteten")
    assert tekst == "Naar der\nophentes sekret af caviteten"
    assert any("uafsluttet" in n.lower() for n in noter)


def test_laegens_egne_spoergsmaalstegn_roeres_ikke():
    """`(Scarlatina?)` og `Pneunomia?` er skrevet af LAEGEN paa siden, ikke
    af transskribenten. De er tekst og skal med i facit -- opmaerkningens
    spoergsmaalstegn fjernes kun inde i en klamme."""
    tekst, _ = ren_laesetekst("Ingen Exanth. (Scarlatina?) - Pneunomia? DB?")
    assert tekst == "Ingen Exanth. (Scarlatina?) - Pneunomia? DB?"

def test_overskydende_slutklamme_flages_og_fjernes():
    tekst, noter = ren_laesetekst("Belægningerne er udbredte, mis] far-")
    assert tekst == "Belægningerne er udbredte, mis far-"
    assert any("overskydende" in n.lower() for n in noter)


# --- orddeling -------------------------------------------------------------


def test_orddeling_samles_hen_over_linjeskift():
    assert saml_orddeling("Kvælnings-\nanfald, som synes") == "Kvælningsanfald, som synes"


def test_linjeskift_uden_bindestreg_bliver_til_mellemrum():
    assert saml_orddeling("Hun har hostet\nen Del Slim") == "Hun har hostet en Del Slim"


def test_bindestreg_med_efterfoelgende_mellemrum_samles_ogsaa():
    """Transskriptionerne har ofte et mellemrum efter bindestregen ved linjeslut."""
    assert saml_orddeling("Belæg- \nninger") == "Belægninger"


def test_tankestreg_mellem_ord_er_ikke_orddeling():
    """`- ` midt paa linjen er tegnsaetning; kun bindestreg SIDST paa linjen deler."""
    assert saml_orddeling("Stærk foetor oris - stærke belagt\nTunge") == (
        "Stærk foetor oris - stærke belagt Tunge"
    )


def test_tom_linje_bliver_ikke_til_dobbelt_mellemrum():
    assert saml_orddeling("første\n\nanden") == "første anden"


def test_ulaeselighedsmaerke_hen_over_linjeskift_samles_ikke_fejlagtigt():
    assert saml_orddeling("Stadig [?]-\nlig ringe") == "Stadig [?]-lig ringe"


def test_bindestreg_foran_stort_begyndelsesbogstav_er_tegnsaetning_ikke_orddeling():
    """Materialet bruger `-` som punktum: "enkelte Rhonchi-" efterfulgt af ny
    saetning. Danske orddelinger fortsaetter derimod med lille bogstav."""
    assert saml_orddeling("enkelte Rhonchi-\nIngen Snue") == "enkelte Rhonchi- Ingen Snue"


def test_orddeling_med_lille_begyndelsesbogstav_samles_stadig():
    assert saml_orddeling("Expira-\ntionerne forcerede") == "Expirationerne forcerede"


def test_flere_tomme_linjer_i_traek_bliver_til_en():
    """Positionsmaerker i margenen skabte tre blanke linjer i traek."""
    tekst, _ = ren_laesetekst("Indkommen kl. 17\n[bottom left side] Rp.\n[bottom right side] Sol.")
    assert tekst == "Indkommen kl. 17\nRp.\nSol."


def test_indledende_mellemrum_efter_fjernet_maerke_falder_vaek():
    tekst, _ = ren_laesetekst("noget\n[right side of page] Rp. Damp")
    assert tekst == "noget\nRp. Damp"


# --- alt-hvad-der-staar-udgaven --------------------------------------------


def test_overstreget_bliver_staaende_naar_vi_beder_om_alt_der_staar():
    """Modellen promptes til at laese hele siden, ogsaa det overstregede.
    Maaler vi mod en tekst uden det, straffer vi den for at laese rigtigt."""
    tekst, _ = ren_laesetekst(
        "hostet en Del. [crossed out]En Del [written instead]Lidt at Drikke",
        behold_overstreget=True,
    )
    assert tekst == "hostet en Del. En Del Lidt at Drikke"


def test_overstreget_uden_erstatning_bliver_ogsaa_staaende():
    tekst, _ = ren_laesetekst(
        "Puls c. 140, lille. [crossed out]Resp c.\nI fauces", behold_overstreget=True
    )
    assert tekst == "Puls c. 140, lille. Resp c.\nI fauces"


def test_overstreget_ulaeseligt_bliver_til_et_ulaeselighedsmaerke():
    """`[crossed out][?]` er et sted, transskribenten hverken kunne laese
    eller ville beholde. Laeser modellen hele siden, ER der noget dér."""
    tekst, _ = ren_laesetekst(
        "Ret stærkt [crossed out][?] [continued on line]Tp. 38.3", behold_overstreget=True
    )
    assert tekst == "Ret stærkt [?] Tp. 38.3"


def test_alt_udgaven_roerer_ikke_de_oevrige_maerker():
    """Kun overstregningen behandles anderledes; understregningsnoter og
    positionsmaerker falder stadig vaek."""
    tekst, _ = ren_laesetekst(
        "Rask indtil [this line is underlined]\n[right side of page]Rp. Damp",
        behold_overstreget=True,
    )
    assert tekst == "Rask indtil\nRp. Damp"


# --- tastefejl i opmaerkningen ---------------------------------------------


@pytest.mark.parametrize(
    "form,forventet",
    [
        # Samtlige afvigende stavemaader, optaellingen af alle 39 filer fandt.
        # Fanges én af dem ikke, ender et maerke som almindelig tekst midt i
        # facit, uden at nogen opdager det.
        ("crossed out", "overstreget"),
        ("crossedout", "overstreget"),
        ("continued on line", "fortsaet"),
        ("continuded on line", "fortsaet"),
        ("continued under line", "fortsaet"),
        ("added over line", "indskud"),
        ("added under line", "indskud"),
        ("added on top of line", "indskud"),
        ("addet on top of line", "indskud"),
        ("added overline", "indskud"),
        ("added between lines", "indskud"),
        ("note added between main lines", "indskud"),
        ("this line is underlined", "understregning"),
        ("this lines is underlined", "understregning"),
        ("»ning« is underline", "understregning"),
        ("right side of page", "position"),
        ("right side og page", "position"),
        ("left side og page", "position"),
        ("right side page", "position"),
        ("note added right side page", "position"),
        ("mid page", "position"),
        ("midpage", "position"),
        ("top right corner", "position"),
        ("right top corner", "position"),
        ("bottom right corner", "position"),
        ("top page left", "position"),
        ("top page right", "position"),
        ("bottom left side", "position"),
        ("bottom right side", "position"),
        ("left top corner", "position"),
        ("left side of page", "position"),
    ],
)
def test_alle_stavemaader_i_materialet_fanges_af_moenstrene(form, forventet):
    assert klassificer_klamme(form)[0] == forventet


# --- understregning gemmes for sig -----------------------------------------


def test_hel_linje_understregning_gemmes_med_sit_linjenummer():
    raa = "foerste linje [this line is underlined]\nanden linje\ntredje [this line is underlined]"
    assert understregninger(raa) == [
        {"linje": 0, "slags": "hel_linje", "tekst": ""},
        {"linje": 2, "slags": "hel_linje", "tekst": ""},
    ]


def test_citat_understregning_gemmer_selve_citatet():
    raa = "da hun fik [»gullig Belægning« is underlined] noget"
    assert understregninger(raa) == [
        {"linje": 0, "slags": "citat", "tekst": "gullig Belægning"}
    ]


def test_noten_om_understregning_taeller_ikke_selv_som_citat():
    """Ordene "is underlined" staar UDEN for citattegnene og er ikke tekst,
    der er understreget paa siden."""
    fund = understregninger("noget [»urin« is underlined]")
    assert [f["tekst"] for f in fund] == ["urin"]


def test_to_citater_i_samme_note_gemmes_hver_for_sig():
    raa = "noget [»Velnært. Barn« and »spredte« is underlined]"
    assert [f["tekst"] for f in understregninger(raa)] == ["Velnært. Barn", "spredte"]


def test_understregning_i_margentekst_faar_margenens_linjenummer():
    """Positionsmaerket laegger en ny linje ind; understregningen bagefter
    hoerer til DEN linje, ikke til journalteksten foer den."""
    raa = "journaltekst [right side of page]Rp. Damp [this line is underlined]"
    assert understregninger(raa) == [{"linje": 1, "slags": "hel_linje", "tekst": ""}]


def test_linjenumrene_passer_paa_laesetekstens_egne_linjer():
    """Kontrakten mellem de to: `linje` skal kunne bruges som indeks i den
    linjeopdelte laesetekst. Ellers peger understregningen paa noget andet."""
    raa = "en [this line is underlined]\nto\ntre [»tre« is underlined]"
    tekst, _ = ren_laesetekst(raa, behold_overstreget=True)
    linjer = tekst.split("\n")
    for fund in understregninger(raa):
        assert 0 <= fund["linje"] < len(linjer)
    assert linjer[understregninger(raa)[1]["linje"]] == "tre"


def test_citat_findes_paa_en_tidligere_linje_naar_noten_staar_forsinket():
    """Transskribenten satte nogle gange noten efter den NAESTE linje. Citatet
    er sidens egen tekst, saa vi kan finde den rigtige linje ved at lede efter
    det -- ellers peger oplysningen paa en linje, der intet har med den at goere."""
    raa = "Urinen: middelm.\nSublim. Guajac. [»Urinen« is underlined]"
    assert understregninger(raa) == [
        {"linje": 0, "slags": "citat", "tekst": "Urinen"}
    ]


def test_citat_med_anden_brug_af_store_bogstaver_findes_stadig():
    raa = "Der er tykke gullige Belæg- [»Tykke gullige Belæg« is underlined]"
    assert understregninger(raa)[0]["linje"] == 0


def test_citat_der_slet_ikke_findes_bliver_paa_notens_egen_linje():
    """Vi flytter kun noten, naar vi faktisk finder teksten. Ellers gaetter vi."""
    raa = "foerste\nanden [»noget helt tredje« is underlined]"
    assert understregninger(raa)[0]["linje"] == 1
