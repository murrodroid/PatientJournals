"""Tests for den baandvise beskaering -- snittet foelger falsen i stedet for
at vaere én lodret linje.

Hvorfor det er noedvendigt: siden krummer ind mod falsen, og skriveren skrev
helt ud. Et lodret snit skal derfor enten skaere tekst af i den ene ende af
siden eller tage naboopslaget med i den anden. Maalt paa 273104_001639 er de
nederste linjer skaaret over, mens de oeverste er rene.

Snittet foelger FALSEN, ikke teksten. Falsen er en fysisk graense og ligger
stille; tekstens yderkant hopper fra linje til linje og ville give et
urolig snit.

Den baerende test er `test_ingen_tekst_tabes_ved_en_skraa_fals` -- den kan
ikke bestaas af et lodret snit, uanset hvor det laegges.
"""

import pytest
from PIL import Image

from andenside.skraa import baandkanter, beskaer_langs_fals, fals_graense
from andenside.masterlist import Side

HVID = 255
SORT = 0


def _side(counter: int) -> Side:
    return Side(
        image_name="prøve",
        folder_name="273098",
        page_type="journal page",
        month="05",
        year="1896",
        patient_page_counter=counter,
        group_id="1",
    )


def _skraa_fals(
    bredde: int = 1000, hoejde: int = 1200, top_x: int = 880, bund_x: int = 800,
    tykkelse: int = 40,
) -> Image.Image:
    """Billede med en fals, der HAELDER fra `top_x` foroven til `bund_x` forneden."""
    img = Image.new("L", (bredde, hoejde), HVID)
    px = img.load()
    for y in range(hoejde):
        x0 = int(top_x + (bund_x - top_x) * y / hoejde)
        for x in range(x0, min(x0 + tykkelse, bredde)):
            px[x, y] = SORT
    return img


def _saet_markoer(img: Image.Image, x: int, y: int, hoejde: int = 24) -> None:
    """En kraftig lodret streg -- 'et ord, der naar helt ud til falsen'."""
    px = img.load()
    for dy in range(hoejde):
        for dx in range(3):
            px[x + dx, y + dy] = SORT


def _markoer_findes(img: Image.Image, forventet_x: int, y: int,
                    hoejde: int = 24) -> bool:
    """Er markoeren ved (forventet_x, y) stadig i billedet?

    Der kigges KUN i markoerens egne raekker. Kiggede vi i hele kolonnens
    hoejde, ville vi finde falsen et andet sted paa siden og tro, det var
    markoeren -- falsen beholdes med vilje delvist, fordi bufferen flytter
    snittet ind i den.
    """
    px = img.load()
    for x in range(max(0, forventet_x - 6), min(img.width, forventet_x + 9)):
        moerke = sum(1 for dy in range(hoejde)
                     if 0 <= y + dy < img.height and px[x, y + dy] < 100)
        if moerke >= hoejde * 0.6:
            return True
    return False


# ----------------------------------------------------------------- kanterne

def test_baandkanterne_foelger_haeldningen():
    img = _skraa_fals(top_x=880, bund_x=800)
    kanter = baandkanter(img, _side(1), antal=8)
    fundne = [x for _, x in kanter if x is not None]
    assert len(fundne) >= 6, "falsen blev ikke fundet i de fleste baand"
    # Foroven skal kanten ligge laengere til hoejre end forneden.
    assert fundne[0] > fundne[-1] + 40, (
        f"kanterne foelger ikke haeldningen: {fundne[0]} -> {fundne[-1]}"
    )


def test_en_lodret_fals_giver_lodrette_kanter():
    img = _skraa_fals(top_x=850, bund_x=850)
    fundne = [x for _, x in baandkanter(img, _side(1), antal=8) if x is not None]
    assert max(fundne) - min(fundne) <= 12, (
        f"en lodret fals gav spredte kanter: {fundne}"
    )


def test_baand_uden_troevaerdig_kant_udfyldes_fra_naboerne():
    """Et baand uden fals (fx et blankt omraade) maa ikke give et hul.

    Ellers ville graensen springe ind over siden netop dér.
    """
    img = _skraa_fals(top_x=880, bund_x=800)
    px = img.load()
    for y in range(500, 650):                 # slet falsen i ét baand
        for x in range(700, 1000):
            px[x, y] = HVID
    graense = fals_graense(img, _side(1), antal=8)
    assert len(graense) == img.height
    # Ingen raekke maa ligge markant inde paa siden i forhold til naboraekker.
    for y in range(1, img.height):
        assert abs(graense[y] - graense[y - 1]) < 20


# --------------------------------------------------------- selve beskæringen

def test_ingen_tekst_tabes_ved_en_skraa_fals():
    """Den baerende test: et lodret snit KAN ikke bestaa den.

    To markoerer placeres lige inden for falsen -- én foroven ved x=870, én
    forneden ved x=790. Et lodret snit maa enten miste den nederste (snit ved
    870) eller tage fals med foroven (snit ved 790). Et baandvist snit
    beholder begge.
    """
    img = _skraa_fals(top_x=880, bund_x=800)
    _saet_markoer(img, 866, 40)      # oeverst, taet paa falsen
    _saet_markoer(img, 786, 1140)    # nederst, taet paa falsen

    beskaaret, _ = beskaer_langs_fals(img, _side(1))
    assert _markoer_findes(beskaaret, 866, 40), "den OEVERSTE markoer gik tabt"
    assert _markoer_findes(beskaaret, 786, 1140), "den NEDERSTE markoer gik tabt"


def test_naboens_tekst_kommer_ikke_med():
    """Markoerne ligger KLAR af falsen (som er 40 px tyk).

    Ligger de inde i falsen selv, maaler testen ikke naboens tekst men
    skyggen -- og den beholdes med vilje delvist, fordi bufferen flytter
    snittet ind i den, vaek fra vores egen tekst.
    """
    img = _skraa_fals(top_x=880, bund_x=800, tykkelse=40)
    _saet_markoer(img, 975, 40)      # naboens side foroven (falsen: 880-920)
    _saet_markoer(img, 895, 1140)    # naboens side forneden (falsen: 800-840)
    beskaaret, _ = beskaer_langs_fals(img, _side(1))
    assert not _markoer_findes(beskaaret, 975, 40), "naboens tekst kom med foroven"
    assert not _markoer_findes(beskaaret, 895, 1140), "naboens tekst kom med forneden"


def test_recto_beskaeres_den_anden_vej():
    img = _skraa_fals(top_x=120, bund_x=200, bredde=1000)
    _saet_markoer(img, 190, 40)      # vores side (til hoejre for falsen), foroven
    _saet_markoer(img, 270, 1140)
    beskaaret, _ = beskaer_langs_fals(img, _side(2))
    assert beskaaret.width < img.width
    assert _markoer_findes(beskaaret, 190 - (img.width - beskaaret.width), 40)


def test_hoejden_bevares():
    img = _skraa_fals()
    beskaaret, _ = beskaer_langs_fals(img, _side(1))
    assert beskaaret.height == img.height


def test_maalingen_fortaeller_hvor_skaev_falsen_var():
    """Hældningen skal kunne aflæses -- ellers kan vi ikke se, hvornaar den
    baandvise beskaering overhovedet gjorde en forskel."""
    skraa = beskaer_langs_fals(_skraa_fals(top_x=880, bund_x=800), _side(1))[1]
    lodret = beskaer_langs_fals(_skraa_fals(top_x=850, bund_x=850), _side(1))[1]
    assert skraa.haeldning_px > 40
    assert lodret.haeldning_px <= 12


def test_ukendt_recto_verso_fejler_frem_for_at_gaette():
    with pytest.raises(ValueError, match="recto/verso"):
        beskaer_langs_fals(_skraa_fals(), _side(None))


def test_bufferen_findes_og_flytter_vaek_fra_vores_tekst():
    """Uden buffer ligger snittet PRAECIS paa falsens kant.

    Det er for taet: den udglattede profil ser ikke de sidste par pixels af
    et bogstav, der straekker sig lidt laengere end resten. Maalt paa
    rigtige sider redder bufferen de linjer, hvor skriften loeber ud i
    papirets krumning.
    """
    img = _skraa_fals(top_x=880, bund_x=800)
    uden = fals_graense(img, _side(1), buffer_andel=0.0)
    med = fals_graense(img, _side(1))
    flyt = [m - u for m, u in zip(med, uden)]
    assert min(flyt) > 0, "bufferen flytter ikke snittet vaek fra vores tekst"
    # 2% af 1000 px = 20 px; der maa ikke vaere skruet ned bag om ryggen.
    assert min(flyt) >= 15, f"bufferen er skrumpet til {min(flyt)} px"


def test_manglende_kant_i_TOPBAANDET_giver_ikke_et_spring():
    """Huller i ENDERNE er det farlige tilfaelde.

    Mangler et baand i midten, kan graensen interpoleres mellem naboerne.
    Mangler det oeverste, er der ingen nabo ovenfor -- og uden udfyldning
    ville graensen for de oeverste raekker blive hentet fra et baand langt
    nede paa siden, altsaa springe ind over vores egen tekst.
    """
    img = _skraa_fals(top_x=880, bund_x=800)
    px = img.load()
    for y in range(0, 200):                    # slet falsen i toppen
        for x in range(700, 1000):
            px[x, y] = HVID
    graense = fals_graense(img, _side(1), antal=8)
    assert len(graense) == img.height
    # Graensen foroven skal stadig ligge ude ved falsen, ikke inde paa siden.
    assert graense[0] > 830, (
        f"graensen sprang ind paa siden foroven: x={graense[0]}"
    )
    for y in range(1, img.height):
        assert abs(graense[y] - graense[y - 1]) < 20


def test_graensen_FOELGER_siden_ned_og_er_ikke_konstant():
    """Graensen skal aendre sig med hoejden -- det er hele pointen.

    Uden dette kunne interpolationen erstattes af én fast vaerdi, og
    beskaeringen ville i praksis vaere lodret igen, mens alle oevrige tests
    stadig gik igennem.
    """
    img = _skraa_fals(top_x=880, bund_x=800)
    graense = fals_graense(img, _side(1))
    assert len(graense) == img.height
    spaend = max(graense) - min(graense)
    assert spaend > 50, f"graensen er praktisk talt konstant (spaend {spaend} px)"
    # Og den skal foelge falsens retning: hoejt oppe laengst til hoejre.
    assert graense[10] > graense[-10] + 40


def test_en_LODRET_fals_giver_en_naesten_konstant_graense():
    """Modstykket: variationen maa ikke vaere stoej.

    Er falsen lodret, skal graensen ogsaa vaere det -- ellers foelger den
    noget andet end falsen.
    """
    graense = fals_graense(_skraa_fals(top_x=850, bund_x=850), _side(1))
    assert max(graense) - min(graense) <= 12
