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

from andenside.skraa import (GANG_SPRED, baandkanter, beskaer_langs_fals,
                             fals_graense)
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
    # Gulvet vogter RETNINGEN og at bufferen ikke skrumper bort bag om
    # ryggen -- ikke den praecise stoerrelse. Den er lead's visuelle valg og
    # har vaeret 1 %, 2 % og nu 0,5 %; en test, der pinner tallet, ville
    # blot skulle rettes hver gang han ser paa arkene igen.
    assert min(flyt) >= 4, f"bufferen er skrumpet til {min(flyt)} px"


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


# --- udskridende baand ---------------------------------------------------


def _fals_med_spike(bredde: int = 1000, hoejde: int = 1200, x: int = 820,
                    tykkelse: int = 40) -> Image.Image:
    """En lige fals -- og ét baand, hvor noget andet moerkt ligger langt inde.

    Det virkelige tilfaelde, maalt paa leveringen 2026-08-30: paa 9 af 307
    sider fandt ALLE 24 baand en kant, men de laa ikke paa samme linje. Et
    enkelt baand kunne pege 400 px inde paa siden, og fordi `fals_graense`
    interpolerer frit mellem baandene, trak den ene maaling snittet med sig
    og skar tvaers gennem teksten. `sikker` sagde ja, for den taeller kun,
    om baandene fandt NOGET.
    """
    img = Image.new("L", (bredde, hoejde), HVID)
    px = img.load()
    for y in range(hoejde):
        for dx in range(tykkelse):
            px[x + dx, y] = SORT
    # En moerk klat inde paa siden, kun i nogle faa baand. NB: den skal
    # ligge INDEN FOR soegevinduet (de ydre 40 % af bredden), ellers ses
    # den slet ikke, og proeven maaler ingenting.
    # Klatten skal daekke et helt baand (~120 raekker) for at slaa falsen i
    # baandets egen profil -- ellers vinder falsen, og proeven maaler intet.
    for y in range(420, 700):
        for dx in range(60):
            px[640 + dx, y] = SORT
    return img


def test_et_udskridende_baand_traekker_ikke_snittet_med_sig():
    """Den baerende test mod takkede snit.

    Uden frasortering foelger graensen den enlige klat 400 px ind paa siden.
    Falsen er maalt til at afvige hoejst 11 px fra en ret linje paa 90 % af
    leveringens 307 sider, saa et baand, der peger 400 px vaek, er en fejl
    -- ikke en krumning.
    """
    img = _fals_med_spike()
    graense = fals_graense(img, _side(1))
    assert graense
    assert min(graense) > 760, (
        f"snittet blev traukket ind paa siden af de faa baand med klatten: "
        f"{min(graense)}-{max(graense)}"
    )


def test_en_side_med_uenige_baand_maerkes_usikker():
    """Et maal, der siger ja til alt, er ikke et maal.

    Paa leveringen stod alle 9 gale sider som `sikker=ja`, fordi kolonnen kun
    taalte baand med et fund. Den skal ogsaa se paa, om de er ENIGE.
    """
    bredde, hoejde = 1000, 1200
    img = Image.new("L", (bredde, hoejde), HVID)
    px = img.load()
    rng = __import__("random").Random(0)
    # Kanten staar fast inden for hvert baand, men hopper mellem baandene --
    # ellers udglattes den til ingenting, og hvert baand melder blot 'intet
    # fund'. Proeven skal ramme det tilfaelde, hvor ALLE baand finder noget,
    # og de bare ikke er enige.
    #
    # ÉN blok pr. baand, ikke én pr. ti: med ti brede blokke laa der tilfaeldigt
    # tolv baand naesten paa linje (afvigelse 24 mod en tolerance paa 25), og
    # proeven bestod paa det haeld frem for paa sin egen praemis. Det kom frem
    # 2026-09-01, da et andet skridt flyttede kanterne 40 px og haeldet vendte.
    for blok in range(0, hoejde, 50):
        x0 = rng.randrange(620, 900)
        for y in range(blok, min(blok + 50, hoejde)):
            for dx in range(40):
                px[x0 + dx, y] = SORT
    _, maaling = beskaer_langs_fals(img, _side(1))
    assert not maaling.sikker, "en side uden nogen ret fals blev erklaeret sikker"


# --------------------------------------------------- snittet gennem folden

def _fals_med_krumning(
    bredde: int = 1000, hoejde: int = 1200, krumning_start: int = 800,
    krumning: int = 15, kerne: int = 10,
) -> Image.Image:
    """Falsen som den faktisk ser ud: en BRED moerkning, ikke en skarp streg.

    Papiret krummer ind mod folden og vender vaek fra lyset, saa det bliver
    gradvist moerkere, laenge foer selve folden. Skriveren skrev ud i den
    krumning. Derfor:

        vores side (hvid) | vores krumning (graa) | folden (sort) |
        naboens krumning (graa) | naboens side (hvid)

    Graatonen 120 ligger under `_profil_i_baand`s taerskel paa 180 og taeller
    altsaa som moerk -- praecis som paa de rigtige sider.

    Baeltet er 40 px paa en side paa 1000, altsaa 4 % af bredden. Det er med
    vilje: paa de rigtige sider er det 1,5-4,2 %, og loftet i
    `skraa.FOLD_LOFT_ANDEL` er 6 %. Et bredere baelte i proeven ville ligge
    over loftet og dermed maale loftet i stedet for reglen.
    """
    img = Image.new("L", (bredde, hoejde), HVID)
    px = img.load()
    graenser = [(krumning_start, krumning_start + krumning, 120),
                (krumning_start + krumning, krumning_start + krumning + kerne, SORT),
                (krumning_start + krumning + kerne,
                 krumning_start + 2 * krumning + kerne, 120)]
    for x0, x1, tone in graenser:
        for y in range(hoejde):
            for x in range(x0, min(x1, bredde)):
                px[x, y] = tone
    return img


def test_ordenderne_i_krumningen_overlever():
    """Snittet skal ligge paa den ANDEN side af foldens moerkning.

    Lead fandt 2026-09-01 tolv sider, hvor snittet laa ved moerkningens
    BEGYNDELSE og barberede de sidste bogstaver af linjerne -- dér, hvor
    siden krummer ind i folden og skriveren skrev helt ud. Moerkningen er
    45-70 px bred paa de sider; bufferen paa 0,5 % er 8-10 px og raekker
    ikke.

    Samme erkendelse staar allerede i `yderkant._kandidater_i_profil`:
    kanten hoerer til i faldets BUND, ikke ved dets begyndelse.
    """
    img = _fals_med_krumning()
    _saet_markoer(img, 806, 40)     # et ord, der naar ud i krumningen
    _saet_markoer(img, 806, 1100)
    beskaaret, _ = beskaer_langs_fals(img, _side(1))
    assert _markoer_findes(beskaaret, 806, 40), "ordenden foroven blev skaaret af"
    assert _markoer_findes(beskaaret, 806, 1100), "ordenden forneden blev skaaret af"


def test_naboens_tekst_kommer_stadig_ikke_med_naar_der_skaeres_gennem_folden():
    """Prisen for at gaa gennem folden maa ikke vaere naboens skrift.

    Naboens egen tekst staar paa hans flade del, uden for hans krumning.
    Folden slutter her ved 800+15+10+15 = 840.
    """
    img = _fals_med_krumning()
    _saet_markoer(img, 880, 40)
    beskaaret, _ = beskaer_langs_fals(img, _side(1))
    assert not _markoer_findes(beskaaret, 950, 40), "naboens tekst kom med"


def test_snittet_loeber_ikke_loebsk_naar_moerket_aldrig_slutter():
    """Er hele resten af billedet moerkt, maa snittet ikke vandre ud i det.

    Sker paa sider, hvor folden gaar i ét med en moerk baggrund. Uden et loft
    ville snittet ende ved billedkanten og tage hele naboopslaget med.
    """
    bredde, hoejde = 1000, 1200
    img = Image.new("L", (bredde, hoejde), HVID)
    px = img.load()
    for y in range(hoejde):
        for x in range(800, bredde):
            px[x, y] = SORT
    graense = fals_graense(img, _side(1))
    assert graense
    assert max(graense) < 900, (
        f"snittet vandrede ud i det moerke: {min(graense)}-{max(graense)}"
    )


def test_en_blød_overgang_stopper_ikke_gangen_gennem_folden():
    """Moerkningen begynder blødt -- gangen skal stadig naa igennem.

    Springet findes dér, hvor moerkningen BEGYNDER, og dér er kolonnen kun
    fx 20 % moerk. Et krav om, at NAESTE kolonne allerede er over halvt
    moerk, ville stoppe gangen med det samme. Set paa 273109_000082, hvor
    syv af 24 baand derfor slet ikke rykkede sig, mens resten gik 45-85 px.
    """
    bredde, hoejde = 1000, 1200
    img = Image.new("L", (bredde, hoejde), HVID)
    px = img.load()
    for y in range(hoejde):
        for x in range(800, 820):        # bloed rampe: hvid -> sort over 20 px
            px[x, y] = int(HVID * (1 - (x - 800) / 20))
        for x in range(820, 840):
            px[x, y] = SORT
        for x in range(840, 860):        # naboens krumning, spejlvendt rampe
            px[x, y] = int(HVID * (x - 840) / 20)
    graense = fals_graense(img, _side(1), buffer_andel=0.0)
    assert graense
    assert min(graense) > 845, (
        f"gangen stoppede i rampen i stedet for at naa gennem folden: "
        f"{min(graense)}-{max(graense)}"
    )


def test_et_moerkt_hjoerne_forneden_giver_ikke_snittet_en_hale():
    """Foldens bredde aendrer sig jaevnt -- et enkelt baand maa ikke stikke af.

    Forneden paa mange sider gaar folden i ét med affotograferingens skygge,
    saa moerkningen aldrig slipper, og gangen loeber videre. Lead saa det som
    en 'hale' paa snittet i bunden af 273102_001066, _001074 og 273024_001127,
    hvor de nederste baand gik 100-121 px mod 60-75 px paa resten af siden.
    """
    img = _fals_med_krumning()                        # baelte: 800-840
    px = img.load()
    for y in range(1100, 1200):          # moerkt baand tvaers over bunden
        for x in range(800, 1000):
            px[x, y] = SORT
    graense = fals_graense(img, _side(1), buffer_andel=0.0)
    spredning = max(graense) - min(graense)
    # Grænsen er den samme, koden selv tillader: foldens bredde SKAL kunne
    # variere lidt ned gennem siden. Tallet er maalt, ikke valgt -- se
    # `skraa.GANG_SPRED`. Halen, lead saa, var 40-60 px.
    assert spredning <= GANG_SPRED, (
        f"snittet fik en hale forneden: {min(graense)}-{max(graense)}"
    )


def test_en_side_hvor_folden_aldrig_slipper_maerkes_usikker():
    """Kan foldens anden side ikke findes, er loftet et gaet -- sig det.

    Paa fire sider (fx 273035_000244 med 17 af 24 baand paa loftet) gaar
    folden i ét med en moerk baggrund hele siden ned. Der er intet at gaa
    efter, og snittet endte langt inde paa naboen. Saa skaeres der som foer,
    og siden maerkes i stedet.
    """
    bredde, hoejde = 1000, 1200
    img = Image.new("L", (bredde, hoejde), HVID)
    px = img.load()
    for y in range(hoejde):
        for x in range(800, bredde):
            px[x, y] = SORT
    _, maaling = beskaer_langs_fals(img, _side(1))
    assert not maaling.sikker, "en side uden en findbar foldkant blev kaldt sikker"
