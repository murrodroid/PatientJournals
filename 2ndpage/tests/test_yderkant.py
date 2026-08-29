"""Tests for detektionen af sidens YDRE kant -- den modsatte af falsen.

Falsbeskaeringen (`skraa.py`) renser den ene kant. Paa den modsatte ligger
enten bogsnittet (bogblokkens sammenpressede sidekanter, harmloest) eller et
blad laengere inde i bindet, som er faldet fladt ud og fotograferet med, saa
der staar FREMMED haandskrift langs kanten. Facit: 7 af 118 oevesider har
fremmed tekst dér (`stages/04_billedforberedelse/output/yderkant_facit.csv`).

De baerende tests:

- `test_kanten_findes_ved_bladets_skygge_ikke_ved_baggrunden` -- et blad
  udenfor giver en SMAL moerk stribe med papir paa begge sider. Den naive
  loesning (find det moerkeste punkt) rammer den sorte baggrund i stedet og
  ville tage hele det fremmede blad med.
- `test_graensen_varierer_ned_gennem_siden` -- en konstant graense skal ikke
  kunne bestaa. Netop den mutation slap igennem i `skraa.py` 2026-08-27,
  til to tests blev foejet til.
- `test_langsom_afdaempning_giver_ingen_kant` -- billedernes lysstyrke
  falder jaevnt ud mod kanten. Et rent "faldt nok i alt"-krav ville
  fejlagtigt skaere midt paa vores egen side.
"""

import numpy as np
import pytest
from PIL import Image

from andenside.masterlist import Side
from andenside.yderkant import (
    baandkanter_ydre,
    beskaer_ydre,
    har_fremmed_blad,
    papir_profil,
    ydre_graense,
    ydre_vindue,
)

PAPIR = 210
SKYGGE = 150      # bladkantens skygge -- moerkere end papir, langt lysere end baggrund
BAGGRUND = 35


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


RECTO = _side(2)   # tredjeside: fals venstre, YDERKANT HOEJRE
VERSO = _side(1)   # andenside:  fals hoejre, YDERKANT VENSTRE


def _side_med_blad(
    bredde: int = 1000,
    hoejde: int = 1200,
    kant_top: int = 700,
    kant_bund: int = 700,
    skygge_bredde: int = 12,
    blad_slut: int = 900,
) -> Image.Image:
    """Vores side, en smal skyggestribe, det fremmede blad, saa baggrund.

    `kant_top`/`kant_bund` lader skyggestriben haelde, som et blad der ligger
    skaevt i bindet gør det.
    """
    a = np.full((hoejde, bredde), PAPIR, dtype=np.uint8)
    for y in range(hoejde):
        x0 = int(kant_top + (kant_bund - kant_top) * y / hoejde)
        a[y, x0 : x0 + skygge_bredde] = SKYGGE
        a[y, blad_slut:] = BAGGRUND
    return Image.fromarray(a, mode="L")


def _side_med_bogsnit(bredde: int = 1000, hoejde: int = 1200, kant: int = 700,
                      snip: int = 20) -> Image.Image:
    """Vores side, en SMAL lys snip, saa bogsnittets moerke baand, saa baggrund.

    Snippen er det, der gør prøven skarp. Maalt paa `273098_001496` -- en
    side helt uden udragende blad -- kommer papiret nemlig kortvarigt igen
    lige uden for kanten, ca. 20 px. En regel, der spørger "kommer papiret
    igen?", siger derfor ja til bogsnittet ogsaa. Kun bæltets BREDDE
    skiller de to ad.
    """
    a = np.full((hoejde, bredde), PAPIR, dtype=np.uint8)
    a[:, kant : kant + 8] = SKYGGE          # sidens egen kantskygge
    a[:, kant + 8 : kant + 8 + snip] = PAPIR - 4   # den smalle lyse snip
    a[:, kant + 8 + snip :] = 90            # bogsnittet: moerkt, ingen papirflade
    a[:, kant + 128 + snip :] = BAGGRUND
    return Image.fromarray(a, mode="L")


# --- soegevinduet: spejlet af falsens ------------------------------------


def test_ydre_vindue_ligger_modsat_falsen():
    """Recto har falsen til venstre, altsaa yderkanten til hoejre."""
    recto = ydre_vindue(RECTO, 1000)
    verso = ydre_vindue(VERSO, 1000)
    assert recto.slut == 1000 and recto.start > 500, "recto: yderkanten er hoejre"
    assert verso.start == 0 and verso.slut < 500, "verso: yderkanten er venstre"
    # Den reelle risiko er, at falsens vindue bliver genbrugt uaendret:
    from andenside.bogryg import soegevindue

    assert recto.start != soegevindue(RECTO, 1000).start
    assert verso.start != soegevindue(VERSO, 1000).start


def test_ukendt_recto_verso_afvises():
    with pytest.raises(ValueError):
        ydre_vindue(_side(None), 1000)


# --- selve kantfindingen -------------------------------------------------


def test_kanten_findes_ved_bladets_skygge_ikke_ved_baggrunden():
    """Den baerende test: skyggestriben, ikke det moerkeste punkt.

    Baggrunden er langt moerkere end skyggen. En detektion, der leder efter
    det stoerste fald eller det laveste niveau, lander paa baggrunden ved
    x=900 og tager dermed hele det fremmede blad med videre til modellen.
    """
    img = _side_med_blad(kant_top=700, kant_bund=700)
    kanter = [x for _, x in baandkanter_ydre(img, RECTO) if x is not None]
    assert kanter, "ingen kant fundet overhovedet"
    assert all(690 <= x <= 715 for x in kanter), f"kanter uden for skyggen: {sorted(set(kanter))}"


def test_snittet_laegges_paa_kantens_YDRE_side():
    """Retningen er ikke symmetrisk, og lead afgjorde hvilken vej.

    Faldets begyndelse ligger inde paa VORES side; et snit dér barberer
    ordenderne (paavist 2026-08-28 paa de rigtige billeder). Et snit lidt
    for langt ude lader derimod kun en flig af naboen staa. Derfor lægges
    kanten i faldets bund.

    Faldet er her jaevnt over 10 px, saa begyndelse og bund ikke kan
    forveksles -- med en brat, flad skygge falder de sammen, og prøven
    ville intet vogte.
    """
    bredde, hoejde = 1000, 1200
    a = np.full((hoejde, bredde), PAPIR, dtype=np.uint8)
    for i in range(10):                       # jaevnt fald fra x=700 til x=709
        a[:, 700 + i] = PAPIR - 6 * (i + 1)
    a[:, 710:900] = PAPIR - 60                # skyggen fortsaetter
    a[:, 900:] = BAGGRUND
    kanter = [x for _, x in baandkanter_ydre(Image.fromarray(a, mode="L"), RECTO)
              if x is not None]
    assert kanter
    assert all(x >= 707 for x in kanter), (
        f"snittet ligger paa den indre side af kanten: {sorted(set(kanter))}"
    )


def test_bufferen_lader_kun_en_flig_af_naboen_staa():
    """Bufferen er margen, ikke kompensation -- og maa ikke sluge naboen.

    Da kanten blev meldt ved faldets BEGYNDELSE, laa den op til 12 kolonner
    inde paa vores side, og bufferen skulle daekke den skaevhed. Nu meldes
    kanten i faldets bund, altsaa paa selve soemmen, saa bufferen er ren
    margen. Lead paapegede 2026-08-28, at den derfor gav for meget af det
    fremmede blad tilbage, og den blev halveret.

    Prøven vogter den ende af afvejningen: af et blad paa 188 px maa der
    hoejst blive en tiendedel tilbage inden for snittet.
    """
    img = _side_med_blad(kant_top=700, kant_bund=700)   # blad fra x=712 til x=900
    graense = ydre_graense(img, RECTO)
    tilbage = max(graense) - 712
    assert tilbage <= 19, f"{tilbage} px af det fremmede blad blev staaende"


def test_graensen_varierer_ned_gennem_siden():
    """En konstant graense skal ikke kunne bestaa.

    Bladet ligger skaevt: skyggen staar ved x=700 foroven og x=780 forneden.
    Erstattes interpolationen af et enkelt tal, falder denne test.
    """
    img = _side_med_blad(kant_top=700, kant_bund=780)
    graense = ydre_graense(img, RECTO)
    assert len(graense) == img.height
    assert max(graense) - min(graense) > 40, (
        f"graensen er praktisk talt konstant: {min(graense)}-{max(graense)}"
    )
    # og den skal haelde den rigtige vej -- ikke bare vaere uroligt
    assert graense[100] < graense[-100]


def test_bufferen_flytter_snittet_UDAD_ikke_ind_over_vores_tekst():
    """Bufferens fortegn er den farligste enkeltfejl i modulet.

    Vender den forkert, skæres der ind i vores egen side i stedet for vaek
    fra den -- og det ville ramme alle 118 sider paa én gang, ogsaa de 110
    der ikke fejler noget i dag.
    """
    img = _side_med_blad(kant_top=700, kant_bund=700)
    raa = [x for _, x in baandkanter_ydre(img, RECTO) if x is not None]
    graense = ydre_graense(img, RECTO)
    assert min(graense) >= max(raa), (
        f"snittet ({min(graense)}) ligger inden for den fundne kant ({max(raa)})"
    )


def _side_med_to_blade(bredde: int = 1000, hoejde: int = 1200) -> Image.Image:
    """Vores side, saa ET blad, saa ET TIL -- og vores egen skygge er svag.

    Det virkelige tilfaelde, maalt paa `37554_001494` og `_001496`: skyggen
    mellem vores side og det foerste blad er saa svag, at nogle baand ikke
    ser den og i stedet finder det NAESTE blads kant, 90 px laengere ude.
    Snittet zigzaggede derfor frem og tilbage. En sidekant er en ret linje;
    detektionen skal bruge det til at forkaste de udskridende baand.
    """
    a = np.full((hoejde, bredde), PAPIR, dtype=np.uint8)
    # Vores egen kant ses KUN i den oeverste tredjedel. Variationen skal
    # vaere grovere end baandenes hoejde -- ellers blander percentilen de
    # svage og de kraftige raekker, og proeven maaler ikke det, den skal.
    a[:, 850:860] = SKYGGE          # det naeste blads kant: kraftig, overalt
    a[:, 920:] = BAGGRUND
    # Vores egen kant kaster skygge i HELE hoejden -- en rigtig papirkant
    # goer altid det -- men er kun kraftig nok til at blive fundet baandvist
    # foroven. Tidligere var den helt fravaerende nedefter, og det er ikke,
    # hvad et rigtigt billede viser.
    a[:, 760:766] = PAPIR - 10
    a[: hoejde // 3, 760:766] = PAPIR - 20
    return Image.fromarray(a, mode="L")


def test_snittet_skrider_ikke_ud_til_det_naeste_blad():
    """Den baerende test for den rette linje.

    Uden krav om, at baandene skal ligge paa samme linje, springer snittet
    mellem x=700 og x=790, og resultatet blev set zigzagge paa de rigtige
    billeder. Med kravet skal alle raekker ligge ved vores egen kant.
    """
    img = _side_med_to_blade()
    graense = ydre_graense(img, RECTO)
    assert graense, "ingen graense fundet"
    assert max(graense) < 820, (
        f"snittet skred ud mod det naeste blad: {min(graense)}-{max(graense)}"
    )


def test_uenige_baand_goer_siden_usikker():
    """Et maal, der aldrig siger nej, er ikke et maal.

    Falsbeskaeringens `sikker`-kolonne gav 10/10 paa alle 118 sider og
    skilte dermed ingenting fra. Her skal en side, hvor baandene IKKE kan
    laegges paa én linje, faktisk blive mærket.
    """
    bredde, hoejde = 1000, 1200
    a = np.full((hoejde, bredde), PAPIR, dtype=np.uint8)
    rng = np.random.default_rng(0)   # fast froe: proeven skal vaere den samme hver gang
    for y in range(hoejde):
        x = int(rng.integers(620, 860))     # kanten hopper tilfaeldigt rundt
        a[y, x : x + 10] = SKYGGE
    ud, maaling = beskaer_ydre(Image.fromarray(a, mode="L"), RECTO)
    assert not maaling.sikker, "en side uden nogen ret kant blev erklaeret sikker"


def test_langsom_afdaempning_giver_ingen_kant():
    """Jaevnt fald ud mod kanten er belysning, ikke en kant.

    Faldet her er 60 niveauer i alt -- langt over enhver rimelig taerskel --
    men fordelt over 300 px. Kraeves faldet ikke inden for et kort spand,
    skaerer detektionen midt paa vores egen side.
    """
    bredde, hoejde = 1000, 1200
    a = np.full((hoejde, bredde), PAPIR, dtype=np.uint8)
    for x in range(700, bredde):
        a[:, x] = PAPIR - int(60 * (x - 700) / (bredde - 700))
    img = Image.fromarray(a, mode="L")
    kanter = [x for _, x in baandkanter_ydre(img, RECTO) if x is not None]
    assert not kanter, f"fandt kant i en ren gradient: {sorted(set(kanter))}"


def test_egen_tekst_taet_paa_kanten_giver_ingen_kant():
    """Blaek maa ikke kunne tages for en kant.

    Derfor maales papirets grundlyshed (hoej percentil pr. kolonne), ikke
    gennemsnittet. En tekstlinje gør en kolonne moerkere i gennemsnit, men
    rører ikke percentilen, saa laenge papiret ses mellem bogstaverne.
    """
    bredde, hoejde = 1000, 1200
    a = np.full((hoejde, bredde), PAPIR, dtype=np.uint8)
    # kraftige lodrette blaekstreger taet ved kanten, men kun paa hver 5. raekke
    for y in range(0, hoejde, 5):
        a[y, 750:790] = 20
    img = Image.fromarray(a, mode="L")
    kanter = [x for _, x in baandkanter_ydre(img, RECTO) if x is not None]
    assert not kanter, f"blaek blev taget for en kant: {sorted(set(kanter))}"


def test_papir_profil_ser_bort_fra_blaek():
    """Selve maalingen: percentilen skal ligge paa papiret, ikke paa blaekket."""
    a = np.full((100, 50), PAPIR, dtype=np.uint8)
    a[::4, 20] = 0          # en fjerdedel af kolonne 20 er sort blaek
    profil = papir_profil(np.asarray(a, dtype=float), 0, 100)
    assert profil[20] > PAPIR - 5, f"blaekket traak profilen ned: {profil[20]}"


# --- er der overhovedet et fremmed blad? (forsoeg B) ---------------------


def test_fremmed_blad_kendes_paa_at_papiret_kommer_igen():
    """Efter skyggen kommer papir igen -- saa ligger der et blad udenfor."""
    fund = har_fremmed_blad(_side_med_blad(), RECTO)
    assert fund.er_blad
    assert fund.niveau_efter > fund.niveau_min + 30


def test_bogsnit_er_ikke_et_fremmed_blad():
    """Bogsnittets lyse snip er for smal til at vaere et blad.

    Prøven er skarp, fordi papiret FAKTISK kommer igen her -- bare kun i
    20 px. En regel, der spørger "kommer papiret igen?", falder i;
    bæltets bredde gør ikke.
    """
    fund = har_fremmed_blad(_side_med_bogsnit(), RECTO)
    assert not fund.er_blad, f"bogsnittets {fund.baelte_bredde} px blev taget for et blad"
    assert fund.baelte_bredde < 45


def test_bladets_baelte_er_mange_gange_bredere_end_bogsnittets():
    """De to skal ligge langt fra hinanden, ikke lige omkring taersklen."""
    blad = har_fremmed_blad(_side_med_blad(), RECTO).baelte_bredde
    snit = har_fremmed_blad(_side_med_bogsnit(), RECTO).baelte_bredde
    assert blad > 4 * snit, f"for taet paa hinanden: blad={blad}, bogsnit={snit}"


# --- selve beskaeringen --------------------------------------------------


def test_beskaeringen_fjerner_det_fremmede_blad():
    img = _side_med_blad(kant_top=700, kant_bund=700)
    ud, maaling = beskaer_ydre(img, RECTO)
    assert ud.width < img.width
    assert maaling.bredde_efter == ud.width
    # intet af bladet maa vaere tilbage: alt uden for graensen er hvidt
    yderste = np.asarray(ud.convert("L"))[:, -1]
    assert (yderste > 200).all(), "der staar stadig noget uden for snittet"


def test_beskaeringen_beholder_billedets_farvetilstand():
    """En graa udgave ville vaere en skjult aendring af det, modellen ser.

    Netop den fejl blev begaaet i falsbeskaeringen 2026-08-27.
    """
    farve = _side_med_blad().convert("RGB")
    ud, _ = beskaer_ydre(farve, RECTO)
    assert ud.mode == "RGB"


def test_side_uden_kant_maerkes_usikker_og_skaeres_ikke():
    """Findes ingen kant, skal siden mærkes -- ikke skæres paa slump.

    Stage-kontrakten: 'Registrér usikre tilfælde frem for at gætte.'
    """
    blank = Image.new("L", (1000, 1200), PAPIR)
    ud, maaling = beskaer_ydre(blank, RECTO)
    assert ud.width == blank.width, "en side uden kant blev beskaaret alligevel"
    assert not maaling.sikker
    assert maaling.baand_med_kant == 0


def test_verso_beskaeres_i_den_anden_ende():
    """Spejlingen skal virke hele vejen igennem, ikke kun i vinduet."""
    img = _side_med_blad(kant_top=700).transpose(Image.FLIP_LEFT_RIGHT)
    ud, maaling = beskaer_ydre(img, VERSO)
    assert maaling.bredde_efter < img.width
    # vores side laa til hoejre efter spejlingen; den skal vaere i behold
    hoejre = np.asarray(ud.convert("L"))[:, -1]
    assert (hoejre > 200).all()


# --- soem-maalingen for sig selv ----------------------------------------


def test_soem_dybde_ser_forskel_paa_en_kant_og_blankt_papir():
    """Selve maalingen bag soem-kravet.

    Kravet aendrer intet paa oevemaengdens 118 sider -- det er maalt. Det
    staar der som vaern for materiale, vi ikke har set endnu, og saa maa
    selve maalingen i det mindste vaere efterproevet: en smal moerk stribe
    skal give et tydeligt udslag, blankt papir naesten intet.
    """
    from andenside.yderkant import soem_dybde

    blankt = np.full((600, 400), PAPIR, dtype=np.uint8)
    med_kant = blankt.copy()
    med_kant[:, 198:204] = SKYGGE

    assert soem_dybde(np.asarray(blankt, dtype=float), 200, 0) < 2
    assert soem_dybde(np.asarray(med_kant, dtype=float), 200, 0) > 40


def test_soem_dybden_falder_naar_linjen_ikke_foelger_kanten():
    """En skaev linje hen over en lodret kant ligger kun delvis i soemmen."""
    from andenside.yderkant import soem_dybde

    a = np.full((600, 400), PAPIR, dtype=np.uint8)
    a[:, 198:204] = SKYGGE
    graa = np.asarray(a, dtype=float)
    assert soem_dybde(graa, 200, 0) > soem_dybde(graa, 200, 80)


def test_bufferen_findes_faktisk():
    """Bufferen skal flytte snittet maalbart -- ikke bare staa i koden.

    En buffer paa nul ville ikke fejle nogen anden proeve, men den er det
    eneste, der skiller snittet fra selve kanten.

    Kravet er et FAST antal pixels, ikke `BUFFER_ANDEL` ganget op. Laeser
    proeven konstanten, kan den aldrig fange, at konstanten bliver skruet
    ned -- den ville bare saenke sit eget krav tilsvarende.
    """
    img = _side_med_blad(kant_top=700, kant_bund=700)     # 1000 px bred
    raa = [x for _, x in baandkanter_ydre(img, RECTO) if x is not None]
    graense = ydre_graense(img, RECTO)
    assert min(graense) - max(raa) >= 4, (
        f"bufferen er kun {min(graense) - max(raa)} px paa et 1000 px billede"
    )
