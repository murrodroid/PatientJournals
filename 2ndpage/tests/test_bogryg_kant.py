"""Tests for den stigningsbaserede kantdetektion i `find_snitpunkt`.

Metoden blev lagt om 2026-08-27. Den gamle ledte efter et NIVEAU: blaekmaengden
skulle krydse 0,30. Det virker kun, saa laenge der er en kraftig skygge nede i
falsen -- og i de velfotograferede bind er der naesten ingen. Maalt paa
oevemaengden topper falsen paa de svageste sider ved 0,291/0,293, mens den
vaerste haandskrifts-stoej paa vejen frem naar 0,300. Intervallerne overlapper,
saa INGEN absolut taerskel kan skille fals fra haandskrift.

Den nye metode leder efter et BRAT SPRING i stedet. Testene her holder netop
den forskel oppe -- de skal blive roede, hvis nogen en dag skifter tilbage til
et niveau.
"""

import pytest
from PIL import Image

from andenside.bogryg import find_snitpunkt
from andenside.masterlist import Side

HVID = 255
SORT = 0
HOEJDE = 400


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


def _billede(bredde: int = 1000) -> Image.Image:
    return Image.new("L", (bredde, HOEJDE), HVID)


def _maal_kolonner(img: Image.Image, fra: int, til: int, andel: float) -> None:
    """Gør `andel` af raekkerne sorte i kolonnerne [fra, til).

    Andelen bliver til blaekmaengden i profilen: 1,0 = helt sort kolonne.
    """
    pixels = img.load()
    antal = int(HOEJDE * andel)
    for x in range(fra, til):
        for y in range(antal):
            pixels[x, y] = SORT


def test_en_SVAG_men_skarp_kant_findes():
    """Selve pointen: falsen behoever ikke vaere moerk, bare skarp.

    Kanten her ligger paa 0,29 -- under den gamle taerskel paa 0,30, saa
    niveau-metoden ville have opgivet siden helt.
    """
    img = _billede()
    _maal_kolonner(img, 850, 1000, 0.29)   # svag, men brat begyndende
    resultat = find_snitpunkt(img, _side(1))  # verso: indhold venstre
    assert resultat.styrke > 0.0, "en svag men skarp kant blev ikke fundet"
    assert 830 <= resultat.x <= 900


def test_et_HOEJT_niveau_uden_spring_er_ikke_en_kant():
    """En jaevnt stigende rampe er haandskrift/skygge, ikke en sidekant.

    Den naar helt op paa 1,0 -- langt over enhver niveau-taerskel -- men den
    stiger saa langsomt, at der ikke er nogen kant at finde.
    """
    img = _billede()
    for i, x in enumerate(range(600, 1000)):
        _maal_kolonner(img, x, x + 1, i / 400)
    resultat = find_snitpunkt(img, _side(1))
    assert resultat.styrke == 0.0, (
        "en jaevn rampe blev forvekslet med en kant -- metoden er faldet "
        "tilbage til at maale niveau"
    )


def test_den_skarpe_kant_vinder_over_den_kraftige_rampe():
    """Det afgoerende tilfaelde, og grunden til at metoden blev lagt om.

    Billedet rummer BAADE en kraftig, langsom rampe (som naar 1,0) og en
    svag, brat kant laengere ude. Den gamle niveau-metode ville snitte i
    rampen -- altsaa inde i vores egen side. Den rigtige kant er den bratte.
    """
    img = _billede()
    for i, x in enumerate(range(600, 800)):      # kraftig, langsom rampe
        _maal_kolonner(img, x, x + 1, i / 200)
    _maal_kolonner(img, 800, 900, 0.0)           # pause
    _maal_kolonner(img, 900, 1000, 0.29)         # svag, brat kant

    resultat = find_snitpunkt(img, _side(1))
    assert resultat.x > 850, (
        f"snittet landede paa x={resultat.x} -- inde i rampen, ikke ved den "
        f"bratte kant"
    )


def test_en_helt_flad_profil_giver_intet_snit():
    resultat = find_snitpunkt(_billede(), _side(1))
    assert resultat.styrke == 0.0


def test_styrken_er_springets_stoerrelse_ikke_blaekmaengden():
    """Styrken skal foelge SPRINGET.

    Ellers kan `beskaer.py` ikke skelne en svag-men-sikker kant fra en
    kraftig, langsom skygge -- og det er netop den skelnen, den bruger.
    """
    svag = _billede()
    _maal_kolonner(svag, 850, 1000, 0.29)
    kraftig = _billede()
    _maal_kolonner(kraftig, 850, 1000, 1.0)
    assert find_snitpunkt(kraftig, _side(1)).styrke > find_snitpunkt(svag, _side(1)).styrke


def test_recto_soeger_i_venstre_kant_og_verso_i_hoejre():
    """Retningen maa aldrig gaettes -- den foelger recto/verso.

    Samme billede, to sidetyper: kanten skal findes hver sin vej.
    """
    for counter, fra, til, forventet in [(1, 850, 1000, 850), (2, 0, 150, 150)]:
        img = _billede()
        _maal_kolonner(img, fra, til, 0.5)
        resultat = find_snitpunkt(img, _side(counter))
        assert resultat.styrke > 0.0
        assert abs(resultat.x - forventet) < 80, (
            f"counter={counter}: snit x={resultat.x}, ventede omkring {forventet}"
        )


def test_bufferen_flytter_snittet_vaek_fra_vores_egen_tekst():
    img = _billede()
    _maal_kolonner(img, 850, 1000, 0.5)
    uden = find_snitpunkt(img, _side(1), buffer_andel=0.0)
    med = find_snitpunkt(img, _side(1), buffer_andel=0.05)
    # Verso: vores tekst er til venstre, saa bufferen skal flytte snittet HOEJRE.
    assert med.x > uden.x


def test_ukendt_recto_verso_fejler_frem_for_at_gaette():
    with pytest.raises(ValueError, match="recto/verso"):
        find_snitpunkt(_billede(), _side(None))


def test_styrken_er_springet_og_ikke_blaekmaengden_paa_stedet():
    """Styrken skal vaere SPRINGET, ikke hvor moerkt der er.

    De to falder ikke sammen: ved en svag kant er blaekmaengden mange gange
    stoerre end springet, fordi udglatningen fordeler springet over ~9
    kolonner. Forveksles de, faar `beskaer.py` et forkert billede af, hvor
    sikkert snittet er.
    """
    img = _billede()
    _maal_kolonner(img, 850, 1000, 0.29)
    resultat = find_snitpunkt(img, _side(1))
    assert 0.0 < resultat.styrke < 0.10, (
        f"styrke={resultat.styrke:.3f} ligner blaekmaengden (~0,29) mere end "
        f"springet"
    )


def test_den_STOERSTE_kant_vinder_ikke_bare_den_foerste():
    """Pinning af et bevidst designvalg -- laes noten.

    Naar der er flere kanter i vinduet, vaelges den KRAFTIGSTE, ikke den
    foerste paa vejen. Maalt paa oevemaengden giver det et snit paa alle 118
    sider og rammer de fire oejenbekraeftede snit.

    Afvejningen er reel: "foerste kant" ville stoppe ved vores egen sidekant
    og kan derfor ikke hoppe forbi falsen, mens "kraftigste kant" i princippet
    kan lande paa naboens kant, hvis den er kraftigere end vores egen. Det er
    ikke sket paa noget af det materiale, vi har -- men skifter nogen dette,
    skal det vaere med vilje, ikke ved et uheld.
    """
    img = _billede()
    _maal_kolonner(img, 700, 780, 0.25)    # en tidligere, moderat kant
    _maal_kolonner(img, 900, 1000, 1.0)    # en senere, langt kraftigere kant
    resultat = find_snitpunkt(img, _side(1))
    assert resultat.x > 850, (
        f"snit x={resultat.x}: den foerste kant vandt over den kraftigste"
    )


def test_soegevinduet_er_bredt_nok_til_de_maalte_falser():
    """Vinduet blev udvidet fra 30% til 40% 2026-08-27.

    Grunden er maalt, ikke skoennet: paa 14 af oevemaengdens sider ligger
    falsen 34-36% inde i billedet. Med et 30%-vindue blev de aldrig set, og
    siderne kunne slet ikke beskaeres. Snaevres vinduet igen, forsvinder de
    sider -- derfor staar tallet fast her.
    """
    from andenside.bogryg import STRIMMEL_ANDEL

    assert STRIMMEL_ANDEL >= 0.36, (
        "soegevinduet er for smalt til de falser, der er maalt paa 34-36% "
        "af bredden"
    )
