"""Tests for det daekningsuafhaengige tal i scripts/maal_koersel.py.

Hovedtallet maaler kun de linjer, forankringen kunne finde. Det giver en
rabat, der vokser med hvor meget modellens tekst afviger -- og paa stage 06's
foerste fire varianter vendte den rangordenen HELT om: den variant, der saa
bedst ud paa hovedtallet, var daarligst uden rabat, fordi den havde tabt
dobbelt saa meget tekst ud af maalingen.

Uden dette tal vaelger vi den variant, der faar modellen til at afvige mest.
"""

import importlib.util
import json
from pathlib import Path

import pytest

ROD = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def script():
    sti = ROD / "scripts" / "maal_koersel.py"
    spec = importlib.util.spec_from_file_location("maal_koersel", sti)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


def _post(navn, linjer):
    return {"image_name": navn, "alt_linjer": linjer}


def test_et_perfekt_svar_giver_nul(script):
    linjer = ["Hun har sovet godt i Nat", "og drukket noget Maelk"]
    poster = [_post("a", linjer)]
    fejl, tabt = script.uden_rabat(poster, {"a": "\n".join(linjer)})
    assert fejl == 0.0 and tabt == 0.0


def test_en_tabt_linje_taeller_som_helt_forkert(script):
    """Den egentlige pointe.

    Hovedtallet ville slet ikke se den manglende linje -- den falder ud af
    maalingen, og de tilbageblevne ser perfekte ud. Her skal den koste.

    Tallet pinnes EKSAKT, ikke som en ulighed: naevneren skal vaere alle
    facits tegn. Bliver den ved et uheld kun de MAALTE tegn, er tallet stadig
    stort, og en ulighed ville lade det passere -- men saa er varianter med
    forskellig daekning igen usammenlignelige, hvilket er hele grunden til at
    tallet findes.
    """
    linjer = ["Hun har sovet godt i Nat", "og drukket noget Maelk"]
    poster = [_post("a", linjer)]
    # Modellen skrev kun den foerste linje; den anden er helt tabt.
    fejl, tabt = script.uden_rabat(poster, {"a": linjer[0]})

    alle_tegn = len(linjer[0]) + len(linjer[1])
    ventet = len(linjer[1]) / alle_tegn
    assert fejl == pytest.approx(ventet, abs=1e-9), (
        f"{fejl:.4f} != {ventet:.4f} -- naevneren er formentlig ikke alle "
        f"facits tegn"
    )
    assert tabt == pytest.approx(ventet, abs=1e-9)


def test_variant_der_afviger_mere_straffes_ikke_belaennes(script):
    """Rangordenen skal vende, naar rabatten fjernes.

    `pyntet` laeser den ene linje helt forbi, saa den falder ud af
    forankringen; dens forankrede rest er til gengaeld fejlfri. `aerlig`
    laeser begge linjer med smaa fejl. Paa hovedtallet vinder `pyntet`; uden
    rabat skal `aerlig` vinde.
    """
    linjer = ["Hun har sovet godt i Nat", "og drukket noget Maelk"]
    poster = [_post("a", linjer)]

    pyntet = linjer[0] + "\nfuldstaendig anden tekst uden lighed overhovedet"
    aerlig = "Hun har sovet godt i Nit\nog drukket noget Malk"

    fejl_pyntet, tabt_pyntet = script.uden_rabat(poster, {"a": pyntet})
    fejl_aerlig, tabt_aerlig = script.uden_rabat(poster, {"a": aerlig})

    assert tabt_pyntet > tabt_aerlig, "pyntet skulle tabe mest tekst"
    assert fejl_aerlig < fejl_pyntet, (
        f"aerlig {fejl_aerlig:.3f} skulle slaa pyntet {fejl_pyntet:.3f}"
    )


def test_sider_uden_facit_traekkes_ikke_ind(script):
    """En side uden facit kan ikke maales og maa ikke fortynde tallet.

    Svaret paa den maalte side rummer med vilje en fejl. Var det fejlfrit,
    ville tallet vaere nul uanset hvad naevneren indeholdt, og testen kunne
    ikke se, om den ekstra side sneg sig ind i den.
    """
    linje = "Hun har sovet godt i Nat"
    poster = [_post("a", [linje])]
    med_fejl = "Hun har sovet godt i Nit"          # ét tegn galt
    uden_ekstra = script.uden_rabat(poster, {"a": med_fejl})
    med_ekstra = script.uden_rabat(
        poster, {"a": med_fejl, "b": "en side vi ikke har facit for"})

    assert uden_ekstra[0] == pytest.approx(1 / len(linje), abs=1e-9)
    assert med_ekstra == uden_ekstra, "siden uden facit fortyndede tallet"


def test_tom_maengde_giver_nul_og_ikke_en_division_med_nul(script):
    assert script.uden_rabat([], {}) == (0.0, 0.0)


def test_raekkefoelgen_paavirker_ikke_tallet(script):
    """Sider gennemloebes sorteret, saa to koersler giver samme tal."""
    poster = [_post("b", ["anden side her"]), _post("a", ["foerste side her"])]
    svar = {"a": "foerste side her", "b": "anden side her"}
    assert script.uden_rabat(poster, svar) == script.uden_rabat(
        list(reversed(poster)), dict(reversed(list(svar.items()))))
