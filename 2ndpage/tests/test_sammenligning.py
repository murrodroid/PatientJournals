"""Tests for sammenligningen paa faelles sider i scripts/maal_koersel.py.

En koersel kan mangle sider: en frist der loeb ud, eller et kald der fejlede.
Det er ikke et haendeligt uheld -- 2026-08-30 fik to af seks varianter kun
henholdsvis 8 og 4 af 12 sider igennem, fordi modellen brugte over 90 sekunder
paa dem.

Maales hver variant paa sit eget saet, sammenlignes de paa forskelligt
materiale, og forskellen kan lige saa godt vaere sidernes som variantens. Det
er den fejl, denne funktion findes for at forhindre.
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


def _lav_koersel(rod: Path, navn: str, svar: dict[str, str]) -> Path:
    """En koersel paa disken i den form, `laes_koersel` forventer."""
    mappe = rod / navn
    (mappe / "svar").mkdir(parents=True)
    (mappe / "opsaetning.json").write_text(json.dumps({
        "dato": "2026-08-30T12:00:00",
        "model": "en-model",
        "promptversion": navn,
        "prompt": "en prompt",
        "variant": "beskaaret",
        "temperatur": 0.0,
    }), encoding="utf-8")
    for side, tekst in svar.items():
        (mappe / "svar" / f"{side}.txt").write_text(tekst, encoding="utf-8")
    return mappe


@pytest.fixture
def facit(script, monkeypatch, tmp_path):
    """To sider med facit, saa maalingen har noget at maale imod."""
    sider = {
        "a_000001": ["Hun har sovet godt i Nat", "og drukket noget Maelk"],
        "b_000002": ["Ingen Snue eller Hoste", "Tungen er belagt"],
    }
    fil = tmp_path / "facit.jsonl"
    fil.write_text("\n".join(
        json.dumps({"image_name": n, "alt_linjer": l, "forside": n})
        for n, l in sider.items()), encoding="utf-8")
    monkeypatch.setattr(script, "FACIT", fil)
    monkeypatch.setattr(script, "sikr_oevemaengde", lambda navne: None)
    return sider


def test_kun_de_sider_alle_koersler_har_bliver_maalt(script, facit, tmp_path,
                                                     capsys):
    """Den egentlige pointe.

    Den ene koersel mangler den svaere side. Maales hver for sig, ser den
    bedre ud -- ikke fordi den laeste bedre, men fordi den slap for siden.
    """
    rod = tmp_path / "koersler"
    fuld = _lav_koersel(rod, "fuld", {
        "a_000001": "Hun har sovet godt i Nat\nog drukket noget Maelk",
        "b_000002": "Ingen Snue eller Hoste\nTungen er belagt",
    })
    # Den anden fik kun den lette side igennem.
    delvis = _lav_koersel(rod, "delvis", {
        "a_000001": "Hun har sovet godt i Nat\nog drukket noget Maelk",
    })

    script.sammenlign([delvis, fuld])
    ud = capsys.readouterr().out

    assert "paa 1 faelles sider" in ud
    assert "b_000002" in ud, "den udeladte side skal naevnes, ikke skjules"
    assert ud.count("  1 ") or " 1 " in ud


def test_koersler_uden_faelles_sider_stopper_i_stedet_for_at_dele_nul(
        script, facit, tmp_path):
    """Nul faelles sider er ikke en sammenligning -- det er en fejl."""
    rod = tmp_path / "koersler"
    en = _lav_koersel(rod, "en", {"a_000001": "Hun har sovet godt i Nat"})
    to = _lav_koersel(rod, "to", {"b_000002": "Ingen Snue eller Hoste"})

    with pytest.raises(SystemExit, match="ingen sider til faelles"):
        script.sammenlign([en, to])


def test_der_advares_naar_de_to_tal_peger_hver_sin_vej(script, facit,
                                                       tmp_path, capsys):
    """Hovedtallet og tallet uden rabat kan udpege hver sin vinder.

    Sker det, er det daekningen man ser, og det skal staa der -- ellers
    vaelger man den variant, der faar modellen til at afvige mest.
    """
    rod = tmp_path / "koersler"
    # `pyntet` laeser den anden linje helt forbi: den falder ud af
    # forankringen, og resten er fejlfri. `aerlig` laeser begge med smaa fejl.
    pyntet = _lav_koersel(rod, "pyntet", {
        "a_000001": "Hun har sovet godt i Nat\nintet som helst der ligner",
        "b_000002": "Ingen Snue eller Hoste\nslet ikke nogen lighed her",
    })
    aerlig = _lav_koersel(rod, "aerlig", {
        "a_000001": "Hun har sovet godt i Nit\nog drukket noget Malk",
        "b_000002": "Ingen Snue eller Hosti\nTungen er belogt",
    })

    script.sammenlign([aerlig, pyntet])
    ud = capsys.readouterr().out

    assert "BEMAERK" in ud, ud
    assert "pyntet" in ud and "aerlig" in ud
