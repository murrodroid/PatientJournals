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


def test_der_advares_naar_den_strenge_maaling_er_hoejere_end_hovedtallet(
        script, monkeypatch, tmp_path, capsys):
    """Beslutning 44: den strenge maaling udelader hele linjer med et `[?]`
    og boer derfor normalt vaere LAVERE end hovedtallet -- den maaler mindre
    af siden, og det, den ikke maaler, er netop det svaereste.

    Er den i stedet HOEJERE, er det, fordi den svaere linje blev laest helt
    rigtigt (den koster intet i nogen af de to maalinger), mens resten af
    siden -- den, den strenge stadig maaler paa -- blev laest daarligt. Fjern
    en fejlfri linje fra baade taeller og naevner, og den tilbagevaerende
    fejlrate STIGER. Det skal opdages og siges, ikke bare passere som et
    hvilket som helst tal.
    """
    sider = {
        "a_000001": ["Ingen Snue eller [?] Hoste", "Tungen er belagt"],
    }
    facit_fil = tmp_path / "facit.jsonl"
    facit_fil.write_text("\n".join(
        json.dumps({"image_name": n, "alt_linjer": l, "forside": n})
        for n, l in sider.items()), encoding="utf-8")
    monkeypatch.setattr(script, "FACIT", facit_fil)
    monkeypatch.setattr(script, "sikr_oevemaengde", lambda navne: None)

    rod = tmp_path / "koersler"
    # `pyntet` laeser den svaere linje helt rigtigt (jokerfeltet sluger
    # gaettet gratis i begge maalinger) og den rene linje helt forkert.
    # Fjernes den svaere linje fra naevneren, som den strenge maaling goer,
    # staar kun den forkerte linje tilbage -- og raten stiger.
    pyntet = _lav_koersel(rod, "pyntet", {
        "a_000001": "Ingen Snue eller Snue Hoste\n"
                    "Slet ingen lighed med originalen overhovedet",
    })
    # `aerlig` laeser begge linjer helt rigtigt -- intet at advare om.
    aerlig = _lav_koersel(rod, "aerlig", {
        "a_000001": "Ingen Snue eller Hoste\nTungen er belagt",
    })

    script.sammenlign([aerlig, pyntet])
    ud = capsys.readouterr().out

    bemaerk = [l for l in ud.splitlines() if "BEMAERK" in l]
    assert bemaerk, ud
    assert "pyntet" in bemaerk[0] and "aerlig" not in bemaerk[0]
