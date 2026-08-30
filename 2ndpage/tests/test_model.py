"""Tests for modellaget -- kun de deterministiske dele.

Selve modelkaldet testes ikke (svar kan ikke laases fast; de logges i stedet
som resultater i stagens output). Det der KAN gaa galt uden at nogen opdager
det, er de to stykker plumbing omkring kaldet:

  1. **Noegleopslaget.** Noeglefilen laeses af koden, aldrig af os. Vi kender
     ikke feltnavnet indeni, saa opslaget maa finde det selv -- og fejle
     tydeligt frem for at gaette, hvis der ikke er et.
  2. **Sammenfoejningen af svaret.** Modellen svarer linje for linje efter et
     skema. Maaleapparatet vil have én tekst. Gaar noget tabt i den
     oversaettelse, ser det ud som om modellen laeste daarligt.
"""

import pytest

from andenside.model import (
    hent_noegle,
    NoegleFejl,
    find_noegle,
    tekst_af_sider,
)


# ---------------------------------------------------------------- noeglen

def test_noeglen_findes_uanset_hvad_feltet_hedder():
    # Vi kender ikke feltnavnet i brugerens fil, saa opslaget skal taale
    # de almindelige stavemaader.
    for navn in ("gemini", "GEMINI_API_KEY", "google", "genai_key", "Gemini Api Key"):
        assert find_noegle({navn: "xxx", "openai": "yyy"}) == "xxx"


def test_den_rigtige_udbyder_vaelges_naar_der_er_flere():
    data = {"openai": "aaa", "anthropic": "bbb", "gemini": "ccc"}
    assert find_noegle(data) == "ccc"


def test_manglende_noegle_fejler_tydeligt_og_naevner_feltnavnene():
    """Fejlen skal kunne handles paa uden at nogen aabner filen.

    Beskeden maa naevne, hvilke felter der ER i filen -- ellers er man
    henvist til at kigge i en fil, vi netop ikke vil kigge i.
    """
    with pytest.raises(NoegleFejl) as fejl:
        find_noegle({"openai": "aaa", "anthropic": "bbb"})
    assert "openai" in str(fejl.value)
    assert "anthropic" in str(fejl.value)


def test_selve_noeglen_lækker_aldrig_ud_i_fejlbeskeden():
    with pytest.raises(NoegleFejl) as fejl:
        find_noegle({"openai": "hemmelig-vaerdi-1234"})
    assert "hemmelig-vaerdi-1234" not in str(fejl.value)


def test_tom_noegle_taeller_ikke_som_fundet():
    # Et tomt felt er vaerre end intet felt: kaldet ville fejle langt senere.
    with pytest.raises(NoegleFejl):
        find_noegle({"gemini": "   "})


def test_indlejrede_noegler_findes_ogsaa():
    # Nogle noeglefiler grupperer pr. udbyder.
    assert find_noegle({"providers": {"gemini": {"api_key": "xxx"}}}) == "xxx"


# ------------------------------------------------------- svaret til tekst

def test_linjerne_bliver_til_én_tekst_i_raekkefoelge():
    sider = [{"text": "Rask indtil for 8 Dage siden"}, {"text": "da hun fik Brystkatarrh."}]
    assert tekst_af_sider(sider) == "Rask indtil for 8 Dage siden\nda hun fik Brystkatarrh."


def test_tomme_linjer_bevares_som_linjer():
    """En tom linje i svaret er stadig et linjeskift.

    Smider vi den vaek, skrider linjeparringen i maaleapparatet.
    """
    sider = [{"text": "en"}, {"text": ""}, {"text": "to"}]
    assert tekst_af_sider(sider) == "en\n\nto"


def test_margin_metadata_kommer_ikke_med_i_teksten():
    """Margendatoer staar i `metadata`, ikke i `text`.

    De TAELLER med i facit som tekst (lead 2026-08-20), men de staar dér,
    hvor de staar paa siden -- de maa ikke limes ind et vilkaarligt sted af
    os. Kommer de ikke med, er det en promptsag, ikke en oprydningssag.
    """
    sider = [{"text": "Rask indtil", "metadata": "18/12"}]
    assert tekst_af_sider(sider) == "Rask indtil"


def test_manglende_tekstfelt_er_en_fejl_ikke_en_tom_linje():
    # En linje uden `text` betyder, at skemaet ikke blev fulgt. Det skal
    # opdages, ikke maskeres som en tom linje.
    with pytest.raises(ValueError, match="text"):
        tekst_af_sider([{"metadata": "18/12"}])


def test_et_tomt_svar_er_en_fejl():
    """Nul linjer betyder, at modellen ikke svarede -- ikke at siden er tom.

    Gik det igennem som "", ville maalingen se det som en side, modellen
    laeste perfekt daarligt, i stedet for som en fejlet koersel.
    """
    with pytest.raises(ValueError, match="tomt"):
        tekst_af_sider([])


# ----------------------------------------- projektet har sin egen noegle

def test_projektets_noegle_er_IKKE_den_private(monkeypatch, tmp_path):
    """Projektet bruger sin egen noeglefil og ingen anden.

    Skellet skal haandhaeves, ikke blot beskrives: falder koden tilbage til
    en anden noeglefil paa maskinen den dag projektets egen mangler, sker det
    lydloest. Derfor maa ingen anden noeglesti optraede i modulet, og en
    manglende noeglefil skal FEJLE frem for at lede videre.
    """
    import andenside.model as m
    from pathlib import Path

    kilde = Path(m.__file__).read_text(encoding="utf-8")
    assert "api_keys" not in kilde, (
        "modulet peger paa en anden noeglefil end projektets egen"
    )

    mangler = tmp_path / "findes-ikke.json"
    with pytest.raises(NoegleFejl) as fejl:
        hent_noegle(mangler)
    # Beskeden skal pege paa projektets egen fil, saa ingen 'loeser' det ved
    # at kopiere en anden ind.
    assert str(mangler) in str(fejl.value)


def test_noeglefilens_placering_kan_saettes_med_en_miljoevariabel(monkeypatch):
    """Så maskinen kan bestemme stien uden at koden skal aendres."""
    import importlib
    import andenside.model as m

    monkeypatch.setenv("ANDENSIDE_NOEGLEFIL", r"D:\et\andet\sted.json")
    genindlaest = importlib.reload(m)
    try:
        assert str(genindlaest.NOEGLEFIL) == r"D:\et\andet\sted.json"
    finally:
        monkeypatch.delenv("ANDENSIDE_NOEGLEFIL", raising=False)
        importlib.reload(m)


# ---------------------------------------------------------------------------
# Timeout paa det enkelte kald
#
# 2026-08-30 stod en koersel paa 12 sider stille i over ti minutter, mens et
# enkeltkald samtidig svarede paa 14 sekunder. Uden en timeout hverken lykkes
# eller fejler et haengende kald, saa fejlhaandteringen pr. side i
# `koer_pilot.py` udloeses aldrig -- koerslen bliver bare vaek.
#
# Selve haengningen kan ikke fremkaldes offline. Det, der KAN testes, er
# ledningen: at timeouten faktisk gives videre til klienten. Det er ogsaa den,
# der lydloest forsvinder, hvis nogen rydder op i opsaetningen.
# ---------------------------------------------------------------------------

def _fang_klientopsaetning(monkeypatch, tmp_path):
    """Bytter google.genai ud og returnerer de argumenter, klienten fik."""
    import sys
    import types as pytypes

    fanget = {}

    class FalskKlient:
        def __init__(self, **kwargs):
            fanget.update(kwargs)
            self.models = self

        def generate_content(self, **kwargs):
            # Baade klientens og kaldets argumenter fanges. Opsaetningen af
            # kaldet (`config`) er det, temperatur og skema ender i, og den
            # ligger IKKE paa klienten.
            fanget.update(kwargs)
            return pytypes.SimpleNamespace(
                text='{"page_lines": [{"text": "en linje"}]}'
            )

    class FalskHttpOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    falsk_types = pytypes.SimpleNamespace(
        HttpOptions=FalskHttpOptions,
        GenerateContentConfig=lambda **k: k,
        Part=pytypes.SimpleNamespace(from_bytes=lambda **k: k),
    )
    falsk_genai = pytypes.SimpleNamespace(Client=FalskKlient, types=falsk_types)
    monkeypatch.setitem(sys.modules, "google.genai", falsk_genai)
    monkeypatch.setitem(sys.modules, "google", pytypes.SimpleNamespace(genai=falsk_genai))

    billede = tmp_path / "side.png"
    billede.write_bytes(b"ikke et rigtigt billede")
    return fanget, billede


def test_kaldet_faar_en_timeout_med(monkeypatch, tmp_path):
    """Uden denne ledning kan ét kald blokere en hel koersel i det uendelige."""
    from andenside import model as m

    fanget, billede = _fang_klientopsaetning(monkeypatch, tmp_path)
    m.transskriber(billede, "prompt", noegle="x")

    assert "http_options" in fanget, "klienten fik ingen http_options"
    assert fanget["http_options"].kwargs.get("timeout"), "ingen timeout sat"


def test_timeouten_er_i_millisekunder_ikke_sekunder(monkeypatch, tmp_path):
    """Biblioteket vil have millisekunder.

    Sendes 180 i stedet for 180000, faar hvert kald 0,18 sekunder og ALT
    fejler -- eller, hvis biblioteket tolker det som sekunder, er vaernet
    tusind gange for slapt. Begge dele er tavse fejl.
    """
    from andenside import model as m

    fanget, billede = _fang_klientopsaetning(monkeypatch, tmp_path)
    m.transskriber(billede, "prompt", noegle="x")

    assert (fanget["http_options"].kwargs["timeout"]
            == int(m.KALD_TIMEOUT_SEKUNDER * 1000))


# ---------------------------------------------------------------------------
# Temperaturen udelades, hvor den skader
#
# Maalt 2026-08-30 paa samme side: ren tekst tager 79-135 sekunder, mens
# skemabundet tager 8-12. Med `temperature=0.0` gik den skemaloese vej over
# serverens egen frist paa ca. 180 sekunder og fejlede 3 ud af 3; uden
# indstillingen lykkedes den 3 ud af 3.
#
# `temperatur=None` skal derfor udelade indstillingen HELT. Sendes 0.0 i
# stedet, virker den skemaloese variant slet ikke -- og fejlen er tavs, fordi
# den ligner et netvaerksproblem.
# ---------------------------------------------------------------------------

def test_temperatur_none_udelader_indstillingen_helt(monkeypatch, tmp_path):
    from andenside import model as m

    fanget, billede = _fang_klientopsaetning(monkeypatch, tmp_path)
    m.transskriber(billede, "prompt", temperatur=None, skema=None, noegle="x")

    # `GenerateContentConfig` er byttet ud med en dict i proeveopsaetningen.
    assert "temperature" not in fanget["config"], fanget["config"]
    assert fanget["config"]["response_mime_type"] == "text/plain"


def test_en_temperatur_paa_nul_sendes_stadig_naar_den_er_bedt_om(
        monkeypatch, tmp_path):
    """`None` og `0.0` maa ikke forveksles.

    Var de det samme, kunne en koersel ikke laengere bindes til en temperatur,
    og to koersler ville ikke kunne gentages ens.
    """
    from andenside import model as m

    fanget, billede = _fang_klientopsaetning(monkeypatch, tmp_path)
    m.transskriber(billede, "prompt", temperatur=0.0, noegle="x")

    assert fanget["config"]["temperature"] == 0.0
