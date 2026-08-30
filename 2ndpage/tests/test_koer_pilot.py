"""Tests for pilotkoerslens sikkerhedsnet.

Projektets regel: alt med eksterne bivirkninger er toerloeb som standard, og
ingen fuld koersel uden leads go. Den regel er kun noget vaerd, hvis den er
HAANDHAEVET i koden -- ikke kun beskrevet i en docstring. Et toerloeb, der
alligevel kalder modellen, koster penge og roerer materiale, vi ikke har faaet
lov til at roere.

Derfor byttes selve modelkaldet ud med en faelde, der sprænger, hvis den
overhovedet bliver roert.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROD = Path(__file__).resolve().parents[1]


def _indlaes_script():
    """Loader scripts/koer_pilot.py som modul (det er ikke en pakke)."""
    sti = ROD / "scripts" / "koer_pilot.py"
    spec = importlib.util.spec_from_file_location("koer_pilot", sti)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


class Faelde(AssertionError):
    """Rejses, hvis modelkaldet roeres."""


@pytest.fixture
def script(monkeypatch):
    """Scriptet med modelkaldet byttet ud med en taellende faelde.

    Faelden RAISER ogsaa, saa der aldrig gemmes et opdigtet svar. Scriptet
    fanger med vilje fejl pr. side (en enkelt fejlet side maa ikke koste de
    foregaaende svar), saa selve taellingen er det, testene ser paa.
    """
    modul = _indlaes_script()
    modul._kald = []

    def spraeng(billede, prompt, **_):
        modul._kald.append(billede)
        raise Faelde("modelkaldet blev roert")

    monkeypatch.setattr(modul, "transskriber", spraeng)
    return modul


def _koer(modul, monkeypatch, argv: list[str]):
    monkeypatch.setattr(sys, "argv", ["koer_pilot.py", *argv])
    modul.main()


def test_uden_yes_kaldes_modellen_ALDRIG(script, monkeypatch, capsys):
    _koer(script, monkeypatch, ["--variant", "beskaaret"])
    ud = capsys.readouterr().out
    assert script._kald == [], "et tørløb kaldte modellen"
    assert "TØRLØB" in ud
    assert "der er ikke kaldt noget" in ud


def test_uden_yes_gemmes_der_ingen_koersel(script, monkeypatch, tmp_path):
    koersler = script.UD / "koersler"
    foer = set(koersler.iterdir()) if koersler.exists() else set()
    _koer(script, monkeypatch, ["--variant", "beskaaret"])
    efter = set(koersler.iterdir()) if koersler.exists() else set()
    assert foer == efter, "et tørløb må ikke efterlade en gemt kørsel"


def test_med_yes_roeres_modelkaldet_faktisk(script, monkeypatch, capsys):
    """Modstykket: med --yes SKAL modellen kaldes.

    Uden denne test kunne de to ovenfor bestaas af et script, der aldrig
    kalder modellen overhovedet -- og saa maalte de ingenting.
    """
    _koer(script, monkeypatch, ["--variant", "beskaaret", "--yes"])
    capsys.readouterr()
    assert script._kald, "--yes kaldte ikke modellen"


def test_en_fejlet_side_stopper_ikke_de_oevrige(script, monkeypatch, capsys):
    """Faelden fejler paa HVER side, og alle skal alligevel vaere forsoegt.

    En enkelt afvist side maa ikke koste de svar, der allerede er hentet og
    betalt for.
    """
    _koer(script, monkeypatch, ["--variant", "beskaaret", "--yes"])
    ud = capsys.readouterr().out
    assert len(script._kald) > 1
    assert "Ingen sider lykkedes" in ud


def test_toerloeb_viser_et_prisoverslag(script, monkeypatch, capsys):
    # Uden en pris er "kør ikke uden go" ikke en oplyst beslutning.
    _koer(script, monkeypatch, ["--variant", "beskaaret"])
    assert "Prisoverslag" in capsys.readouterr().out


def test_prompten_hentes_ud_af_promptfilen(script):
    prompt = script._prompt("textpage_uaendret")
    assert "expert archivist" in prompt
    # Den menneskelaesbare indramning omkring kodeblokken maa ikke med.
    assert "Kollegaens egen prompt" not in prompt
    assert not prompt.startswith("#")


def test_alle_promptfiler_kan_laeses_og_giver_ren_prompttekst(script):
    """Hver variant skal kunne hentes, og indramningen skal blive udenfor.

    Promptfilerne er skrevet til et menneske: begrundelsen staar rundt om
    selve teksten. Slipper indramningen med ud, sender vi vores egne noter
    til modellen som en del af prompten -- og maaler saa noget andet, end vi
    tror.
    """
    filer = sorted(p.stem for p in script.PROMPTER.glob("*.md"))
    assert len(filer) >= 3, filer
    for navn in filer:
        prompt = script._prompt(navn)
        assert prompt and not prompt.startswith("#"), navn
        assert "```" not in prompt, navn
        assert "Teksten" not in prompt, navn


def test_ukendt_promptnavn_fejler_med_en_liste_over_dem_der_findes(script):
    """En stavefejl i et variantnavn maa ikke koere en tilfaeldig prompt."""
    with pytest.raises(SystemExit) as fejl:
        script._prompt("findes_ikke")
    assert "textpage_uaendret" in str(fejl.value)


def test_ukendt_variant_afvises_af_argumentparseren(script, monkeypatch):
    with pytest.raises(SystemExit):
        _koer(script, monkeypatch, ["--variant", "halvt_opslag"])


def test_en_proeveside_i_listen_STOPPER_koerslen(script, monkeypatch, tmp_path):
    """Den laaste proevemaengde maa aldrig koeres paa ved et uheld.

    `pilotsider.csv` indeholder i dag kun oevesider, saa vaernet ville aldrig
    blive proevet af sig selv. Her lægges en proeveside ind med vilje: koerslen
    skal doe, og modellen skal ALDRIG roeres -- heller ikke for de lovlige
    sider i samme liste.
    """
    from andenside.vaern import ProeveMaengdeFejl

    import csv as _csv
    import json as _json

    # Find en rigtig proeveside via opdelingen -- ikke et opdigtet id, som
    # vaernet ville afvise af en helt anden grund.
    opdeling = ROD / "stages" / "02_facit" / "output" / "opdeling.csv"
    proeve_forsider = {r["forside"] for r in _csv.DictReader(
        opdeling.open(encoding="utf-8")) if r["maengde"] == "proeve"}
    facit = ROD / "stages" / "02_facit" / "output" / "facit.jsonl"
    proeveside = next(
        _json.loads(l)["image_name"]
        for l in facit.read_text(encoding="utf-8").splitlines()
        if _json.loads(l)["forside"] in proeve_forsider
    )

    rigtig = script.UD / "pilotsider.csv"
    lovlige = list(_csv.DictReader(rigtig.open(encoding="utf-8")))
    falsk_ud = tmp_path / "output"
    falsk_ud.mkdir()
    with (falsk_ud / "pilotsider.csv").open("w", encoding="utf-8", newline="") as f:
        skriver = _csv.DictWriter(f, fieldnames=list(lovlige[0]))
        skriver.writeheader()
        skriver.writerows(lovlige[:2])
        skriver.writerow({**lovlige[0], "billede": proeveside})

    monkeypatch.setattr(script, "UD", falsk_ud)
    with pytest.raises(ProeveMaengdeFejl) as fejl:
        _koer(script, monkeypatch, ["--variant", "beskaaret", "--yes"])

    assert proeveside in str(fejl.value)
    assert script._kald == [], "modellen blev kaldt trods en prøveside i listen"


# ---------------------------------------------------------------------------
# Fristen pr. side
#
# 2026-08-30: to koersler paa 12 sider stod stille i henholdsvis ti og seks
# minutter, mens enkeltkald samtidig svarede paa 10-12 sekunder. Bibliotekets
# egen http-timeout afbroed ikke kaldet. Uden en frist, vi selv haandhaever,
# forsvinder en koersel bare -- den hverken lykkes eller fejler, og
# fejlhaandteringen pr. side udloeses aldrig.
# ---------------------------------------------------------------------------

def test_en_haengende_side_stopper_ikke_de_oevrige(script, monkeypatch, capsys):
    """Den side, der haenger, skal opgives -- resten skal koeres faerdig.

    Uden fristen bliver ALLE svar vaek, ogsaa dem der allerede var hentet,
    fordi koerslen aldrig naar frem til at gemme.
    """
    import time

    kaldte = []

    def haenger_paa_den_anden(sti, prompt, **kwargs):
        navn = Path(sti).stem
        kaldte.append(navn)
        if len(kaldte) == 2:
            time.sleep(30)          # laengere end fristen nedenfor
        return "en linje", {"page_lines": [{"text": "en linje"}]}

    monkeypatch.setattr(script, "transskriber", haenger_paa_den_anden)
    monkeypatch.setattr(script, "FORSOEGSFRIST_SEKUNDER", 1)
    monkeypatch.setattr(script, "FORSOEG_PR_SIDE", 1)   # ingen gentagelse her

    gemte = {}
    monkeypatch.setattr(script, "gem_koersel",
                        lambda rod, o, svar, **k: (gemte.update(svar),
                                                   _tom_mappe(rod))[1])

    _koer(script, monkeypatch, ["--antal", "3", "--yes"])
    ud = capsys.readouterr().out

    assert "OPGIVET" in ud
    # De to andre sider skal vaere gemt -- ikke tabt sammen med den haengende.
    assert len(gemte) == 2, gemte


def _tom_mappe(rod):
    rod.mkdir(parents=True, exist_ok=True)
    return rod


# ------------------------------------------------------------------
# Gentagelse ved serverfejl
#
# Maalt 2026-08-30 paa 16 kald: et kald, der lykkes, tager 7-12 sekunder i
# ALLE varianter. Fejler det, haenger det til serverens egen frist paa ca.
# 180 sekunder og svarer 504. Fejlen er binaer, og fejlraten afhaenger
# staerkt af prompt og skema -- 0 % paa 56 kald for nogle kombinationer,
# 61 % for andre.
#
# Uden gentagelse maales varianterne derfor paa FORSKELLIGE sider, og
# forskellen mellem dem bliver sidernes i stedet for variantens. Det skete:
# to af seks varianter fik kun 8 og 4 af 12 sider igennem, og
# faellesmaengden faldt til 3 sider.
# ------------------------------------------------------------------


def test_en_side_proeves_igen_efter_en_serverfejl(script, monkeypatch,
                                                  capsys):
    """En 504 er forbigaaende. Anden gang skal siden vaere med."""
    forsoeg = {}

    def fejler_foerste_gang(sti, prompt, **kwargs):
        navn = Path(sti).stem
        forsoeg[navn] = forsoeg.get(navn, 0) + 1
        if forsoeg[navn] == 1:
            raise RuntimeError("504 DEADLINE_EXCEEDED")
        return "en linje", {"page_lines": [{"text": "en linje"}]}

    monkeypatch.setattr(script, "transskriber", fejler_foerste_gang)
    gemte = {}
    monkeypatch.setattr(script, "gem_koersel",
                        lambda rod, o, svar, **k: (gemte.update(svar),
                                                   _tom_mappe(rod))[1])

    _koer(script, monkeypatch, ["--antal", "3", "--yes"])
    ud = capsys.readouterr().out

    assert len(gemte) == 3, f"alle tre sider skulle vaere reddet: {gemte}"
    assert "forsoeg 2" in ud, ud
    assert "OPGIVET" not in ud


def test_en_side_der_ALTID_fejler_opgives_uden_at_haenge_koerslen(
        script, monkeypatch, capsys):
    """Gentagelse maa ikke blive uendelig.

    Et kald, der aldrig lykkes, skal opgives efter det aftalte antal
    forsoeg -- ellers bytter vi en haengende koersel for en uendelig.
    """
    kald = []
    # Den daarlige side udpeges som den ANDEN, scriptet naar frem til, ikke
    # ved navn: hvilke sider `--antal` vaelger afhaenger af pilotsider.csv, og
    # en test, der binder sig til et billed-id, gaar stille i staa den dag
    # udvalget aendrer sig -- den ville saa bestaa uden at proeve noget.
    daarlig: list[str] = []

    def fejler_altid_paa_den_anden(sti, prompt, **kwargs):
        navn = Path(sti).stem
        if not daarlig and len(set(kald)) == 1 and navn not in kald:
            daarlig.append(navn)
        kald.append(navn)
        if navn in daarlig:
            raise RuntimeError("504 DEADLINE_EXCEEDED")
        return "en linje", {"page_lines": [{"text": "en linje"}]}

    monkeypatch.setattr(script, "transskriber", fejler_altid_paa_den_anden)
    monkeypatch.setattr(script, "FORSOEG_PR_SIDE", 3)
    gemte = {}
    monkeypatch.setattr(script, "gem_koersel",
                        lambda rod, o, svar, **k: (gemte.update(svar),
                                                   _tom_mappe(rod))[1])

    _koer(script, monkeypatch, ["--antal", "3", "--yes"])
    ud = capsys.readouterr().out

    assert daarlig, "testen naaede aldrig at udpege en daarlig side"
    assert "OPGIVET efter 3 forsoeg" in ud, ud
    assert len([n for n in kald if n == daarlig[0]]) == 3
    assert len(gemte) == 2, "de oevrige sider skulle stadig vaere gemt"
    assert daarlig[0] not in gemte
