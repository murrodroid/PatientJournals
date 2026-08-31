"""Maaler en gemt koersel mod facit og skriver rapporten ved siden af svarene.

Det er broen mellem stage 05 og stage 03: piloten gemmer modelsvar, dette
script foerer dem gennem maaleapparatet uden at roere hverken facit eller
svarene. Der kaldes INGEN model -- alt materiale ligger allerede paa disken,
saa en maaling kan koeres om og gentages uden at koste noget.

    .venv/Scripts/python.exe scripts/maal_koersel.py            # nyeste koersel
    .venv/Scripts/python.exe scripts/maal_koersel.py 20260830_141937_beskaaret
    .venv/Scripts/python.exe scripts/maal_koersel.py --alle     # alle koersler

## Laes tallene: hele siden er altid maalt

Maaleapparatet soeger ikke laengere facits linjer frem i modellens tekst --
det sammenligner facits fulde tekst med modellens fulde tekst i ét straek, fra
top til bund (se modul-docstringen i `src/andenside/maal.py`). Der er derfor
ingen "dækning" at laese tallene med haanden paa: hele facit staar altid i
naevneren, for alle varianter, paa alle koersler.

Den ENE undtagelse er den strenge maaling (`cer_raa_streng`): den udelader
med vilje hele linjer med et `[?]`, fordi der ikke findes noget forankret
modstykke at maale dem mod. Hvor stor en del af facit den strenge maaling
overhovedet ser, staar ved siden af som `andel_facit_i_streng` -- en FAST
brøk, ens for alle varianter, ikke en glidende rabat der vokser med, hvor
meget en variant afviger. Sammenlignes koersler paa hovedtallet, er det
allerede den fulde side; den strenge er en ekstra, strengere maaling ved
siden af, ikke en mere daekkende variant af hovedtallet.

## Hvad der skrives

    <koersel>/rapport.md   maaletallene i den aftalte form
    <koersel>/gab.csv      hvad modellen skrev, hvor facit ikke kunne laeses
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROD / "src"))

from andenside.koersel import find_koersler, laes_koersel  # noqa: E402
from andenside.maal import maal_saet  # noqa: E402
from andenside.rapport import skriv_gab, skriv_rapport  # noqa: E402
from andenside.skemaer import tekst_af_svar  # noqa: E402
from andenside.vaern import sikr_oevemaengde  # noqa: E402

FACIT = ROD / "stages" / "02_facit" / "output" / "facit.jsonl"
KOERSLER = ROD / "stages" / "05_foerste_transskription" / "output" / "koersler"


def _andel(m, slags: str) -> str:
    """Fejlandelen som decimaltal. Nul tegn giver 0, ikke en division med nul."""
    if slags == "tegn":
        return f"{m.tegnafstand / m.facit_tegn:.4f}" if m.facit_tegn else "0.0000"
    return f"{m.ordafstand / m.facit_ord:.4f}" if m.facit_ord else "0.0000"


def _facit_for(navne: set[str]) -> list[dict]:
    poster = [json.loads(l) for l in FACIT.read_text(encoding="utf-8").splitlines()]
    return sorted((p for p in poster if p["image_name"] in navne),
                  key=lambda p: p["image_name"])


def _svar_med_metadata(mappe: Path, svar: dict[str, str]) -> dict[str, str] | None:
    """Samme svar, men med margendatoen foldet ind i linjens tekst.

    Kollegaens app laegger datoen i `metadata`; facit har den inline. En
    variant, der samler sine egne dele (`linjefelter`), faar den altsaa med,
    mens `beskrevet` taber den -- og saa vinder den foerste delvist paa
    udfoldningen i stedet for paa skemaet. Her foldes det GEMTE svar ud paa
    begge maader uden et nyt modelkald, saa begge tal kan staa side om side.

    Returnerer None, naar varianten ikke har et metadata-felt at folde ind --
    saa er der intet at sammenligne.
    """
    raa_fil = mappe / "raa_skemasvar.json"
    if not raa_fil.exists():
        return None
    raa = json.loads(raa_fil.read_text(encoding="utf-8"))
    if not any("metadata" in l for d in raa.values()
               for l in d.get("page_lines", [])):
        return None
    anderledes = {}
    for navn, d in raa.items():
        tekst = tekst_af_svar(d, med_metadata=True)
        if tekst != svar.get(navn):
            anderledes[navn] = tekst
    if not anderledes:
        return None
    return {**svar, **anderledes}


def maal_en(mappe: Path) -> dict:
    opsaetning, svar = laes_koersel(mappe)
    # `laes_koersel` giver ikke datoen tilbage, men den staar i filen, og
    # rapporten skal baere den: uden dato kan et tal ikke stedfaestes.
    dato = json.loads((mappe / "opsaetning.json").read_text(encoding="utf-8"))["dato"]
    sikr_oevemaengde(sorted(svar))

    poster = _facit_for(set(svar))
    uden_facit = sorted(set(svar) - {p["image_name"] for p in poster})
    if uden_facit:
        # En side uden facit kan ikke maales. Den skal naevnes, ikke skjules:
        # ellers ser daekningen bedre ud, end den er.
        print(f"  {len(uden_facit)} side(r) uden facit, ikke maalt: "
              + ", ".join(uden_facit))
    if not poster:
        print("  ingen sider med facit -- intet at maale")
        return {}

    saet = maal_saet(poster, svar)
    (mappe / "rapport.md").write_text(
        skriv_rapport(saet, titel=f"Koersel {mappe.name}",
                      model=opsaetning.model,
                      promptversion=opsaetning.promptversion,
                      dato=dato, noter=opsaetning.noter),
        encoding="utf-8")
    (mappe / "gab.csv").write_text(skriv_gab(saet), encoding="utf-8")

    # Samme svar, foldet ud med margendatoen inde i teksten. Koster intet
    # ekstra kald og gør, at varianter med og uden `metadata` kan stilles op
    # mod hinanden uden at udfoldningen afgør sammenligningen.
    med_md = _svar_med_metadata(mappe, svar)
    cer_med_md = ""
    if med_md is not None:
        saet_md = maal_saet(poster, med_md)
        cer_med_md = _andel(saet_md.fladet["raa"], "tegn")

    return {
        "koersel": mappe.name,
        "model": opsaetning.model,
        "variant": opsaetning.variant,
        "sider": len(saet.sider),
        # `raa` er den strengeste og den, der ikke kan pyntes; `arbejdstal`
        # er stage 03's aftalte hovedtal. Begge staar med, saa ingen af dem
        # kan vaelges bagefter, alt efter hvad der ser bedst ud.
        "cer_raa": _andel(saet.fladet["raa"], "tegn"),
        "cer_arbejdstal": _andel(saet.fladet["arbejdstal"], "tegn"),
        "wer_raa": _andel(saet.fladet["raa"], "ord"),
        "cer_raa_streng": _andel(saet.rene["raa"], "tegn"),
        "cer_raa_med_metadata": cer_med_md,
        # Hvor stor en del af facit den strenge maaling overhovedet ser -- en
        # FAST brøk, ens for alle varianter (se `SaetMaaling.andel_af_facit_i_rene`).
        "andel_facit_i_streng": f"{saet.andel_af_facit_i_rene:.4f}",
        # Linjetrofastheden (beslutning 35): hvor mange facit-linjer der har
        # et genkendeligt modstykke, og hvor mange af dem der stod i en
        # anden raekkefoelge.
        "linjer_i_alt": saet.linjer_i_alt,
        "linjer_identiske": saet.identiske_linjer,
        "linjer_omrokeret": saet.linjer_omrokeret,
    }


def sammenlign(mapper: list[Path]) -> None:
    """Stiller koersler op mod hinanden paa de sider, de ALLE har.

    En koersel kan mangle sider -- en frist der loeb ud, eller et kald der
    fejlede. Maales hver koersel paa sit eget saet, sammenlignes varianter paa
    forskelligt materiale, og forskellen kan lige saa godt vaere sidernes som
    variantens. Her skaeres alle ned til faellesmaengden foerst.

    Hovedtallet og den strenge maaling vises side om side. Reglen fra stage 03
    (beslutning 44) gaelder stadig: den strenge udelader linjer med et `[?]`,
    saa den boer normalt vaere LAVERE end hovedtallet. Er den i stedet
    hoejere, har de reddede stumper omkring de ulaeselige steder pyntet paa
    hovedtallet, og saa er det den strenge, der gaelder -- det advares der om
    nedenfor.
    """
    koersler = []
    for mappe in mapper:
        opsaetning, svar = laes_koersel(mappe)
        koersler.append((mappe, opsaetning, svar))

    faelles = set.intersection(*(set(s) for _, _, s in koersler))
    alle = set.union(*(set(s) for _, _, s in koersler))
    if not faelles:
        raise SystemExit("koerslerne har ingen sider til faelles")

    poster = _facit_for(faelles)
    faelles = {p["image_name"] for p in poster}
    print()
    print(f"Sammenligning paa {len(faelles)} faelles sider "
          f"(af {len(alle)} i alt paa tvaers af koerslerne)")
    udeladt = sorted(alle - faelles)
    if udeladt:
        print("  udeladt, fordi mindst én koersel mangler dem: "
              + ", ".join(udeladt))

    print()
    print(f"{'variant':<34} {'sider':>5} {'hovedtal':>9} {'streng':>9} "
          f"{'facit i streng':>15}")
    raekker = []
    for mappe, opsaetning, svar in koersler:
        kun = {n: t for n, t in svar.items() if n in faelles}
        saet = maal_saet(poster, kun)
        raekker.append((opsaetning.promptversion, mappe.name, len(kun),
                        float(_andel(saet.fladet["raa"], "tegn")),
                        float(_andel(saet.rene["raa"], "tegn")),
                        saet.andel_af_facit_i_rene))
    for navn, _, n, hoved, streng, daek_streng in sorted(
            raekker, key=lambda r: r[3]):
        print(f"{navn:<34} {n:>5} {hoved:>8.2%} {streng:>9.2%} "
              f"{daek_streng:>14.1%}")

    # Beslutning 44: den strenge maaling udelader netop de linjer, hvor et
    # ulaeseligt sted lod en kendt stump slippe billigt igennem. Er den
    # ALLIGEVEL hoejere end hovedtallet, har den redning pyntet paa
    # hovedtallet, og saa er det den strenge, der gaelder (PROGRESS.md,
    # stage 03) -- ogsaa naar to koersler stilles op mod hinanden.
    pyntede = [navn for navn, _, _, hoved, streng, _ in raekker
               if streng > hoved]
    if pyntede:
        print()
        print("  BEMAERK: for " + ", ".join(pyntede) + " er den strenge "
              "maaling HOEJERE end hovedtallet.")
        print("  Reddede stumper omkring ulaeselige steder har pyntet paa "
              "hovedtallet -- den strenge gaelder her.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("koersel", nargs="?",
                   help="mappenavn under output/koersler/; standard er den nyeste")
    p.add_argument("--alle", action="store_true",
                   help="maal alle koersler og skriv en samlet oversigt")
    p.add_argument("--sammenlign", nargs="*", metavar="KOERSEL",
                   help="stil koersler op mod hinanden paa deres faelles "
                        "sider; uden navne bruges alle")
    args = p.parse_args()

    alle = find_koersler(KOERSLER)
    if not alle:
        raise SystemExit(f"ingen koersler i {KOERSLER}")

    if args.sammenlign is not None:
        valgte = ([m for m in alle if m.name in args.sammenlign]
                  if args.sammenlign else alle)
        if len(valgte) < 2:
            raise SystemExit("der skal mindst to koersler til en sammenligning")
        sammenlign(sorted(valgte, key=lambda m: m.name))
        return

    if args.alle:
        valgte = alle
    elif args.koersel:
        valgte = [m for m in alle if m.name == args.koersel]
        if not valgte:
            raise SystemExit(f"findes ikke: {args.koersel}\n"
                             + "\n".join(f"  {m.name}" for m in alle))
    else:
        valgte = [max(alle, key=lambda m: m.name)]

    raekker = []
    for mappe in sorted(valgte, key=lambda m: m.name):
        print(f"\n=== {mappe.name} ===")
        r = maal_en(mappe)
        if not r:
            continue
        raekker.append(r)
        print(f"  model {r['model']}, {r['sider']} sider")
        print(f"  tegnfejl raa        {float(r['cer_raa']):.2%}")
        print(f"  tegnfejl arbejdstal {float(r['cer_arbejdstal']):.2%}")
        print(f"  tegnfejl streng     {float(r['cer_raa_streng']):.2%} "
              f"(facit set {float(r['andel_facit_i_streng']):.1%})")
        print(f"  ordfejl raa         {float(r['wer_raa']):.2%}")
        print(f"  linjer identiske    {r['linjer_identiske']}/{r['linjer_i_alt']} "
              f"({r['linjer_identiske'] / r['linjer_i_alt']:.1%})"
              if r["linjer_i_alt"] else "  linjer identiske    0/0")
        print(f"  linjer omrokeret    {r['linjer_omrokeret']}")
        if r["cer_raa_med_metadata"]:
            print(f"  tegnfejl m. margendato "
                  f"{float(r['cer_raa_med_metadata']):.2%}")
        print(f"  rapport  {(mappe / 'rapport.md').relative_to(ROD)}")

    if len(raekker) > 1:
        sti = KOERSLER / "oversigt.csv"
        with sti.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(raekker[0]))
            w.writeheader()
            w.writerows(raekker)
        print(f"\nSamlet oversigt: {sti.relative_to(ROD)}")


if __name__ == "__main__":
    main()
