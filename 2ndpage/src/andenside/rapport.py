"""Rapportformatet -- én maaling skrevet ud, saa den kan laeses af et menneske.

Formatet er aftalt FOER foerste modelkald, saa tallene ikke bliver formet
efter, hvad der ser godt ud. To ting er obligatoriske og maa ikke kunne
slaas fra:

1. **Daekningen staar ved hvert eneste tal.** De linjer, vi ikke maaler, er
   ikke tilfaeldige -- det er de svaereste linjer paa siden, og det er derfor,
   transskribenten ikke kunne laese dem. Tallet er systematisk for pænt.
2. **Facit rummer selv fejl** (beslutning 37). Én er bekraeftet. En enkelt
   uenighed mellem model og facit er derfor ikke automatisk modellens fejl.

Rapporten udpeger desuden de vaerste enkeltsider, saa de kan ses efter med
oejnene frem for kun at indgaa i et gennemsnit.
"""
from __future__ import annotations

from andenside import cer
from andenside.maal import SaetMaaling, SideMaaling

VAERSTE_SIDER = 10
GAB_I_RAPPORTEN = 15

FORBEHOLD = """> **Sådan læses tallene.** Dækningen står ved hvert tal, fordi de linjer,
> der ikke er målt, er de sværeste på siden — dem transskribenten selv ikke
> kunne læse. Tallet er derfor systematisk for pænt, og forskellen bliver
> større, jo lavere dækningen er.
>
> Facit rummer selv fejl (beslutning 37). Én er bekræftet ved kontrol:
> `37554_001491` skriver "for 2 Dage siden", hvor der på siden står "for 3
> Dage siden". En enkelt uenighed mellem model og facit er altså ikke i sig
> selv modellens fejl.
>
> `raa` er tallet, leverancen står ved. `arbejdstal` (uden versaler og
> tegnsætning) er det, vi træffer valg ud fra. De øvrige varianter viser,
> hvor meget af fejlen der er ortografisk støj frem for egentlige læsefejl —
> ingen af dem må vælges, fordi den klæder resultatet."""


SAADAN_MAALES_DER = """## Sådan er der målt

Rapporten bruger ordet **forankring** hele vejen igennem, så her er hvad det
betyder, i almindeligt sprog:

Facits tekst søges frem i modellens tekst, én linje ad gangen, fra toppen af
siden og nedefter. Søgningen tåler læsefejl — den leder efter det stykke
modeltekst, der ligner facit-linjen mest, og godtager det, hvis det ikke
afviger for meget. Er stykket fundet, er linjen *forankret*, og de to
tekststykker kan sammenlignes tegn for tegn. Kan linjen ikke findes, går den
helt ud af målingen, i begge tekster.

Der søges i modellens **rå tekst uden hensyn til dens linjeskift**. Derfor er
det ligegyldigt for tallene, om modellen følger sidens linjer eller laver sine
egne — og derfor kan vi måle bagefter, hvad den faktisk gjorde (se
*Linjetrofasthed* nedenfor) i stedet for at gætte på forhånd.

Hvor facit siger `[?]` — et sted transskribenten ikke kunne læse — deles
linjen, og de kendte stumper på hver side søges hver for sig. Det, modellen
skrev i mellemrummet, måles ikke; der findes ingen sandhed at måle det imod.
Men det gemmes, både fordi længden siger noget om, hvor tilbøjelig modellen er
til at digte, og fordi det er dens bud på et sted, ingen har kunnet læse.

**Ordforklaring:** *tegnafstand* = hvor mange enkelttegn der skal rettes,
indsættes eller slettes for at nå fra modellens tekst til facits. *CER* er den
afstand delt med antallet af tegn i facit; *WER* er det samme regnet på hele
ord, og den er derfor altid et større tal — ét forkert bogstav gør hele ordet
forkert. *Fladet tekst* betyder, at linjeskiftene er taget ud og ord, der er
delt hen over et linjeskift, er sat sammen igen."""


def _pct(x: float) -> str:
    return f"{x * 100:.2f} %".replace(".", ",")


def _varianttabel(maal: dict[str, cer.Maaltal]) -> list[str]:
    linjer = [
        "| Variant | Tegnfejl (CER) | Ordfejl (WER) | Tegnafstand | Facit-tegn |",
        "|---|---:|---:|---:|---:|",
    ]
    for navn in cer.VARIANTER:
        m = maal[navn]
        linjer.append(
            f"| `{navn}` | {_pct(m.cer)} | {_pct(m.wer)} | {m.tegnafstand} | {m.facit_tegn} |"
        )
    return linjer


def _sidelinje(side: SideMaaling) -> str:
    m = side.fladet["arbejdstal"]
    return (
        f"| `{side.image_name}` | {_pct(m.cer)} | {_pct(side.daekning)} | "
        f"{side.linjer_maalt}/{side.linjer_i_alt} | {side.model_tegn_uforankret} |"
    )


def skriv_rapport(
    saet: SaetMaaling,
    *,
    titel: str,
    model: str,
    promptversion: str,
    dato: str,
    noter: str = "",
) -> str:
    """Bygger hele rapporten som markdown.

    Bogholderiet oeverst (model, promptversion, dato) er der, for at en
    koersel kan genfindes -- modelsvar er ikke deterministiske, saa tallene
    er kun meningsfulde sammen med, hvad der frembragte dem.
    """
    fladet = saet.fladet
    pr_linje = saet.pr_linje
    sider = saet.sider

    # Opsummeringer regnes ét sted, saa de samme tal ikke kan komme til at
    # staa forskelligt i to afsnit af samme rapport.
    model_i_alt = sum(s.model_tegn_i_alt for s in sider)
    uforankret = sum(s.model_tegn_uforankret for s in sider)
    gab = saet.gab
    gabtegn = sum(len("".join(g.model_tekst.split())) for _, g in gab)
    svaere = sum(s.svaere_linjer for s in sider)
    reddet = sum(s.svaere_linjer_reddet for s in sider)
    uden_skift = sum(s.uden_linjeskift_indeni for s in sider)
    egen = sum(s.egen_modellinje for s in sider)
    maalte_linjer = sum(s.linjer_maalt for s in sider)

    ud: list[str] = [
        f"# {titel}",
        "",
        "| Bogholderi | |",
        "|---|---|",
        f"| Model | `{model}` |",
        f"| Promptversion | `{promptversion}` |",
        f"| Dato | {dato} |",
        f"| Sider målt | {len(sider)} |",
        f"| Facit-udgave | `alt_*` (beslutning 24) |",
        "",
        FORBEHOLD,
        "",
        SAADAN_MAALES_DER,
        "",
        "## Hovedtal — fladet tekst",
        "",
        f"Målt på **{_pct(saet.daekning)} af facits tegn** "
        f"({_pct(saet.linjedaekning)} af linjerne). "
        "De udeladte er de sværeste.",
        "",
    ]
    ud += _varianttabel(fladet)
    ud += [
        "",
        f"Af de {svaere} linjer med mindst ét `[?]` kunne forankringen redde "
        f"**{reddet}** ind i målingen ved at måle de kendte stumper omkring "
        "det ulæselige sted. Grundreglen er ellers, at hele linjen går ud "
        "(beslutning 38), så uden det trin ville dækningen have været "
        "væsentligt lavere.",
    ]

    # Den strenge maaling ved siden af hovedtallet. Staar HER, lige efter det,
    # fordi den er svaret paa "er hovedtallet skaevt?" -- ikke et sidetal.
    streng = saet.rene
    ud += [
        "",
        "## Uden de linjer, der rummer et ulæseligt sted",
        "",
        "Hovedtallet ovenfor tager de kendte stumper med fra linjer, hvor",
        "transskribenten gav op — teksten på hver side af et `[?]`. Det er",
        "netop dér, både modellen og opdelingen er mest usikre, så det kan",
        "trække tallet skævt. Her er den samme måling med de linjer helt ude.",
        "",
        f"Den strenge måling ser **{_pct(saet.andel_af_facit_i_rene)} af facits tegn** "
        f"(resten ligger på linjer med mindst ét `[?]`) og fik fat i "
        f"{_pct(saet.rene_daekning)} af dem.",
        "",
    ]
    ud += _varianttabel(streng)
    ud += [
        "",
        f"**Sammenlign de to.** Hovedtallet er {_pct(fladet['arbejdstal'].cer)}, "
        f"den strenge er {_pct(streng['arbejdstal'].cer)} (`arbejdstal`) — "
        f"en forskel på {_pct(abs(streng['arbejdstal'].cer - fladet['arbejdstal'].cer))}.",
        "",
        "Ligger de tæt, tilfører redningen af de svære linjer ingen skævhed, og",
        "hovedtallet kan bruges, fordi det hviler på mest tekst. Er den strenge",
        "**lavere**, er de reddede stumper sværere end resten, og hovedtallet er",
        "for pessimistisk. Er den strenge **højere**, har redningen pyntet, og så",
        "er det den strenge, der gælder.",
    ]

    ud += [
        "",
        "## Pr. linje",
        "",
        "Samme tekst, men målt linje for linje efter at linjerne er parret via",
        "forankringen. Skrider ikke ved et afvigende linjebrud, fordi parringen",
        "sker på indhold og ikke på linjenummer.",
        "",
    ]
    ud += _varianttabel(pr_linje)
    ud += [
        "",
        f"Linjer der er nøjagtig rigtige (`arbejdstal`): "
        f"{pr_linje['arbejdstal'].stykker_identiske} af "
        f"{pr_linje['arbejdstal'].stykker_maalt} = "
        f"{_pct(pr_linje['arbejdstal'].andel_identiske)}",
    ]

    # Kontroltallet. Uden det kan man ikke vide, om forankringen pynter.
    kontrol = saet.sider_med_fuldsidekontrol
    ud += ["", "## Kontrol — hele siden uden forankring", ""]
    if kontrol:
        her = saet.fuldside["arbejdstal"].cer
        der = fladet["arbejdstal"].cer
        forskel = her - der
        ud += [
            f"På de **{kontrol} sider uden et eneste `[?]`** kan hele siden",
            "sammenlignes direkte, uden forankring og med fuld dækning. Det er",
            "den eneste måling i rapporten, der ikke kan pynte på noget.",
            "",
        ]
        ud += _varianttabel(saet.fuldside)
        ud += [
            "",
            f"Kontrollen ligger på **{_pct(her)}** mod hovedtallets **{_pct(der)}** "
            f"(`arbejdstal`) — en forskel på {_pct(abs(forskel))}.",
            "",
            "**Ligger kontrollen væsentligt HØJERE, måler forankringen ikke alt.**",
            "Den ser hverken tekst, modellen har fundet på, eller tekst, den har",
            "sprunget over — kun det, der kunne parres. Forskellen er altså ikke",
            "støj, den er den del af fejlen, hovedtallet lader ligge, og den skal",
            "læses sammen med opdigtningstallene nedenfor.",
            "",
            "Ligger de to tæt, måler hovedtallet reelt hele teksten, og forskellen",
            "mellem dem er blot, at kontrollen kun dækker de nemmeste sider — dem",
            "helt uden ulæselige steder.",
        ]
    else:
        ud += ["Ingen af de målte sider er helt uden `[?]`, så kontrollen kan ikke køres."]

    ud += [
        "",
        "## Opdigtning",
        "",
        "To signaler. Ingen af dem er et korrekthedsmål — der findes ingen sandhed",
        "at måle imod dér, hvor facit siger `[?]` — men de siger, om modellen",
        "skriver noget, den ikke har dækning for. Det tredje sted at kigge er",
        "kontroltallet ovenfor: det er det eneste, der tæller opdigtet tekst med",
        "som egentlige fejl.",
        "",
        "| Signal | Værdi |",
        "|---|---:|",
        f"| Modeltekst uden modstykke i facit | {uforankret} tegn "
        f"= {_pct(uforankret / model_i_alt if model_i_alt else 0)} af modellens tekst |",
        f"| Tekst skrevet dér hvor facit siger `[?]` | {gabtegn} tegn fordelt på {len(gab)} steder |",
        "",
        "**\"Uden modstykke\" har et gulv og er ikke nul, selv når intet er digtet.**",
        "Modellen skriver noget dér, hvor facit siger `[?]`, og den skriver også de",
        "linjer, forankringen ikke kunne parre. Målt på facit mod facit selv ligger",
        "gulvet omkring 2.500 tegn for øvemængden (se `selvtest.md`). Tallet skal",
        "derfor læses som et tillæg til det gulv, ikke som et absolut mål for",
        "opdigtning.",
        "",
        "## Linjetrofasthed",
        "",
        "Svaret på det, der indtil nu har været en formodning (beslutning 35):",
        "laver modellen sine egne linjeskift, eller følger den sidens?",
        "",
        "| Mål | Værdi |",
        "|---|---:|",
        f"| Facit-linjer der ligger inden for én af modellens linjer | {uden_skift} af {maalte_linjer} |",
        f"| Facit-linjer der får deres egen modellinje | {egen} af {maalte_linjer} |",
        "",
        "**Sådan læses de to tal.** Er de begge lig antallet af målte linjer,",
        "har modellen skrevet sidens linjer, som de står — én facit-linje pr.",
        "modellinje. Er det FØRSTE tal højt og det andet lavt, har modellen",
        "samlet flere af sidens linjer i én af sine egne. Er det første tal lavt,",
        "løber facits linjer hen over modellens linjeskift, altså laver modellen",
        "sine egne brud. Ingen af delene er en fejl i sig selv, og ingen af dem",
        "påvirker tallene ovenfor — men svaret afgør, om linjeskiftene kan",
        "afleveres videre til kollegaens `PageLine`-skema, og det er værd at vide.",
    ]

    # De vaerste sider, saa de kan ses efter med oejnene.
    vaerste = sorted(sider, key=lambda s: (-s.fladet["arbejdstal"].cer, s.image_name))
    ud += [
        "",
        f"## De {min(VAERSTE_SIDER, len(vaerste))} værste sider",
        "",
        "Sorteret efter `arbejdstal`. Se dem efter med øjnene, før tallet tros —",
        "en enkelt side med en fejlagtig parring kan trække hele hovedtallet.",
        "",
        "| Side | Tegnfejl | Dækning | Linjer målt | Modeltekst uden modstykke |",
        "|---|---:|---:|---:|---:|",
    ]
    ud += [_sidelinje(s) for s in vaerste[:VAERSTE_SIDER]]

    # Listen ovenfor kan ikke staa alene. En side, hvor kun en fjerdedel af
    # teksten kunne forankres, faar et flot tegnfejlstal -- der er jo naesten
    # ikke maalt paa den -- og lander i BUNDEN af listen, hvor ingen kigger.
    # Lav daekning er et vaerre tegn end hoej tegnfejl, fordi den betyder, at
    # tallet for siden ikke betyder noget.
    tyndest = sorted(sider, key=lambda s: (s.daekning, s.image_name))
    ud += [
        "",
        f"## De {min(VAERSTE_SIDER, len(tyndest))} tyndest målte sider",
        "",
        "Lav dækning er et værre tegn end høj tegnfejl: her er der næsten ikke",
        "målt på siden, så dens tal betyder ikke noget. En side, hvor modellen",
        "sprang det meste over eller skrev noget helt andet, dukker op HER — ikke",
        "i listen ovenfor, hvor den tværtimod ser god ud.",
        "",
        "| Side | Dækning | Tegnfejl | Linjer målt | Modeltekst uden modstykke |",
        "|---|---:|---:|---:|---:|",
    ]
    for side in tyndest[:VAERSTE_SIDER]:
        m = side.fladet["arbejdstal"]
        ud.append(
            f"| `{side.image_name}` | {_pct(side.daekning)} | {_pct(m.cer)} | "
            f"{side.linjer_maalt}/{side.linjer_i_alt} | {side.model_tegn_uforankret} |"
        )

    if gab:
        ud += [
            "",
            "## Hvad modellen skrev, hvor facit siger `[?]`",
            "",
            "Skrives ud, fordi det er modellens bud på steder, transskribenten",
            "ikke kunne læse. Det er IKKE facit og må aldrig skrives ind i det —",
            "arbejdsgangen med udklip og ja/nej hører i stage 07.",
            "",
            "| Side | Facit | Modellens bud |",
            "|---|---|---|",
        ]
        for navn, g in gab[:GAB_I_RAPPORTEN]:
            bud = " ".join(g.model_tekst.split()) or "*(intet)*"
            ud.append(f"| `{navn}` | `{g.facit_mellem}` | {bud} |")
        if len(gab) > GAB_I_RAPPORTEN:
            ud.append(f"| … | | *{len(gab) - GAB_I_RAPPORTEN} steder mere* |")

    if noter:
        ud += ["", "## Noter", "", noter]

    return "\n".join(ud) + "\n"


def skriv_gab(saet: SaetMaaling) -> str:
    """Alle gab som CSV -- ikke kun de faa, rapporten har plads til.

    Kontrakten siger, at gabene skal skrives til en fil (rod-CONTEXT.md
    2026-08-21, "Hvad der IKKE bygges nu"): der bygges ingen gennemsyns-app i
    denne stage, men materialet til den skal ligge klar. Arbejdsgangen med
    taette udklip og ja/nej hoerer i stage 07.

    Filen er IKKE facit og maa aldrig skrives ind i det. Den er en liste over,
    hvad modellen skrev de steder, transskribenten ikke kunne laese.

    Raekkefoelgen foelger sidernes navne og derefter positionen paa siden, saa
    to koersler giver samme fil.
    """
    linjer = ["side,facit_mellem,model_tekst,model_start,model_slut"]
    for navn, g in saet.gab:
        felter = [
            navn,
            g.facit_mellem,
            " ".join(g.model_tekst.split()),
            str(g.model_start),
            str(g.model_slut),
        ]
        linjer.append(",".join(_csv_felt(f) for f in felter))
    return "\n".join(linjer) + "\n"


def _csv_felt(vaerdi: str) -> str:
    if any(c in vaerdi for c in ',"\n'):
        return '"' + vaerdi.replace('"', '""') + '"'
    return vaerdi
