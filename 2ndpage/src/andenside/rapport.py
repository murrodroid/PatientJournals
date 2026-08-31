"""Rapportformatet -- én maaling skrevet ud, saa den kan laeses af et menneske.

Formatet er aftalt FOER foerste modelkald, saa tallene ikke bliver formet
efter, hvad der ser godt ud.

Den store aendring 2026-08-31: forankringen er vaek (se `maal.py`s docstring).
Dermed er ogsaa **daekning** og **rabat** vaek som begreber. Hele facit er
altid i naevneren, saa der findes ikke laengere et "maalt paa X % af tegnene"
at saette ved hvert tal, ingen liste over "tyndest maalte sider", og ingen
fuldside-kontrol -- kontrollen og det, den kontrollerede, er blevet det samme
tal. Skriver nogen de ord ind igen, er det en fejl, ikke en udeladelse.

Ét obligatorisk forbehold staar tilbage: **facit rummer selv fejl**
(beslutning 37). Én er bekraeftet. En enkelt uenighed mellem model og facit er
derfor ikke automatisk modellens fejl.

Rapporten udpeger desuden de vaerste enkeltsider, saa de kan ses efter med
oejnene frem for kun at indgaa i et gennemsnit.
"""
from __future__ import annotations

from andenside import cer
from andenside.maal import SaetMaaling, SideMaaling
from andenside.sidemaaling import JOKER_LOFT

VAERSTE_SIDER = 10
GAB_I_RAPPORTEN = 15

FORBEHOLD = """> **Sådan læses tallene.** Facit rummer selv fejl (beslutning 37). Én er
> bekræftet ved kontrol: `37554_001491` skriver "for 2 Dage siden", hvor der
> på siden står "for 3 Dage siden". En enkelt uenighed mellem model og facit
> er altså ikke i sig selv modellens fejl.
>
> `raa` er tallet, leverancen står ved. `arbejdstal` (uden versaler og
> tegnsætning) er det, vi træffer valg ud fra. De øvrige varianter viser,
> hvor meget af fejlen der er ortografisk støj frem for egentlige læsefejl —
> ingen af dem må vælges, fordi den klæder resultatet."""


SAADAN_MAALES_DER = f"""## Sådan er der målt

**Hele siden sammenlignes i ét stræk, fra øverste linje til nederste.** Facits
tekst og modellens tekst stilles op mod hinanden som to lange tekster, og der
tælles, hvor mange enkelttegn der skal rettes, indsættes eller slettes for at
komme fra den ene til den anden. Linjeskiftene er taget ud på begge sider, og
ord, der er delt hen over et linjeskift, er sat sammen igen.

**Der bliver ikke søgt.** Ingen linje bliver ledt op inde i modellens tekst.
Der er kun én vej gennem siden, oppefra og ned, og hele facit er altid med.
Det er den vigtige forskel fra de tidligere rapporter: dengang blev hver
facit-linje søgt frem i modelsvaret, og en linje, der ikke kunne findes, faldt
helt ud af regnestykket i begge tekster. Et gentaget ord kunne dermed sende
søgningen langt ned på siden og tage alle de mellemliggende linjer med sig ud
af målingen. Det kan ikke længere ske, og der findes derfor heller ikke
længere noget tal for, hvor stor en del af siden der blev målt: svaret er
altid hele siden.

**Rækkefølgen tæller med.** Skriver modellen sidens linjer i en anden orden,
end de står, koster det. Det er med vilje: rækkefølgen er data i en
patientjournal. Hvor meget af fejlen der skyldes netop dét, står i afsnittet
*Rækkefølge og linjer* nedenfor.

**Hvor facit siger `[?]`** — et sted transskribenten ikke kunne læse — må
modellen skrive noget, uden at det koster. Der findes jo ingen sandhed at måle
det imod. Men fribilletten har et loft: op til {JOKER_LOFT} tegn indhold
(mellemrum tæller ikke med) er gratis, og skriver modellen mere, koster
overskuddet ét point pr. tegn. Loftet er der, fordi et sted uden loft ville
lade en model springe vilkårligt langt frem i sin egen tekst gratis og dermed
få rigtige fejl slugt af det ulæselige sted ved siden af. Det, modellen skrev
de steder, gemmes og står nederst i rapporten.

**Ordforklaring:** *tegnafstand* = antallet af enkelttegn, der skal rettes,
indsættes eller slettes. *CER* (tegnfejl) er den afstand delt med antallet af
tegn i facit; *WER* (ordfejl) er det samme regnet på hele ord og er derfor
altid et større tal — ét forkert bogstav gør hele ordet forkert. *Fladet
tekst* betyder, at linjeskiftene er taget ud og delte ord samlet igen."""


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
    """Én raekke i tabellen over de vaerste sider.

    Kolonnerne er valgt, saa en daarlig side kan afkodes uden at slaa op:
    tegnfejlen, hvor mange af sidens linjer der rummer et ulaeseligt sted, hvor
    mange linjer modellen skrev i forkert orden, og om den overhovedet skrev
    lige saa meget tekst, som der stod paa siden.
    """
    m = side.fladet["arbejdstal"]
    return (
        f"| `{side.image_name}` | {_pct(m.cer)} | "
        f"{side.svaere_linjer}/{side.linjer_i_alt} | "
        f"{side.omrokering.antal_flyttede} | "
        f"{side.model_tegn_i_alt}/{side.facit_tegn_i_alt} |"
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
    streng = saet.rene
    sider = saet.sider

    # Opsummeringer regnes ét sted, saa de samme tal ikke kan komme til at
    # staa forskelligt i to afsnit af samme rapport.
    model_i_alt = sum(s.model_tegn_i_alt for s in sider)
    facit_i_alt = sum(s.facit_tegn_i_alt for s in sider)
    svaere = sum(s.svaere_linjer for s in sider)
    gab = saet.gab
    jokertegn = saet.joker_tegn_i_alt
    overskud = saet.joker_overskud

    linjer_i_alt = saet.linjer_i_alt
    parret = saet.linjer_parret
    uparret = linjer_i_alt - parret
    flyttede = saet.linjer_omrokeret
    identiske = saet.identiske_linjer

    ud: list[str] = [
        f"# {titel}",
        "",
        "| Bogholderi | |",
        "|---|---|",
        f"| Model | `{model}` |",
        f"| Promptversion | `{promptversion}` |",
        f"| Dato | {dato} |",
        f"| Sider målt | {len(sider)} |",
        "| Facit-udgave | `alt_*` (beslutning 24) |",
        "",
        FORBEHOLD,
        "",
        SAADAN_MAALES_DER,
        "",
        "## Hovedtal — hele siden",
        "",
        f"Målt på alle {len(sider)} siders fulde tekst — {facit_i_alt} tegn i alt,",
        f"fordelt på {linjer_i_alt} linjer, hvoraf {svaere} rummer mindst ét `[?]`.",
        "Intet er udeladt.",
        "",
        "Alle seks varianter står side om side (beslutning 26); ingen af dem må",
        "vælges efter, hvilken der klæder resultatet bedst. **Tegnfejl er",
        "beslutningstallet**, ordfejl står ved siden af som et groft mål for, hvor",
        "mange ord der overhovedet er ramt.",
        "",
    ]
    ud += _varianttabel(fladet)

    # Den strenge maaling staar HER, lige efter hovedtallet, fordi den er
    # svaret paa "er hovedtallet skaevt?" -- ikke et sidetal.
    ud += [
        "",
        "## Den strenge måling — uden linjer med et ulæseligt sted",
        "",
        "Hovedtallet ovenfor har alle linjer med, også dem hvor transskribenten",
        "gav op midt i og skrev `[?]`. Her er den samme måling, hvor hele den",
        "slags linje er taget ud af facit, så modellen hverken kan straffes eller",
        "belønnes for dem. Det er samtidig konventionen i faget: Transkribus og",
        "beslægtede værktøjer udelader hele linjen ved ulæselige steder, så netop",
        "dette tal kan sammenlignes med anden forskning.",
        "",
        f"Den strenge måling ser **{_pct(saet.andel_af_facit_i_rene)} af facits "
        "tegn**; resten ligger på linjer med mindst ét `[?]`.",
        "",
        "**Udeladelsen er FAST.** Den afhænger udelukkende af facit — af hvilke",
        "linjer transskribenten satte et `[?]` i — og er derfor nøjagtig den",
        "samme for alle seks varianter og for alle modeller, vi nogensinde måler.",
        "Det er den afgørende forskel fra den *dækning*, de tidligere rapporter",
        "opgjorde: dén flyttede sig, alt efter hvor meget af siden søgningen",
        "kunne genfinde i det enkelte modelsvar, og gav dermed mest rabat til den",
        "model, der afveg mest. Det væltede konklusionen 30. august. Sådan et tal",
        "findes ikke længere nogen steder i rapporten.",
        "",
    ]
    ud += _varianttabel(streng)

    hoved_cer = fladet["arbejdstal"].cer
    streng_cer = streng["arbejdstal"].cer
    ud += [
        "",
        f"**Sammenlign de to.** Hovedtallet er {_pct(hoved_cer)}, den strenge er "
        f"{_pct(streng_cer)} (`arbejdstal`) — en forskel på "
        f"{_pct(abs(streng_cer - hoved_cer))}.",
        "",
        "**Er den strenge lavere**, er de svære linjer sværere end resten af",
        "teksten — det ventede. Hovedtallet kan bruges, som det står, fordi det",
        "hviler på al teksten.",
        "",
        "**Er den strenge højere, gælder den strenge.** Så har modellen fået",
        "noget forærende af de ulæselige steder: den skrev noget dér, som slap",
        "gratis igennem under loftet, og det pynter kun på hovedtallet. Den",
        "strenge måling kan ikke rammes af det, fordi den slet ikke ser de",
        "linjer. Vælg derfor altid det højeste af de to, når de er uenige.",
        "",
        "## Rækkefølge og linjer",
        "",
        "Målingen ovenfor er streng om rækkefølgen: skriver modellen sidens",
        "linjer i en anden orden, tæller det som fejl på lige fod med forkert",
        "læste ord. Tallene her viser, hvor meget af fejlen der er af den slags.",
        "De regnes ved at parre hver facit-linje med den modellinje, den ligner",
        "mest, og se efter, hvilken orden de parrede linjer så står i.",
        "",
        "| Mål | Værdi |",
        "|---|---:|",
        f"| Facit-linjer i alt | {linjer_i_alt} |",
        f"| Linjer med et genkendeligt modstykke hos modellen | {parret} |",
        f"| Linjer uden modstykke (modellen sprang dem over eller læste noget helt andet) | {uparret} |",
        f"| Parrede linjer, der står i forkert indbyrdes rækkefølge | {flyttede} |",
        f"| Linjer modellen ramte nøjagtigt | {identiske} |",
        "",
        "\"Ramte nøjagtigt\" betyder ord for ord ens, når man ser bort fra",
        "versaler, accenter og tegnsætning.",
        "",
        "> **Forbehold — det her er vejledende tal, ikke beslutningstal.** De",
        "> kommer ikke fra hovedmålingen, men fra en parring af linjer lavet",
        "> alene til formålet. Parringen tager facit-linjerne oppefra og ned og",
        "> giver hver af dem den bedste ledige modellinje. Det har en kendt",
        "> svaghed: står der flere næsten ens linjer på siden — og det gør der",
        "> tit i journalmateriale, hvor de samme vitale værdier gentages — kan en",
        "> tidlig facit-linje nå at lægge beslag på en modellinje, der rettelig",
        "> hørte til en senere facit-linje. Så bliver både \"uden modstykke\" og",
        "> \"forkert rækkefølge\" en anelse for høje. En rigtig løsning kræver en",
        "> global optimal tildeling og er ikke lavet. Brug tallene til at forstå",
        "> tegnfejlen, ikke til at træffe beslutninger.",
        "",
        "## Opdigtning",
        "",
        "Signaler for, om modellen skriver noget, den ikke har dækning for.",
        "Ingen af dem er et korrekthedsmål — dér hvor facit siger `[?]`, findes",
        "der ingen sandhed at måle imod.",
        "",
        "| Signal | Værdi |",
        "|---|---:|",
        f"| Tekst henført til de ulæselige steder (øvre grænse) | {jokertegn} tegn fordelt på {len(gab)} steder |",
        f"| Heraf over fribilletten, og altså talt som fejl | {overskud} tegn |",
        f"| Modellens tekst i alt mod facits | {model_i_alt} mod {facit_i_alt} tegn |",
        "",
        f"**De {jokertegn} tegn er en ØVRE grænse, ikke et mål for opdigtning.**",
        "Fribilletten er gratis indtil loftet, og målingen har derfor ingen grund",
        "til at holde igen: den lader gerne det ulæselige sted æde et par af",
        "nabordene med, når de alligevel er gratis. En del af tallet er altså",
        "tekst, modellen har læst helt rigtigt. Det er efterprøvet — på rigtige",
        "sider lægger tallet sig lige præcis op ad loftet, netop fordi den sidste",
        "plads bliver fyldt op med korrekt nabotekst.",
        "",
        f"**Det skarpe signal er de {overskud} tegn over fribilletten.** Dem har",
        "modellen skrevet ud over, hvad et ulæseligt sted overhovedet kan dække,",
        f"og de er talt som fejl. Er det tal stort, skriver modellen lange",
        "passager, hvor transskribenten kun kunne se ét ord — og så er hovedtallet",
        "i forvejen mildt over for den, fordi den første del af hvert sted var",
        "gratis.",
        "",
        "Den sidste linje er det groveste, men også det mest robuste signal:",
        "skriver modellen væsentligt flere tegn end der står på siden, har den",
        "lagt noget til; skriver den væsentligt færre, har den sprunget noget",
        "over. Begge dele er allerede talt med i tegnfejlen ovenfor — linjen her",
        "siger blot, hvilken af de to slags fejl der dominerer.",
    ]

    # De vaerste sider, saa de kan ses efter med oejnene.
    vaerste = sorted(sider, key=lambda s: (-s.fladet["arbejdstal"].cer, s.image_name))
    ud += [
        "",
        f"## De {min(VAERSTE_SIDER, len(vaerste))} værste sider",
        "",
        "Sorteret efter tegnfejl (`arbejdstal`). Se dem efter med øjnene, før",
        "tallet tros. Kolonnen *Linjer med `[?]`* siger, hvor svær siden var at",
        "læse i første omgang; *Linjer i forkert orden* siger, om fejlen er",
        "omrokering frem for forkert læsning; *Modeltegn/facittegn* siger, om",
        "modellen skrev for meget eller for lidt.",
        "",
        "| Side | Tegnfejl | Linjer med `[?]` | Linjer i forkert orden | Modeltegn/facittegn |",
        "|---|---:|---:|---:|---:|",
    ]
    ud += [_sidelinje(s) for s in vaerste[:VAERSTE_SIDER]]

    if gab:
        ud += [
            "",
            "## Hvad modellen skrev, hvor facit siger `[?]`",
            "",
            "Skrives ud, fordi det er modellens bud på steder, transskribenten",
            "ikke kunne læse. Det er IKKE facit og må aldrig skrives ind i det —",
            "arbejdsgangen med udklip og ja/nej hører i stage 07.",
            "",
            "Facits egne ord på hver side af det ulæselige sted står med, så",
            "stedet kan findes igen på siden med det blotte øje. Den fulde liste",
            "ligger i gab-filen.",
            "",
            "| Side | Facit før | Modellens bud | Facit efter | Tegn |",
            "|---|---|---|---|---:|",
        ]
        for navn, g in gab[:GAB_I_RAPPORTEN]:
            bud = " ".join(g.model_tekst.split()) or "*(intet)*"
            ud.append(
                f"| `{navn}` | {g.facit_foer} | {bud} | {g.facit_efter} | "
                f"{g.indholdstegn} |"
            )
        if len(gab) > GAB_I_RAPPORTEN:
            ud.append(f"| … | | *{len(gab) - GAB_I_RAPPORTEN} steder mere* | | |")

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
    linjer = ["side,facit_foer,model_tekst,facit_efter,indholdstegn"]
    for navn, g in saet.gab:
        felter = [
            navn,
            g.facit_foer,
            " ".join(g.model_tekst.split()),
            g.facit_efter,
            str(g.indholdstegn),
        ]
        linjer.append(",".join(_csv_felt(f) for f in felter))
    return "\n".join(linjer) + "\n"


def _csv_felt(vaerdi: str) -> str:
    if any(c in vaerdi for c in ',"\n'):
        return '"' + vaerdi.replace('"', '""') + '"'
    return vaerdi
