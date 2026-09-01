# Rapportformat — eksempel på en færdig måling

| Bogholderi | |
|---|---|
| Model | `INGEN — konstrueret prøve, ikke et modelsvar` |
| Promptversion | `—` |
| Dato | 2026-08-22 |
| Sider målt | 118 |
| Facit-udgave | `alt_*` (beslutning 24) |

> **Sådan læses tallene.** Facit rummer selv fejl (beslutning 37). Én er
> bekræftet ved kontrol: `37554_001491` skriver "for 2 Dage siden", hvor der
> på siden står "for 3 Dage siden". En enkelt uenighed mellem model og facit
> er altså ikke i sig selv modellens fejl.
>
> `raa` er tallet, leverancen står ved. `arbejdstal` (uden versaler og
> tegnsætning) er det, vi træffer valg ud fra. De øvrige varianter viser,
> hvor meget af fejlen der er ortografisk støj frem for egentlige læsefejl —
> ingen af dem må vælges, fordi den klæder resultatet.

## Sådan er der målt

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
det imod. Men fribilletten har et loft: op til 15 tegn indhold
(mellemrum tæller ikke med) er gratis, og skriver modellen mere, koster
overskuddet ét point pr. tegn. Loftet er der, fordi et sted uden loft ville
lade en model springe vilkårligt langt frem i sin egen tekst gratis og dermed
få rigtige fejl slugt af det ulæselige sted ved siden af. Det, modellen skrev
de steder, gemmes og står nederst i rapporten.

**Ordforklaring:** *tegnafstand* = antallet af enkelttegn, der skal rettes,
indsættes eller slettes. *CER* (tegnfejl) er den afstand delt med antallet af
tegn i facit; *WER* (ordfejl) er det samme regnet på hele ord og er derfor
altid et større tal — ét forkert bogstav gør hele ordet forkert. *Fladet
tekst* betyder, at linjeskiftene er taget ud og delte ord samlet igen.

## Hovedtal — hele siden

Målt på alle 118 siders fulde tekst — 54284 tegn i alt,
fordelt på 2586 linjer, hvoraf 297 rummer mindst ét `[?]`.
Intet er udeladt.

Alle seks varianter står side om side (beslutning 26); ingen af dem må
vælges efter, hvilken der klæder resultatet bedst. **Tegnfejl er
beslutningstallet**, ordfejl står ved siden af som et groft mål for, hvor
mange ord der overhovedet er ramt.

| Variant | Tegnfejl (CER) | Ordfejl (WER) | Tegnafstand | Facit-tegn |
|---|---:|---:|---:|---:|
| `raa` | 8,46 % | 23,58 % | 5368 | 63455 |
| `uden_versaler` | 8,46 % | 23,58 % | 5367 | 63455 |
| `uden_diakritika` | 8,46 % | 23,58 % | 5368 | 63455 |
| `uden_tegnsaetning` | 8,73 % | 23,21 % | 5242 | 60054 |
| `arbejdstal` | 8,73 % | 23,21 % | 5241 | 60054 |
| `lempeligst` | 8,73 % | 23,21 % | 5241 | 60054 |

## Den strenge måling — uden linjer med et ulæseligt sted

Hovedtallet ovenfor har alle linjer med, også dem hvor transskribenten
gav op midt i og skrev `[?]`. Her er den samme måling, hvor hele den
slags linje er taget ud af facit, så modellen hverken kan straffes eller
belønnes for dem. Det er samtidig konventionen i faget: Transkribus og
beslægtede værktøjer udelader hele linjen ved ulæselige steder, så netop
dette tal kan sammenlignes med anden forskning.

Den strenge måling ser **87,84 % af facits tegn**; resten ligger på linjer med mindst ét `[?]`.

**Udeladelsen er FAST.** Den afhænger udelukkende af facit — af hvilke
linjer transskribenten satte et `[?]` i — og er derfor nøjagtig den
samme for alle seks varianter og for alle modeller, vi nogensinde måler.
Det er den afgørende forskel fra den *dækning*, de tidligere rapporter
opgjorde: dén flyttede sig, alt efter hvor meget af siden søgningen
kunne genfinde i det enkelte modelsvar, og gav dermed mest rabat til den
model, der afveg mest. Det væltede konklusionen 30. august. Sådan et tal
findes ikke længere nogen steder i rapporten.

| Variant | Tegnfejl (CER) | Ordfejl (WER) | Tegnafstand | Facit-tegn |
|---|---:|---:|---:|---:|
| `raa` | 8,72 % | 23,55 % | 4953 | 56827 |
| `uden_versaler` | 8,71 % | 23,55 % | 4952 | 56827 |
| `uden_diakritika` | 8,72 % | 23,55 % | 4953 | 56827 |
| `uden_tegnsaetning` | 8,90 % | 23,55 % | 4792 | 53863 |
| `arbejdstal` | 8,89 % | 23,55 % | 4790 | 53863 |
| `lempeligst` | 8,89 % | 23,55 % | 4790 | 53863 |

**Sammenlign de to.** Hovedtallet er 8,73 %, den strenge er 8,89 % (`arbejdstal`) — en forskel på 0,17 %.

**Er den strenge lavere**, er de svære linjer sværere end resten af
teksten — det ventede. Hovedtallet kan bruges, som det står, fordi det
hviler på al teksten.

**Er den strenge højere, gælder den strenge.** Så har modellen fået
noget forærende af de ulæselige steder: den skrev noget dér, som slap
gratis igennem under loftet, og det pynter kun på hovedtallet. Den
strenge måling kan ikke rammes af det, fordi den slet ikke ser de
linjer. Vælg derfor altid det højeste af de to, når de er uenige.

## Rækkefølge og linjer

Målingen ovenfor er streng om rækkefølgen: skriver modellen sidens
linjer i en anden orden, tæller det som fejl på lige fod med forkert
læste ord. Tallene her viser, hvor meget af fejlen der er af den slags.
De regnes ved at parre hver facit-linje med den modellinje, den ligner
mest, og se efter, hvilken orden de parrede linjer så står i.

| Mål | Værdi |
|---|---:|
| Facit-linjer i alt | 2586 |
| Linjer med et genkendeligt modstykke hos modellen | 2510 |
| Linjer uden modstykke (modellen sprang dem over eller læste noget helt andet) | 76 |
| Parrede linjer, der står i forkert indbyrdes rækkefølge | 6 |
| Linjer modellen ramte nøjagtigt | 1000 |

"Ramte nøjagtigt" betyder ord for ord ens, når man ser bort fra
versaler, accenter og tegnsætning.

> **Forbehold — det her er vejledende tal, ikke beslutningstal.** De
> kommer ikke fra hovedmålingen, men fra en parring af linjer lavet
> alene til formålet. Parringen tager facit-linjerne oppefra og ned og
> giver hver af dem den bedste ledige modellinje. Det har en kendt
> svaghed: står der flere næsten ens linjer på siden — og det gør der
> tit i journalmateriale, hvor de samme vitale værdier gentages — kan en
> tidlig facit-linje nå at lægge beslag på en modellinje, der rettelig
> hørte til en senere facit-linje. Så bliver både "uden modstykke" og
> "forkert rækkefølge" en anelse for høje. En rigtig løsning kræver en
> global optimal tildeling og er ikke lavet. Brug tallene til at forstå
> tegnfejlen, ikke til at træffe beslutninger.

## Opdigtning

Signaler for, om modellen skriver noget, den ikke har dækning for.
Ingen af dem er et korrekthedsmål — dér hvor facit siger `[?]`, findes
der ingen sandhed at måle imod.

| Signal | Værdi |
|---|---:|
| Tekst henført til de ulæselige steder (øvre grænse) | 4134 tegn fordelt på 354 steder |
| Heraf over fribilletten, og altså talt som fejl | 168 tegn |
| Modellens tekst i alt mod facits | 59305 mod 54284 tegn |

**De 4134 tegn er en ØVRE grænse, ikke et mål for opdigtning.**
Fribilletten er gratis indtil loftet, og målingen har derfor ingen grund
til at holde igen: den lader gerne det ulæselige sted æde et par af
nabordene med, når de alligevel er gratis. En del af tallet er altså
tekst, modellen har læst helt rigtigt. Det er efterprøvet — på rigtige
sider lægger tallet sig lige præcis op ad loftet, netop fordi den sidste
plads bliver fyldt op med korrekt nabotekst.

**Det skarpe signal er de 168 tegn over fribilletten.** Dem har
modellen skrevet ud over, hvad et ulæseligt sted overhovedet kan dække,
og de er talt som fejl. Er det tal stort, skriver modellen lange
passager, hvor transskribenten kun kunne se ét ord — og så er hovedtallet
i forvejen mildt over for den, fordi den første del af hvert sted var
gratis.

Den sidste linje er det groveste, men også det mest robuste signal:
skriver modellen væsentligt flere tegn end der står på siden, har den
lagt noget til; skriver den væsentligt færre, har den sprunget noget
over. Begge dele er allerede talt med i tegnfejlen ovenfor — linjen her
siger blot, hvilken af de to slags fejl der dominerer.

## De 10 værste sider

Sorteret efter tegnfejl (`arbejdstal`). Se dem efter med øjnene, før
tallet tros. Kolonnen *Linjer med `[?]`* siger, hvor svær siden var at
læse i første omgang; *Linjer i forkert orden* siger, om fejlen er
omrokering frem for forkert læsning; *Modeltegn/facittegn* siger, om
modellen skrev for meget eller for lidt.

| Side | Tegnfejl | Linjer med `[?]` | Linjer i forkert orden | Modeltegn/facittegn |
|---|---:|---:|---:|---:|
| `273105_001572` | 33,33 % | 0/4 | 0 | 99/76 |
| `273108_001538` | 28,43 % | 1/4 | 0 | 122/93 |
| `273106_001695` | 27,88 % | 1/5 | 0 | 128/99 |
| `273104_001635` | 25,66 % | 0/7 | 0 | 126/103 |
| `273100_001308` | 24,76 % | 0/3 | 0 | 114/91 |
| `273108_001557` | 23,74 % | 1/6 | 0 | 157/128 |
| `273110_001529` | 21,37 % | 1/6 | 0 | 160/124 |
| `273105_001714` | 19,18 % | 0/9 | 0 | 153/130 |
| `273100_001260` | 18,75 % | 2/8 | 0 | 197/161 |
| `273111_001377` | 17,11 % | 0/8 | 0 | 185/162 |

## Hvad modellen skrev, hvor facit siger `[?]`

Skrives ud, fordi det er modellens bud på steder, transskribenten
ikke kunne læse. Det er IKKE facit og må aldrig skrives ind i det —
arbejdsgangen med udklip og ja/nej hører i stage 07.

Facits egne ord på hver side af det ulæselige sted står med, så
stedet kan findes igen på siden med det blotte øje. Den fulde liste
ligger i gab-filen.

| Side | Facit før | Modellens bud | Facit efter | Tegn |
|---|---|---|---|---:|
| `273098_001496` | rachitisk. Der er rigelig | gtydelige | Udflod fra Næsen, ingen | 9 |
| `273098_001498` | noget har ikke været | utydeligt | , drikker Kun lidt, | 9 |
| `273098_001498` | at afstødes. Foetor idag | utydelsgt | mindre. P. nogenlunde regelm. | 9 |
| `273098_001498` | Rp. Serum dan. 10cbctm. | utydeligt | ufor. Rp. | 9 |
| `273098_001498` | ufor. Rp. | utyrecigt | . sol. nitr. arg. | 9 |
| `273098_001499` | varm. Pulsen er regelmæssig | utydeligt | kraftig. Døser ikke. Igår | 9 |
| `273098_001499` | middelstærke rester af belægning | utydeligt | (infiltration) Udseendet er i | 9 |
| `273098_001502` | Hun er varm p | utydeligt | Extremiteterne nu, køligere p | 9 |
| `273098_001502` | gennem næsen. 2 dr | uty |  | 3 |
| `273098_001502` |  | deligt uwydeligt | P. c 120 lille. | 15 |
| `273098_001503` | begyndte at drikke; med | utydeligj | kl 1 Tiden begyndte | 9 |
| `273098_001503` | Tiden begyndte hun at | utydeligt | Skrigeture med c. 10 | 9 |
| `273098_001503` | v. Varmedunke. Ingen Kramper, | utydeligt | Opkastn. | 9 |
| `273098_001503` | Opkastn. | utydeligt | 2 | 9 |
| `273098_001503` | 2 | utydeligt | Aabn. og Vandladn. i | 9 |
| … | | *339 steder mere* | | |

## Noter

**Dette er ikke en måling af en model.** "Modelteksten" er facit selv med 5 % af bogstaverne byttet tilfældigt og et opdigtet afsnit sat til sidst på hver side, så alle rapportens felter har noget at vise. Formatet er aftalt her, før første modelkald, så tallene ikke bliver formet efter, hvad der ser godt ud.

Tallene selv betyder derfor ingenting. Det, der skal tages stilling til, er om det er DE FELTER, der skal træffes valg ud fra.

To ting i tabellerne er artefakter af den konstruerede prøve og vil se anderledes ud ved et rigtigt modelsvar: `raa`, `uden_versaler` og `uden_diakritika` er ens, fordi forvanskningen hverken ændrer store bogstaver eller omlyde — og linjetrofastheden er 100 %, fordi "modellen" her per konstruktion skriver facits egne linjeskift.

Kørt på øvemængden. Prøvemængden røres først ved den endelige bedømmelse.
