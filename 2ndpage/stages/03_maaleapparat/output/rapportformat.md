# Rapportformat — eksempel på en færdig måling

| Bogholderi | |
|---|---|
| Model | `INGEN — konstrueret prøve, ikke et modelsvar` |
| Promptversion | `—` |
| Dato | 2026-08-22 |
| Sider målt | 118 |
| Facit-udgave | `alt_*` (beslutning 24) |

> **Sådan læses tallene.** Dækningen står ved hvert tal, fordi de linjer,
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
> ingen af dem må vælges, fordi den klæder resultatet.

## Sådan er der målt

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
delt hen over et linjeskift, er sat sammen igen.

## Hovedtal — fladet tekst

Målt på **97,20 % af facits tegn** (93,58 % af linjerne). De udeladte er de sværeste.

| Variant | Tegnfejl (CER) | Ordfejl (WER) | Tegnafstand | Facit-tegn |
|---|---:|---:|---:|---:|
| `raa` | 3,72 % | 19,71 % | 2350 | 63148 |
| `uden_versaler` | 3,72 % | 19,71 % | 2350 | 63148 |
| `uden_diakritika` | 3,72 % | 19,71 % | 2350 | 63148 |
| `uden_tegnsaetning` | 3,92 % | 19,82 % | 2349 | 59857 |
| `arbejdstal` | 3,92 % | 19,82 % | 2349 | 59857 |
| `lempeligst` | 3,92 % | 19,82 % | 2349 | 59857 |

Af de 297 linjer med mindst ét `[?]` kunne forankringen redde **280** ind i målingen ved at måle de kendte stumper omkring det ulæselige sted. Grundreglen er ellers, at hele linjen går ud (beslutning 38), så uden det trin ville dækningen have været væsentligt lavere.

## Pr. linje

Samme tekst, men målt linje for linje efter at linjerne er parret via
forankringen. Skrider ikke ved et afvigende linjebrud, fordi parringen
sker på indhold og ikke på linjenummer.

| Variant | Tegnfejl (CER) | Ordfejl (WER) | Tegnafstand | Facit-tegn |
|---|---:|---:|---:|---:|
| `raa` | 3,84 % | 19,47 % | 2350 | 61208 |
| `uden_versaler` | 3,84 % | 19,47 % | 2350 | 61208 |
| `uden_diakritika` | 3,84 % | 19,47 % | 2350 | 61208 |
| `uden_tegnsaetning` | 4,07 % | 19,58 % | 2349 | 57736 |
| `arbejdstal` | 4,07 % | 19,58 % | 2349 | 57736 |
| `lempeligst` | 4,07 % | 19,58 % | 2349 | 57736 |

Linjer der er nøjagtig rigtige (`arbejdstal`): 988 af 2420 = 40,83 %

## Kontrol — hele siden uden forankring

På de **25 sider uden et eneste `[?]`** kan hele siden
sammenlignes direkte, uden forankring og med fuld dækning. Det er
den eneste måling i rapporten, der ikke kan pynte på noget.

| Variant | Tegnfejl (CER) | Ordfejl (WER) | Tegnafstand | Facit-tegn |
|---|---:|---:|---:|---:|
| `raa` | 9,78 % | 23,54 % | 1050 | 10737 |
| `uden_versaler` | 9,78 % | 23,54 % | 1050 | 10737 |
| `uden_diakritika` | 9,78 % | 23,54 % | 1050 | 10737 |
| `uden_tegnsaetning` | 10,07 % | 23,55 % | 1024 | 10164 |
| `arbejdstal` | 10,07 % | 23,55 % | 1024 | 10164 |
| `lempeligst` | 10,07 % | 23,55 % | 1024 | 10164 |

Kontrollen ligger på **10,07 %** mod hovedtallets **3,92 %** (`arbejdstal`) — en forskel på 6,15 %.

**Ligger kontrollen væsentligt HØJERE, måler forankringen ikke alt.**
Den ser hverken tekst, modellen har fundet på, eller tekst, den har
sprunget over — kun det, der kunne parres. Forskellen er altså ikke
støj, den er den del af fejlen, hovedtallet lader ligge, og den skal
læses sammen med opdigtningstallene nedenfor.

Ligger de to tæt, måler hovedtallet reelt hele teksten, og forskellen
mellem dem er blot, at kontrollen kun dækker de nemmeste sider — dem
helt uden ulæselige steder.

## Opdigtning

To signaler. Ingen af dem er et korrekthedsmål — der findes ingen sandhed
at måle imod dér, hvor facit siger `[?]` — men de siger, om modellen
skriver noget, den ikke har dækning for. Det tredje sted at kigge er
kontroltallet ovenfor: det er det eneste, der tæller opdigtet tekst med
som egentlige fejl.

| Signal | Værdi |
|---|---:|
| Modeltekst uden modstykke i facit | 5527 tegn = 9,32 % af modellens tekst |
| Tekst skrevet dér hvor facit siger `[?]` | 894 tegn fordelt på 92 steder |

**"Uden modstykke" har et gulv og er ikke nul, selv når intet er digtet.**
Modellen skriver noget dér, hvor facit siger `[?]`, og den skriver også de
linjer, forankringen ikke kunne parre. Målt på facit mod facit selv ligger
gulvet omkring 2.500 tegn for øvemængden (se `selvtest.md`). Tallet skal
derfor læses som et tillæg til det gulv, ikke som et absolut mål for
opdigtning.

## Linjetrofasthed

Svaret på det, der indtil nu har været en formodning (beslutning 35):
laver modellen sine egne linjeskift, eller følger den sidens?

| Mål | Værdi |
|---|---:|
| Facit-linjer der ligger inden for én af modellens linjer | 2420 af 2420 |
| Facit-linjer der får deres egen modellinje | 2420 af 2420 |

**Sådan læses de to tal.** Er de begge lig antallet af målte linjer,
har modellen skrevet sidens linjer, som de står — én facit-linje pr.
modellinje. Er det FØRSTE tal højt og det andet lavt, har modellen
samlet flere af sidens linjer i én af sine egne. Er det første tal lavt,
løber facits linjer hen over modellens linjeskift, altså laver modellen
sine egne brud. Ingen af delene er en fejl i sig selv, og ingen af dem
påvirker tallene ovenfor — men svaret afgør, om linjeskiftene kan
afleveres videre til kollegaens `PageLine`-skema, og det er værd at vide.

## De 10 værste sider

Sorteret efter `arbejdstal`. Se dem efter med øjnene, før tallet tros —
en enkelt side med en fejlagtig parring kan trække hele hovedtallet.

| Side | Tegnfejl | Dækning | Linjer målt | Modeltekst uden modstykke |
|---|---:|---:|---:|---:|
| `273100_001258` | 6,36 % | 99,03 % | 23/23 | 34 |
| `273108_001554` | 6,11 % | 94,90 % | 20/23 | 83 |
| `273105_001571` | 5,95 % | 95,09 % | 22/23 | 91 |
| `273103_001463` | 5,80 % | 97,30 % | 22/28 | 40 |
| `273108_001557` | 5,71 % | 97,66 % | 6/6 | 32 |
| `273100_001294` | 5,54 % | 99,21 % | 22/22 | 23 |
| `273103_001437` | 5,54 % | 96,30 % | 24/26 | 51 |
| `273109_000081` | 5,53 % | 94,72 % | 20/20 | 65 |
| `37554_001498` | 5,52 % | 98,50 % | 12/14 | 28 |
| `273103_001436` | 5,24 % | 98,09 % | 26/26 | 45 |

## De 10 tyndest målte sider

Lav dækning er et værre tegn end høj tegnfejl: her er der næsten ikke
målt på siden, så dens tal betyder ikke noget. En side, hvor modellen
sprang det meste over eller skrev noget helt andet, dukker op HER — ikke
i listen ovenfor, hvor den tværtimod ser god ud.

| Side | Dækning | Tegnfejl | Linjer målt | Modeltekst uden modstykke |
|---|---:|---:|---:|---:|
| `273104_001633` | 74,93 % | 4,15 % | 21/26 | 203 |
| `273108_001555` | 89,17 % | 4,20 % | 18/18 | 114 |
| `273110_001529` | 90,32 % | 2,40 % | 6/6 | 47 |
| `273100_001306` | 90,59 % | 4,19 % | 21/26 | 70 |
| `273105_001570` | 91,21 % | 3,24 % | 24/26 | 115 |
| `273105_001708` | 92,04 % | 3,23 % | 26/28 | 114 |
| `273105_001711` | 92,95 % | 4,06 % | 21/23 | 101 |
| `273104_001640` | 93,14 % | 3,20 % | 23/25 | 92 |
| `273098_001503` | 93,45 % | 3,39 % | 16/17 | 42 |
| `273103_001467` | 93,75 % | 2,70 % | 23/25 | 90 |

## Hvad modellen skrev, hvor facit siger `[?]`

Skrives ud, fordi det er modellens bud på steder, transskribenten
ikke kunne læse. Det er IKKE facit og må aldrig skrives ind i det —
arbejdsgangen med udklip og ja/nej hører i stage 07.

| Side | Facit | Modellens bud |
|---|---|---|
| `273098_001498` | `[?]` | utydeligt |
| `273098_001498` | `[?]` | utydelsgt |
| `273098_001499` | `[?]` | utydeligt |
| `273098_001503` | `[?]` | utydeligj |
| `273098_001503` | `[?]` | utydeligt |
| `273098_001503` | `[?]` | utydeligt |
| `273098_001503` | `[?]. [?]` | utydeligt. utydeuigt |
| `273098_001509` | `[?] [?]` | utydelijt utydeligt |
| `273098_001512` | `[?]` | utydeligt |
| `273098_001513` | `[?]` | utydeligt |
| `273100_001294` | `[?]` | utydeligt |
| `273100_001295` | `[?]` | utydeliet |
| `273100_001306` | `[?]` | utydeligt |
| `273100_001306` | `[?]` | utydelimt |
| `273100_001306` | `[?]` | utydeligt |
| … | | *77 steder mere* |

## Noter

**Dette er ikke en måling af en model.** "Modelteksten" er facit selv med 5 % af bogstaverne byttet tilfældigt og et opdigtet afsnit sat til sidst på hver side, så alle rapportens felter har noget at vise. Formatet er aftalt her, før første modelkald, så tallene ikke bliver formet efter, hvad der ser godt ud.

Tallene selv betyder derfor ingenting. Det, der skal tages stilling til, er om det er DE FELTER, der skal træffes valg ud fra.

To ting i tabellerne er artefakter af den konstruerede prøve og vil se anderledes ud ved et rigtigt modelsvar: `raa`, `uden_versaler` og `uden_diakritika` er ens, fordi forvanskningen hverken ændrer store bogstaver eller omlyde — og linjetrofastheden er 100 %, fordi "modellen" her per konstruktion skriver facits egne linjeskift.

Kørt på øvemængden. Prøvemængden røres først ved den endelige bedømmelse.
