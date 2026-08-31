# Koersel 20260830_162704_beskaaret

| Bogholderi | |
|---|---|
| Model | `gemini-3.1-pro-preview` |
| Promptversion | `textpage_uaendret/beskrevet` |
| Dato | 2026-08-30T16:27:04 |
| Sider målt | 8 |
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

Målt på **87,71 % af facits tegn** (87,77 % af linjerne). De udeladte er de sværeste.

| Variant | Tegnfejl (CER) | Ordfejl (WER) | Tegnafstand | Facit-tegn |
|---|---:|---:|---:|---:|
| `raa` | 9,64 % | 27,51 % | 419 | 4345 |
| `uden_versaler` | 8,79 % | 24,35 % | 382 | 4345 |
| `uden_diakritika` | 9,64 % | 27,51 % | 419 | 4345 |
| `uden_tegnsaetning` | 8,71 % | 23,79 % | 360 | 4133 |
| `arbejdstal` | 7,82 % | 20,06 % | 323 | 4133 |
| `lempeligst` | 7,82 % | 20,06 % | 323 | 4133 |

Af de 47 linjer med mindst ét `[?]` kunne forankringen redde **43** ind i målingen ved at måle de kendte stumper omkring det ulæselige sted. Grundreglen er ellers, at hele linjen går ud (beslutning 38), så uden det trin ville dækningen have været væsentligt lavere.

## Uden de linjer, der rummer et ulæseligt sted

Hovedtallet ovenfor tager de kendte stumper med fra linjer, hvor
transskribenten gav op — teksten på hver side af et `[?]`. Det er
netop dér, både modellen og opdelingen er mest usikre, så det kan
trække tallet skævt. Her er den samme måling med de linjer helt ude.

Den strenge måling ser **73,68 % af facits tegn** (resten ligger på linjer med mindst ét `[?]`) og fik fat i 93,24 % af dem.

| Variant | Tegnfejl (CER) | Ordfejl (WER) | Tegnafstand | Facit-tegn |
|---|---:|---:|---:|---:|
| `raa` | 9,66 % | 27,22 % | 328 | 3394 |
| `uden_versaler` | 8,90 % | 23,84 % | 302 | 3394 |
| `uden_diakritika` | 9,66 % | 27,22 % | 328 | 3394 |
| `uden_tegnsaetning` | 8,80 % | 23,71 % | 284 | 3229 |
| `arbejdstal` | 7,99 % | 20,14 % | 258 | 3229 |
| `lempeligst` | 7,99 % | 20,14 % | 258 | 3229 |

**Sammenlign de to.** Hovedtallet er 7,82 %, den strenge er 7,99 % (`arbejdstal`) — en forskel på 0,17 %.

Hvor stor en forskel der skal til, før den betyder noget, kan aflæses
af selvtesten: dér forvanskes alle bogstaver lige meget, så de svære
linjer er netop IKKE sværere end resten, og alligevel skiller de to tal
sig 0,08 procentpoint ad (7,92 % mod 7,84 % ved 10 % forvanskning). En
forskel af den størrelse er maskineriet selv. Er forskellen flere gange
større, kommer den fra materialet, og så siger retningen følgende:

**Er den strenge lavere**, er de reddede stumper sværere end resten af
teksten. Det er det ventede — de ligger op ad de steder, transskribenten
selv gav op over for. Hovedtallet er da en smule for pessimistisk og kan
bruges som det står, fordi det hviler på mest tekst.

**Er den strenge højere, er det et advarselstegn**, og så er det den
strenge, der gælder. Grunden er, at en reddet stump kun tæller med i
hovedtallet, hvis den overhovedet kunne findes i modellens svar. Har
modellen slagtet stumpen — læst noget helt andet, eller sprunget stedet
over — kan søgningen ikke genkende den, og linjen falder UD af målingen
i stedet for at tælle som en fejl. Tilbage bliver kun de stumper,
modellen klarede. Hovedtallet opgør altså de vellykkede redninger og
kommer til at se pænere ud, jo dårligere modellen faktisk læste de
svære steder. Den strenge måling kan ikke rammes af det, fordi den slet
ikke ser de linjer.

## Pr. linje

Samme tekst, men målt linje for linje efter at linjerne er parret via
forankringen. Skrider ikke ved et afvigende linjebrud, fordi parringen
sker på indhold og ikke på linjenummer.

| Variant | Tegnfejl (CER) | Ordfejl (WER) | Tegnafstand | Facit-tegn |
|---|---:|---:|---:|---:|
| `raa` | 10,27 % | 28,49 % | 432 | 4208 |
| `uden_versaler` | 9,36 % | 25,37 % | 394 | 4208 |
| `uden_diakritika` | 10,27 % | 28,49 % | 432 | 4208 |
| `uden_tegnsaetning` | 9,21 % | 23,74 % | 367 | 3986 |
| `arbejdstal` | 8,25 % | 20,05 % | 329 | 3986 |
| `lempeligst` | 8,25 % | 20,05 % | 329 | 3986 |

Linjer der er nøjagtig rigtige (`arbejdstal`): 79 af 165 = 47,88 %

## Kontrol — hele siden uden forankring

Ingen af de målte sider er helt uden `[?]`, så kontrollen kan ikke køres.

## Opdigtning

To signaler. Ingen af dem er et korrekthedsmål — der findes ingen sandhed
at måle imod dér, hvor facit siger `[?]` — men de siger, om modellen
skriver noget, den ikke har dækning for. Det tredje sted at kigge er
kontroltallet ovenfor: det er det eneste, der tæller opdigtet tekst med
som egentlige fejl.

| Signal | Værdi |
|---|---:|
| Modeltekst uden modstykke i facit | 397 tegn = 10,05 % af modellens tekst |
| Tekst skrevet dér hvor facit siger `[?]` | 76 tegn fordelt på 12 steder |

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
| Facit-linjer der ligger inden for én af modellens linjer | 160 af 165 |
| Facit-linjer der får deres egen modellinje | 164 af 165 |

**Sådan læses de to tal.** Er de begge lig antallet af målte linjer,
har modellen skrevet sidens linjer, som de står — én facit-linje pr.
modellinje. Er det FØRSTE tal højt og det andet lavt, har modellen
samlet flere af sidens linjer i én af sine egne. Er det første tal lavt,
løber facits linjer hen over modellens linjeskift, altså laver modellen
sine egne brud. Ingen af delene er en fejl i sig selv, og ingen af dem
påvirker tallene ovenfor — men svaret afgør, om linjeskiftene kan
afleveres videre til kollegaens `PageLine`-skema, og det er værd at vide.

## De 8 værste sider

Sorteret efter `arbejdstal`. Se dem efter med øjnene, før tallet tros —
en enkelt side med en fejlagtig parring kan trække hele hovedtallet.

| Side | Tegnfejl | Dækning | Linjer målt | Modeltekst uden modstykke |
|---|---:|---:|---:|---:|
| `273108_001555` | 13,09 % | 89,17 % | 18/18 | 61 |
| `273103_001462` | 11,76 % | 89,52 % | 24/27 | 58 |
| `273104_001639` | 11,32 % | 77,81 % | 19/22 | 78 |
| `273102_001064` | 10,08 % | 96,61 % | 23/24 | 12 |
| `273105_001570` | 5,75 % | 73,30 % | 18/26 | 90 |
| `273107_001864` | 3,60 % | 92,91 % | 25/29 | 37 |
| `273110_001527` | 3,35 % | 98,16 % | 23/25 | 18 |
| `273098_001503` | 2,23 % | 87,46 % | 15/17 | 43 |

## De 8 tyndest målte sider

Lav dækning er et værre tegn end høj tegnfejl: her er der næsten ikke
målt på siden, så dens tal betyder ikke noget. En side, hvor modellen
sprang det meste over eller skrev noget helt andet, dukker op HER — ikke
i listen ovenfor, hvor den tværtimod ser god ud.

| Side | Dækning | Tegnfejl | Linjer målt | Modeltekst uden modstykke |
|---|---:|---:|---:|---:|
| `273105_001570` | 73,30 % | 5,75 % | 18/26 | 90 |
| `273104_001639` | 77,81 % | 11,32 % | 19/22 | 78 |
| `273098_001503` | 87,46 % | 2,23 % | 15/17 | 43 |
| `273108_001555` | 89,17 % | 13,09 % | 18/18 | 61 |
| `273103_001462` | 89,52 % | 11,76 % | 24/27 | 58 |
| `273107_001864` | 92,91 % | 3,60 % | 25/29 | 37 |
| `273102_001064` | 96,61 % | 10,08 % | 23/24 | 12 |
| `273110_001527` | 98,16 % | 3,35 % | 23/25 | 18 |

## Hvad modellen skrev, hvor facit siger `[?]`

Skrives ud, fordi det er modellens bud på steder, transskribenten
ikke kunne læse. Det er IKKE facit og må aldrig skrives ind i det —
arbejdsgangen med udklip og ja/nej hører i stage 07.

| Side | Facit | Modellens bud |
|---|---|---|
| `273098_001503` | `[?]` | ved |
| `273098_001503` | `[?]` | Hosten, |
| `273098_001503` | `[?]. [?]` | Orden. - Skjælven |
| `273102_001064` | `[?]` | kroupøs |
| `273103_001462` | `[?]` | Svulst |
| `273104_001639` | `[?] [?]` | Snøvlen synes |
| `273104_001639` | `[?]` | sovet |
| `273105_001570` | `[?] [?] [?]` | *(intet)* |
| `273107_001864` | `[?]` | *(intet)* |
| `273108_001555` | `[?]` | Stridor, |
| `273108_001555` | `[?]` | Nakken |
| `273110_001527` | `[?]` | ætsedes |

## Noter

pilot, 12 af 15 sider, én pr. bind, prompt=textpage_uaendret, skema=beskrevet
