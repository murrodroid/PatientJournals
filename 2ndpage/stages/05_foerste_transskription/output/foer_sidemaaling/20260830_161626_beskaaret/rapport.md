# Koersel 20260830_161626_beskaaret

| Bogholderi | |
|---|---|
| Model | `gemini-3.1-pro-preview` |
| Promptversion | `layoutviden/linjefelter` |
| Dato | 2026-08-30T16:16:26 |
| Sider målt | 12 |
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

Målt på **85,26 % af facits tegn** (83,21 % af linjerne). De udeladte er de sværeste.

| Variant | Tegnfejl (CER) | Ordfejl (WER) | Tegnafstand | Facit-tegn |
|---|---:|---:|---:|---:|
| `raa` | 7,97 % | 24,45 % | 498 | 6247 |
| `uden_versaler` | 7,22 % | 21,48 % | 451 | 6247 |
| `uden_diakritika` | 7,88 % | 23,87 % | 492 | 6247 |
| `uden_tegnsaetning` | 7,17 % | 20,73 % | 426 | 5940 |
| `arbejdstal` | 6,40 % | 17,36 % | 380 | 5940 |
| `lempeligst` | 6,30 % | 16,78 % | 374 | 5940 |

Af de 60 linjer med mindst ét `[?]` kunne forankringen redde **52** ind i målingen ved at måle de kendte stumper omkring det ulæselige sted. Grundreglen er ellers, at hele linjen går ud (beslutning 38), så uden det trin ville dækningen have været væsentligt lavere.

## Uden de linjer, der rummer et ulæseligt sted

Hovedtallet ovenfor tager de kendte stumper med fra linjer, hvor
transskribenten gav op — teksten på hver side af et `[?]`. Det er
netop dér, både modellen og opdelingen er mest usikre, så det kan
trække tallet skævt. Her er den samme måling med de linjer helt ude.

Den strenge måling ser **76,94 % af facits tegn** (resten ligger på linjer med mindst ét `[?]`) og fik fat i 89,08 % af dem.

| Variant | Tegnfejl (CER) | Ordfejl (WER) | Tegnafstand | Facit-tegn |
|---|---:|---:|---:|---:|
| `raa` | 7,81 % | 24,42 % | 391 | 5004 |
| `uden_versaler` | 7,11 % | 21,26 % | 356 | 5004 |
| `uden_diakritika` | 7,69 % | 23,69 % | 385 | 5004 |
| `uden_tegnsaetning` | 7,00 % | 20,58 % | 333 | 4755 |
| `arbejdstal` | 6,27 % | 17,30 % | 298 | 4755 |
| `lempeligst` | 6,14 % | 16,57 % | 292 | 4755 |

**Sammenlign de to.** Hovedtallet er 6,40 %, den strenge er 6,27 % (`arbejdstal`) — en forskel på 0,13 %.

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
| `raa` | 8,46 % | 25,07 % | 512 | 6054 |
| `uden_versaler` | 7,66 % | 22,14 % | 464 | 6054 |
| `uden_diakritika` | 8,36 % | 24,50 % | 506 | 6054 |
| `uden_tegnsaetning` | 7,57 % | 20,65 % | 434 | 5733 |
| `arbejdstal` | 6,75 % | 17,32 % | 387 | 5733 |
| `lempeligst` | 6,65 % | 16,75 % | 381 | 5733 |

Linjer der er nøjagtig rigtige (`arbejdstal`): 122 af 233 = 52,36 %

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
| Modeltekst uden modstykke i facit | 1023 tegn = 16,47 % af modellens tekst |
| Tekst skrevet dér hvor facit siger `[?]` | 122 tegn fordelt på 20 steder |

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
| Facit-linjer der ligger inden for én af modellens linjer | 229 af 233 |
| Facit-linjer der får deres egen modellinje | 233 af 233 |

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
| `273108_001555` | 13,83 % | 89,17 % | 18/18 | 59 |
| `273103_001462` | 11,08 % | 96,63 % | 26/27 | 46 |
| `273102_001064` | 9,62 % | 96,61 % | 23/24 | 13 |
| `273104_001639` | 8,96 % | 95,03 % | 21/22 | 38 |
| `273099_001445` | 8,08 % | 93,73 % | 20/24 | 52 |
| `273105_001570` | 7,24 % | 76,78 % | 20/26 | 153 |
| `273107_001864` | 5,56 % | 6,34 % | 3/29 | 484 |
| `273110_001527` | 3,35 % | 98,16 % | 23/25 | 18 |
| `273109_000081` | 3,09 % | 92,53 % | 19/20 | 52 |
| `273106_001694` | 2,35 % | 99,48 % | 22/22 | 12 |

## De 10 tyndest målte sider

Lav dækning er et værre tegn end høj tegnfejl: her er der næsten ikke
målt på siden, så dens tal betyder ikke noget. En side, hvor modellen
sprang det meste over eller skrev noget helt andet, dukker op HER — ikke
i listen ovenfor, hvor den tværtimod ser god ud.

| Side | Dækning | Tegnfejl | Linjer målt | Modeltekst uden modstykke |
|---|---:|---:|---:|---:|
| `273107_001864` | 6,34 % | 5,56 % | 3/29 | 484 |
| `273105_001570` | 76,78 % | 7,24 % | 20/26 | 153 |
| `273098_001503` | 87,46 % | 1,67 % | 15/17 | 45 |
| `273108_001555` | 89,17 % | 13,83 % | 18/18 | 59 |
| `273109_000081` | 92,53 % | 3,09 % | 19/20 | 52 |
| `273099_001445` | 93,73 % | 8,08 % | 20/24 | 52 |
| `273100_001306` | 94,31 % | 1,40 % | 23/26 | 51 |
| `273104_001639` | 95,03 % | 8,96 % | 21/22 | 38 |
| `273102_001064` | 96,61 % | 9,62 % | 23/24 | 13 |
| `273103_001462` | 96,63 % | 11,08 % | 26/27 | 46 |

## Hvad modellen skrev, hvor facit siger `[?]`

Skrives ud, fordi det er modellens bud på steder, transskribenten
ikke kunne læse. Det er IKKE facit og må aldrig skrives ind i det —
arbejdsgangen med udklip og ja/nej hører i stage 07.

| Side | Facit | Modellens bud |
|---|---|---|
| `273098_001503` | `[?]` | ved |
| `273098_001503` | `[?]` | Hylen, |
| `273098_001503` | `[?]. [?]` | Orden. - Skjælven |
| `273100_001306` | `[?]` | Stuegang |
| `273100_001306` | `[?]` | Hvidske |
| `273100_001306` | `[?]` | st |
| `273102_001064` | `[?]` | kroupøs |
| `273103_001462` | `[?]` | Svulst |
| `273104_001639` | `[?] [?]` | Snøven synes |
| `273104_001639` | `[?]` | døset |
| `273104_001639` | `[?]` | stadigt |
| `273105_001570` | `[?] [?] [?]` | *(intet)* |
| `273105_001570` | `[?]` | ith |
| `273108_001555` | `[?]` | stenos. |
| `273108_001555` | `[?]` | Nakken |
| … | | *5 steder mere* |

## Noter

pilot, 12 af 15 sider, én pr. bind, prompt=layoutviden, skema=linjefelter
