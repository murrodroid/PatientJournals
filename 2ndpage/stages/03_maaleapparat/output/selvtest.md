# Selvtest af måleapparatet

Kørt på **øvemængdens 118 sider**. Ingen modelkald — hver
"modeltekst" er facit selv, forvansket på en måde hvor det rigtige svar
er kendt på forhånd. Kør igen med `scripts/selvtest_maaleapparat.py`.

Forvanskningerne er konstruerede, ikke repræsentative. Det er meningen:
data der fremkalder en bestemt fejl er sjældent typiske.

## Tallene

| Forvanskning | raa | uden_versaler | uden_diakritika | uden_tegnsætn. | arbejdstal | arbejdstal, strengt | Dækning | Modeltekst uden modstykke |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| facit mod sig selv | 0,00 % | 0,00 % | 0,00 % | 0,00 % | 0,00 % | 0,00 % | 97,58 % | 2519 tegn |
| alle ø skrevet som ö | 0,75 % | 0,75 % | 0,00 % | 0,79 % | 0,79 % | 0,80 % | 97,58 % | 2519 tegn |
| alt med små bogstaver | 5,30 % | 0,00 % | 5,30 % | 5,59 % | 0,00 % | 0,00 % | 97,58 % | 2519 tegn |
| al tegnsætning fjernet | 5,04 % | 5,04 % | 5,04 % | 0,10 % | 0,10 % | 0,00 % | 97,55 % | 2447 tegn |
| hele siden som ét afsnit | 0,00 % | 0,00 % | 0,00 % | 0,00 % | 0,00 % | 0,00 % | 97,58 % | 2519 tegn |
| hvert linjebrud flyttet ét ord | 0,00 % | 0,00 % | 0,00 % | 0,00 % | 0,00 % | 0,00 % | 97,58 % | 2519 tegn |
| et opdigtet afsnit tilføjet | 0,00 % | 0,00 % | 0,00 % | 0,00 % | 0,00 % | 0,00 % | 97,58 % | 7475 tegn |
| 2 % af bogstaverne byttet | 1,51 % | 1,51 % | 1,51 % | 1,59 % | 1,59 % | 1,56 % | 97,25 % | 2735 tegn |
| 10 % af bogstaverne byttet | 7,50 % | 7,50 % | 7,50 % | 7,92 % | 7,92 % | 7,84 % | 97,16 % | 2916 tegn |
| den midterste tredjedel sprunget over | 0,45 % | 0,44 % | 0,45 % | 0,42 % | 0,40 % | 0,36 % | 61,38 % | 3900 tegn |

Kolonnen **arbejdstal, strengt** er den samme måling med linjer, der
rummer et `[?]`, helt ude (beslutning 44). Her i selvtesten forvanskes
alle bogstaver med samme sandsynlighed, så de svære linjer er IKKE
sværere end de andre — de to tal bør derfor ligge tæt. Gør de det,
ved vi, at selve maskineriet ikke skaber en forskel, og at en forskel
på rigtige data kommer fra materialet, ikke fra måden at måle på.

### Hvad hver linje skal vise

- **facit mod sig selv** — Nul fejl i alle varianter. Sætter samtidig **gulvet** for kolonnen "modeltekst uden modstykke": den er ikke nul her, selvom intet er digtet. Det, der står, er ordet `utydeligt` dér hvor facit har `[?]` plus teksten på de linjer, der ikke kunne forankres. Ved en rigtig måling skal tallet læses som et tillæg til dette gulv, ikke som et absolut mål for opdigtning.
- **alle ø skrevet som ö** — `raa` får fejl; `uden_diakritika` og `lempeligst` skal være nul. Det er den hyppigste enkeltforveksling i materialet, og den er ortografisk støj, ikke en læsefejl.
- **alt med små bogstaver** — `raa` får fejl; `uden_versaler` og `arbejdstal` skal være nul.
- **al tegnsætning fjernet** — `raa` får fejl; `uden_tegnsaetning` og `arbejdstal` skal være tæt på nul. Bindestregen er bevidst ladt stå — fjernes den, forsvinder orddelingen med den, og så måler prøven to ting på én gang.
- **hele siden som ét afsnit** — Samme tal som facit mod sig selv. Målingen må ikke afhænge af, om modellen laver sine egne linjeskift (beslutning 35).
- **hvert linjebrud flyttet ét ord** — Samme tal som facit mod sig selv. Uden forankringen ville alt efter det første brud være forkert — det er hele grunden til, at linjerne parres på indhold og ikke på linjenummer.
- **et opdigtet afsnit tilføjet** — Tegnfejlen ser det ikke. Kun "modeltekst uden modstykke" gør, og den springer fra gulvet på ~2.500 tegn til ~7.500. Det er derfor det tal skal stå ved siden af hovedtallet i enhver rapport.
- **2 % af bogstaverne byttet** — Målt tegnafstand skal ligge tæt på antallet af indlagte fejl — se næste tabel for hvor tæt.
- **10 % af bogstaverne byttet** — Samme, men her begynder dækningen at falde: de hårdest forvanskede linjer kan ikke forankres.
- **den midterste tredjedel sprunget over** — Dækningen skal falde til omkring to tredjedele. **Tegnfejlen bliver IKKE nul**, og det er et målt fund, ikke en forventning — se afsnittet "Falske forankringer" nedenfor for hvad der faktisk sker.

## Hvor meget apparatet finder af det, vi selv lagde ind

Den vigtigste tabel i hele selvtesten. Venstre kolonne er bogstaver, vi
selv byttede om; midterkolonnen er den tegnafstand, målingen fandt. Er de
ikke ens, er forskellen **skævheden i tallet** — og den peger altid samme
vej: målingen finder mindre, end der er, fordi de linjer den ikke kan
forankre, er de hårdest ramte.

| Forvanskning | Fejl vi lagde ind | Fejl målingen fandt | Fundet |
|---|---:|---:|---:|
| alle ø skrevet som ö | 477 | 477 | 100,00 % |
| alt med små bogstaver | 3388 | 3360 | 99,17 % |
| al tegnsætning fjernet | 3260 | 3192 | 97,91 % |
| 2 % af bogstaverne byttet | 1015 | 953 | 93,89 % |
| 10 % af bogstaverne byttet | 5087 | 4737 | 93,12 % |

Tallet kan ikke nå 100 %. Tre grunde, alle kendte:

1. **Uforankrede linjer falder ud** — de hårdest forvanskede først.
2. **Stumper under fem tegn bruges ikke** til forankring, så teksten
   omkring et `[?]` er ikke altid med.
3. **Levenshtein kan være billigere end vores ombytninger** — to fejl
   ved siden af hinanden kan af og til rettes med ét greb.

Det er derfor, dækningen skal stå ved hvert tal. Et tal på 5 % tegnfejl
målt på 88 % af teksten er ikke det samme som 5 % på det hele.

## Falske forankringer

Springer modellen en del af siden over, bliver tegnfejlen ikke nul,
selvom hvert eneste ord, den faktisk skrev, er rigtigt. Første forklaring var
en formodning; her er hvad der faktisk sker, efterprøvet linje for linje
på forvanskningen "den midterste tredjedel sprunget over":

**1. En manglende linje forankrer sig i en linje, der ligner.** Facits
`Hendes tilstand er i løbet af natten bleven` findes ikke i modellen, men
`I løbet af natten` gør — og stumpen lander dér. `Tungen` lander i
`Lunge`. `ingen Appetit, ligget hen og døset,` lander i `Det ligger hen
og døser,`.

**2. Og det skader de EFTERFØLGENDE linjer.** Det var ikke med i den
første forklaring, og det er den vigtigere halvdel. Forankringen går fra
venstre mod højre, så et falsk træf flytter søgepunktet frem forbi det
sted, hvor de næste linjer i virkeligheden står. De finder så kun en
afskåret rest af sig selv: `begge Lunger overalt en Mængde fugtige` blev
målt mod `r overalt en Mængde fugtige`, selvom modellen havde skrevet
hele linjen rigtigt.

Prisen er lille på dette materiale — 181 tegn fordelt på 27 af de 118
sider — men den vokser med, hvor meget modellen springer over. Derfor:
**en side med lav dækning skal ses efter med øjnene**, ikke bare tros.
Rapporten har sin egen liste over de tyndest målte sider netop derfor.

## Knappen `MAKS_AFVIGELSE`

Hvor meget en stump må afvige og stadig regnes for fundet. Tabellen står
her, fordi knappen kan bruges til at pynte: sættes den lavere, falder
dækningen, og de linjer der bliver tilbage, er de letteste. Tegnfejlen ser
bedre ud og måler mindre og mindre repræsentativt materiale.

Målt på forvanskningen "10 % af bogstaverne byttet".

| MAKS_AFVIGELSE | raa | Dækning | Linjer målt | Fundet af de indlagte fejl |
|---:|---:|---:|---:|---:|
| 0,2 | 7,13 % | 94,90 % | 2347 af 2586 | 86,40 % |
| 0,4 | 7,50 % | 97,16 % | 2420 af 2586 | 93,12 % |
| 0,6 | 7,55 % | 97,32 % | 2425 af 2586 | 93,85 % |

Projektets værdi er **0,4**. Den er sat
rundhåndet med vilje. Læg mærke til, at den strengeste indstilling giver den
*laveste* tegnfejl — den ser bedst ud og er mest misvisende.

## Hvad forankringen henter hjem

Beslutning 38 skærer hele linjen fra, når den rummer et `[?]`.
Forankringen henter de kendte stumper på linjen tilbage i målingen.

| Mål | Værdi |
|---|---:|
| Linjer i øvemængden | 2586 |
| Heraf med mindst ét `[?]` | 297 = 11,48 % |
| Svære linjer forankringen redder | 281 = 94,61 % af dem |
| Dækning med forankring | 97,58 % |
| Gab fundet (modellens bud på et `[?]`) | 95 |

**Bemærk at dette er en øvre grænse.** Her er "modellen" facit selv, så
hver stump findes ordret. En rigtig model læser dårligere, og færre
stumper vil kunne forankres. Det rigtige tal kommer først i stage 05.
