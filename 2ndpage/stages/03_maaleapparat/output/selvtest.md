# Selvtest af måleapparatet

Kørt på **øvemængdens 118 sider**. Ingen modelkald — hver
"modeltekst" er facit selv, forvansket på en måde hvor det rigtige svar
er kendt på forhånd. Kør igen med `scripts/selvtest_maaleapparat.py`.

Forvanskningerne er konstruerede, ikke repræsentative. Det er meningen:
data der fremkalder en bestemt fejl er sjældent typiske.

Målingen er én redigeringsafstand over hele siden, i rækkefølge, uden
søgning. Der findes derfor ikke længere nogen dækning: hele facit er
altid i nævneren, og en linje kan ikke falde ud af regnestykket.

## Tallene

| Forvanskning | raa | uden_versaler | uden_diakritika | uden_tegnsætn. | arbejdstal | arbejdstal, strengt | Model-tegn af facit-tegn | Omrokerede linjer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| facit mod sig selv | 0,01 % | 0,01 % | 0,01 % | 0,00 % | 0,00 % | 0,00 % | 56591 af 54284 | 3 af 2586 |
| alle ø skrevet som ö | 0,76 % | 0,76 % | 0,01 % | 0,79 % | 0,79 % | 0,80 % | 56591 af 54284 | 3 af 2586 |
| alt med små bogstaver | 5,36 % | 0,02 % | 5,36 % | 5,65 % | 0,00 % | 0,01 % | 56591 af 54284 | 3 af 2586 |
| al tegnsætning fjernet | 5,16 % | 5,16 % | 5,16 % | 0,00 % | 0,00 % | 0,00 % | 53331 af 54284 | 3 af 2586 |
| hele siden som ét afsnit | 0,58 % | 0,58 % | 0,58 % | 0,30 % | 0,30 % | 0,28 % | 56591 af 54284 | 0 af 2586 |
| hvert linjebrud flyttet ét ord | 0,52 % | 0,52 % | 0,52 % | 0,27 % | 0,27 % | 0,25 % | 56591 af 54284 | 21 af 2586 |
| et opdigtet afsnit tilføjet | 8,75 % | 8,74 % | 8,75 % | 9,03 % | 9,02 % | 9,64 % | 61547 af 54284 | 3 af 2586 |
| 2 % af bogstaverne byttet | 1,50 % | 1,50 % | 1,50 % | 1,58 % | 1,58 % | 1,55 % | 56591 af 54284 | 3 af 2586 |
| 10 % af bogstaverne byttet | 7,55 % | 7,55 % | 7,55 % | 7,97 % | 7,97 % | 7,90 % | 56591 af 54284 | 5 af 2586 |
| den midterste tredjedel sprunget over | 32,73 % | 32,69 % | 32,73 % | 32,70 % | 32,65 % | 31,12 % | 37690 af 54284 | 27 af 2586 |
| et gentaget ord læst en anelse forkert | 0,30 % | 0,30 % | 0,30 % | 0,31 % | 0,31 % | 0,33 % | 56591 af 54284 | 5 af 2586 |

Kolonnen **arbejdstal, strengt** er den samme måling med linjer, der
rummer et `[?]`, helt ude (beslutning 44). Her i selvtesten forvanskes
alle bogstaver med samme sandsynlighed, så de svære linjer er IKKE
sværere end de andre — de to tal bør derfor ligge tæt. Gør de det,
ved vi, at selve maskineriet ikke skaber en forskel, og at en forskel
på rigtige data kommer fra materialet, ikke fra måden at måle på.

**Model-tegn af facit-tegn** er, om modellen overhovedet skrev lige så
meget tekst, som der stod på siden — tegn uden mellemrum. Den erstatter
den gamle kolonne "modeltekst uden modstykke", som kun gav mening,
mens der blev forankret. **Omrokerede linjer** er linjer, modellen
skrev i en anden orden end facit; den måles for sig af `orden.py`,
fordi hovedtallet er strengt om rækkefølge og ellers ville skjule,
hvor stor en del af fejlen der bare er ombytning.

### Hvad hver linje skal vise

- **facit mod sig selv** — Nul fejl i alle varianter. Det ord, "modellen" skriver dér hvor facit har `[?]`, er kortere end jokerfeltets loft og skal derfor slippe helt gratis igennem. Er tallet ikke nul her, er alt andet i tabellen ligegyldigt.
- **alle ø skrevet som ö** — `raa` får fejl; `uden_diakritika` og `lempeligst` skal være nul. Det er den hyppigste enkeltforveksling i materialet, og den er ortografisk støj, ikke en læsefejl.
- **alt med små bogstaver** — `raa` får fejl; `uden_versaler` og `arbejdstal` skal være nul.
- **al tegnsætning fjernet** — `raa` får fejl; `uden_tegnsaetning` og `arbejdstal` skal være tæt på nul. Bindestregen er bevidst ladt stå — fjernes den, forsvinder orddelingen med den, og så måler prøven to ting på én gang.
- **hele siden som ét afsnit** — Tæt på facit mod sig selv, men **ikke helt nul**, og resten er et målt fund: uden linjeskift kan et ord, facit har delt hen over to linjer, ikke samles igen, så `Infektions- sygdomme` bliver stående som to ord (efterprøvet linje for linje). Målingen må ellers ikke afhænge af, om modellen laver sine egne linjeskift (beslutning 35).
- **hvert linjebrud flyttet ét ord** — Samme lille rest og samme årsag: bindestregen står nu midt på en linje i stedet for sidst, og så samles det delte ord ikke. Linjeskiftene er taget ud på begge sider før målingen, så hvor de sad, må ellers ikke kunne ses i tallet.
- **et opdigtet afsnit tilføjet** — Her ses forskellen fra de gamle rapporter tydeligst: det opdigtede afsnit koster nu ét point pr. indsat tegn i selve tegnfejlen. Under forankringen var afsnittet gratis, fordi det ikke havde nogen facit-linje at blive parret med.
- **2 % af bogstaverne byttet** — Målt tegnafstand skal ligge tæt på antallet af indlagte fejl — se næste tabel for hvor tæt.
- **10 % af bogstaverne byttet** — Samme prøve, ti gange så hårdt. Med så mange fejl tæt på hinanden begynder redigeringsafstanden at kunne finde en billigere vej end vores egne ombytninger, og det skal kunne ses i næste tabel.
- **den midterste tredjedel sprunget over** — Den sprungne tredjedel koster nu direkte: hvert tegn, modellen ikke skrev, er en sletning. Tegnfejlen skal derfor ligge omkring en tredjedel. Under forankringen faldt de manglende linjer helt ud af regnestykket og kostede næsten ingenting.
- **et gentaget ord læst en anelse forkert** — Prøven på netop dét, der væltede den gamle måling. To bogstaver byttet i **første** forekomst af et ord, der står på mindst to linjer i facit; de senere forekomster står urørt. Begge fejl skal findes igen. Antallet af sider, der overhovedet har sådan et ord, står under tabellen — er det lavt, er prøven svag.

Af øvemængdens 118 sider har **94** et ord på mindst 5 tegn, der står på to forskellige
linjer. Kun de sider bidrager med indlagte fejl i prøven
"et gentaget ord læst en anelse forkert" — resten leveres urørt. Er
tallet lavt, er prøven tilsvarende svag, og det står her frem for at
blive gemt bag procenten.

## Hvor meget apparatet finder af det, vi selv lagde ind

Den vigtigste tabel i hele selvtesten. Venstre kolonne er tegn, vi selv
byttede eller fjernede; midterkolonnen er den tegnafstand, målingen
fandt. Er de ikke ens, er forskellen **skævheden i tallet**, og den skal
stå her frem for at være et skjult fradrag.

Tallet kan ligge på begge sider af 100 %. Under 100 %: redigerings-
afstanden fandt en billigere vej end vores egne ombytninger — to fejl
ved siden af hinanden kan af og til rettes med ét greb — eller fejlen
landede inde i et jokerfelt og slap gratis igennem (se næste afsnit).
Over 100 %: en ombytning kan have gjort teksten dyrere at rette end de
enkelttegn, vi ændrede.

| Forvanskning | Fejl vi lagde ind | Fejl målingen fandt | Fundet |
|---|---:|---:|---:|
| alle ø skrevet som ö | 477 | 482 | 101,05 % |
| alt med små bogstaver | 3388 | 3402 | 100,41 % |
| al tegnsætning fjernet | 3260 | 3273 | 100,40 % |
| 2 % af bogstaverne byttet | 1015 | 953 | 93,89 % |
| 10 % af bogstaverne byttet | 5087 | 4790 | 94,16 % |
| et gentaget ord læst en anelse forkert | 188 | 193 | 102,66 % |

## Fejl, der forsvinder ned i et jokerfelt

Hvor facit siger `[?]`, må modellen skrive hvad som helst op til
jokerfeltets loft, uden at det koster. Det er en aftalt fribillet — der
findes ingen sandhed at måle stedet imod — men den er samtidig
målingens egen skævhed: en indlagt fejl, der tilfældigvis rammer inde i
det ord, "modellen" skrev på et `[?]`, kan ikke findes igen.

Her er den talt op i stedet for antaget. Optællingen gentager de samme
ombytninger tegn for tegn og ser efter, hvor de landede.

| Forvanskning | Indlagte fejl | Heraf inde i et jokerfelt | Andel |
|---|---:|---:|---:|
| 2 % af bogstaverne byttet | 1015 | 67 | 6,60 % |
| 10 % af bogstaverne byttet | 5087 | 302 | 5,94 % |

De øvrige forvanskninger kan ikke ramme et jokerfelt. Omlyd, små
bogstaver og fjernet tegnsætning rører ikke det ord, der står på et
`[?]` — det har hverken ø, versaler eller tegnsætning — og det
gentagne ord vælges udtrykkeligt blandt ord uden `[?]` i.

Tallet er et loft for, hvad fribilletten koster i selvtesten, ikke et
skøn over rigtige data. En rigtig model skriver noget andet og længere
på et ulæseligt sted, og hvad den så gør, kan kun ses i gab-filen.

## De ulæselige steder i øvemængden

| Mål | Værdi |
|---|---:|
| Linjer i øvemængden | 2586 |
| Heraf med mindst ét `[?]` | 297 = 11,48 % |
| Jokerfelter i alt | 354 |
| Tegn "modellen" lagde i dem | 3811 |
| Tegn ud over loftet (det der kostede) | 0 |

Målt på "facit mod sig selv", altså med det korte ord `utydeligt` på
hvert `[?]`. Det ligger under loftet og koster derfor ingenting. En
rigtig model kan skrive mere, og så begynder overskuddet at tælle —
det tal er derfor ikke en forudsigelse, men et udgangspunkt at måle
de rigtige kørsler op imod.
