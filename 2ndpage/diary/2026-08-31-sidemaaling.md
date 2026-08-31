# Dagbog 2026-08-31: forankringen er fjernet

Stage 03 blev låst 23. august med én betingelse: *"genåbnes uden tøven, hvis der
viser sig et hul ved de første rigtige tal."* Hullet var der, og det er nu lukket.
Alle seks trin i `stages/03_maaleapparat/PLAN_sidemaaling.md` er gennemført.

Det her er dagbogen for selve ombygningen. Resultaterne af wordpicking-forsøget,
målt forfra, står nederst.

---

## Hvad der blev bygget

Måleapparatet søgte hidtil hver facit-linje frem i modellens tekst. Nu
sammenlignes **facits fulde tekst med modellens fulde tekst i ét stræk, fra top
til bund**. Der er kun én vej gennem siden, og hele facit står altid i nævneren.

Den eneste afvigelse fra en almindelig redigeringsafstand: hvor facit siger
`[?]`, kunne transskribenten ikke læse stedet, og modellen må skrive noget dér
uden at det koster — op til et loft.

**Tre begreber forsvandt med søgningen:** *dækning*, *rabat* og *pr. linje*.
De fandtes alle for at beskrive, hvor meget søgningen tabte. Der tabes ikke
længere noget. Også fuldside-kontrollen udgik: den sammenlignede hele siden
direkte som kontrol MOD forankringen, og nu er kontrollen blevet det samme tal
som det, den kontrollerede.

---

## Fire ting gik anderledes end planen forudsatte

### 1. `pr_linje` og `fuldside` blev fjernet, ikke bygget om

Planen sagde "bygges om, ikke fjernes". Det viste sig ikke at kunne lade sig
gøre meningsfuldt. `pr_linje` summerede en måling pr. **parret** linje — og
parringen VAR forankringen. Den kunne genskabes fra `orden.py`s linjeparring,
men så ville et hovedtal arve dén parrings kendte svaghed: et grådigt
venstre-mod-højre valg mellem næsten ens linjer. Journalmateriale er fuldt af
næsten ens linjer.

Det er et bevidst fravalg, ikke en forglemmelse, og det kan omgøres.

### 2. Den strenge måling måtte bygges på en ny mekanisme

Den strenge måling (beslutning 44) udelader hele linjer med et `[?]`. Under
forankringen kunne den udelade **modellens modstykke** til sådan en linje, fordi
forankringen udpegede det. Uden forankring findes der intet modstykke.

Løsningen: hele linjen erstattes af ét jokermærke. Målingen bliver den samme ene
gennemgang af siden, blot med de svære linjer gjort gratis, og nævneren falder
med præcis de linjers tegn.

Så skal mærket have et **loft** — og det er en knap. Projektet har erfaring med
knapper, der pynter: `MAKS_AFVIGELSE` kunne sænke tegnfejlen fra 7,50 % til
7,13 % og se pænere ud, mens færre og lettere linjer blev målt.

**Derfor blev knappen målt, ikke valgt** (`output/jokerloft.md`, alle 16 gemte
kørsler). To fund, som ikke var til at se på forhånd:

- **"Linjens længde + 15" gav nøjagtig samme tal som SLET INTET LOFT** i 15 af
  de 16 kørsler. De 15 tegn var altså ikke et mildt slæk — loftet blev aldrig
  bindende. Jeg havde selv lagt dem oveni ved analogi til det inline-loft, og
  analogien var forkert.
- **Et fast loft på 15 fordobler tallet** (til 15-24 %). De 15 tegn er udledt af
  ordlængden i materialet og er det rigtige mål for et `[?]` inde i en linje. En
  hel linje er 25-45 tegn, så modellens tekst løber langt over.

Linjens egen længde er den eneste af de fire kandidater, der **ikke** er
systematisk mildere end hovedtallet — og en strengere måling, der er mildere end
den almindelige, kan ikke bruges som kontrol af den.

### 3. En reel fejl, fundet af en test

Testen "facit mod sig selv giver nul fejl" fejlede med **2 tegn**. En side helt
uden læsefejl skal give nul.

Årsagen: orddelingsreglen (beslutning 42) blev brugt på facit, men ikke på
modelteksten. Facits `"Infektions-"` / `"sygdomme."` blev samlet til ét ord;
modellens samme to linjer stod som to. Modellen blev straffet for en forskel,
der kun er facits egen typografi — præcis dét, beslutning 42 findes for at
forhindre.

Fejlen var i ny kode og nåede aldrig et resultat. Men den er værd at notere,
fordi den slap forbi en gennemlæsning og først faldt for den mest banale test i
filen.

### 4. Joker-tegntallet er en øvre grænse, ikke et mål for opdigtning

Rapporten opgør, hvor mange tegn modellen lagde på de ulæselige steder. Ved
efterprøvning på rigtige sider lagde tallet sig **præcis op ad loftet** — aldrig
over. Et konstrueret tilfælde med 40 tegn opkrævede korrekt de 25 over loftet,
så mekanismen virker.

Forklaringen: fribilletten er gratis indtil loftet, så målingen har ingen grund
til at holde igen — den lader gerne det ulæselige sted æde et par korrekt læste
naboord med, når de alligevel er gratis. **En del af tallet er altså tekst,
modellen har læst helt rigtigt.**

Rapporten siger det nu. Det skarpe signal er overskuddet over fribilletten, ikke
det samlede tal.

---

## To åbne punkter, der kræver lead

### Beslutning 35 kan ikke længere besvares

*Laver modellen sine egne linjeskift, eller følger den sidens?* Det var
`uden_linjeskift_indeni` og `egen_modellinje`, og begge var forankringstal.

`orden.py`s linjeparring kan tælle, hvor mange facit-linjer der har et
genkendeligt modstykke, og hvor mange der står i forkert rækkefølge. Men den kan
**ikke** skelne "modellen slog to af sidens linjer sammen" fra "modellen brød dem
et andet sted". Parringsraten bærer signalet groft, men blander linjestruktur
sammen med læsekvalitet: en linje, modellen læste elendigt, tæller som "uden
modstykke" på lige fod med en linje, den slog sammen med naboen.

Det er ikke et akademisk hul. Beslutning 35 er forudsætningen for at aflevere
`PageLine`-poster videre til kollegaens app, hvor hver linje skal være sin egen
post. **Skal afklares, før leverancen.**

### Beslutning 44's formulering holder ikke længere

Reglen siger: *"Er den strenge højere end hovedtallet, har redningen pyntet, og
så er det den strenge, der gælder."*

Den redning fandtes kun under forankringen. Målt på alle 16 gemte kørsler er den
strenge måling nu **konsekvent ca. 1 procentpoint højere — uden en eneste
undtagelse.** Reglen ville altså udløses hver gang og siger dermed ingenting.

Forskellen har en uskyldig forklaring: hovedtallet har en fribillet ved hvert
`[?]`, den strenge har ingen, fordi den slet ikke ser de linjer. At hovedtallet
ligger lavere er forventeligt.

Selve tallene er der ikke noget galt med. Det er ordlyden, der skal skiftes.

---

## Om de gamle tal — og en fejl i min egen rækkefølge

De gamle rapporter er arkiveret i
`stages/05_foerste_transskription/output/foer_sidemaaling/` med en README, der
forklarer, hvorfor de ikke gælder længere. De bevares, fordi dagens vigtigste
læring **er**, at det gamle apparat kunne vende en rangorden — og den læring kan
kun efterprøves, hvis tallene stadig findes.

**Arkiveringen skete for sent.** Jeg satte en genberegning i gang, før arkivet
var lavet, og tolv af de gamle rapporter blev overskrevet. De var ikke i git.

Intet gik dog tabt: modelsvarene blev aldrig rørt, den gamle kode ligger i commit
`3391110`, og det gamle apparat er deterministisk. Rapporterne er genskabt ved at
køre den arkiverede kode på de samme svar. Fire af de seksten er de originale
filer.

Lektien er banal og skal alligevel skrives ned: **arkivér før du genberegner**,
også når genberegningen "bare" er en prøvekørsel. Planen sagde det; jeg tog
trinnene i forkert rækkefølge.

---

## Testene

Alle nye tests er set fejle mod en genindført fejl, som `_config/tdd.md` kræver.
De vigtigste er de tre, der vogter selve rabatten: genindføres den gamle regel om,
at nævneren kun er de linjer, modellen faktisk skrev, falder præcis de tre tests,
der skal fange den.

`tests/test_forankring.py` og `tests/test_uden_rabat.py` er slettet — begge
testede udelukkende mekanismer, der ikke findes mere.

---

## Resultaterne, målt forfra

Forsøgets seks varianter, målt igen med sidemålingen på de 11 sider, alle
kørsler har til fælles. Margendatoen er foldet ind for alle, så ingen variant
straffes for at have læst datoen rigtigt.

| Variant | Prompt + skema | Ny måling | Gammel måling |
|---|---|---:|---:|
| **V3** | `layoutviden` + kollegaens skema | **11,56 %** | 11,51 % |
| V4 | `layoutviden` + linjen delt i felter | 11,68 % | 11,63 % |
| V5 | `layoutviden` + usikkerhedsfelt | 12,28 % | 12,25 % |
| V1 | kollegaens prompt + kollegaens skema | 12,44 % | 12,41 % |
| V0 | kollegaens prompt + bart skema | 12,84 % | 12,80 % |
| V2 | ren tekst, intet skema | 14,24 % | 14,07 % |

**Det vigtigste ved tabellen er, hvor lidt der skete.** Rangordenen er den
samme, og de fem øverste tal flytter sig 0,03-0,05 procentpoint. Isoleringen
holder også:

- **Prompten alene** (V1 → V3, samme skema): 12,44 % → 11,56 %. **0,88 pp.**
  Med det gamle apparat: 0,90 pp.
- **Skemaet alene** (V3 → V4, samme prompt): 11,56 % → 11,68 %. **Intet** —
  en anelse værre, som før.

Konklusionen fra 30. august står altså uændret: **layoutviden i prompten
virker, feltopdeling gør ikke.** Det er værd at holde fast i, at det ikke var
givet på forhånd. Det gamle apparat kunne vende en rangorden på en enkelt
side; at det så ikke gjorde det på netop dette forsøg, kunne vi først vide,
da det nye apparat var bygget.

Forbeholdet fra 30. august står også: 11 sider er for få. Jackknife-spændet
var 1,1-1,5 pp, og effekten er 0,88.

### Sammenligningen havde den samme fejl som rapporten

Første gennemløb gav et andet resultat: `linjefelter` lå **1,08 pp foran** alle
andre. Det var ikke et fund, det var fejlen fra RETTELSE 2 én gang til.
`maal_en` foldede margendatoen ind, før den skrev rapporten; `sammenlign`
gjorde det ikke. Kollegaens app lægger datoen i `metadata` og uden for `text`,
facit har den inline, og `linjefelter` samler selv sine dele — så den fik
datoen gratis med, mens de andre blev straffet for at have læst den rigtigt.

Rettet i commit `e66d116`, med en test der først er set fejle: to kørsler, der
begge har læst siden fejlfrit, skal begge måle nul. Uden rettelsen målte
`beskrevet` 12,96 %.

Det er anden gang på to dage, at netop denne udfoldning har vendt et resultat.
Den hører til på listen over ting, der skal tjekkes, hver gang et nyt tal
sammenlignes på tværs af skemaer.
