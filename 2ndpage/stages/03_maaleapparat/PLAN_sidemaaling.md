# Plan: forankringen erstattes af én sidemåling

**Status: alle fem åbne spørgsmål afklaret 2026-08-30. Klar til trin 1.
Ikke påbegyndt.**
Skrevet 2026-08-30 efter en grill-session. Der grilles videre, før noget køres.

## Hvorfor

Stage 03 blev låst 2026-08-23 med den udtrykkelige betingelse: *"Genåbnes uden
tøven, hvis der viser sig et hul ved de første rigtige tal."* Hullet er der.

Målingen parrer facits linjer med modellens tekst forfra og nedefter, og
søger fremad fra sidste træf. Rammer den forkert og lander for langt nede,
rykker søgepunktet med, og alt før det punkt er derefter uden for rækkevidde.

Det skete på `273107_001864` i variant V4:

```text
facit linje  1: 'Ingen Snue.'                       -> forankret ved position 564 af 617
facit linje 26: 'Tg. ikke [?] suspect ingen Snue.'
```

Modellen læste linje 1 som `Ingen Hoste.` (en almindelig læsefejl) og linje 26
som `G. ikke Rødt. ingen Snue.` (en rimelig læsning). Sætningen "ingen Snue"
står altså **to gange på siden i facit selv** — det er ikke noget modellen fandt
på. Søgningen foretrak det ordrette match i linje 26 frem for det halvdårlige i
linje 1 og smed linje 2-25 væk. **26 af 29 linjer tabt.**

Målt på alle 71 sider i de seks varianter: ét tilfælde. Men det kostede hele
siden og vendte forsøgets rangorden om — V4 gik fra dårligst til bedst, da den
side blev udeladt.

Leads holdning (2026-08-30): *"jeg vil ikke have en model der søger noget frem
længere nede, orden vs orden måling"* og *"stage 03 skal være mere konservativ
og ikke inviterende til at hoppe rundt"*.

## Hvad der skal bygges

**Én global sammenligning af hele siden, i rækkefølge, uden søgning.**
Facits fulde tekst mod modellens fulde tekst, fra top til bund, i ét stræk.

Den eneste afvigelse fra en almindelig editeringsafstand: **hvor facit siger
`[?]`, må modellen skrive hvad som helst uden at det koster.** Det var den
eneste grund til at forankre — 11,5 % af facits linjer har et sted,
transskribenten ikke kunne læse, og modellen må ikke straffes for at gætte dér.

Følger af det:
- Der er kun én vej gennem siden. Et gentaget ord kan ikke flytte noget.
- **"Dækning" og "uden rabat" forsvinder som begreber.** Hele siden er altid
  målt. Dermed forsvinder også den fælde, der fik den forkerte konklusion
  igennem to gange på én dag.
- Gabene (hvad modellen skrev på et `[?]`) falder stadig ud af målingen, og
  mere direkte end før: det er præcis den modeltekst, der blev stillet op mod
  et jokertegn.

## Trin

Hvert trin kan kommenteres for sig. Rækkefølgen er bindende — trin 2 kan ikke
prøves af uden trin 1.

### Trin 1 — måleoperationen alene, uden at røre noget bestående

Ny funktion i `cer.py` eller et nyt modul: editeringsafstand mellem to
strenge, hvor facit-siden kan indeholde jokermærker der matcher vilkårligt
mange tegn til pris nul. Returnerer afstand, facit-længde, og hvad der lå i
hvert joker-felt.

Testes for sig mod konstruerede tilfælde, herunder netop det gentagne ord fra
`273107_001864`. **Testen på det tilfælde skal ses fejle mod den nuværende
`forankr()`.**

Rører intet bestående. Kan kasseres uden følger.

### Trin 2 — kør begge målinger på dagens seks gemte kørsler

Ingen modelkald; svarene ligger gemt. Sammenlign de to måleapparaters tal og
rangorden.

Formålet er ikke at vælge — valget er truffet — men at vide **hvor meget** de er
uenige, før de gamle tal kasseres. Er de enige overalt undtagen på den ene
side, er det et resultat i sig selv.

### Trin 3 — indsæt den nye måling i `maal.py`

`forankr()` fjernes. `SideMaaling` og `SaetMaaling` mister felterne om
dækning og forankring. Det er en ægte forenkling — de fleste af de nuværende
felter findes kun for at beskrive, hvor meget forankringen tabte.

Skal bevares, fordi de er kontraktbundne eller efterspurgte:
- **Gabene** (`skriv_gab`) — kontrakt fra rod-CONTEXT 2026-08-21.
- **Linjetrofastheden** — svaret på beslutning 35, og forudsætning for at
  aflevere `PageLine`-poster videre.
- **De seks varianter** (`raa` … `lempeligst`) — beslutning 26, alle
  rapporteres side om side.

### Trin 4 — rapporten skrives om

Afsnittet om dækning udgår. Afsnittet om den strenge måling **bevares** (se
spørgsmål 2), men skal skrive tydeligt, at dens "dækning" er en fast
udeladelse på 12,4 %, ens for alle varianter — ikke den glidende rabat, der
væltede konklusionen 30. august.

Nyt afsnit: hvor mange tegn modellen lagde i joker-felterne, og hvor mange
linjer der stod i en anden orden end facit.

Afsnittet "Sådan er der målt" skal beskrive den nye metode i almindeligt
sprog. Det skal kunne læses uden `CONTEXT.md` ved hånden — det krav gjaldt
også den gamle rapport.

### Trin 5 — selvtesten køres om

`scripts/selvtest_maaleapparat.py` ødelægger facit med vilje og måler, hvor
meget apparatet finder igen. Skævheden var 93,1 % med forankring. Den skal
måles på ny, og forvanskningerne skal udvides med **gentagne ord** — det er
den fejltype, der væltede den gamle måling, og selvtesten indeholdt den ikke.

Tager omkring ni minutter. Ingen modelkald.

### Trin 6 — dagens seks varianter måles forfra

Gratis; svarene ligger gemt. Først dér ved vi, hvad wordpicking-forsøget
faktisk viste.

Det gamle output arkiveres samtidig i `output/foer_sidemaaling/` med en
README (se spørgsmål 5).

**Forventning, skrevet ned på forhånd så den kan tages fejl:** V4
(`linjefelter`) rykker op, fordi dens dårlige tal skyldtes ét falsk træf på
`273107_001864`. Bliver den ikke bedre, er der noget andet galt, som vi ikke
har forstået endnu.

## Afklarede spørgsmål

Alle fem blev afklaret med lead 2026-08-30, efter en grill-session og
research på, hvordan andre håndterer ulæselige steder i facit.

1. ~~**Hvor meget må et joker-felt sluge?**~~ **AFKLARET 2026-08-30 (lead):
   loft på 15 tegn, og det slugte opgøres ved siden af.**

   Tallet er hentet fra materialet, ikke valgt pænt: 15 ligger over 99.
   percentil for et enkelt ord i øvefacit (14 tegn), så et `[?]`, der dækkede
   ét ord, altid slipper gratis igennem — også et langt sammensat ord. 250 af
   de 354 mærker står alene på deres linje og dækker efter alt at dømme netop
   ét ord. Skriver modellen mere end 15 tegn på ét ulæseligt sted, koster
   overskuddet som fejl.

   Rapporten skal desuden skrive ud, **hvor mange tegn modellen i alt lagde i
   joker-felterne**. Det er det eneste sted, tilbøjeligheden til at digte kan
   ses, og det koster ingenting at regne ud.

   Baggrund: HTR-traditionen (Transkribus) udelader hele linjen ved ulæselige
   steder — strengere end begge de muligheder, der blev stillet op, men det
   ville koste 12,4 % af facits tegn, systematisk det sværeste materiale.
   Nyere OCR-benchmarks (olmOCR-Bench) lægger omvendt eksplicitte fradrag ind
   for hallucineret tekst, fordi almindelig CER ikke fanger den fejltype
   sprogmodeller laver. Talegenkendelse har et fortilfælde for netop et loft:
   løb af indsættelser kappes ved et fast antal.

2. ~~**Overlever den strenge måling (beslutning 44)?**~~ **AFKLARET
   2026-08-30 (lead): ja, den bevares.**

   Den blev indført 2026-08-23 som værn mod forankringens rabat, og den rabat
   findes ikke længere. Men researchen viste, at netop den fremgangsmåde er
   **konventionen i HTR**: Transkribus-praksis udelader hele linjen ved
   ulæselige steder, både ved træning og måling. Beholdes tallet, kan vores
   resultater sammenlignes med anden forskning.

   Følge: "dækning" vender tilbage som begreb — men **kun for det ene tal**,
   og med en kendt, fast værdi (12,4 % af øvefacits tegn ligger på linjer med
   et `[?]`). Det er ikke den glidende, variantafhængige rabat, der væltede
   konklusionen; det er en fast udeladelse, der er ens for alle varianter.
   Den forskel skal stå skrevet i rapporten, så de to slags "dækning" ikke
   forveksles.

3. ~~**Tegn eller ord som hovedtal?**~~ **AFKLARET 2026-08-30 (lead):
   tegnfejl forbliver beslutningstallet, ordfejl står ved siden af.**

   Begge falder ud af samme udregning, så det koster ingenting at have begge.
   Tegnfejl er finere: en model, der rammer fire ud af fem bogstaver i et
   svært ord, får credit for det, og på 1800-tals lægehåndskrift med latinske
   forkortelser er den forskel reel. Ordfejl er desuden altid et meget større
   tal — i dag cirka 25 % mod 9 % — fordi ét forkert bogstav gør hele ordet
   forkert. Det er let at fejllæse udefra som "modellen er elendig".

4. ~~**Hvad med omrokerede linjer?**~~ **AFKLARET 2026-08-30 (lead): måles
   strengt i rækkefølge. Omrokering opgøres som sit eget tal ved siden af.**

   Lead havde først ønsket, at omrokering blev tilgivet: *"vi skal tillade
   omrokering af hele linjer eller rækker af ord hvis de tydeligt bare er
   blevet sat i forkert rækkefølge, men det er svært at implementere
   ordentligt."*

   To grunde til ikke at bygge det ind i tallet:

   **Teknisk:** editeringsafstand har ingen "flyt"-operation. Tillader man
   vilkårlige blokflytninger, bliver problemet uoverskueligt at regne eksakt,
   og enhver tilnærmelse indfører netop den slags skøn, vi lige har fjernet —
   hvornår er noget *tydeligt bare* omrokeret? Det er samme ladeport som
   søgningen fremad, i en anden form.

   **Vigtigere:** rækkefølgen er noget, projektet har brug for. Journalen
   læses kronologisk, og leverancen er `PageLine`-poster, hvor rækkefølgen
   bærer betydning. En ombyttet linje er en fejl, der skal kunne ses.

   Målingen skal derfor opgøre **antal linjer, der står i en anden orden end
   facit**, som et selvstændigt tal. Det kan altid besluttes, at det ikke
   betyder noget; det omvendte kan ikke besluttes, hvis tallet har skjult det.

5. ~~**Hvad sker der med de eksisterende tal i `output/`?**~~ **AFKLARET
   2026-08-30 (lead): arkivér med en note, og genberegn alt.**

   De gamle filer flyttes til `output/foer_sidemaaling/` med en README, der
   siger hvorfor de ikke gælder længere. Alt genberegnes med den nye måling.

   Begrundelsen er ikke pietet. Dagens vigtigste læring **er**, at det gamle
   apparat kunne vende en rangorden — og den læring kan kun efterprøves, hvis
   de gamle tal stadig findes. Slettes de, står dagbogen med tal, ingen kan
   kontrollere.

## Facit kan blive rettet senere — det skal planen tåle

Lead (2026-08-30): *"i fremtiden kunne jeg godt forestille mig at gå ind og
læse selv og tilrette vores gt facitter."*

Det er ikke en fjern mulighed, det er allerede aktuelt. Der er én bekræftet
fejllæsning i det leverede facit (`37554_001491`: facit skriver "for 2 Dage
siden", på siden står "for 3 Dage siden"), og hyppigheden er ukendt — ét fund
på tretten stikprøver. Og 354 `[?]`-mærker er 354 steder, hvor en mere øvet
læser kan komme videre.

Tre følger for den her plan:

- **Målingen må ikke antage, at `[?]` er permanent.** Antallet af jokerfelter
  vil falde, hvis facit rettes, og tallene skal kunne regnes om uden andet end
  en ny kørsel af måleapparatet på de gemte svar. Det er de i forvejen — men
  det må ikke bygges væk.
- **Gab-filen bliver et arbejdsredskab, ikke kun et måletal.** Den viser, hvad
  modellen skrev netop dér, hvor transskribenten gav op. Det er den korteste
  vej til en liste over steder, der er værd at kigge på med egne øjne. Det er
  et argument for at bevare den, uanset hvad der ellers falder bort.
- **Ingen tal må gemmes uden den facit-udgave, de er regnet på.** Rettes facit,
  er gamle tal ikke sammenlignelige med nye, og det skal kunne ses frem for at
  skulle huskes.

## Kortlægning af hvad der falder med `forankr()` (efterprøvet 2026-08-30)

Lavet ved delegeret gennemlæsning og derefter **stikprøvet i koden**. To af
kortlæggerens påstande holdt ikke og er rettet nedenfor — linjehenvisninger
herfra skal slås efter, ikke tros.

### Felter i `SideMaaling`, delt efter om de overlever

**Forsvinder med forankringen** (findes kun for at beskrive, hvad den tabte):
`linjer`, `facit_tegn_maalt`, `linjer_maalt`, `svaere_linjer_reddet`,
`model_tegn_daekket`, `rene_linjer_maalt`, `rene_tegn_maalt`,
`uden_linjeskift_indeni`, `egen_modellinje`, samt egenskaberne `daekning` og
`model_tegn_uforankret`.

**Overlever uændret** (regnes af facit eller modeltekst alene):
`image_name`, `facit_tegn_i_alt`, `linjer_i_alt`, `svaere_linjer`,
`model_tegn_i_alt`, `rene_linjer_i_alt`, `rene_tegn_i_alt`.

**Skal bygges om, ikke fjernes:** `fladet`, `pr_linje`, `rene`, `fuldside`,
`gab`. De er alle regnet på forankrede linjer i dag, men har hver især en
meningsfuld udgave under sidemålingen.

I `SaetMaaling` er kun `sider` et felt; resten er egenskaber, der summerer
ovenstående og følger med.

### To fejlslutninger fra kortlægningen, rettet

**"Gab-mekanismen kan ikke fungere uden forankring."** Rigtigt om den
NUVÆRENDE kode — et gab dannes mellem to *fundne* stumper, og uden søgning er
ingen stump fundet. Men forkert om den nye måling: dér er et gab netop den
modeltekst, der blev stillet op mod et `[?]`-jokerfelt. Det falder direkte ud
af sammenligningen og kræver hverken søgning eller stumper. Gabene bliver
**lettere** at danne, ikke sværere.

**"Hele rapporten bliver meningsløs."** Overdrevet. Afsnittene om dækning og
om de tyndest målte sider udgår, fordi begreberne forsvinder. Resten
overlever i omskrevet form: hovedtallet, den strenge måling (bevaret, se
spørgsmål 2), opdigtning (nu fra jokerfelternes indhold), linjetrofasthed (nu
fra `orden.py`), og de værste sider.

### Verificeret ved stikprøve

- Gab-dannelsen står i `maal.py` omkring linje 346-356 — bekræftet ordret.
- Feltlisten for `SideMaaling` stemmer med dataklassen selv.
- **Henvisningen "rapport.py:636" findes ikke** — filen er 393 linjer.
  `fuldside` bruges i virkeligheden i linje 226, 229 og 238. Kortlæggerens
  øvrige linjenumre er derfor ikke efterprøvet og skal slås efter ved brug.

## Kan testene køre hurtigere? (målt 2026-08-30)

Hele pakken tager 67 sekunder. Den er kørt et tocifret antal gange på én dag,
så det er ikke pynt at gøre noget ved. **Målt** med `pytest --durations=12`,
ikke gættet:

| | |
|---|---|
| Hele pakken | 67 s |
| De 11 langsomste tests (alle billedbaserede) | ~57 s |
| Den enkeltværste, `test_soem_gulvet_ligger_under_alle_godkendte_snit` | 22,6 s |

### Årsagen er ikke billederne — det er masterlisten

`load_masterlist()` læser en CSV på **47 MB med 570.519 rækker**, og den
genindlæses fra bunden ved hvert eneste kald: målt 2,3 s første gang, 3,1 s
anden gang. Ingen cache.

`tests/test_yderkant_rigtige_billeder.py` kalder den 12 gange — seks
parametriserede tests plus seks gennemløb i løkketesten, som behandler
nøjagtig de samme seks sider én gang til. Cirka **36 af de 67 sekunder er ren
genindlæsning af den samme fil**.

### Forslag, i den rækkefølge de betaler sig

1. **`@lru_cache` på `load_masterlist()`.** Én linje. Efterprøvet: ingen
   kalder muterer det returnerede opslag (den eneste `index[...]` er et
   opslag inde i `lookup`), så cachen kan ikke forgifte nogen. Forventet
   gevinst: langt størstedelen af de 36 sekunder.
   Bonus: **tre scripts har hver sin private omgåelse** af netop dette
   (`_INDEX`-globaler i `beskaer_alle.py`, `beskaer_levering.py` og
   `fals_kvalitet.py`). De kan falde bort bagefter — samme løsning skrevet
   tre steder, fordi den manglede ét sted.
2. **Cache `_graense_og_soem()` pr. side i testfilen.** De seks sider
   behandles to gange. En `lru_cache` fjerner den ene omgang.
3. **Parallelkørsel.** Maskinen har 22 kerner; `pytest-xdist` med `-n auto`
   ville dele arbejdet. Kræver en ny udviklingsafhængighed i et miljø, hvor
   der med vilje kun er numpy og PIL — og gevinsten er begrænset af den
   enkeltværste test (Amdahls lov). **Tag punkt 1 og 2 først**; er pakken
   nede omkring 20 sekunder, er dette formentlig ikke besværet værd.
4. **Del pakken op.** Mærk billedtestene (`@pytest.mark.billeder`) og lad
   dem være fra som standard, med hele pakken før commit. Giver en hurtig
   inderste løkke, men indfører en risiko for, at nogen glemmer at køre
   det hele. Kun værd at overveje, hvis 1-3 ikke rækker.

### Om at vektorisere: pas på

Et glidende vindue over båndene med kumulative summer i stedet for en løkke
er en oplagt tanke, og den kan være rigtig i selve detektionskoden. Men
projektet har allerede prøvet den slags én gang: **vektorisering af
linjesøgningen gjorde den LANGSOMMERE** (0,5 → 0,7 s pr. side), fordi
tabellerne er 24×5, og numpys kaldsomkostning er større end regnestykket
(dagbog 2026-08-29). Mål før og efter, hver gang — og husk at gevinsten her
efter alt at dømme ligger i CSV-indlæsningen, ikke i billedbehandlingen.

### Ikke gjort

Ingen af punkterne er udført. Punkt 1 rører `masterlist.py`, som hører til
den låste stage 01/04-kode, og bør besluttes særskilt frem for at glide med
som en sidegevinst i en måleapparats-ombygning.

## Hvad der IKKE er i planen

- Ingen ændring i facit (stage 02 er låst og urørt).
- Ingen modelkald. Alle trin kan køres på gemte svar.
- Ingen ændring i prøvemængdens status. Den er stadig låst.
