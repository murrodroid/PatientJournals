# Stage 06 — Prompt og model

## Formål

Selve læse-implementeringen: finde den bedste kombination af model, prompt
og opløsning. Dette er projektets kerneleverance, adskilt fra stage 05's
engangs-check af billedforberedelsen og fra stage 07's uenighedslag.

## Inputs

| Type | Sti |
|---|---|
| Testregler | `../../_config/tdd.md` |
| Første tal | `../05_foerste_transskription/output/foerste_tal.md` |
| Måleapparat | `../03_maaleapparat/output/rapportformat.md` |
| Anbefalinger | `../00_forundersoegelse/output/forundersoegelse.md` |

## Process

1. Én akse ad gangen, kun hvis den foregående er afklaret: model,
   promptformulering, **billedskarphed** (se stage 00's fund — dette er
   ikke en mindre detalje; litteraturen finder markant dårligere
   maskinlæsning under en vis skarphedsgrænse, og vores kildebilleder
   ligger sandsynligvis dér eller under), og forbehandling med
   linjedetektion (baselines) enten som udklip eller som hjælp til at
   vide, hvilke linjer der findes på siden.
1a. **Hold Transkribus/specialiseret håndskriftsgenkendelse som reel
    sammenligning, ikke kun baggrundsviden** (stage 00-anbefaling): et
    dansk forsøg fandt Transkribus foran både ChatGPT og Gemini på dansk
    håndskrift, og et uafhængigt studie fandt Claude/GPT-4o på 41-60 %
    tegnfejl på et andet historisk materiale — langt fra Humphries'
    5-7 %. Antag ikke Gemini vinder, før stage 05's egne tal viser det.
2. **Prompt som backup for ufuldkommen beskæring**: stage 04's snit behøver
   ikke være perfekt. Test eksplicit, om en instruktion om at ignorere
   delvis synlig nabotekst i kanten løser resten af problemet billigere end
   at forfine beskæringen yderligere.
3. **Vær opmærksom på fremmed bogstavbleed fra HELT andre opslag**, ikke kun
   det umiddelbare naboopslag. Lead: på en verso-side, fx side 101, kan man
   i affotograferingen nogle gange se stumper af en helt anden, langt
   tidligere verso-side (fx side 51) stikke ind i billedet — formentlig fordi
   bogen ikke ligger helt fladt, og en bagvedliggende side skinner igennem
   eller er synlig i en anden vinkel. Det er en ANDEN forureningskilde end
   den nære nabo-strimmel, stage 04 håndterer, og kan ende forkert i
   transskriptionen, hvis intet fanger det. Undersøg om det sker konsekvent
   på samme kant, og om en prompt-instruktion kan bede modellen ignorere
   løsrevne, ude-af-kontekst bogstaver/ord, der ikke hænger sammen med den
   omkringliggende tekst.
4. Alle kørsler på øvemængden. Den låste prøvemængde røres ikke.
5. Gem altid rå modelsvar + fuld opsætning + dato.
6. Ingen fuld kørsel uden dit udtrykkelige go.

### Låst efter leads svar 2026-08-20/21

- **Overstregning er et selvstændigt forsøg her, ikke i stage 05.** leads
  eget forslag til, hvordan det kan gribes an: en separat model eller prompt,
  der udelukkende leder efter, hvad der KAN være streget ud, hvorefter de to
  svar kombineres, og det overstregede skilles fra, før tegnfejlene regnes.
  Den kendte faldgrube, der skal måles direkte: modellen forveksler
  understregning med overstregning. Facit har begge dele gemt —
  `understreget`-feltet rummer 409 understregninger med linjenummer, og
  `alt_*` mod `rettet_*` viser de 33 overstregninger.
- **Margentekstens placering er udskudt hertil** (beslutning 25). Facit
  placerer margentekst dér, hvor transskribenten satte mærket. Finder
  modellen teksten i en anden rækkefølge, skal det håndteres — men det tages
  op her, ikke før.
- **Er de ulæselige steder ulæselige, eller bare dårligt fotograferet?**
  20-30 tætte udklip i højeste tilgængelige opløsning, forelagt lead.
  Kildeviseren kan ikke levere mere end de ~900-1.000 pixels pr. tekstside,
  vi allerede har (API'et har ingen størrelsesparameter, efterprøvet
  2026-08-21) — så det kræver kollegaens originalscanninger. Falder de 498
  mærker markant, bliver både dækningen og facit bedre. Ingen kode, kun et
  forsøg.

### Baseline-aksen — udskrevet 2026-08-27

Den ene bullet i proces-punkt 1 ("linjedetektion (baselines)") dækkede over et
helt forsøg. Her er, hvad det forsøg består af.

**Hvad en baseline er her.** En baseline er en detekteret grundlinje under hver
skrevet tekstlinje på siden — altså maskinens bud på, hvor linjerne fysisk
ligger, før noget som helst er læst. Vi har dem ikke i forvejen: facit har
ingen koordinater, kun tekst pr. linjenummer, og det er netop derfor
måleapparatet ikke bygger på linjeforankring mod billedet (beslutning 10).

**Hvorfor det overhovedet er en akse.** Helsidelæsning lader modellen selv
afgøre, hvor mange linjer der er, og hvor de begynder. To fejltyper i stage
05's tal kan skyldes præcis det: sprunget-over tekst (linjer modellen aldrig
så) og sammenblanding med naboopslaget. Baselines kan i princippet fjerne
begge, fordi linjeinddelingen så kommer udefra og ikke fra læsningen.

**De to varianter, der skal testes hver for sig** — de er ikke samme sag, og
kun den ene kræver billedbehandling:

1. **Baselines som oplysning i prompten.** Hele siden sendes som nu, men
   prompten får at vide, hvor mange tekstlinjer detektoren fandt, og hvor de
   ligger. Billig at teste, ingen ny billedpipeline. Måler, om modellen retter
   sig efter en linjetælling, den får udefra.
2. **Udklip pr. linje.** Hver detekteret linje klippes ud og sendes for sig,
   og svarene sættes sammen bagefter. Dyrere: kræver udklipning, rækkefølge,
   sammensætning og et bogholderi pr. linje frem for pr. side. Til gengæld er
   linjenummereringen givet per konstruktion, hvilket rammer lige ned i
   måleapparatets forankringstrin.

**Hvad der skal til, for at aksen kaldes en gevinst.** Ikke bare et lavere
hovedtal. Tre ting skal ses samlet i `output/resultater.csv`:

- fuldside-kontrollen (beslutning 43) skal ikke blive dårligere — den er den
  eneste måling, der ser fundet-på og sprunget-over tekst;
- antallet af linjer, forankringen ikke kunne parre, skal falde;
- gevinsten skal holde på mere end én bog, ellers er det en detektor, der
  tilfældigvis passer til én håndskrift.

Bliver hovedtallet bedre, mens fuldside-kontrollen bliver værre, er sporet
forkastet, ikke forbedret.

**Hvor koden kommer fra.** Riksarkivets YOLO-linjemodel, som den er vendoret i
magresprot (`baselines_detect.py`, `baselines.py`) — se
`references/icm_metodik.md` afsnit 2. To forbehold, der skal afklares FØR
noget bygges: magresprots kopi hænger på en lokal sti-dependency
(`pagexml-tools`), der ikke findes på denne maskine, og den er skrevet til at
detektere linjer *inden i givne regioner*. Vi har ingen regioner. Fuldside-
detektion findes formentlig kun i kollegaens
`CopenhagenCityArchives/python-yolo-segmentation`. Er fuldside-varianten ikke
til at få til at køre på et par timer, stopper aksen dér og noteres som
uafprøvet — den er ikke kerneleverancen.

**Hvornår det bliver en stage for sig.** Variant 1 bliver i stage 06; det er
en promptvariant. Vælger vi variant 2, bliver udklipningen reelt et
forbehandlingstrin på niveau med stage 04, med egne mellemfiler og eget mål —
og så flyttes den ud i sin egen stage frem for at gemme sig som en kolonne i
`resultater.csv`. Beslutningen tages, når variant 1's tal ligger.

**Transkribus hører til her.** Det er den anden måde at få specialiseret
håndskriftsgenkendelse i tale på (proces-punkt 1a), og magresprot har allerede
en fungerende Transkribus-klient. Kører Transkribus samme sider gennem det
samme måleapparat, er det en direkte sammenligning og ikke en fornemmelse.

## Outputs

| Fil | Beskrivelse |
|---|---|
| `output/koersler/` | Rå modelsvar pr. kørsel med fuld angivelse af opsætning |
| `output/resultater.csv` | Én række pr. kørsel: opsætning og alle seks måletal |
| `output/sammenligning.md` | Hvad hver akse viste, og hvad der blev valgt |

## Test Contract

Modelsvar kan ikke testes fast. Bogholderiet skal testes: en kørsel gemmes
altid med sin opsætning, og to kørsler med samme opsætning kan skelnes på
dato. En test skal sikre, at kode som læser den låste prøvemængde fejler,
medmindre den udtrykkeligt får lov.

## Handoff

Næste stage er `07_anden_stemme`. Reviewed betyder, at du har set
sammenligningen og udpeget den bedste kombination af model, prompt og
opløsning.
