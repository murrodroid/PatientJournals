# CONTEXT.md — beslutningslog for 2ndpage

Kun tilføjelser; daterede afsnit. Status/tjekliste ligger i PROGRESS.md.
Hvad projektet ER lige nu: README.md. Detaljeret baggrundsviden: references/.

## 2026-08-18 — Projektstart: mål, fund og første beslutninger

### Mål (fra kontekstone.md, leads oplæg)

Transskribere **andensiderne** i håndskrevne patientjournaler fra
Blegdamshospitalet (1889-97). Forsiderne er allerede Gemini-transskriberet via
murrodroid/PatientJournals med objekt-/felt-prompts; andensiderne er
fortløbende modtagenotater skrevet linje for linje, så felt-tilgangen
forventes ikke at bære. Kernen i projektet er **systematisk metodetest**:
en testfilstruktur hvor metoder kan køres og evalueres mod håndlavet ground
truth. Vinderen skal kunne loades ind som en metode i kollegaens app.

### Beslutninger truffet i dag

1. **Eget repo i c:\Work\2ndpage** — eksperimenterne bor her med egen
   stage-struktur; PatientJournals bruges som læse-reference, og den vindende
   metode porteres/plugges ind til sidst. (Alternativ, forkastet: udvikle
   direkte på branch i PatientJournals — binder eksperimenterne til deres
   struktur for tidligt.)
2. **Designfilosofi: ICM** (Interpretable Context Methodology) fra
   magresprot_xmltools — nummererede stages med test-håndhævet seks-afsnits
   CONTEXT.md-kontrakt og menneskelig review mellem stages. Se
   references/icm_metodik.md. ICM-skelettet oprettes EFTER /grill-me, så
   stage-inddelingen afspejler den stress-testede plan.
3. **Backends i første testrunde** (leads valg): Gemini (samme som
   forsiderne), Claude (primært som uoverensstemmelses-flagger, jf.
   Humphries), samt baseline-baserede metoder fra magresprot/kollega-repoet
   (YOLO-baselines → udklip til model / Transkribus / Riksarkiv-HTR — sidste
   vurderet mest usandsynlig). Baselines kan også være ren præ-behandling før
   Gemini får hele siden eller udklip.
4. **Eval-metrik**: CER/WER i to varianter (strict + modified à la Humphries)
   mod de manuelle transskriptioner. Findes ikke i PatientJournals (bekræftet
   fraværende både i marts-klonen og upstream/main) — det er vores nye bidrag.

### Dagens vigtigste fund (detaljer i references/)

- **Masterlisten løser side-identifikationen**: Blegdam_master_list.csv
  (570.519 rækker) har `patient_page_counter`; `== 2` giver 71.380 andensider
  direkte. Forsider: 71.378. leads teori bekræftet: listen indekserer en
  billedmappe med samme struktur, som kan skaffes.
- **De lokale billeder er kun udvalgte opslag**: `C:\Work\Alle
  patientjournaler_komprimeret` (23.820 JPG, intake_front_*/intake_dead_*)
  springer 4-8 sekvensnumre ad gangen — andensiderne ligger sandsynligvis
  IKKE lokalt endnu og skal hentes/synkes.
- **Ground truth er hele indlæggelser**: 39 RTF'er ("full journal") hvor
  andensiden er første `[page]`-blok efter forside-markøren. Opmærkning rig
  men ustandardiseret ([?] vs. gæt?, crossed out/written instead, ~8
  positions-tags) ⇒ eval kræver RTF-parser + normaliseringslag. Mulig formel
  spec i naboen `Transcription codebooks` (ikke undersøgt).
- **Kollegaen er tættere på end antaget**: upstream/main har allerede
  `TextPage`/`PageLine`-skema + `textpage`-prompt (linje-for-linje, ignorér
  modstående side) OG en fuld webapp (stdlib http.server) med jobs, batch
  (Gemini/Vertex + Anthropic), UI-redigerbare versionerede skemaer og
  validator-leaderboard. En "metode" = skema + prompt-nøgle + modelnavn.
  Det reelle integrationshul er **billedudvælgelse**: kun `fp_mode`/`_fp`-
  suffiks skelner sidetyper — ingen kobling til masterlisten endnu.
- **Humphries' opskrifter** (references/humphries_generative_history.md):
  Gemini 3 Pro @ temp 0, høj opløsning, minimal thinking = 0,69 % modified
  CER; transcribe-then-correct; og især **model-uenigheds-flagning** (maj
  2026-posten): to modelfamilier gen-transskriberer, kun uenige passager
  (~4 %) til menneskelig review — fangede 76 % af restfejlene. Claude-rollen
  er "anden modelfamilie", ikke "bedre transskribent".
- **Genbrugskandidater**: magresprots tre baseline-moduler (vendoret
  Riksarkivet-YOLO) ligger kun på magresprots origin/main og har en lokal
  sti-dependency (pagexml-tools) der ikke findes på denne maskine — kopiér
  frit af den. Fuldside-baseline-detektion (uden regioner) findes muligvis kun
  i CopenhagenCityArchives/python-yolo-segmentation.

### Kendte faldgruber (fra kontekstone.md + fund)

- Opslag viser nabosider → afgræns primærsiden (CV, prompting,
  klassifikation eller kombination — skal testes; `textpage`-prompten siger
  allerede "Primary Page Only").
- Overstregninger: modeller ser "igennem" dem; ren ignorér-prompt erfaret
  utilstrækkelig; separat flagnings-prompt er en mulighed — testes.
- Mærkelig tegnsætning, super-/subscript, forkortelser; prompts holdes så
  simple som muligt (understøttet af Humphries' fund).

### Åbne spørgsmål

- Hvor kommer andenside-BILLEDERNE fysisk fra (kildemappen masterlisten
  indekserer — sti/synk)?
- Skal ground truth-andensider matches mod eksisterende lokale billeder
  (intake_dead_* dækker formentlig dødsfaldene 1896-97 = deaths-RTF'erne)?
- `Transcription codebooks` og `Automatic transcription versions` +
  `validation data` (naboer til Manual transcriptions) — indhold ukendt.
- leads fork har umerget `origin/patch-1` og lokal `Severity_prompt` —
  stadig relevante, eller overhalet af upstream?
- Koordinering med kollegaen: sidetype-udvælgelse i appen (fp_mode vs.
  masterliste-kobling) og om `textpage`-sporet allerede er testet af ham.

### Næste skridt

/grill-me på planen fra en Opus-agent (lead skifter model manuelt), derefter
ICM-skelet + stage-plan.

## 2026-08-18 (senere) — Grill: rettelser og låste beslutninger

### To rettelser til afsnittet ovenfor

1. **Andensider er `patient_page_counter == 1`, ikke 2.** Tælleren begynder på
   0 ved forsiden. Verificeret i rækkerne omkring en kendt forside. Det, der
   ovenfor kaldes andensider, er tredjesider. Rigtigt tal: 71.391 andensider.
2. **Der er nul andenside-billeder lokalt.** Alle 38 GT-forsider findes blandt
   de 23.783 lokale JPG'er, ingen af de 38 tilhørende andensider gør.
   Kildemappen `C:\Work\Alle patientjournaler` findes ikke længere. Intet
   eksperiment kan køres, før billeder er skaffet.

### Nye fund

- **Ét billede er ét opslag med to sider.** Set visuelt: venstre halvdel er
  forrige blads bagside, teksten står til højre; på forsideopslag er venstre
  side blank. Opløsning kun ~900-1.000 pixels pr. tekstside, hvilket er lavt
  og en åben risiko.
- **Ground truth dækker hele forløb, ikke kun andensiden.** De 38 patienter har
  257 journalsider ud over forsiderne, alle med `[page]`-blokke i RTF'erne.
- **`group_id` løber over bindgrænser** — filtrér altid også på `folder_name`
  og `page_type`.
- **Masterlisten dækker 1880-1910**, bredere end de transskriberede 1889-97.
- **StadsCER (`J-Hoffi/StadsCER`) er næsten det måleapparat, vi skal bruge** —
  se `references/stadscer.md`. Fem-varianters måling, dansk-tilpasset
  tegnfoldning. Mangler orddeling hen over linjeskift, som er vores bidrag.
- **Bogryg-detektion findes ikke** i noget af leads repoer; kun en skreven
  specifikation i magresprots separator-research plus to brugbare byggeklodser
  (kolonnevis blækprofil, og indsigten at ryggen er en top og ligger fast i x).
- Kollegaens modelregister har nu `gemini-3.1-pro`, `gemini-3.5-flash`,
  `claude-opus-4-6`, `gpt-5` m.fl.

### Låste beslutninger

| # | Beslutning | Begrundelse |
|---|---|---|
| 5 | Billeder skaffes som filoverførsel efter en navngiven liste; skyadgang senere | Kollegaen kan levere med det samme; se `billedanmodning/` |
| 6 | Anmodningen omfatter **alle 257 sider med facit** plus 50 andensider fra andre patienter, 1889-97 | Alt målbart materiale først; de 50 giver et blik på variationen |
| 7 | Output er **ren læsetekst**; iagttagelser om understregning, margen og udstregning er et senere spørgsmål | Enkel prompt, ét entydigt mål. Overstregningsfælden måles alligevel, fordi facit beholder erstatningen og ikke det overstregede |
| 8 | Måling på **fladet tekststrøm** som beslutningstal, men **linjeskift bevares** i modellens svar | Linjeskift er værdifulde både som hallucinationsindikator og ved aflevering til KSA |
| 9 | **StadsCERs fem varianter** overtages frem for Humphries' to | Allerede dansk-tilpasset og i brug |
| 10 | Særskilt **hallucinationskontrol** i stedet for ren linjeforankring | StadsCERs argument for linjeforankring forudsætter givne baselines; det har vi ikke |
| 11 | Første forsøg varierer **kun billedforberedelsen**; model og prompt holdes fast | Dobbeltopslaget er det, der gør materialet særligt |
| 12 | Billedforberedelsen får **eget mål og egne tests**, låses før modelsammenligning | Et forkert snit må aldrig kunne forveksles med en dårlig læsning |
| 13 | Scope er **transskription**. NLP-detektion af udsagn er senere eller en kollegas arbejde | leads afgrænsning; fagords-mål holdes som valgfri diagnose |
| 14 | Forundersøgelsen er **én grundig gennemgang**, kan sættes i gang igen ved tvivl | Feltet flytter sig, men et løbende spor koster mere end det giver nu |
| 15 | **Syv stages** (fin opdeling) | Facit-forståelsen skal låses, før måleapparat bygges oven på den |
| 16 | **Eget tyndt kaldelag** til modellerne | Leverancen til kollegaen er prompt, skema og bevis — ikke kørselskode |

### Det store mål bag transskriptionen

leads formulering: at kunne detektere udsagn som "mæslinger i hjemmet",
"underernæret ved ankomst", "har tidligere haft mæslinger". Det er ikke dette
projekts scope, men det forklarer, hvorfor rå tekst nogle gange skal kunne
bruges uden forbehold, og andre gange med viden om usikkerhed — derfor er
usikkerhedsmarkering et lag oven på teksten, ikke vævet ind i den.

### Fortsat åbne punkter

- Findes der skarpere originaler end ~1.650-2.000 px bredde pr. opslag?
- Hvor står kollegaens dashboard-arbejde, og har han allerede afprøvet sin
  egen `textpage`-prompt på andensider?
- leads fork har umerget `origin/patch-1` og lokal `Severity_prompt`.

## 2026-08-18 (senere igen) — Selvbetjent billedadgang fundet mens vi ventede

Mens billedanmodningen afventede kollegaen, blev der bygget et midlertidigt
hente-script mod kbharkiv.dk's kildeviser, og en blokerende antagelse blev
afklaret undervejs.

### Fund: kbharkiv.dk kan hentes direkte, uden om kollegaen

Kildeviseren (`kildeviser.kbharkiv.dk`) er en SvelteKit-app, der taler med et
åbent, udokumenteret API:

- `GET https://api.kbharkiv.dk/pages?unit_id=<bind-id>` → liste af
  `{id, page_number, image_url, ...}` for hele bindet.
- `GET https://api.kbharkiv.dk/file/<id>` → selve billedet (WebP), ingen
  nøgle krævet.

**Forskydning fundet og verificeret to gange uafhængigt**: kildeviserens
`page_number` = masterlistens `<bind>_<counter>` minus 1 (kildeviseren
tæller fra 0 ved et omslagsfoto). Verificeret ved at hente den beregnede
side for to kendte facit-forsider og læse indholdet: `273098_001471`
(Christiane Marie Andersen, Croup) og `273099_001359` (Esther Engstrøm,
diphtheritis) — begge matchede facit ord for ord, inklusive en tredje
kontrol af selve andensidens tekst mod RTF-facit. Antagelsen er stærk, men
kun afprøvet på to bind — bør stikprøvekontrolleres ved brug på nye bind.

Scriptet ligger i `scripts/kbharkiv_hent.py` (uden for ICM-stagestrukturen,
markeret midlertidigt). Har en hård grænse på 20 billeder pr. kørsel som
sikkerhedsnet mod utilsigtede fulde kørsler. 16 rigtige andensider er hentet
til `stages/01_datagrundlag/output/proeve_opslag/` og brugt til at afklare
opslagsstrukturen (se næste punkt).

**Konsekvens for beslutning 5**: filoverførslen fra kollegaen står ved magt
som den rene, langsigtede kanal (hans egne filnavne, ingen tvivl om
kalibrering), men vi er ikke længere blokeret på den. Billedanmodningen bør
stadig sendes.

### Opslagsstrukturen — grundigt revideret efter at have set flere rigtige billeder

Første antagelse (fra det allerførste, lokale forsidebillede) var, at hvert
billede viser et **symmetrisk dobbeltopslag** med to omtrent lige brede
sider og bogryggen i midten. Det holder IKKE for almindelige
fortsættelsesopslag — kun for selve forsideopslaget, hvor formatet ser
anderledes ud (se nedenfor).

**Ny, evidensbaseret model** (set i tre uafhængige billeder:
`273098_001496`, `273098_001508`, `273099_001362`, alle i
`stages/01_datagrundlag/output/proeve_opslag/`): hvert billede er
**asymmetrisk beskåret omkring ÉN målside**, som fylder ca. 75-85% af
rammen, med kun en SMAL strimmel af naboopslaget synlig i den ene kant —
præcis det lead selv skrev i det allerførste oplæg: man "kan risikere at
se NOGET af" nabosider, ikke en hel side. Faktisk indhold pr. `page_counter`
er fuldt ud til stede i sit eget billede (verificeret ord for ord tre gange
mod facit: 1361, 1362, 1363 i bind 273099).

**LØST 2026-08-18 (senere): recto/verso-reglen, forklaret af lead.**
Alle patientjournaler var oprindeligt løse blade, senere indbundet — og når
foldede blade indbindes, starter man altid på en enkelt recto. Derfor er
**forsiden altid recto, andensiden altid verso, tredjesiden altid recto**,
og så fremdeles (lige `patient_page_counter` = recto, ulige = verso, med
forsiden som `0`). Det forklarer strimmel-siden fuldstændigt og passer på
ALLE observerede billeder, ikke kun de fleste:

| Side | Recto/verso | Primærindhold | Strimmel af nabo |
|---|---|---|---|
| Forside (0) | recto | højre | venstre (forrige patients tomme rest) |
| Andenside (1) | verso | venstre | højre (tredjesidens begyndelse) |
| Tredjeside (2) | recto | højre | venstre (andensidens hale) |

Krydstjekket eksplicit mod alle fem billeder, der er set indtil nu:
`273098_001471` (forside/recto → indhold højre ✓), `273098_001496`/`_001508`
(andensider/verso → indhold venstre ✓), `273099_001361` (tredjeside/recto →
indhold højre, fremmed hale-strimmel venstre ✓), `273099_001362`
(andenside/verso → indhold venstre, forhåndsvisning af tredjeside højre ✓),
`273099_001363` (tredjeside/recto → indhold højre, andensidens hale-strimmel
venstre ✓). Alle fem stemmer.

**Konsekvens for stage 04**: hvilken kant der skal beskæres væk, kan
**udledes direkte af `patient_page_counter`s paritet** — ingen CV-gætteri
nødvendigt for selve sidevalget. Opgaven bliver at finde SELVE snitpunktet
(hvor strimlen holder op og hovedsiden begynder) inden for den kendte kant,
ikke at afgøre hvilken kant. Forsideopslagets bredere, mere symmetriske
format hænger sammen med, at forrige patients afsluttede sag ofte efterlader
en næsten tom verso — ikke en systematisk anden fototype.

**Scope udvidet**: Lead ønsker BÅDE andenside (verso) OG tredjeside (recto)
med i projektet, ikke kun andensiden alene.

## 2026-08-18 (endnu senere) — Projektomfang bekræftet; størrelsesoverslag afblæst

- **Fuldt korpus, 1880-1910**, ikke kun de 1889-97-årgange forsiderne
  dækker. De 50 "ekstra andensider" i billedanmodningen dækker kun 1889-97
  — bør udvides ved en senere anmodning, hvis generalisering til de øvrige
  årgange (1880-88, 1898-1910) skal testes for alvor.
- **Størrelsesoverslag for TIFF/komprimeret er nedprioriteret på ubestemt
  tid.** Begrundelse: transskriptionen kommer ikke til at køre lokalt — den
  sker via kollegaens webapp/cloud-pipeline (GCS-bucket). Lokal
  diskplads er derfor ikke en reel begrænsning for dette projekt. Tages op
  igen, hvis/når det bliver relevant; ingen grund til at jage scriptet nu.

## 2026-08-18 (aftentimer) — Snitpunkt-detektion løst og verificeret 8/8

Efter recto/verso-reglen blev afklaret, blev søgningen efter snitpunktet
gjort simpel: begræns den kolonnevise blækprofil til KUN den kant,
paritetsreglen allerede har udpeget (30% af bredden), og find den lyseste
(mindst blækfyldte) kolonne dér — det er den fysiske rille mellem sidens
egen tekst og naboopslagets strimmel.

**Implementeret** i `src/andenside/bogryg.py` (`soegevindue`,
`find_snitpunkt`) og kørt via `src/andenside/kontaktark.py`, som producerer
`stages/04_billedforberedelse/output/snit.csv` og annoterede kontaktark.

**Verificeret med egne øjne på alle 8 billeder i pilotmaterialet** — ikke
kun 1-2 stikprøver denne gang. Alle otte røde linjer lander præcist i den
fysiske rille mellem hovedsiden og naboopslagets strimmel:
`273098_001472/73/96/97/508/509`, `273099_001360/61`. Se
`stages/04_billedforberedelse/output/kontaktark/`.

Låst med en regressionstest (`tests/test_bogryg_real_billeder.py`) mod fire
af de øjenbekræftede positioner, så en fremtidig ændring af algoritmen
opdages, hvis den flytter snittet væk fra det bekræftede bånd.

**Kendt begrænsning, IKKE skjult**: alle 8 billeder stammer fra samme to
bind og måneder (maj-juni 1896), formentlig samme fotograferingssession.
Metoden er ikke afprøvet på bredere materiale, og der er ingen
usikkerheds-flagning implementeret endnu, fordi intet eksempel i
pilotmaterialet fejler — se `stages/04_billedforberedelse/output/usikre.md`.

## 2026-08-18 (natten) — "8/8 perfekt" var forkert; rettet af leads gennemsyn

Lead gennemgik selv kontaktarkene og fandt, at 4 af de 8 (`273098_001496`,
`273098_001497`, `273099_001360`, `273099_001361`) var reelt dårlige — den
røde streg gik igennem starten af naboopslagets tekst, ikke i ryggen.
Min egen visuelle "8/8"-vurdering fra tidligere var altså for overfladisk.

**Diagnose**: algoritmen ledte efter det ABSOLUT lyseste punkt i hele
kant-vinduet ("dal"). Det kan sagtens lande langt inde i NABOENS egen
blanke margen, forbi selve ryggen, hvis naboens tekst ikke fylder hele
vinduets bredde. Et mellemliggende forsøg (udelad top/bund 5% for at
undgå affotograferingens baggrundsskygge) ændrede stort set intet i
tallene — bekræftet numerisk, ikke kun antaget.

**Løsning**: dumpede de rå blækprofil-tal i vinduet og så, at ryggen
rent faktisk viser sig som en **kraftig top** (0,5-1,0 — langt over
håndskrifts normale 0,05-0,15), præcis som magresprots oprindelige
research sagde fra start (se `references/icm_metodik.md`). Det første,
allerførste forsøg denne aften (før recto/verso-reglen var kendt) ledte
også efter en top, men fejlede, fordi det søgte i HELE billedets bredde
uden at vide hvilken kant der var relevant — nu hvor kanten er kendt,
virker top-søgning præcis. `find_snitpunkt` går fra vores egen, betroede
side og ind mod naboopslaget, og snitter ved første kolonne der krydser
en ryg-tærskel (0,30) — det bevarer hele vores egen side og skærer
både ryg og nabo-strimmel fra.

Alle 8 billeder gennemset igen efter rettelsen, inklusive de 4 tidligere
fejlende — alle otte lander nu i selve ryggen. Regressionstesten er
opdateret til de nye, korrekte positioner.

**Læring at tage med videre**: en visuel gennemgang, jeg selv laver, er
ikke nok alene — leads gennemsyn fangede noget, mit eget ikke gjorde.
Stol ikke på egen "set med øjne"-erklæring som endegyldig, når brugeren
kan og vil se efter selv.

**Buffer tilføjet**: snittet flyttes nu en anelse (1% af billedbredden) væk
fra vores egen tekst og ind mod ryggen, efter leads anmodning — uden buffer
risikerer et snit lige på grænsen at skære bittesmå udløbere af bogstaver,
som den udglattede profil ikke fanger. Implementeret i
`find_snitpunkt`s `buffer_andel`-parameter.

**Terminologi rettet**: det hedder "ryg", ikke "rygning" — rettet
gennemgående i kode, tests og denne fil.

### Ny forureningskilde opdaget: bleed fra HELT ANDRE, fjerne opslag

Lead: ud over strimlen fra det UMIDDELBARE naboopslag (som stage 04
allerede håndterer) kan affotograferingen nogle gange vise stumper af en
helt anden, langt tidligere side i samme bind. Eksempel: på en verso-side,
fx side 101, kan man nogle gange se lidt af en helt anden verso-side, fx
side 51, stikke ind i billedet. Formentlig fordi bogen ikke ligger helt
fladt under fotograferingen, og en bagvedliggende side bliver delvist
synlig. Det er en ANDEN og mere lumsk fejlkilde end nabo-strimlen, fordi
den ikke sidder på en forudsigelig kant og ikke kan udledes af
recto/verso-paritet — og den kan ende forkert i transskriptionen, hvis intet
fanger den. Lagt ind som opgave i `stages/06_prompt_og_model/CONTEXT.md`:
undersøg om det sker konsekvent, og om en prompt-instruktion om at ignorere
løsrevne, ude-af-kontekst bogstaver kan løse det billigere end at forsøge at
detektere det billedmæssigt.

### Stage 04 godkendt og låst

Lead gennemgik de opdaterede kontaktark (med buffer) og bekræftede: "de er
perfekte nu". Stage 04 er hermed låst som forudsætning for stage 05 på det
nuværende pilotmateriale (8 billeder, 2 bind). Den kendte begrænsning —
kun afprøvet på ét fotograferingssession, ingen bred test — står stadig
åben og kan afdække nyt, når flere billeder hentes; godkendelsen dækker
metoden på det materiale, den er set imod, ikke en garanti for hele korpuset.

## 2026-08-18 (sent) — Stage 00 gennemført: fem forskningsspørgsmål

Efter lead præciserede rækkefølgen (stage 00 skal laves før stage 05,
ikke springes over), blev fem spørgsmål afgrænset og undersøgt af
selvstændige agenter: overstregningshåndtering, nordisk/dansk
LLM-HTR-arbejde, kvalitetsmål ud over CER/WER, "fjernt bleed" fra andre
opslag, og øvrige benchmarks + opløsning. Fuldt notat med kilder:
`stages/00_forundersoegelse/output/forundersoegelse.md`; åbne punkter i
`aabne_spoergsmaal.md`.

**De to vigtigste, mest konsekvensfulde fund** (kilder med årstal i
`stages/00_forundersoegelse/output/forundersoegelse.md`):

1. **Billedernes skarphed er en målt risiko, ikke kun en formodning.** Et
   studie fra 2025 fandt, at maskinlæsning bliver markant dårligere under
   en vis skarphedsgrænse. Vores sider (~900-1.600 billedpunkter brede)
   ligger sandsynligvis i eller under den grænse. Skal måles konkret og
   testes som egen akse i stage 06 — ikke antages uvæsentligt, bare fordi
   Humphries ikke nævnte det som et problem på sit eget, formentlig
   skarpere materiale.
2. **Humphries' gode tal (5-7 % tegnfejl) må ikke antages at gælde dansk
   håndskrift.** Et uafhængigt studie fandt Claude på 41 % tegnfejl og
   GPT-4o på omkring 60 % tegnfejl på et andet historisk materiale — samme
   modeller som os, radikalt andre tal. Et dansk forsøg (én person, ikke
   et formelt studie) fandt, at ChatGPT direkte digtede indhold på et
   dansk 1844-dokument, og at Gemini fejlede på navne, mens det
   specialiserede program Transkribus slog begge. **Konsekvens**: stage
   05's første rigtige tal skal tages meget alvorligt og ikke antages gode
   på forhånd; Transkribus/specialiseret håndskriftsgenkendelse bør
   forblive en reel sammenligning i stage 06, ikke kun baggrundsviden.

Øvrige fund: overstregnings-litteraturen for LLM'er er reelt tom (vi
bliver de første til at teste det empirisk); et klinisk
nøgleords-overlevelsesmål er en billig, veldokumenteret måde at supplere
CER/WER på uden at bygge fuld NLP-detektion; "fjernt bleed" fra sider
langt fra hinanden er IKKE et dokumenteret fænomen under de kendte
betegnelser (show-through/bleed-through dækker kun samme blads
for-/bagside) — mest sandsynlige forklaring er fysisk forstyrrelse af
bindet, værd at spørge arkivet om, hvis det viser sig systematisk.

### Stage-plan udvidet fra syv til ni stages

Lead påpegede, at kun én stage (den tidligere "05 metodeforsøg") dækkede
selve læse-implementeringen, mens forarbejdet (00-04) havde fem — uforeneligt
med den "fin opdeling"-begrundelse, der blev valgt under grillen (fang fejl
tæt på deres oprindelse). Splittet i tre: `05_foerste_transskription`
(lukker sløjfen på stage 04 med rigtige tal), `06_prompt_og_model` (selve
læse-implementeringen: model, prompt, opløsning), `07_anden_stemme`
(uenighedslag, egen fase siden det er kvalitetskontrol oven på læsningen,
ikke selve læsningen). `08_integration` er den gamle `06`, rykket.

## 2026-08-18 (natten, fortsat) — Snitpræcision behøver ikke være perfekt

leads pointe: selv hvis billedforberedelsen ikke rammer 100% rent, kan
resten løses ved at prompte sig ud af det i stage 05 — bede modellen om
udtrykkeligt at ignorere en delvis synlig nabotekst i kanten af billedet.
Det sænker kravet til stage 04's præcision uden at gøre arbejdet
overflødigt: en renere beskæring er stadig billigere (færre tokens,
mindre risiko for distraktion), men er ikke længere en hård forudsætning
for at komme videre til stage 05.

### Første, mislykkede forsøg på automatisk rygdetektion (læring, ikke løsning)

Et hurtigt prototype-script (`scripts/bogryg_profil.py`, midlertidigt) blev
afprøvet med tre forskellige kolonneprofil-metoder på `273099_001362` —
blækmængde-top, længste sammenhængende mørke løb, og blækmængde-dal.
**Alle tre landede forkert**, midt i selve håndskriften, ikke ved en
fysisk kant. Det bekræftede undervejs den reviderede model ovenfor (der er
ingen tydelig bogryg midt i billedet at finde, fordi billedet ikke er
symmetrisk). Scriptet er bevaret som udgangspunkt, men rygdetektion i
stage 04 skal designes ud fra den asymmetriske model, ikke som en
"midte-søgning", og skal have flere rigtige facit-labels at teste imod, før
den regnes for pålidelig.

## 2026-08-20 — Stage 02 bygget: facit-læseren og klammekortlægningen

Stage 02 er bygget færdig og afventer nu din gennemgang. Koden ligger i
`src/andenside/facit.py` (tolkningen) og `src/andenside/facit_bygger.py`
(samling og skrivning). 45 tests i `tests/test_facit_parser.py` og
`tests/test_facit_rtf.py`, plus syv kontrakttests i
`tests/test_facit_kontrakt.py`. Alle fire outputfiler i
`stages/02_facit/output/` er skrevet.

### Fund, der ændrer noget

- **Der er facit for 168 sider, ikke 257.** De 39 patienter har 268 sider ud
  over forsiden i masterlisten (de 257 fra dødsfaldsmapperne plus Andrea
  Olsens 11). RTF-filerne mærker 208 af dem med et sidemærke, og 40 af de 208
  sidemærker står uden tekst efter sig. Tilbage står **168 sider med faktisk
  transskriberet tekst**. Påstanden ovenfor fra 18. august om at alle 257
  sider har `[page]`-blokke er altså forkert — transskriptionerne stopper
  tidligere end indlæggelserne gør for en del patienter. Billedanmodningen på
  257 billeder er stadig rigtig at sende (de resterende sider er stadig
  gyldige andensider, bare uden facit), men **det målbare materiale er 168
  sider**, ikke 257.
- **To filer ville være tabt af en for snæver søgning.** Bind 37554 har et
  femcifret bind-id, ikke et sekscifret som alle de andre. En regex på seks
  cifre tabte begge filer uden at sige noget. Der er nu en test, der er set
  fejle netop på det.
- **<patientnavn>-filen bruger en anden sidemærkning** — `[273104_001637]`
  uden ordet "page". Alle 39 filer læses nu ens.
- **Klammer kan ligge inden i hinanden**, fx en understregningsnote der
  citerer et ulæseligt sted. Læseren tæller derfor dybde i stedet for at
  bruge en søgning efter enkeltklammer.
- **Fire filer har en tastefejl i opmærkningen** (en klamme der aldrig lukkes,
  tre løse slutklammer). De gættes ikke på plads — teksten bevares, og de
  seks flagede steder står i `output/udeladte.md` til gennemsyn.

### Beslutninger truffet undervejs

| # | Beslutning | Hvorfor |
|---|---|---|
| 17 | **Egen RTF-afkodning frem for et færdigt bibliotek** | Linjeskiftene ER data (de svarer til linjerne på siden), og vi skal styre præcist, hvad der bliver til et linjeskift. Filerne er meget ensartede TextEdit-RTF uden tabeller eller billeder. |
| 18 | **Mærker matches på mønster, ikke på faste strenge** | Opmærkningen er skrevet i hånden over flere år og rummer tastefejl: `crossedout`, `continuded on line`, `right side og page`, `addet on top of line`, `is underline`. En liste over faste strenge ville tabe dem i stilhed. |
| 19 | **Overstreget tekst løber til linjeskiftet, hvis intet mærke lukker den** | Aflæst af materialet: overstregningen lukkes enten af `[written instead]`, af `[continued on line]`, af et håndskrevet linjeskift eller af et rigtigt linjeskift. Der findes ingen forekomst, der løber videre til næste linje. |
| 20 | **Indskud og margentekst BEHOLDES i læseteksten; noter om understregning og placering fjernes** | Ordene i et indskud og i margenen er faktisk skrevet på siden — en model, der læser siden, ser dem. Noten "dette er understreget" omtaler derimod tekst, der allerede står der, og er ikke selv tekst. Margentekst får sin egen linje, så den ikke klistrer sig til journallinjen. |
| 21 | **Bindestreg sidst på en linje samles kun, når næste linje fortsætter med lille bogstav** | Materialet bruger også bindestreg som punktum ("enkelte Rhonchi-" efterfulgt af en ny sætning). Uden reglen blev "Rhonchi- Ingen Snue" til "RhonchiIngen Snue". Står bindestregen efter fx `[?]`, bliver den stående — vi ved ikke, hvad der blev delt. |
| 22 | **Prøvemængden er hver tredje patient, sorteret efter forsidens billed-id** | Ingen lodtrækning betyder ingen frøkerne at glemme og samme opdeling hver eneste gang. Id'erne løber kronologisk gennem bindene, så de 13 prøvepatienter fordeler sig af sig selv over hele perioden maj 1896 – august 1897. Opdelingen er pr. patient, aldrig pr. side: to sider fra samme indlæggelse ligner hinanden i håndskrift, blæk og ordforråd og ville lække fra øve- til prøvemængde. |

### Kendt begrænsning, ikke løst

Transskribenten deler nogle gange et ord hen over to linjer **uden** at sætte
bindestreg — "ved Inspira" / "tion under affekt" er "Inspiration". Der findes
ingen regel, der kan skille det fra to virkelige ord uden at kunne dansk, så
udfladningen lader dem stå som to ord. Effekten på måletallet er lille, men
den er der, og den rammer facit, ikke modellen. Din gennemgang af læseteksten
er stedet, hvor den slags skal fanges.

### Sidegevinst: `pillow` var aldrig erklæret som afhængighed

`pyproject.toml` havde `dependencies = []`, mens `bogryg.py`,
`kontaktark.py` og `opslagsregister.py` alle importerer `PIL`. Stage 04's
tests kunne derfor ikke køre i et frisk miljø. `pillow>=11.0` er nu erklæret,
og hele testsamlingen (101 tests) kører grønt.

### Samme dag, senere — de tre spørgsmål til lead besvaret

Jeg havde bygget tre antagelser ind i facit uden at spørge. De blev lagt
frem, og her er svarene:

1. **Margentekst og indskud tæller MED i facit** som tekst, modellen skal
   ramme. Beslutning 20 ovenfor står altså uændret. Begrundelsen holder:
   ordene er faktisk skrevet på siden, så en model, der læser hele siden,
   ser dem — udelader vi dem fra facit, straffes modellen for at gøre det
   rigtige, hver gang den læser en recept i margenen.
2. **De 40 tomme sidemærker er afklaret, ikke længere tvetydige.** lead:
   "hvis der er tekst i billedet, så er det bare fordi det ikke er
   transskriberet af kollega". Kontrolleret: ti af de fyrre blev hentet fra
   kbharkiv, spredt over alle syv berørte patienter. Alle ti har blæk i
   samme mængde som sider, vi ved har tekst (0,14–0,17 mod 0,14–0,23 på
   stage 01's pilotbilleder). To blev set efter med øjnene — begge fulde af
   tekst, den ene ender med "døde Kl. 8¾", altså den sidste side i
   forløbet. De holdes ude af facit, som de gjorde, men nu af en kendt
   grund frem for en formodning.
3. **Prøvemængdens størrelse er mit valg.** Den bliver stående på en
   tredjedel (13 patienter, 50 sider).

Mønstret bag de 40 er værd at holde fast i: de ligger i kun 7 af de 39
patienter, altid som en sammenhængende hale sidst i forløbet — én patient
har 14 utransskriberede sider i træk. Transskriptionen stopper altså
tidligere end indlæggelsen gør, snarere end at springe enkeltsider over.

## 2026-08-20 (aften) — leads svar på otte spørgsmål om måling og facit

Otte spørgsmål blev lagt frem, hver med situation, valgmuligheder, afvejning
og en anbefaling. Svarene er låst her.

| # | Spørgsmål | leads svar |
|---|---|---|
| 23 | **Ulæselige steder (`[?]`, 496 stk.)** | Skæres ud af begge tekster, før der måles. Stedet tæller hverken for eller imod. |
| 24 | **Overstreget tekst** | **Modellen skal bare læse hvad der står** — også det overstregede, og også det der står efter. Den promptes IKKE til at afgøre, hvad der skulle stå i stedet. Lead har dårlige erfaringer med at få en sprogmodel til at genkende noget som specifikt overstreget. Forsøg med dette senere, ikke nu. |
| 25 | **Margentekstens placering** | Lades ligge indtil videre (mulighed c). Måles ikke for sig nu; tages op senere. |
| 26 | **Hvilket af de fem tegnfejlstal** | (b) uden store/små og uden tegnsætning som arbejdstal, men (a) rå og (c) mest lempelige rapporteres ved siden af — som StadsCER selv gør det. |
| 27 | **Hvornår er det godt nok** | Ingen grænse sættes nu; det første rigtige tal ses først. Derefter sættes grænser pr. brug. |
| 28 | **Måling ud over tegnfejl** | Tegnfejl + ordfejl. Ingen målrettet opmærkning af datoer, temperaturer og medicin — det kræver et større apparat, og **det er ikke dét, lead skal bruge**. Se nedenfor. |
| 29 | **Egen billedhentning** | Hent hele øvemængden (118 sider). |
| 30 | **Billedanmodningen** | Er sendt til kollegaen. |

### Beslutning 24 omgør beslutning 7 delvist — facit har nu to udgaver

Beslutning 7 (18. august) sagde, at facit beholder erstatningen og ikke det
overstregede. Det holder stadig som den *historisk rigtige* tekst, men det
kan ikke være det, vi måler på: når modellen bliver bedt om at læse alt hvad
der står, vil den skrive de overstregede ord, og så ville facit kalde det en
fejl 33 steder — modellen ville blive straffet for at gøre præcis det, vi bad
om.

`facit.jsonl` rummer derfor nu begge udgaver pr. side:

- **`alt_linjer` / `alt_fladet`** — alt hvad der står på siden, også det
  overstregede. **Det er den, målingen bruger.**
- **`rettet_linjer` / `rettet_fladet`** — den rettede læsning, hvor det
  overstregede er fjernet og kun erstatningen står tilbage. Det er den
  historisk rigtige tekst og den, et færdigt datasæt skal rumme.

De to er identiske på alle sider uden overstregning, og en kontrakttest
vogter det.

### Vigtigste steer fra svar 28: hvad teksten skal BRUGES til

leads egne ord: det er "ofte raw tekst jeg skal bruge, altså natural text
som beskriver sådan 'har haft mæslinger', 'underernæret' eller andet.
Temperaturer og medicin er ikke ligeså vigtigt, det er de mere miljøbundne
variable jeg henter i p 2 og 3."

Det er en retningsangivelse, der rækker længere end måleapparatet. Værdien i
anden- og tredjesiderne ligger i **den fortællende optagelsestekst om
patientens forhistorie og levevilkår** — sygdomme, ernæringstilstand,
boligforhold, smittekilde — ikke i de daglige kliniske målinger. Det bør
farve, hvad vi prioriterer at få læst rigtigt, hvilke sider vi ser på først,
og på sigt hvordan margenrecepterne vægtes (spørgsmål 25, udskudt).

## 2026-08-20 (aften, fortsat) — Sideforskydningen bekræftet på alle bind, og en fejl fundet i selve facit

### Forskydningen holder i alle 15 bind

Antagelsen bag hente-scriptet — at kildeviserens interne sidenummer er
`counter - 1` — var kun efterprøvet på bind 273098 og 273099. Efter at
hele øvemængden var hentet, blev de resterende 13 bind kontrolleret: ét
billede pr. bind, håndskriften på målsiden læst og sammenlignet med facits
tekst for samme billed-id.

**Alle 13 bekræftet, ingen tegn på forskydning nogen steder.** Verso/recto-
reglen (ulige forskel fra forsiden = indhold til venstre, lige = til højre)
stemte også hver gang. Det var en reel risiko: en forskydning i ét bind ville
have parret billeder med den forkerte sides facittekst, og ingen test kan
fange den slags.

### Fundet undervejs: facit er ikke fejlfri

Ved kontrollen af `37554_001491` blev der fundet en uoverensstemmelse, som
derefter blev efterprøvet direkte på billedet:

- **Facit skriver:** "Blev syg for **2** Dage siden"
- **På siden står der:** "Blev syg for **3** Dage siden"

Tallet er entydigt på billedet: linje 1 på samme side har "Morbilli for 2
Aar" med et tydeligt fladbundet 2, mens linjen fire linjer nede har et tal
med den dobbeltbuede top, der kendetegner et 3 i den skrivehånd. De to
glyffer ligner ikke hinanden. Resten af blokken matcher ordret.

**Hvorfor det betyder noget:** facit er vores målestok. Rummer målestokken
selv fejl, sætter det et gulv under, hvor lav en fejlprocent vi overhovedet
kan måle — en model, der læser "3 Dage" korrekt, bliver talt forkert. Det
gør ikke projektet ugyldigt, men det betyder, at en meget lav måling skal
mødes med skepsis frem for begejstring, og at de allerbedste resultater bør
efterses manuelt, før de tros.

Det er ét fund fra én stikprøve på tretten sider. Vi ved ikke, hvor hyppigt
det er. Spørgsmålet om, hvordan den slags skal håndteres — rettes i facit,
tælles som støj, eller opgøres særskilt — er lagt til lead og er endnu
ubesvaret.

## 2026-08-20 (sent) — leads rettelser til facit-reglerne, og understregningen reddet

### Fire ændringer i læsereglerne

| # | Ændring | Hvorfor |
|---|---|---|
| 31 | **`[..rede?]` bliver til `[?]`, ikke til teksten "..rede"** | lead: det er ikke en plausibel læsning, men "et ord der ender på -rede". Bogstaverne er ukendte, så stedet kan ikke måles på — en tegnfejlsmåling ville regne på punktummerne. Reglen er generel: enhver klamme med prikker eller ellipse som pladsholder bliver et ulæselighedsmærke. |
| 32 | **En klamme uden spørgsmålstegn er også et læseforslag** | lead: "8 er at se som 7 bare uden ?". `[gangrenerede]` bliver til teksten `gangrenerede`. Kategorien "uforstået" er dermed næsten tom. |
| 33 | **En uafsluttet klamme repareres ved første mellemrum** | lead om de fire tastefejl: "se det som forslag, men for nu bare fjern `[`, `]`, `?`". `[ophentes? sekret af...` bliver til `ophentes sekret af...`. Vi rører kun de klammer, der ALDRIG lukkes — der findes én lovlig klamme, som spænder over et linjeskift. |
| 34 | **Lægens egne spørgsmålstegn røres ikke** | Der er syv `?` uden for klammer, og de er skrevet af LÆGEN: `(Scarlatina?)`, `(injectionssted?)`, `Pneunomia?`, `DB?`. Det er tekst på siden. "Fjern `?`" kan derfor kun gælde transskribentens opmærkning, aldrig teksten. |

### Understregningen bliver gemt for sig i stedet for at gå tabt

Lead: "understregning er ret godt til at benchmarke i senere forsøg, men ikke
så vigtigt lige nu". Reglen om at fjerne noten fra læseteksten står — men
oplysningen om HVAD der var understreget blev hidtil kastet væk sammen med
noten. Den ville være dyr at grave frem igen.

`facit.jsonl` har nu et felt `understreget` pr. side: én post pr.
understregning med linjenummer i `alt_linjer`, om det er hele linjen eller et
citat, og selve citatet. **409 understregninger** er bevaret: 253 hele linjer
og 156 citater.

Linjenummeret er ikke gættet. Det sættes af et usynligt mærke, der følger med
gennem hele oprydningen af teksten, så det peger på den færdige tekst og ikke
på en midlertidig udgave. Transskribenten satte nogle gange noten først EFTER
den næste linje; i de tilfælde ledes der baglæns efter citatet i sidens egen
tekst. **Alle 156 citater rammer nu den linje, de faktisk står på** — mod 150
uden den søgning.

### To ting til senere, som lead rejste

- **Modellen forveksler understregning med overstregning.** Det er en fejl,
  Lead har set før, når man prompter en model til at genkende overstreget
  tekst. Det er en selvstændig grund til IKKE at prompte for overstregning nu
  (beslutning 24), og det er noget, et senere forsøg skal måle direkte —
  netop derfor er understregningsdataene værd at have gemt.
- **Hans forslag til overstregning på sigt**: en separat model eller prompt,
  der udelukkende leder efter, hvad der KAN være streget ud, og så kombinerer
  de to svar bagefter og skiller det overstregede fra, før tegnfejlene
  regnes. Det hører hjemme i stage 06 eller 07, ikke nu.
- **Åbent historisk spørgsmål**: Lead ved ikke, hvorfor nogle dele er
  understreget og andre ikke. 253 hele linjer og 156 citater er et materiale
  at undersøge det på, hvis nogen får lyst.

### Krav til stage 03, fastlagt nu

- **Der skal måles pr. linje, ikke kun på den fladede tekst.** lead: "jeg vil
  ikke nødvendigvis have det fladet ud men også sammenligne pr linje". Begge
  udgaver ligger i facit (`alt_linjer` og `alt_fladet`).
- **Fejllæsninger i facit accepteres som udgangspunkt** (svar på spørgsmålet
  rejst tidligere samme dag): "Vi må som udgangspunkt bare acceptere at der
  kan være fejllæsninger."
- **Stavefejlene i opmærkningen fanges af mønstre**, ikke af faste strenge.
  Efter leads udtrykkelige ønske om at være sikker på det er der nu en test,
  der gennemgår samtlige 31 stavemåder, optællingen fandt, og kræver at hver
  eneste havner i den rigtige kategori. Testen er set fejle: fjernes
  fleksibiliteten i ét mønster, falder `continuded on line` og
  `continued under line` igennem med det samme.

### Kildefilerne er og bliver skrivebeskyttede

Lead 2026-08-20: "jeg regner ikke med at du renser op i de filer, jeg regner
med du transformerer dem over i en klon eller kontinuerligt bare bruger et
script på dem". Det er præcis, hvordan det er bygget: `facit.py` læser
RTF-filerne på OneDrive og rører dem aldrig; alt skrives til
`stages/02_facit/output/`.

Det løfte står nu ikke kun i en tekstfil.
`test_facit_filerne_bliver_aldrig_skrevet_til` tager størrelse og
ændringstidspunkt på alle 39 kildefiler, kører hele bygningen, og kræver at
intet har flyttet sig. Testen er set fejle: lægges der en linje ind i `byg()`,
der blot rører én fils tidsstempel, bliver den rød med det samme.

## 2026-08-21 — En ubevist antagelse fanget: laver modellen sine egne linjeskift?

Lead spurgte, om jeg havde belæg for at modellen ikke ville skrive linje for
linje. **Det havde jeg ikke.** Jeg havde fremstillet det som en kendsgerning.

Ved eftersøgning viste det sig, at samme ubeviste antagelse allerede stod i
`references/stadscer.md` som en afgjort sag — formuleret "vores model laver
sine egne linjeskift" — og at den dér er den **erklærede grund** til, at vi
måler på fladet tekst frem for linjeforankret. En formodning var altså blevet
til en præmis for et metodevalg, uden at nogen havde målt noget.

Referencefilen er rettet, så antagelsen står som det, den er.

### Hvad vi faktisk ved

- **Intet måleresultat.** Der er ikke kørt et eneste modelkald i dette projekt
  endnu.
- **Ét holdepunkt der peger den anden vej**: kollegaens app har allerede et
  linje-for-linje-skema (`TextPage` / `PageLine` med `text`, `metadata` og
  autonummereret `page_line_number`). Nogen har altså bygget ud fra, at en
  model kan levere linje for linje. Om det nogensinde blev afprøvet, står som
  et åbent punkt i stage 08 ("afklar om `textpage` allerede er afprøvet") —
  det er værd at spørge kollegaen om, da han kan have svaret liggende.
- **StadsCERs egen erfaring kan ikke overføres**: dér fodres modellen med
  facits baselines, så linjeopdelingen er ens per konstruktion. Det siger
  intet om, hvad der sker, når modellen får en hel side.

### Beslutning 35: måleapparatet må ikke afhænge af svaret

Stage 03 bygges, så det virker uanset: der måles både på den fladede strøm og
pr. linje via parring af linjerne. Er modellen linjetro, er parringen et
nulled og koster intet. Er den ikke, redder den linjemålingen fra at skride
efter det første afvigende linjebrud.

Dermed blokerer spørgsmålet ikke stage 03, og det behøver ikke besvares på
forhånd. **Stage 05's allerførste kørsel svarer på det gratis** — vi skal bare
huske at kigge efter det, og at notere svaret som et måleresultat frem for en
formodning.

## 2026-08-21 — Tre åbne punkter målt op, så de kan besluttes

De tre resterende åbne punkter i stage 02 var beskrevet for løst til at kunne
tages stilling til. Her er tallene bag dem.

### 1. Umarkeret orddeling: 8 tilfælde, ikke et problem

Metode: korpusset bruges som ordbog. Findes stumpen "Inspira" aldrig som
selvstændigt ord i de 168 sider, mens "Inspiration" gør, er linjeskiftet efter
al sandsynlighed midt i et ord. Konservativt skøn — det fanger kun de sikre.

| Mål | Antal |
|---|---|
| Linjepar undersøgt | 3.124 |
| Linjer der slutter med bindestreg (markeret orddeling) | 284 |
| Kandidatpar (slutter på bogstav, næste starter med lille) | 925 |
| **Formodet umarkeret orddeling** | **8** |

De otte: `Inspira`+`tion`, `fau`+`ces`, `hal`+`sen`, `Respira`+`tionen`,
`Legems`+`bygning`, `Dæm`+`pning`, `lø`+`ber`, `udskri`+`ves`.

Umarkeret orddeling er altså ca. **3 % af alle orddelinger** (8 af 292).
Koster den ét ekstra mellemrum hver, er det **0,009 % af tegnene** i facit.
Det er under støjgrænsen for enhver beslutning, vi skal træffe.

### 2. Fejl i facit: lille gulv under tallet, stor betydning for enkeltsager

Grundlaget er tyndt: kontrollen af sideforskydningen sammenlignede kun de
første 3-6 linjer på 13 sider, altså omkring 60 af korpussets 3.526 linjer.
Dér blev fundet én sikker fejl.

Holdt den rate for hele korpusset, ville det være **omkring 60 fejl**. Med én
observation er usikkerheden enorm — det kunne lige så vel være 10 eller 200 —
men det er tydeligvis ikke nul.

**Det afgørende er, hvad 60 fejl betyder.** Er hver fejl ét tegn, er det 60 af
92.604 tegn = **0,06 %**. Gulvet under tegnfejlsprocenten er altså
forsvindende. Fejl i facit rammer derimod hårdt, når man ser på en ENKELT
uenighed mellem model og facit og konkluderer "modellen tog fejl". Det er dér,
forsigtigheden skal ligge — ikke i hovedtallet.

### 3. Ulæselige steder fylder mere end noget andet åbent punkt

| Mål | Værdi |
|---|---|
| Ulæselighedsmærker i facit | 498 |
| Andel af tegnene i facit | **1,61 %** |
| Sider helt uden et eneste | 33 af 168 |
| Median pr. side | 3 |
| Værste enkeltside | 14 |

Til sammenligning: umarkeret orddeling er 0,009 %, og facit-fejl anslås til
0,06 %. **De ulæselige steder er to størrelsesordener større end begge.**
Beslutningen om, hvordan de behandles, er derfor den eneste af de tre, der
reelt kan flytte et måletal — og den er allerede truffet (beslutning 23:
skæres ud af begge tekster). Det, der endnu ikke er afgjort, er *mekanikken*:
hvordan man finder det stykke af modellens tekst, der svarer til facits `[?]`,
netop dér hvor teksterne er sværest at stille op mod hinanden.

## 2026-08-21 (fortsat) — De tre åbne punkter afgjort

| # | Beslutning | leads begrundelse |
|---|---|---|
| 36 | **Umarkeret orddeling lukkes uden ændring.** De otte tilfælde står som to ord i den fladede tekst. | "det kan endda være meningen hvor skriveren måske ikke har sat en `-`". Pointen er god: transskribenten gengiver siden, og manglede bindestregen dér, er facit netop korrekt. Målt til 0,009 % af tegnene. |
| 37 | **Fejl i facit accepteres.** Ingen systematisk gennemlæsning. | Uenighedslisten fra stage 05 kan ikke skelne facit-fejl fra modelfejl af sig selv — se nedenfor. |
| 38 | **Ved ulæselige steder skæres HELE LINJEN fra målingen** (leads forslag), i stedet for at skære selve stedet ud. | Tegn-for-tegn-opstilling forudsætter, at teksterne ellers ligner hinanden. Gør de ikke det — og det må vi regne med, at de ikke altid gør — falder opstillingen fra hinanden netop dér, hvor den skal bruges. |

### Hvad linje-reglen koster, målt

| Mål | Værdi |
|---|---|
| Linjer i alt | 3.526 |
| Linjer med mindst ét ulæseligt sted | 422 (**12,0 %**) |
| Tegn på de linjer | 10.898 af 89.770 (**12,1 %**) |
| Sider hvor mere end halvdelen ryger | **0** af 168 |
| Sider hvor intet ryger | 33 |
| Median andel tabt pr. side | 11 % |

12 % er til at bære, og ingen enkeltside bliver udhulet. Reglen kræver
desuden ingen tegn-for-tegn-opstilling inde i et sted, hvor den ene tekst er
ukendt — den er robust præcis dér, hvor alternativerne er skrøbelige.

### Den skævhed begge beslutninger deler, og som skal stå i enhver rapport

De 12 % linjer, vi skærer fra, er ikke tilfældige: det er **de sværeste
linjer på siden**. Det er jo netop derfor, transskribenten ikke kunne læse
dem. Måler vi på de resterende 88 %, måler vi på det lettere materiale, og
tallet bliver derfor **for pænt**.

Nøjagtig samme skævhed rammer facit-fejlene: en fejl i facit bliver kun
synlig, hvis modellen læser stedet BEDRE end transskribenten gjorde. Læser
modellen forkert på samme måde — hvilket er mest sandsynligt netop på de
svære steder — tælles det som enighed, og fejlen er usynlig for os.

Derfor skal enhver rapport fra stage 03 og frem oplyse **dækningen** ved
siden af tallet: "målt på 88 % af linjerne; de udeladte 12 % er de sværeste".
Uden den oplysning er tallet misvisende, uanset hvor korrekt det er udregnet.

### Om at skelne facit-fejl fra modelfejl (leads spørgsmål)

Lead: "vi kommer jo til at finde andre fejl end dem som er transkriberet
forkert, så kan vi vel ikke skelne". Det er rigtigt, og der er ingen skjult
teknik, der løser det. Uenighedslisten er en **prioriteret læseliste**, ikke
en sorteringsmaskine: den skifter arbejdet fra at genlæse 3.526 linjer til at
se på nogle hundrede steder, hvor der faktisk er noget at afgøre. Selve
afgørelsen kræver et blik på billedet.

Tre ting gør listen kortere, uden at afgøre noget for os:

1. **To uafhængige modeller, der er enige mod facit.** Stage 07 planlægger
   allerede Claude som anden stemme. Er to læsere enige mod en tredje, peger
   det på den tredje. Ikke bevis — modeller kan dele samme forkerte tilbøjelighed.
2. **Hvilken slags forskel det er.** Et ciffer eller et egennavn er et sted,
   hvor en model ikke har nogen grund til at foretrække det ene frem for det
   andet; den slags uenighed er oftere en reel læseforskel end en opdigtning.
   Strukturelle afvigelser — hele indskudte sætninger — er derimod modellens.
3. **Enkeltstående mod systematisk.** Læser en model en bestemt bogstavform
   forkert, sker det mange steder. En smutter hos transskribenten er ét sted.

**Den standardmetode, vi ikke har:** normalt fastslås fejlprocenten i et
facit ved dobbelt-indtastning — to personer transskriberer uafhængigt, og
uenighederne afgøres af en tredje. Vores facit er skrevet én gang af én
person. Det er en kendt begrænsning i denne slags arbejde, ikke en fejl ved
netop dette projekt, men det betyder, at vi **ikke kan sætte et tal** på
facits kvalitet med de midler, vi har. Vi kan finde eksempler; vi kan ikke
måle en rate.

## 2026-08-21 — Fire forslag til fremadrettet arbejde med de ulæselige steder

**Ingen af dem er besluttet.** Beslutning 38 (skær hele linjen fra) står som
den fungerende regel; disse fire er veje, der kan gøre den mindre kostbar
eller gøre problemet mindre. Rækkefølgen nedenfor er min anbefaling.

### Målingerne, de bygger på

| Mål | Værdi |
|---|---|
| Ulæselighedsmærker | 498 |
| Svære linjer (mindst ét mærke) | 422 af 3.526 = 12,0 % |
| **Kendt tekst på de svære linjer** | **9.404 tegn = 86 %** |
| Ulæseligt på de samme linjer | 1.494 tegn = 14 % |
| Kendte stumper mellem mærkerne | 647, median 12 tegn |
| — heraf over 15 tegn (gode holdepunkter) | 252 |
| — heraf under 5 tegn (svage) | 122 |

Vi smider altså 9.404 tegn brugbart facit væk for at undgå 1.494 ukendte.

**Opløsning, efterprøvet:** kbharkiv-API'et har ingen størrelsesparameter —
`/file/<id>` giver ét billede, typisk 1610 × 2205 pixels for et helt opslag,
altså omkring 900-1.000 pixels pr. tekstside. Vejen til højere opløsning går
gennem kollegaens originalscanninger, ikke gennem kildeviseren.

---

### Forslag 1 — Genvind de 86 % ved at forankre i de kendte stumper

I stedet for at kassere hele linjen deles den ved de ulæselige steder, og hver
kendt stump søges i modellens tekst som en ren tekststump. Rammer stumpen,
måles den; rammer den ikke, springes den over.

- **Hvad det giver**: op mod 9.400 tegn tilbage i målingen, og dermed en
  dækning nær 100 % i stedet for 88 %.
- **Hvorfor det ikke er mulighed (a) igen**: (a) krævede en tegn-for-tegn-
  opstilling af HELE teksten, som falder fra hinanden, når modellen fejler
  andre steder. En søgning efter en enkelt stump på 12-30 tegn er langt mere
  robust — den behøver ikke, at resten af siden passer.
- **Risiko**: de 122 stumper under 5 tegn er for korte til at forankre
  entydigt og skal formentlig udelades. Rammer en stump ikke, ved vi ikke om
  modellen læste forkert eller bare skrev det anderledes — så en stump, der
  ikke findes, må ikke tælle som en fejl uden videre.
- **Hvornår**: stage 03, som en udvidelse af linje-reglen.

### Forslag 2 — Brug de svære linjer som hallucinationsprøve

De 422 linjer kasseres i dag. De er samtidig det bedste sted i hele materialet
til at måle den fejl, der er farligst for en historiker: at modellen digter
flydende tekst, hvor siden er ulæselig.

Vi kan ikke måle korrekthed dér (der findes ingen sandhed), men vi kan måle
**adfærd**: skriver modellen noget som helst? Markerer den selv usikkerhed?
Hvor langt er det, den skriver, sammenlignet med det, transskribenten kunne
læse rundt om det?

- **Hvad det giver**: et selvstændigt tal for opdigtningstilbøjelighed, som
  ikke kræver facit — og som er langt mere relevant for, om du tør bruge
  teksten, end en tegnfejlsprocent.
- **Risiko**: det er et adfærdsmål, ikke et korrekthedsmål, og må aldrig
  præsenteres som om det var det sidste.
- **Hvornår**: stage 03 (målet defineres), stage 05 (første tal).

### Forslag 3 — Efterprøv om stederne overhovedet ER ulæselige

Transskribenten arbejdede ud fra de billeder, hun havde. Stage 00 fandt
billedskarphed som en målt risiko for hele projektet. Hvis en del af de 498
mærker skyldes billedkvalitet snarere end blækket på papiret, forsvinder
problemet delvist af sig selv, når kollegaens originalscanninger kommer.

Fremgangsmåde: tag 20-30 ulæselige steder, klip tæt om dem i den højeste
opløsning, vi kan skaffe, og se om et menneske eller en model nu kan læse dem.

- **Hvad det giver**: et svar på, om 498 er et tal om kilden eller om vores
  billedfil. Falder det markant, bliver både dækningen og facit bedre — og det
  gavner hele korpusset, ikke kun målingen.
- **Forudsætning**: kollegaens levering. Kildeviserens API kan ikke levere
  mere end det, vi allerede har (efterprøvet).
- **Hvornår**: når billederne kommer.

### Forslag 4 — Lad modellen være anden transskribent netop dér

De 498 steder er præcis dér, hvor en maskine kan tilføje noget, mennesket ikke
kunne. Er to uafhængige modeller enige om en læsning på et `[?]`, er det et
kvalificeret bud, der kan forelægges dig sammen med et tæt udklip af stedet.

- **Hvad det giver**: problemet vendes fra en måleteknisk byrde til et
  produkt — huller i din egen transskription bliver fyldt ud. Det er også den
  eneste realistiske erstatning for dobbelt-indtastning, som vi ikke har.
- **Risiko**: modeller opdigter overbevisende, og to modeller kan dele samme
  skævhed. Må aldrig gå direkte i facit — kun forelægges med billedet, og du
  taster ja eller nej.
- **Hvornår**: stage 07, hvor Claude allerede er planlagt som anden stemme.

---

**Sammenhængen mellem dem**: 1 og 2 gør det bedste ud af materialet, som det
er. 3 kan fjerne noget af problemet ved roden. 4 vender resten til noget
brugbart. De udelukker ikke hinanden, og ingen af dem ændrer beslutning 38 —
de bygger ovenpå den.

## 2026-08-21 — De fire forslag skæres ned til ÉN funktion

leads indvending mod at bygge alle fire: risiko for oppustet, uigennemskuelig
kode. Den er berettiget — forslagene var skrevet som en liste over funktioner,
ikke som et design.

**Forslag 1, 2 og 4 kræver den samme ene handling:** find facits kendte stumper
i modellens tekst. Alt andet er aflæsninger af dens resultat.

```
Facit-linje:    "væg, men denne var [?], og Canylen"
Kendte stumper: "væg, men denne var"  |  ", og Canylen"
Modelsvar:      "...væg, men denne var tynd, og Canylen..."
                  └── stump 1 ────┘ └gab┘ └── stump 2 ──┘
```

- Stumperne er kendt sandhed → de måles (forslag 1).
- Gabet er det, modellen skrev, hvor facit siger `[?]`; dets længde er
  hallucinations-signalet (forslag 2).
- Samme gab er modellens bud på stedet (forslag 4).

### Hvad der bygges i stage 03

Én funktion, med tests:

    forankr(facit_linje, modeltekst) -> (fundne_stumper, gab, ikke_fundne)

Regler, der holder den lille og ærlig:

- **Stumper under 5 tegn bruges ikke** (122 af de 647). De kan forankre hvor
  som helst og ville give falsk tryghed.
- **Et gab tælles kun**, når stumperne på BEGGE sider er fundet. Ellers ved vi
  ikke, hvor det begynder og slutter.
- **En stump, der ikke findes, er ikke en fejl.** Den er uforankret, og linjen
  falder tilbage til beslutning 38: hele linjen ud af målingen.

Det sidste punkt er vigtigt for tilliden til tallet: **beslutning 38 forbliver
grundreglen**, og forankringen er en forbedring oven på den med en defineret
vej tilbage, når den ikke virker. Der er altså ikke to måder at måle på — der
er én regel med et beskrevet fejltilfælde.

### Hvad der IKKE bygges nu

- **Ingen gennemsyns-app til forslag 4.** Gabene skrives bare til en fil.
  Arbejdsgangen med udklip og ja/nej hører i stage 07 og bygges først, når der
  er noget at se på.
- **Intet mål for om modellen selv markerer usikkerhed.** Det ville kræve, at
  prompten beder om det, og beslutning 24 siger, at vi ikke beder modellen om
  den slags vurderinger. Kun gabets længde og indhold måles — det er gratis.
- **Forslag 3 er ikke kode.** Det er et forsøg, der køres, når kollegaens
  originalscanninger kommer: 20-30 tætte udklip, som lead ser på. Det står
  som et punkt i stage 06's tjekliste, ikke som en opgave i stage 03.

## 2026-08-21 — Stage 02 godkendt og låst

Lead: "alt lyder godt". Tolkningsreglerne i `klammekonventioner.md`, de to
facit-udgaver og læseteksten er bekræftet af historikeren. **Stage 02 er
hermed låst** som grundlag for stage 03.

Låst betyder ikke forseglet: dukker der senere et problem op, der hører
hjemme i facit — fx hvis kollegaens originalscanninger viser, at flere af de
498 ulæselige steder faktisk kan læses — genåbnes stagen uden tøven.

### Beslutningerne er skrevet ud til de stages, de rører

Beslutninger truffet i anden session stod kun i rod-`CONTEXT.md`. De er nu
skrevet ind i de enkelte stage-kontrakter, så en agent, der kun læser sin
egen stage, får dem med:

| Stage | Hvad der er skrevet ind |
|---|---|
| **03 måleapparat** | Hele `Process`-afsnittet er skrevet om: arbejdstal (26), mål på `alt_*` (24), både fladet og pr. linje, den ubeviste linjeskifts-antagelse (35), orddelingsreglen (21), linje-reglen ved ulæselige steder (23 + 38) med målt pris, `forankr()`-funktionen som ÉN ting frem for fire features, ordfejl uden målrettet opmærkning (28), krav om at dækningen står ved hvert tal, ingen kvalitetsgrænse på forhånd (27), og at facit rummer fejl (37) |
| **05 første transskription** | Prompten beder om at læse hvad der står — ikke om at genkende overstregning (24) eller markere ulæselige steder. Kør på øvemængdens 118 sider. To ting at kigge efter i første kørsel: laver modellen sine egne linjeskift, og hvad skriver den på de 422 svære linjer |
| **06 prompt og model** | Overstregning som selvstændigt forsøg med leads eget forslag til fremgangsmåde. Margentekstens placering (25). Forsøget med højere opløsning på de ulæselige steder — kræver kollegaens scanninger, kildeviseren kan ikke levere mere |
| **07 anden stemme** | Uenighedslisten er en læseliste, ikke en sorteringsmaskine. Dens skævhed skal stå i rapporten. To modeller enige mod facit er det stærkeste signal, vi har. Modellens bud på ulæselige steder forelægges her, ikke i stage 03 |

## 2026-08-21 — Gennemgang af begge sessioner: 17 fund, ét i det leverede facit

Lead bad om en gennemgang af alt fra begge sessioner — kontekst, beslutninger,
sammenspil — i stil med et "doven senior"-review. Arbejdet blev delt på to
agenter, én til dokumentationen og én til koden, og deres fund er efterprøvet
her, før der blev rettet.

### Det alvorligste: en fejl i det leverede facit

En uafsluttet klamme **slugte resten af siden**. Reparationen skete først ved
enden af blokken, så dybdetællingen aldrig kom tilbage til nul — og dermed
blev alle SENERE mærker på siden aldrig genkendt.

Det stod i det leverede facit: `273104_001643` havde `[added over line](Fibiger)`
bogstaveligt i både `alt_fladet` og `rettet_fladet`. Værre viste et konstrueret
eksempel, at en ægte overstregning efter stedet heller ikke blev fjernet fra
den rettede udgave — netop den udgave, hvis eneste opgave er at fjerne den.

Rettet ved at sætte de glemte slutklammer ind **før** teksten deles op. Facit
er bygget igen; ingen side har rå klammemærker tilbage.

**Lærdommen**: stage 02 var godkendt og låst, da fejlen blev fundet. Det er
netop derfor "låst" ikke må betyde "forseglet". En stage genåbnes, når der
dukker noget op, der hører hjemme i den.

### Fire mindre fejl i koden

- Et nultegn i kilden ville vælte kørslen med et råt nedbrud, fordi læseren
  selv bruger et nultegn som usynligt mærke. Kilden renses nu for det først.
- Oprydningen af tomme linjer kørte FØR mærkerne blev fjernet, så mærkerne
  forhindrede den i at se hele løbet af nylinjer. Oprydningen er skrevet om og
  blev enklere: linjenumre tildeles først, mærkerne fjernes, og linjerne rykker
  sammen med numrene følgende med.
- Ordgrænsen manglede om "page" i positionsmønstret, så et læseforslag, der
  blot indeholdt bogstavfølgen, forsvandt sporløst. Tastefejlen `midpage` er
  beholdt udtrykkeligt i mønstret.
- Den baglæns søgning efter et understregningscitat matchede kun første ord og
  kunne lande på en forkert linje med samme første ord. Den kræver nu to ord.
  Resultat: 155 citater rammer, 1 rammer forbi.

### Tolv fund i dokumentationen

Det vigtigste mønster: **stage-kontrakterne var løbet fra virkeligheden**. Både
stage 01's og stage 02's egne `CONTEXT.md` beskrev regler, senere beslutninger
havde omgjort. Det er alvorligt netop i denne metodik, hvor en agent kun læser
sin egen stages kontrakt — den ville have bygget efter den forkerte regel.

`CONTEXT.md` rummede desuden afsnittet om leads rettelser to gange, hvoraf den
første kopi var forvansket: en kommando havde fejlet på backticks og nået at
skrive en udgave, hvor filnavne og stier var faldet ud af teksten. Den er
fjernet.

Resten var forældede tal: 35 mod 38 beslutninger, seks mod fire flagede steder,
"ca. 55" mod 50 sider i prøvemængden, 38 mod 39 patienter brugt om hinanden,
404 noter mod 409 poster, en overhalet advarsel om at kun 8 af 39 filer var
læst, og et tomt `research/` listet i README.

### Hvad der IKKE blev rettet, med vilje

- **Daterede afsnit i `CONTEXT.md`, hvor et tal var sandt på datoen.** Filen er
  en append-only beslutningslog; omskrives historikken, holder den op med at
  være en log. "Seks flagede steder" står stadig i afsnittet fra 20. august,
  hvor det var seks — beslutning 32 gjorde det til fire, og det står i sit eget
  afsnit.
- **En midlertidig fil fra review-agenten** (`scratch_out.txt`) blev fejet med
  ind i commit `8244f9b` og fjernet igen i `a6e66d6`. Den findes ikke i HEAD.
  Historikken omskrives ikke for det.

### Om at bruge agenter til gennemgang

Begge agenter leverede fund, der holdt ved efterprøvning, og kode-agentens
vigtigste fund var en fejl, der allerede lå i en leveret fil — den slags findes
ikke ved at læse sin egen kode igennem igen. Men de blev alle efterprøvet her,
før der blev handlet på dem, og ét forslag blev forkastet: at flage
`[continued on line]` og indskudsmærker uden modpart ville have givet støj,
fordi de legitimt optræder alene.

## 2026-08-22 — Stage 03 bygget: måleapparatet

Stagens kontrakt lå fast fra 21. august. Under bygningen dukkede fem valg op,
som kontrakten ikke havde taget stilling til, fordi de først bliver synlige,
når koden skal skrives. De er truffet her og skrevet ind, så de kan ses efter.

### Nye låste beslutninger

| # | Beslutning | Begrundelse |
|---|---|---|
| 39 | **Forankringen søger med tolerance, ikke efter ordret træf.** En facit-stump regnes for fundet, hvis det nærmeste stykke modeltekst afviger med højst 40 % af stumpens længde (`MAKS_AFVIGELSE = 0,4`). | Krævede forankringen ordret træf, ville hver eneste forankret stump per definition have nul fejl. Tallet ville så måle, hvor tit modellen var fejlfri — ikke hvor god den er. Måleapparatet ville bekræfte sig selv. |
| 40 | **Linjeparringen ER forankringen** — der bygges ikke en separat mekanisme til at parre linjer. Hver facit-linje søges i modellens rå tekst uden hensyn til dens linjeskift. | Kontraktens punkt 3 kræver parring, så målingen ikke skrider efter det første afvigende linjebrud. Det falder gratis ud af forankringen, og det holder stagen på ÉN funktion frem for to, der kan komme i modstrid. |
| 41 | **Linjetrofasthed er noget vi MÅLER, ikke noget vi antager.** To tal i rapporten: hvor mange facit-linjer der ligger inden for én af modellens linjer, og hvor mange der får deres egen. | Beslutning 35 sagde, at vi ikke ved, om modellen laver sine egne linjeskift. Nu afhænger målingen ikke af svaret, og svaret kommer af sig selv ved stage 05's første kørsel. |
| 42 | **Orddelingen afgøres på FACITS linjer og bruges på begge sider.** Deler facit et ord over to linjer, samles det både i facit og i modellens tekst. | Ellers straffes en model, der har læst rigtigt: facit deler `Infektions-` / `sygdomme.`, og en model, der skriver `Infektionssygdomme` i ét stykke, ville få en fejl for det. Linjedelingen er sidens artefakt, ikke tekstens indhold. |
| 43 | **Fuldside-kontrollen indføres som fast del af rapporten.** På de sider, der slet ikke har `[?]`, sammenlignes hele siden direkte uden forankring. | Forankringen måler kun det, den kan parre. Den ser hverken tekst, modellen har fundet på, eller tekst, den har sprunget over. Kontrollen er det ene tal i rapporten, der ikke kan pynte på noget, og den skal stå der, uanset om den flatterer. |

### Målt under bygningen

Tallene stammer fra `stages/03_maaleapparat/output/selvtest.md`, kørt på
øvemængdens 118 sider uden et eneste modelkald.

- **Forankringen redder 94,6 % af de svære linjer.** 297 af øvemængdens 2.586
  linjer har mindst ét `[?]`; 281 af dem kan forankres alligevel. Dækningen
  bliver 97,6 % af tegnene i stedet for de knap 88 %, beslutning 38 lagde op
  til. **Men det er en øvre grænse**: i selvtesten er "modellen" facit selv, så
  hver stump findes ordret. En rigtig model læser dårligere.
- **Skævheden er nu et tal.** Bytter vi selv 5.087 bogstaver om, finder
  målingen 4.737 af dem — 93,1 %. Ved 2 % forvanskning: 93,9 %. Målingen
  underrapporterer altså systematisk, fordi de linjer, den ikke kan forankre,
  er de hårdest ramte. Det er præcis den skævhed, kravet om at anføre
  dækningen ved hvert tal skal holde synlig.
- **Knappen kan bruges til at pynte, og det er nu dokumenteret.** Sænkes
  `MAKS_AFVIGELSE` fra 0,4 til 0,2, falder den målte tegnfejl fra 7,50 % til
  7,13 % — den ser bedre ud — mens dækningen falder fra 97,2 % til 94,9 %, og
  andelen af fundne fejl fra 93,1 % til 86,4 %. Den strengeste indstilling
  giver det pæneste og mest misvisende tal. Tabellen står i selvtesten, så
  valget ikke kan træffes ubemærket.
- **Nyt fund: en sprunget-over blok giver fantomfejl.** Springer modellen en
  hel del af siden over, forankrer nogle af de manglende linjer sig fejlagtigt
  i en lignende linje andetsteds. Prisen er lille (0,45 % tegnfejl på en side,
  hvor svaret burde være nul), men den er der. Det er grunden til, at
  rapporten udpeger de værste enkeltsider til gennemsyn med øjnene frem for
  at lade dem gå op i et gennemsnit.

### Rettelse til et tidligere tal

Facit rummer **3.680 linjer**, ikke 3.526. De 3.526 stammer fra opmålingen 21.
august, som blev lavet FØR facit blev bygget igen efter de 17 fund samme dag.
De øvrige tal fra den opmåling holder: 422 svære linjer, 647 stumper, median
12 tegn, 122 under fem tegn. Andelen af svære linjer flytter sig derfor fra
12,0 % til 11,5 %.

### En sjette variant i tabellen

StadsCER rapporterer fem varianter. Vi rapporterer seks: `arbejdstal` (uden
versaler OG uden tegnsætning) er tilføjet, fordi beslutning 26 udpeger netop
den kombination som det tal, vi træffer valg ud fra — og den kan ikke regnes
ud ved at lægge de to enkeltfiltre sammen. `raa` er stadig det tal, leverancen
står ved.

### Hvad der bevidst IKKE blev bygget

- **Ingen gennemsyns-app til gabene.** De skrives til en fil
  (`output/gab_eksempel.csv` viser formatet) og som en afkortet tabel i
  rapporten. Arbejdsgangen med tætte udklip og ja/nej hører i stage 07.
- **Intet mål for om modellen selv markerer usikkerhed.** Det ville kræve, at
  prompten bad om det, og beslutning 24 siger, at vi ikke beder modellen om
  den slags vurderinger.
- **Ingen kvalitetsgrænse.** Beslutning 27 står: Lead ser det første rigtige
  tal, før der sættes grænser.

## 2026-08-22 (senere) — Fantomfejlene efterprøvet: der var en mekanisme mere

Da stage 03 blev skrevet ind ovenfor, stod der som forklaring på fantomfejlene:
"når en hel blok mangler, forankrer nogle af de manglende linjer sig fejlagtigt
i en linje, der ligner, andetsteds på siden." **Det var en formodning skrevet
som en kendsgerning.** Den er nu efterprøvet linje for linje, og den holder —
men den er kun halvdelen.

### Hvad der faktisk sker

**1. En manglende linje forankrer sig i en linje, der ligner.** Bekræftet.
Facits `Hendes tilstand er i løbet af natten bleven` findes ikke i modellen,
men `I løbet af natten` gør, og stumpen lander dér. `Tungen` lander i `Lunge`.
`ingen Appetit, ligget hen og døset,` lander i `Det ligger hen og døser,`.

**2. Og det skader de EFTERFØLGENDE linjer.** Det var ikke med i forklaringen,
og det er den vigtigere halvdel. Forankringen går fra venstre mod højre, så et
falsk træf flytter søgepunktet frem forbi det sted, hvor de næste linjer i
virkeligheden står. De finder derefter kun en afskåret rest af sig selv:
`begge Lunger overalt en Mængde fugtige` blev målt mod
`r overalt en Mængde fugtige`, selvom modellen havde skrevet hele linjen
rigtigt.

### Hvad det betyder

Målt pris på øvemængden: 181 tegn fordelt på 27 af de 118 sider, når hver side
får sin midterste tredjedel skåret væk. Lille — men den **vokser med, hvor
meget modellen springer over**, og den rammer linjer, modellen læste korrekt.

Det ændrer ikke koden. Venstre-mod-højre-rækkefølgen er der af en grund: uden
den kunne en gentaget vending forankre bagud og lave et gab med negativ
længde. Prisen for at fjerne fejlen ville være en større fejl.

**Men den udpeger en knap, hvis det senere viser sig at betyde noget.** Det
mindste konstruerede eksempel, der fremkalder fejlen, bruger en facit-stump på
præcis fem tegn — `Lunge` — som forankrer sig inde i ordet `Lunger` på næste
linje. Det er lige netop grænsen `MINDSTE_STUMP = 5`, som stage 03's punkt 7
satte for at undgå stumper, der "kan forankre hvor som helst" (ikke at
forveksle med beslutning 7, som handler om ren læsetekst). Fem tegn var altså
ikke nok i det tilfælde. Grænsen er ikke hævet: det ville koste dækning på
netop de svære linjer, den er sat for at redde, og der er ingen måling endnu,
der siger at falske forankringer er et problem i praksis. Men viser stage 05,
at de er det, er `MINDSTE_STUMP` stedet at kigge — ikke rækkefølgen. En
ordgrænse-regel duer i øvrigt ikke: facits linjer begynder og slutter
rutinemæssigt midt i et ord (`Kvælnings` / `anfald`).

Begrænsningen er pinnet som en test (`test_falsk_forankring_skader_ogsaa_den_
naeste_linje`), så den ikke kan ændre sig ubemærket.

Men det ændrer, hvordan rapporten skal læses, og det står nu skrevet begge
steder: **en side med lav dækning skal ses efter med øjnene**, ikke tros.
Rapporten fik derfor sin egen liste over de tyndest målte sider ved siden af
listen over de værste — en side, hvor modellen sprang det meste over, får
nemlig et FLOT tegnfejlstal og lander i bunden af "de værste", hvor ingen
kigger.

### Om at skrive formodninger som kendsgerninger

Det er anden gang i dette projekt, at en forklaring er gledet ind i
dokumenterne som noget målt (første gang: "modellen laver sine egne
linjeskift", fanget af lead 21. august, nu beslutning 35 og 41). Begge gange
var forklaringen rimelig, og den ene gang var den endda rigtig. Det er ikke
pointen. Pointen er, at et projekt, hvis hele formål er at skelne målte tal fra
formodninger om en models kvalitet, ikke kan tillade sig at være sjusket med
samme skel i sine egne noter.
