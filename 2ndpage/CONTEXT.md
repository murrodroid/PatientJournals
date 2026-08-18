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
