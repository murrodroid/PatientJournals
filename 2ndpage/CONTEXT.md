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

- Rummer et fortsættelsesopslag tekst på begge halvsider, og dækker facits
  `[page]`-blok hele opslaget eller kun den ene halvdel? Afgøres i stage 01.
- Findes der skarpere originaler end ~900-1.000 pixels pr. tekstside?
- Hvor står kollegaens dashboard-arbejde, og har han allerede afprøvet sin
  egen `textpage`-prompt på andensider?
- leads fork har umerget `origin/patch-1` og lokal `Severity_prompt`.
