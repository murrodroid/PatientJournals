# PROGRESS.md — levende tjekliste for 2ndpage

Status her vinder på "hvad er gjort"; `CONTEXT.md` vinder på "hvorfor".
Videre til næste stage kun efter menneskelig gennemgang.

## Dagbog

- [2026-08-20 12:00](diary/2026-08-20.md) — Projektet 2ndpage bygget fra bunden: kortlægning, ICM-skelet, stage 00/01/04 gennemført
- [2026-08-20 16:20](diary/2026-08-20.md) — Stage 02 bygget: facit-læser, klammekortlægning, øve/prøve-opdeling
- [2026-08-21](diary/2026-08-21.md) — Stage 02 godkendt og låst; 16 beslutninger truffet i dialog
- [2026-08-22](diary/2026-08-22.md) — Stage 03 bygget: måleapparatet; beslutning 39-43; skævheden gjort til et tal
- [2026-08-23](diary/2026-08-23.md) — Stage 03 godkendt og låst; den strenge måling tilføjet (beslutning 44)
- [2026-08-23 21:05](diary/2026-08-23.md) — Gennemgangen afslørede, at rapportens forklaring ikke virker; instrument-tal blev fremlagt som resultater
- [2026-08-27 12:55](diary/2026-08-27.md) — Stage 04 genåbnet: falsdetektionen kunne ikke virke generelt, og piloten afslørede det
- [2026-08-27 15:40](diary/2026-08-27.md) — Beskæringen følger nu falsen bånd for bånd; tre fejlmålinger undervejs

## Fase 0 — Kortlægning og plan (afsluttet 2026-08-18)

- [x] Kortlagt ICM-metodikken, PatientJournals, kollegaens app, facit,
      masterliste, billeder og Humphries → `references/`
- [x] Kortlagt eget forarbejde i StadsCER → `references/stadscer.md`
- [x] Eftersøgt bogryg-kode i egne repoer (findes ikke; kun specifikation)
- [x] Rettet to fejl: andensider er `patient_page_counter == 1`, og der er
      nul andenside-billeder lokalt
- [x] Grill gennemført; 16 beslutninger låst i `CONTEXT.md`
- [x] Billedanmodning skrevet: 307 billeder → `billedanmodning/`
- [x] ICM-skelet oprettet: `AGENTS.md`, `_config/tdd.md`, ni stages, strukturtest
      (05 splittet i 05/06/07 senere samme dag — se stage 05-afsnittet)

## Stage 00 — Forundersøgelse

- [x] Afgrænset fem spørgsmål: overstregninger, nordisk/dansk arbejde,
      kvalitetsmål ud over CER/WER, fjernt bleed, benchmarks + opløsning
- [x] Fem agenter kørt, hver med kilder+år, måling vs. formodning skilt ad
- [x] `output/forundersoegelse.md` — seks handlingsanvisende anbefalinger
- [x] `output/aabne_spoergsmaal.md` — seks punkter kun vi selv kan afklare
- [ ] **Gennemgang ved lead**

**Vigtigste fund**: vores billeders skarphed er en målt, reel risiko —
maskinlæsning bliver markant dårligere under en vis skarphedsgrænse, og
vores sider ligger sandsynligvis dér eller under. Humphries' gode tal
(5-7 % tegnfejl) kan IKKE antages at gælde vores danske materiale; ét
uafhængigt studie fandt 41-60 % tegnfejl på et andet historisk materiale
med de samme modeller. Stage 05's første tal skal tages meget alvorligt,
uanset hvad litteraturen ellers siger.

## Stage 01 — Datagrundlag *(ikke længere blokeret — selvbetjent kanal fundet)*

- [x] **Selvbetjent billedadgang fundet**: kbharkiv.dk's kildeviser har et
      åbent API (`api.kbharkiv.dk/pages?unit_id=`, `/file/<id>`).
      Forskydning `page_number = counter - 1` verificeret to gange.
      Script: `scripts/kbharkiv_hent.py` (midlertidigt, 20-billeders grænse).
- [x] 16 rigtige andensider hentet til `output/proeve_opslag/`, tre af dem
      krydstjekket ord for ord mod facit
- [x] **Billedanmodningen er sendt til kollegaen** (bekræftet af lead
      2026-08-20). Afventer hans levering. **Bemærk**: anmodningen dækker
      de 38 dødsfaldspatienter, ikke <patientnavns> 10 sider — de er
      selvhentet via kbharkiv og ligger i øvemængden, så hullet er lukket.
- [x] **Hele øvemængden hentet selv** (2026-08-20, leads go): 118 sider fra
      15 bind via kbharkiv → `output/oeve_billeder/`. Prøvemængdens 50 sider
      hentes IKKE — de skal først røres ved den endelige bedømmelse.
- [x] **Sideforskydningen efterprøvet på ALLE 15 bind** (2026-08-20): var kun
      verificeret på 2. Ét billede pr. bind læst og sammenlignet med facit.
      `page_number = counter - 1` holder overalt; recto/verso-reglen ligeså.
      Ingen forskydning fundet — risikoen for at parre billede med forkert
      facittekst er dermed lukket for det materiale, vi har.
- [ ] **NYT ÅBENT PUNKT: facit rummer fejl.** Ved kontrollen blev der fundet
      én sikker fejllæsning i facit (`37554_001491`: facit skriver "for 2
      Dage siden", på siden står "for 3 Dage siden"). Det sætter et gulv
      under, hvor lav en fejlprocent vi kan måle. Ét fund fra tretten
      stikprøver — hyppigheden er ukendt. Afventer leads stillingtagen.
- [x] **Første tjek, model revideret**: billederne er IKKE symmetriske
      dobbeltopslag (undtagen forsideopslag) — hvert billede er asymmetrisk
      beskåret om ÉN målside med kun en smal strimmel af naboopslaget i én
      kant. Facit findes altid fuldt i billedet.
- [x] **Strimmel-mønster bekræftet**: løst af recto/verso-reglen (se stage
      04) — ikke en formodning længere.
- [x] `output/opslagsregister.csv`, `output/daekning.md`,
      `output/opslag_struktur.md` — alle skrevet 2026-08-18.
- [ ] **Gennemgang ved lead** — ingen formel godkendelse af selve stage 01
      endnu, kun de enkelte fund undervejs (indirekte dækket af, at
      stage 04 er godkendt på samme materiale)

## Stage 02 — Facit — **GODKENDT OG LÅST 2026-08-21**

- [x] Kortlæg alle klammeformer udtømmende (ikke kun de otte læste filer) —
      alle 39 filer, 8 typer fordelt på 194 skrivemåder, i
      `output/klammekonventioner.md`
- [x] RTF-læser med tests pr. konvention, hver set fejle —
      `src/andenside/facit.py`, 19 + 45 tests
- [x] Ren læsetekst — i TO udgaver efter beslutning 24: `rettet_*` hvor
      overstreget fjernes og erstatningen beholdes, og `alt_*` hvor alt
      hvad der står bliver stående. `alt_*` er den, der måles på.
      `[?]` bevares i begge.
- [x] Fladet udgave med orddeling samlet
- [x] Opdeling i øvemængde (26 patienter) og låst prøvemængde (13 patienter),
      pr. patient — hver tredje patient efter forsidens billed-id, ingen
      lodtrækning
- [x] **Tallene**: 39 filer → 208 sidemærker → 40 uden tekst → **168 sider med
      facit**, fordelt på 39 patienter (de 38 fra dødsfaldsmapperne plus
      <patientnavn>). Fire steder flaget til gennemsyn.
- [x] **leads rettelser til læsereglerne indarbejdet** (2026-08-20,
      beslutning 31-34): klammer med prikker/ellipse bliver `[?]` fordi de
      ikke kan måles på; klammer uden `?` er også læseforslag; uafsluttede
      klammer repareres ved første mellemrum; lægens egne `?` uden for
      klammer røres ikke (7 stk., fx `(Scarlatina?)`).
- [x] **Understregningen gemmes nu for sig** i stedet for at gå tabt: 409
      poster (253 hele linjer, 156 citater) med linjenummer i `alt_linjer`.
      Alle 156 citater rammer den linje, de faktisk står på.
- [x] **Stavefejlene efterprøvet**: test over alle 31 stavemåder,
      optællingen fandt — hver skal havne i den rigtige kategori. Set fejle.
- [x] **Kildefilerne kan ikke røres**: test tager tidsstempel og størrelse
      på alle 39 RTF'er, kører bygningen, kræver at intet har flyttet sig.

- [x] **Gennemgang ved lead — GODKENDT** (2026-08-21): "alt lyder godt".
- [x] **Gennemgang af begge sessioner, 2026-08-21**: 17 fund, alle
      efterprøvet og rettet. Ét lå i det leverede facit — en uafsluttet
      klamme slugte resten af siden, så `[added over line](Fibiger)` stod
      bogstaveligt i teksten på 273104_001643, og en overstregning efter
      stedet blev ikke fjernet fra den rettede udgave. Facit er bygget
      igen. Se CONTEXT.md 2026-08-21.
      Tolkningsreglerne, de to facit-udgaver og læseteksten er bekræftet.
      **Stage 02 er hermed låst** som grundlag for stage 03. Genåbnes uden
      tøven, hvis der senere dukker et problem op, der hører hjemme her.
- [x] **Lukket**: umarkeret orddeling over to linjer (8 tilfælde, 0,009 %
      af tegnene). Lead: kan endda være meningen, hvis skriveren ikke
      satte bindestreg — så er facit netop korrekt. Ingen ændring.
- [x] **Rettelse til tidligere tal**: der er facit for 168 sider, ikke 257.
      De 257 er antallet af SIDER hos de 38 dødsfaldspatienter, ikke
      antallet af
      transskriberede sider. Billedanmodningen er stadig rigtig.
- [x] **De 40 tomme sidemærker er AFKLARET** (2026-08-20, lead + stikprøve):
      siderne er beskrevet i journalen, de er bare ikke transskriberet af
      kollegaen. Ti af dem hentet og målt — alle har blæk som sider, vi ved
      har tekst; to set efter med øjnene. De holdes ude af facit. Mønstret:
      de fyrre ligger i kun 7 patienter, altid som en sammenhængende hale
      sidst i forløbet.
- [x] **leads svar på margentekst** (2026-08-20): margentekst og indskud
      TÆLLER MED i facit som tekst, modellen skal ramme. Ingen ændring.
- [x] **leads svar på prøvemængden**: mit valg — en tredjedel beholdes.
- [x] **Facit har nu TO udgaver pr. side** (2026-08-20, beslutning 24):
      `alt_*` = alt hvad der står, også det overstregede → **den der måles
      på**; `rettet_*` = overstreget fjernet, kun erstatningen → den
      historisk rigtige tekst til et færdigt datasæt. Modellen promptes
      IKKE til at genkende overstregninger.

## Stage 03 — Måleapparat — **GODKENDT OG LÅST 2026-08-23**

- [x] **Mål BÅDE fladet og pr. linje.** Linjeparringen er selve forankringen:
      hver facit-linje søges i modellens rå tekst uden hensyn til dens
      linjeskift, så målingen ikke skrider efter første afvigende brud
      (beslutning 40).
- [x] **Den ubeviste antagelse er nu et målt tal, ikke en formodning.**
      Rapporten opgør, hvor mange facit-linjer der ligger inden for én af
      modellens linjer, og hvor mange der får deres egen. Målingen afhænger
      ikke længere af svaret (beslutning 41). Svaret selv kommer ved stage
      05's første kørsel.
- [x] Overtaget `cer.py` fra StadsCER → `src/andenside/cer.py`. Fem varianter
      plus `arbejdstal` som den sjette. `compare()` er IKKE overtaget: den
      matcher på linje-id, og dem har vi ikke.
- [x] Orddeling hen over linjeskift samles — StadsCERs kendte mangel.
      Beslutningen tages på FACITS linjer og bruges på begge sider, så en
      model, der skrev ordet samlet, ikke straffes (beslutning 42). En test
      holder de to steder op mod hinanden på alle 168 sider.
- [x] Hallucinationskontrol uden krav om identisk linjeopdeling: tre
      uafhængige signaler — modeltekst uden modstykke, tekst skrevet dér hvor
      facit siger `[?]`, og fuldside-kontrollen.
- [x] **Behandlingen af `[?]`** (beslutning 38) er grundreglen; forankringen
      er lagt ovenpå med en defineret vej tilbage.
- [x] **ÉN funktion, ikke fire features**: `forankr()` i
      `src/andenside/maal.py`. Reglerne holder: stumper under 5 tegn bruges
      ikke, gab tælles kun med stumper fundet på begge sider, en uforankret
      linje falder tilbage til beslutning 38. **Nyt valg**: søgningen tåler
      læsefejl inde i stumpen (beslutning 39) — ellers ville hver forankret
      stump per definition have nul fejl, og måleapparatet ville bekræfte
      sig selv.
- [x] Ingen gennemsyns-app. Gabene skrives til en fil — `skriv_gab()`
      lægger dem alle i CSV (`output/gab_eksempel.csv` viser formatet), og
      rapporten viser de første femten. Kontrakten krævede en fil, ikke kun
      en tabel.
- [x] **Dækningen står ved hvert tal** i rapportformatet, sammen med
      forbeholdet om at facit selv rummer fejl.
- [x] Rapportformat, selvtest og gab-fil → `stages/03_maaleapparat/output/`.
      Rapporten forklarer sig selv: et afsnit "Sådan er der målt" i almindeligt
      sprog, før det første tal, plus en ordforklaring af tegnafstand, CER, WER
      og fladet tekst. Den skal kunne læses uden CONTEXT.md ved hånden.
      **Determinismen efterprøvet i fuld skala**: to kørsler på alle 118 sider
      gav samme rapport og samme gab-fil, tegn for tegn — og de stemmer med
      filerne i repoet.
      205 tests grønne; 10 bevidste mutationer af koden blev alle fanget.
      Søgefunktionen er desuden prøvet mod en rå gennemsøgning af alle
      udsnit på 240 tilfældige tilfælde.

**Målt i selvtesten** (118 øvesider, ingen modelkald):

- Forankringen redder **94,6 %** af de svære linjer. Dækning 97,6 % af
  tegnene i stedet for knap 88 %. **Øvre grænse** — her er "modellen" facit
  selv, så hver stump findes ordret. En rigtig model læser dårligere.
- **Skævheden er nu et tal**: bytter vi selv 5.087 bogstaver om, finder
  målingen 4.737 = 93,1 %. Målingen underrapporterer systematisk, fordi de
  linjer, den ikke kan forankre, er de hårdest ramte.
- **Knappen kan pynte, og det er dokumenteret**: strammes `MAKS_AFVIGELSE`
  fra 0,4 til 0,2, falder tegnfejlen fra 7,50 % til 7,13 % — pænere — mens
  dækningen falder og andelen af fundne fejl går fra 93,1 % til 86,4 %.
- **Nyt fund, efterprøvet linje for linje**: springer modellen en hel blok
  over, sker der TO ting. Nogle af de manglende linjer forankrer sig
  fejlagtigt i en linje, der ligner — og det falske træf flytter søgepunktet
  frem, så de EFTERFØLGENDE linjer kun finder en afskåret rest af sig selv,
  selvom modellen læste dem rigtigt. Anden halvdel var ikke med i den første
  forklaring. Målt pris: 181 tegn på 27 af 118 sider. Den vokser med, hvor
  meget modellen springer over. Se CONTEXT.md 2026-08-22 (senere).
- **Rettelse**: facit har 3.680 linjer, ikke 3.526 (tallet var fra FØR
  genbygningen 21. august). De 422 svære linjer holder; andelen bliver
  11,5 % i stedet for 12,0 %.

- [x] **Den strenge måling tilføjet** (beslutning 44, lead 2026-08-23):
      samme måling, men linjer med et `[?]` slet ikke med — heller ikke deres
      kendte stumper. Står lige efter hovedtallet med forskellen skrevet ud,
      fordi de reddede stumper netop ligger op ad de ulæselige steder, hvor
      alt er mest usikkert. Er den strenge højere end hovedtallet, har
      redningen pyntet, og så er det den strenge, der gælder.
- [x] **Gennemgang ved lead — GODKENDT** (2026-08-23): "alt andet virker ok".
      **Stage 03 er hermed låst.** Genåbnes uden tøven, hvis der viser sig et
      hul ved de første rigtige tal.
- [ ] **leads forbehold, prøves af i stage 05**: afsnit 7 og 8 ("de 10 værste
      sider", "de 10 tyndest målte sider") "virker lidt søgte, men vi kan
      prøve". Giver de ikke noget på det første rigtige modelsvar, ryger de ud.
- [x] **Løs ende lukket**: `pyproject.toml`s døde kommando-indgang
      (`andenside.cli:main` uden en `cli.py`) er fjernet efter leads valg.

## Stage 04 — Billedforberedelse

- [x] **Model revideret** (2026-08-18): opgaven er at skære en smal
      forurenende strimmel væk fra ÉN kant, ikke at dele midt over.
- [x] **Kant-reglen LØST** (2026-08-18, lead): recto/verso-paritet af
      `patient_page_counter` afgør entydigt hvilken kant der bærer strimlen
      — andenside=verso=indhold venstre, tredjeside=recto=indhold højre.
- [x] **Scope udvidet**: tredjeside (recto, 71.380 sider) er nu også med.
- [x] **Snitpunkt-detektion LØST, v2** (2026-08-18): første version (dal =
      lyseste punkt i vinduet) blev erklæret "8/8 perfekt" af mig, men
      leads eget gennemsyn fandt 4 reelle fejl (snit gik gennem naboens
      tekst). Rettet: ryggen viser sig som en KRAFTIG TOP i
      blækprofilen, ikke en dal — algoritmen går nu fra vores egen side
      og snitter ved ryggens nære kant. Alle 8 billeder gennemset igen
      efter rettelsen, inklusive de 4 tidligere fejlende. Se CONTEXT.md.
      `src/andenside/bogryg.py` + `kontaktark.py`, låst med opdateret
      regressionstest (`tests/test_bogryg_real_billeder.py`).
- [x] **Beslutning**: snitpræcision behøver ikke være perfekt — stage 05
      kan prompte modellen til at ignorere delvis nabotekst som backup.
- [x] **Buffer + terminologi rettet** (2026-08-18): 1% buffer væk fra egen
      tekst; "ryg" ikke "rygning" gennemgående.
- [x] **Gennemgang ved lead — GODKENDT** (2026-08-18): "de er perfekte nu".
      Alle 8 kontaktark bekræftet. **Stage 04 er hermed låst** som
      forudsætning for stage 05 på det nuværende pilotmateriale.
- [ ] **Kendt begrænsning, stadig åben**: kun afprøvet på 2 bind, samme
      måneder (maj-juni 1896) — bredere test nødvendig, når flere billeder
      er hentet. Kan afdække nye fejltyper (fx det fjerne bleed lead
      nævnte) på trods af godkendelsen ovenfor.
- [ ] Ingen usikkerheds-flagning implementeret endnu (intet fejlende
      eksempel at kalibrere en tærskel mod) — se `output/usikre.md`.
- [ ] Frasortér naboblade der rager usædvanligt langt ind (ikke testet)

## Stage 05 — Første transskription

*(splittet ud af tidligere "05 Metodeforsøg" 2026-08-18 — fin opdeling af
selve læse-implementeringen, samme princip som resten af planen)*

- [ ] Forsøg 1: kun billedforberedelsen varieres (`gemini-3.1-pro`, fast prompt)
- [ ] Bogholderi: rå svar + fuld opsætning gemmes pr. kørsel
- [ ] **Ingen fuld kørsel uden leads go**
- [ ] **Gennemgang ved lead** — er beskæringen god nok til at gå videre?

## Stage 06 — Prompt og model

- [ ] Én akse ad gangen: model, prompt, opløsning, linjedetektion (baselines)
- [ ] **Baseline-aksen** (udskrevet i stage 06's CONTEXT 2026-08-27):
  - [ ] Afklar først, om fuldside-linjedetektion overhovedet kan køre her —
        magresprots kopi kræver givne regioner og en sti-dependency, vi ikke har.
        Kan den ikke, noteres aksen som uafprøvet og lukkes.
  - [ ] Variant 1: baselines som oplysning i prompten (billig, ingen ny pipeline)
  - [ ] Variant 2: udklip pr. linje — kun hvis variant 1 peger den vej; flyttes
        i så fald ud i sin egen stage
  - [ ] Gevinstkrav: fuldside-kontrollen må ikke falde, uparrede linjer skal ned,
        og det skal holde på mere end én bog
- [ ] Transkribus gennem samme måleapparat som reel sammenligning (proces-punkt 1a)
- [ ] Test om prompt kan løse resterende beskæringsufuldkommenheder
- [ ] **Er de ulæselige steder ulæselige, eller bare dårligt fotograferet?**
      20-30 tætte udklip i højeste opløsning, forelagt lead. Kræver
      kollegaens originalscanninger — kildeviseren kan ikke give mere end
      de ~900-1.000 pixels pr. tekstside, vi allerede har (efterprøvet
      2026-08-21). Ingen kode, kun et forsøg.
- [ ] Undersøg fjern-opslags-bleed (se CONTEXT.md, leads fund om side 51 i side 101)
- [ ] **Gennemgang ved lead** — bedste kombination udpeges

## Stage 07 — Anden stemme

- [ ] `claude-opus-4-6` som anden stemme, uenighedsmarkering
- [ ] Mål fejlfangst pr. gennemsynsbyrde (jf. Humphries maj 2026)
- [ ] **Gennemgang ved lead**

## Stage 08 — Integration

- [ ] Leverance: prompt, skema, måletal
- [ ] Forslag til sideudvælgelse via `patient_page_counter` frem for `_fp`
- [ ] Afklar dashboard-status og om `textpage` allerede er afprøvet
- [ ] Efterprøvning: vores tal mod hans app på samme sider
