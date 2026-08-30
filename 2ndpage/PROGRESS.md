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
- [2026-08-27 17:10](diary/2026-08-27.md) — Alle 118 beskåret, 11x hurtigere; udragende blade udpeget som næste arbejde
- [2026-08-28](diary/2026-08-28.md) — Yderkanten: facit for alle 118, detektion bygget, snittet vendt udad efter leads indsigelse
- [2026-08-29](diary/2026-08-29.md) — Måleapparat efterprøvet mod leads domme; svag-bekræftelsen var fejlen og er fjernet; forsøg A valgt
- [2026-08-30](diary/2026-08-30.md) — Leveringen hentet (307 PNG); falssnittet gik galt på 9 sider og er rettet; buffer sat til 0,5 %
- [2026-08-30 senere](diary/2026-08-30.md) — Lead godkendte de 27 kontaktark; stage 04 låst; piloten begynder på 5-10 sider

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
- [x] **Billedanmodningen er LEVERET** (2026-08-30, ekstern harddisk):
      307 PNG i `output/levering_2026-08/`. 223 hentet ind (173 øve + 50
      ekstra uden facit); prøvemængdens 84 bevidst ladt på harddisken.
      Samme opløsning som vores webp, blot ukomprimeret — målt forskel
      PSNR 41-42 dB. Script: `scripts/hent_levering.py`. Prøvemængdens 84
      hentet efter leads ønske (2026-08-30) i `proeve_LAAST/`, adskilt så
      et glob ikke kan samle dem op; de må stadig ikke måles på. **Bemærk**: anmodningen dækker
      de 38 dødsfaldspatienter, ikke de 10 sider under `273104_001636` — de er
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
      `273104_001636`). Fire steder flaget til gennemsyn.
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

## Stage 04 — Billedforberedelse — **GODKENDT OG LÅST 2026-08-30**

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
- [x] **Begge snit kørt på hele leveringen** (2026-08-30, leads go):
      307 sider gennem falssnit + yderkantssnit →
      `output/levering_beskaaret/`. Første gang yderkant-snittet faktisk
      SKRIVES; hidtil var det kun målt. Fjernet i alt: median 28-32 %.
      **Kun 4 af 307 mærket usikre** (`273111_001380`, `273111_001381`,
      `37554_001496`, `273107_001884`). Script: `scripts/beskaer_levering.py`.
- [x] **Bredden afprøvet uden for facit-perioden**: de 50 andensider fra
      1889-1897 gav **nul usikre**. Første tegn på at forberedelsen holder
      bredere — men kun et fravær af alarm, ikke en måling.
- [x] **Falssnittet rettet** (2026-08-30, efter leads gennemgang): båndene
      fandt hver især noget, men lå ikke på samme linje, og interpolationen
      trak snittet med. Målt afviger falsen 5 px fra en ret linje i median
      (90-percentil 11 px); leads gale sider afveg 245-412 px.
      `skraa.fjern_udskridende` kaster nu de uenige bånd, og `sikker`
      kræver enighed. Alle gale sider har nu afvigelse 0.
      **4 af de 9 gale sider lå i øvemængden** og havde været skåret
      forkert hele tiden, uden at nogen måling sagde fra.
- [x] **Kontaktarkene viser nu det bortskårne** (fals rød, yderkant blå) —
      man kan ikke bedømme et snit på resultatet alene.
- [x] **Gennemgang ved lead — GODKENDT** (2026-08-30) af
      `output/levering_beskaaret/*/kontaktark/` (27 ark) under den endelige
      kode. **Stage 04 er hermed låst.** Genåbnes uden tøven, hvis stage 05's
      første modelsvar peger på, at snittene spolerer noget.
- [x] **Falssnittets buffer sat til 0,5 %** (2026-08-30, leads visuelle
      valg mellem 2,0/1,0/0,5 % i `output/buffer_sammenligning/`). Alle 307
      skåret om; alle seks af leads tidligere domme holder.
- [ ] 21 sider fik kastet bånd (`output/fals_kvalitet.csv`) — de ser nu
      rigtige ud, men er ikke set efter én for én.
- [ ] `273111_001380` og `_001381` mærkes stadig usikre på falsen.
- [ ] Ingen usikkerheds-flagning implementeret endnu (intet fejlende
      eksempel at kalibrere en tærskel mod) — se `output/usikre.md`.
- [x] **Frasortér naboblade der rager ud** (2026-08-28) —
      `src/andenside/yderkant.py`. Facit for alle 118 yderkanter lavet
      visuelt (`output/yderkant_facit.csv`): **7 sider har fremmed tekst**
      langs yderkanten, 110 er rene, 1 usikker. Detektionen måler papirets
      grundlyshed pr. kolonne bånd for bånd og vælger den inderste rette
      kant, mindst 6 bånd kan enes om. 19 tests, alle set fejle mod
      muteret kode.
- [x] **Snittet vendt udad** (2026-08-28, leads indsigelse): kanten meldes
      nu i faldets bund og bufferen er hævet til 1,2 % — snittet flyttede
      22 px udad i median, så ordender ikke klippes.
- [x] **Alle 118 snit gennemset**: 116 sidder på sidens kant; alle 7
      problemsider får den fremmede strimmel uden for snittet.
      `273108_001555` skærer gennem skriften (mærket usikker af koden),
      `273103_001463` er omtvistet (ikke mærket).
- [x] **`sikker`-kolonnen siger nu faktisk nej** — 2 af 118 mærkes, mod
      falsbeskæringens 0 af 118.
- [x] **Leads valg truffet (2026-08-29): forsøg A** — skær alle 118 ved
      sidens egen kant.
- [x] **Måleapparat bygget og efterprøvet mod leads egne domme** (2026-08-29):
      sømdybde skiller hans to forkerte snit (3,0 og 5,0) fra hans fire
      rigtige (12-25). Tre tidligere måleforsøg blev kasseret, fordi de
      IKKE bestod den prøve — se `output/yderkant_eval.md`.
- [x] **Fejlen fundet: svag-bekræftelsen**, tilføjet for at redde én side,
      ødelagde to andre. Fjernet igen med sine to konstanter. Begge leads
      forkerte sider er dermed rettet.
- [x] **Sømkrav indbygget som værn** — ændrer målt intet på øvemængden,
      men får siden til at afstå frem for at gætte på ukendt materiale.
- [x] 305 tests grønne; 13 af 14 mutationer fanget (den sidste er
      sømkravet, som beviseligt intet ændrer på det materiale, vi har).
- [x] **Gennemgang ved lead — GODKENDT** (2026-08-30): dækket af
      gennemgangen af de 27 kontaktark, som viser begge snit på samme ark
      (fals rød, yderkant blå).
- [x] Beskårne billeder skrevet for alle 307 sider (2026-08-30, leads go).

## Stage 05 — Første transskription

*(splittet ud af tidligere "05 Metodeforsøg" 2026-08-18 — fin opdeling af
selve læse-implementeringen, samme princip som resten af planen)*

- [ ] **Første skridt: 5-10 sider** (leads valg 2026-08-30, beslutning 52) —
      piloten er dér, prompten formes, ikke en måling af beskæringen. Kilde:
      de ukomprimerede PNG i
      `stages/04_billedforberedelse/output/levering_beskaaret/oeve/beskaarne/`,
      så komprimering ikke er en åben mistanke ved det første tal.
- [ ] Forsøg 1: kun billedforberedelsen varieres (`gemini-3.1-pro`, fast prompt)
- [ ] Bogholderi: rå svar + fuld opsætning gemmes pr. kørsel
- [ ] **Ingen fuld kørsel uden leads go**
- [ ] **Gennemgang ved lead** — er beskæringen god nok til at gå videre?

**Blokeret på to ting, begge leads:**

- [ ] API-nøgle i `C:\Workndpage_keys.json` (feltnavn skal indeholde
      `gemini`, `genai` eller `google`). Indholdet læses aldrig; tjek med
      `hent_noegle()`.
- [ ] Leads go til de første modelkald. **Der er endnu ikke kørt ét eneste
      modelkald i projektet.**

**Åbent, som ikke lukkede med stage 04:** webp mod PNG i fuld skala. Ved
5-10 sider er filstørrelsen ligegyldig; ved en fuld kørsel er de 173 PNG
37 gange større end de 118 webp. Kan afgøres med en måling på de samme
sider i begge formater, når der først findes rigtige modelsvar at måle på.

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
