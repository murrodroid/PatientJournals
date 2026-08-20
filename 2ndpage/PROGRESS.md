# PROGRESS.md — levende tjekliste for 2ndpage

Status her vinder på "hvad er gjort"; `CONTEXT.md` vinder på "hvorfor".
Videre til næste stage kun efter menneskelig gennemgang.

## Dagbog

- [2026-08-20 12:00](diary/2026-08-20.md) — Projektet 2ndpage bygget fra bunden: kortlægning, ICM-skelet, stage 00/01/04 gennemført
- [2026-08-20 16:20](diary/2026-08-20.md) — Stage 02 bygget: facit-læser, klammekortlægning, øve/prøve-opdeling

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
      2026-08-20). Afventer hans levering.
- [x] **Hele øvemængden hentet selv** (2026-08-20, leads go): 118 sider fra
      15 bind via kbharkiv → `output/oeve_billeder/`. Prøvemængdens 50 sider
      hentes IKKE — de skal først røres ved den endelige bedømmelse.
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

## Stage 02 — Facit *(bygget 2026-08-20, afventer gennemgang)*

- [x] Kortlæg alle klammeformer udtømmende (ikke kun de otte læste filer) —
      alle 39 filer, 9 typer fordelt på 194 skrivemåder, i
      `output/klammekonventioner.md`
- [x] RTF-læser med tests pr. konvention, hver set fejle —
      `src/andenside/facit.py`, 19 + 45 tests
- [x] Ren læsetekst: overstreget fjernes, erstatning beholdes, `[?]` bevares
- [x] Fladet udgave med orddeling samlet
- [x] Opdeling i øvemængde (26 patienter) og låst prøvemængde (13 patienter),
      pr. patient — hver tredje patient efter forsidens billed-id, ingen
      lodtrækning
- [x] **Tallene**: 39 filer → 208 sidemærker → 40 uden tekst → **168 sider med
      facit**, fordelt på 39 patienter. Seks steder flaget til gennemsyn.
- [ ] **Gennemgang ved lead** (historikerens bekræftelse af læseteksten) —
      læs `output/klammekonventioner.md`'s tolkningsregler og stikprøv
      `output/facit.jsonl` mod et par sider, du kender
- [ ] **Åben**: transskribenten deler nogle gange ord over to linjer uden
      bindestreg ("Inspira" / "tion"). Kan ikke skilles fra to virkelige ord
      med en regel; står som to ord i den fladede udgave.
- [x] **Rettelse til tidligere tal**: der er facit for 168 sider, ikke 257.
      De 257 er antallet af SIDER hos de 38 patienter, ikke antallet af
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

## Stage 03 — Måleapparat

- [ ] Overtag `cer.py` fra StadsCER med de fem varianter
- [ ] Byg samling af orddeling hen over linjeskift (StadsCERs kendte mangel)
- [ ] Byg hallucinationskontrol uden krav om identisk linjeopdeling
- [ ] Fastlæg behandlingen af `[?]`
- [ ] Rapportformat + selvtest mod facit og forvanskede udgaver
- [ ] **Gennemgang ved lead**

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
- [ ] Test om prompt kan løse resterende beskæringsufuldkommenheder
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
