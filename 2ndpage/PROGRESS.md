# PROGRESS.md — levende tjekliste for 2ndpage

Status her vinder på "hvad er gjort"; `CONTEXT.md` vinder på "hvorfor".
Videre til næste stage kun efter menneskelig gennemgang.

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

- [ ] Afgræns spørgsmålene undersøgelsen skal besvare
- [ ] Kør agenter pr. spørgsmål
- [ ] `output/forundersoegelse.md` med handlingsanvisende anbefalinger
- [ ] `output/aabne_spoergsmaal.md`
- [ ] **Gennemgang ved lead**

## Stage 01 — Datagrundlag *(ikke længere blokeret — selvbetjent kanal fundet)*

- [x] **Selvbetjent billedadgang fundet**: kbharkiv.dk's kildeviser har et
      åbent API (`api.kbharkiv.dk/pages?unit_id=`, `/file/<id>`).
      Forskydning `page_number = counter - 1` verificeret to gange.
      Script: `scripts/kbharkiv_hent.py` (midlertidigt, 20-billeders grænse).
- [x] 16 rigtige andensider hentet til `output/proeve_opslag/`, tre af dem
      krydstjekket ord for ord mod facit
- [ ] Aflever stadig billedanmodningen til kollegaen (ren, langsigtet kanal)
- [x] **Første tjek, model revideret**: billederne er IKKE symmetriske
      dobbeltopslag (undtagen forsideopslag) — hvert billede er asymmetrisk
      beskåret om ÉN målside med kun en smal strimmel af naboopslaget i én
      kant. Facit findes altid fuldt i billedet. Strimlens side (v/h)
      varierer, muligvis efter recto/verso-paritet — IKKE bekræftet.
- [ ] **Kræver leads øjne**: bekræft modellen og strimmel-mønsteret på
      `proeve_opslag/273098_001496/1508` + `273099_001361/62/63.webp`
- [ ] `output/opslagsregister.csv` — billede, masterliste, facit, opløsning
- [ ] `output/daekning.md` og `output/opslag_struktur.md`
- [ ] **Gennemgang ved lead**

## Stage 02 — Facit

- [ ] Kortlæg alle klammeformer udtømmende (ikke kun de otte læste filer)
- [ ] RTF-læser med tests pr. konvention, hver set fejle
- [ ] Ren læsetekst: overstreget fjernes, erstatning beholdes, `[?]` bevares
- [ ] Fladet udgave med orddeling samlet
- [ ] Opdeling i øvemængde og låst prøvemængde, pr. patient
- [ ] **Gennemgang ved lead** (historikerens bekræftelse af læseteksten)

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
- [ ] **Kendt begrænsning**: kun afprøvet på 2 bind, samme måneder (maj-juni
      1896) — bredere test nødvendig, når flere billeder er hentet.
- [ ] Ingen usikkerheds-flagning implementeret endnu (intet fejlende
      eksempel at kalibrere en tærskel mod) — se `output/usikre.md`.
- [ ] Frasortér naboblade der rager usædvanligt langt ind (ikke testet)
- [ ] **Gennemgang ved lead** — se kontaktarkene i
      `output/kontaktark/`, bekræft eller korrigér, før forberedelsen
      låses som forudsætning for stage 05

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
