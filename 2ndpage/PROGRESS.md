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
- [x] ICM-skelet oprettet: `AGENTS.md`, `_config/tdd.md`, syv stages, strukturtest

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
      forurenende strimmel væk fra ÉN kant, ikke at dele midt over — se
      CONTEXT.md. Tre naive kolonneprofil-forsøg (top/løb/dal) landede alle
      forkert i selve teksten; scriptet `scripts/bogryg_profil.py` er
      bevaret som udgangspunkt, ikke en løsning.
- [ ] Afklar strimlens side pr. billede (recto/verso-hypotese, ubekræftet)
- [ ] Design detektion ud fra den asymmetriske model med flere facit-labels
- [ ] Frasortér naboblade der rager ind; find blanke halvsider
- [ ] Afklar forsideopslags afvigende, bredere format
- [ ] Kontaktark med indtegnede snit
- [ ] Usikre snit mærkes og skæres ikke
- [ ] **Gennemgang ved lead** — forberedelsen låses herefter

## Stage 05 — Metodeforsøg

- [ ] Forsøg 1: kun billedforberedelsen varieres (`gemini-3.1-pro`, fast prompt)
- [ ] Bogholderi: rå svar + fuld opsætning gemmes pr. kørsel
- [ ] Senere akser, én ad gangen: model, prompt, opløsning, linjedetektion
- [ ] Anden stemme (`claude-opus-4-6`) og uenighedsmarkering
- [ ] **Ingen fuld kørsel uden leads go**
- [ ] **Gennemgang ved lead** — vinderen udpeges

## Stage 06 — Integration

- [ ] Leverance: prompt, skema, måletal
- [ ] Forslag til sideudvælgelse via `patient_page_counter` frem for `_fp`
- [ ] Afklar dashboard-status og om `textpage` allerede er afprøvet
- [ ] Efterprøvning: vores tal mod hans app på samme sider
