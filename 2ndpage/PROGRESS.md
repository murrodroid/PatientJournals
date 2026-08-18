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

## Stage 01 — Datagrundlag *(blokeret: afventer billeder)*

- [ ] Aflever billedanmodningen til kollegaen
- [ ] Modtag de 307 billeder
- [ ] **Første tjek**: rummer et fortsættelsesopslag tekst på begge halvsider,
      og dækker facits `[page]`-blok hele opslaget?
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

- [ ] Kolonnevis blækprofil; bogryg som top med fast vandret position
- [ ] Deling i venstre og højre side; bekræft læserækkefølgen empirisk
- [ ] Frasortér naboblade der rager ind; find blanke halvsider
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
