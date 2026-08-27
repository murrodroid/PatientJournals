# Billedmateriale + Blegdam_master_list.csv + søsterprojekter

Kortlagt 2026-08-18.

## Blegdam_master_list.csv — nøglen til andensiderne

Sti: `<kilderod>\PID-scapes and Blegdam Patient journals\Patient journals\Meta data\Blegdam_master_list.csv`

570.519 rækker (én pr. opslag/billede), 354 bind. Kolonner: `image_name`
(fx `272944_000001`), `folder_name` (bind-id), `patient_status`,
`page_counter`, `page_type`, `month`, `year`, `patient_page_counter`,
`group_id`, `dataset`, `transcribed`.

| Optælling | Værdi |
|---|---|
| page_type = journal page | 492.367 |
| page_type = front page | 71.378 |
| page_type = preamble | 6.774 |
| patient_page_counter = 0 (= forsider) | 71.378 |
| **patient_page_counter = 1** (= andensider) | **71.391** |
| patient_page_counter = 2 (= tredjesider) | 71.380 |
| transcribed = TRUE | 24.643 |

- **Andensiderne kan filtreres direkte frem**: `patient_page_counter == 1`.
  Tælleren begynder på 0 ved forsiden — verificeret direkte i rækkerne omkring
  en kendt forside, hvor `273098_001471` er front page med tælleren 0 og
  `_001472` er journal page med tælleren 1. (Tidligere note sagde fejlagtigt
  `== 2`; det er tredjesiderne.)
- **Gruppe-id løber over bindgrænser.** `group_id` skifter kun ved en forside,
  så det sidste patientforløb i et bind deler gruppe med næste binds
  indledningssider. Filtrér derfor altid på både `group_id`, `folder_name` og
  `page_type`.
- `transcribed=TRUE` stammer alle fra `dataset`-værdierne
  `patientjournals_v2_1889-97_gemini_almost_complete.jsonl` (22.671),
  `..._final_pages.jsonl` (1.078), `..._remaining_deaths.jsonl` (894) —
  dvs. forsiderne er Gemini-transskriberet for årgangene 1889-97.
- Leads teori bekræftet: masterlisten peger ind i en billedmappe med præcis
  samme struktur (bind-mappe → `image_name`.ext), som kan skaffes.

## C:\Work\Alle patientjournaler_komprimeret (lokale billeder)

23.820 JPG'er (~9,5 GB, typisk 0,3-0,6 MB/stk, JPEG-kvalitet 88 — genereret
2026-04-30 af `C:\Work\compress_journals\compress_journals.py` fra de
ukomprimerede PNG-originaler "Alle patientjournaler").

- 193 mapper: `intake_dead_<bind-id>` (79 stk.) + `intake_front_<bind-id>`
  (114 stk.).
- Filnavne = masterlistens `image_name`: `<bind-id>_<sekvens>.jpg`
  (fx `273012_000769.jpg`).
- **Verificeret 2026-08-18: der er NUL andensider lokalt.** Alle 38
  ground-truth-forsider findes blandt de 23.783 lokale JPG'er; ingen af de 38
  tilhørende andensider gør. Mappen rummer altså kun forsideudvalget, og
  sekvensspringene på 4-8 er netop hullerne, hvor de mellemliggende opslag
  mangler. Kildemappen `C:\Work\Alle patientjournaler`, som
  komprimeringsscriptet læste fra, findes ikke længere på maskinen.
- **Ét billede = ét opslag med to sider.** Verificeret visuelt: venstre halvdel
  er forrige blads bagside, højre halvdel bærer teksten. På forsideopslag er
  venstre side blank. Fortsættelsesopslag rummer efter alt at dømme tekst på
  begge halvdele, hvilket skal bekræftes, så snart de første billeder er hentet.
- **Opløsning: ~1.700-2.000 × 2.200-2.300 pixels for hele opslaget**, altså kun
  omkring 900-1.000 pixels bredde pr. tekstside. Det er lavt for håndskrift, og
  Humphries fandt netop billedopløsning vigtigere end modellens tænketid — så
  det er en reel risiko og et åbent spørgsmål, om der findes skarpere
  originaler hos kbharkiv.

## Hvor billederne skal hentes fra

Kollegaens pipeline (`upstream/main`, `config/settings.py`) peger på en Google
Cloud-spand `data-blegdamsjournaler` med præfikset `pages`, og på en ekstern
harddisk `/Volumes/Expansion/patientjournaler_1889-1897_jpg` på hans egen
maskine. `date_mapping.csv` viser, at originalerne stammer fra kbharkiv.dk med
et permalink pr. arkivenhed. Første leverance sker som filoverførsel efter
`billedanmodning/billedanmodning_2026-08-18.md`.

Bemærk: masterlisten dækker **1880-1910**, bredere end de 1889-97, som
forsidetransskriptionen omfatter.

## Søsterprojekter i C:\Work (hurtig skimning)

| Mappe | Hvad | Status |
|---|---|---|
| `PatientJournals_googledocai` | Gemini-baseret transskriptionspipeline med validering/accuracy-rapport (navnet til trods: IKKE Google Doc AI). uv-projekt, runs/-mappe. | Aktivt eksperiment, sidst rørt 2026-03-11 |
| `PatientJournals_Kraken` | Næsten identisk kodebase (samme README, samme src/-moduler: crops.py, extract.py, ocr.py, models.py); Python 3.13-variant | Videreførelse, sidst rørt 2026-03-12 |
| `patientjournaler_v2_codealong` | Analyse-øvelse (madsp, RUC-praktik): Gemini-transskriptioner → HISCO-klassifikation | Afsluttet/forladt, maj 2026 |
| `compress_journals` | PNG→JPEG-komprimeringsværktøjet der lavede den komprimerede mappe | Engangsværktøj, kørt 2026-04-30 |
