# Stage 01 — Datagrundlag

## Formål

Skaffe billederne og vide præcis hvilke sider vi har, hvilke der har facit, og
hvordan de hænger sammen med masterlisten. Uden dette kan intet måles.

## Inputs

| Type | Sti |
|---|---|
| Testregler | `../../_config/tdd.md` |
| Masterliste | `...\Patient journals\Meta data\Blegdam_master_list.csv` |
| Facit-mappe | `...\Patient journals\Manual transcriptions` |
| Lokale billeder (kun forsider) | `C:\Work\Alle patientjournaler_komprimeret` |
| Billedanmodning | `../../billedanmodning/billedanmodning_2026-08-18.md` |

## Process

1. **To kanaler, og den selvbetjente er den, der har leveret.**
   a. Billedanmodningen til kollegaen (307 billeder) er sendt 2026-08-20
      og afventer levering. Den er den rene, langsigtede kanal.
   b. `scripts/kbharkiv_hent.py` henter selv fra kbharkiv.dk's åbne API.
      Forskydningen er `page_number = counter - 1`, efterprøvet på ALLE
      15 bind 2026-08-21. Herfra kommer de 118 øvebilleder, vi arbejder
      på. **Prøvemængdens 50 sider hentes bevidst ikke** — de røres først
      ved den endelige bedømmelse.
      Kildeviseren kan ikke levere højere opløsning: API'et har ingen
      størrelsesparameter, og vi får ~900-1.000 pixels pr. tekstside.
2. **Første tjek når billederne lander**: se et fortsættelsesopslag med egne
   øjne og afgør, om der står tekst på begge halvsider, og om facits
   `[page]`-blok dækker hele opslaget eller kun den ene halvdel. Alt senere
   arbejde afhænger af svaret.
3. Byg et opslagsregister, der binder billedfil, masterliste-række og
   facit-fil sammen på tværs af de tre kilder.
4. Optæl og rapportér dækning: hvor mange opslag med facit har vi faktisk
   billeder til, og hvor er hullerne.
5. Notér billedernes opløsning pr. fil, så vi kan se, om opløsning er en
   begrænsning, og om der findes skarpere originaler at bede om.

## Outputs

| Fil | Beskrivelse |
|---|---|
| `output/opslagsregister.csv` | Én række pr. opslag: billedfil, masterliste-felter, facit-fil, opløsning |
| `output/daekning.md` | Optælling af hvad vi har og mangler, med huller navngivet |
| `output/opslag_struktur.md` | Svaret på om et opslag rummer to tekstsider, og hvordan facit dækker det |
| `output/proeve_opslag/` | De 8 første selvhentede pilotbilleder (webp), stage 04 er verificeret på dem |
| `output/oeve_billeder/` | **Hele øvemængden: 118 sider fra 15 bind**, selvhentet 2026-08-20 |
| `output/oevemaengde_billeder.txt` | Listen over de 118 billed-id'er, som hente-scriptet fik |
| `output/kbharkiv_kalibrering/` | Materiale fra da forskydningen mellem kildeviserens sidetal og masterlistens billed-id blev fastlagt |

Billedfilerne selv er ikke i git (se `.gitignore`) — de kan hentes igen med
`scripts/kbharkiv_hent.py` ud fra `oevemaengde_billeder.txt`.

## Test Contract

Registerbygningen er deterministisk og skal testes: en række i registret må
aldrig pege på en billedfil, der ikke findes, og aldrig knytte en facit-fil
til et opslag fra et andet patientforløb. Testen skal dække tilfældet, hvor et
patientforløb løber over en bindgrænse, da gruppe-id i masterlisten fortsætter
ind i næste binds indledningssider.

## Handoff

Næste stage er `02_facit`. Reviewed betyder, at du har set svaret på
opslagsspørgsmålet og accepteret dækningsopgørelsen.
