# Stage 08 — Integration

## Formål

Aflevere den valgte metode til kollegaens app i `murrodroid/PatientJournals`,
så andensider kan køres derfra på samme måde som forsiderne.

## Inputs

| Type | Sti |
|---|---|
| Testregler | `../../_config/tdd.md` |
| Valgt metode | `../06_prompt_og_model/output/sammenligning.md` |
| Uenighedslag | `../07_anden_stemme/output/traeffik.md` |
| Kollegaens kontrakt | `../../references/app_interface_upstream.md` |
| Masterliste-viden | `../../references/billeder_og_masterliste.md` |

## Process

1. Aflever tre ting: **prompten**, **skemaet** og **beviset** — måletallene
   mod facit, så han kan se, hvad metoden kan. Ikke kørselskode; hans pakke
   har allerede klienter og batch-spor.
2. Skemaet er efter al sandsynlighed hans eksisterende `TextPage`/`PageLine`,
   eventuelt som en ny version oprettet i hans egen skema-redigering.
   Prompten er en navngiven post i hans `Config.prompts`.
3. **Det egentlige hul er sideudvælgelsen.** Hans pipeline skelner kun
   sidetyper på et `_fp`-mærke i filnavne. Andensider udpeges rettelig af
   masterlistens `patient_page_counter`. Foreslå koblingen, og aftal med ham
   om den hører hjemme hos ham eller hos os.
4. Afklar samtidig de to uafklarede punkter: hvor hans dashboard-arbejde står,
   og om hans egen `textpage`-prompt allerede er afprøvet på andensider — det
   ville spare os for at gentage arbejdet.
5. Kør en lille afprøvning gennem hans app og sammenlign med vores egne tal på
   de samme sider. Er de forskellige, er noget koblet forkert.

## Outputs

| Fil | Beskrivelse |
|---|---|
| `output/leverance.md` | Prompt, skema og måletal samlet i afleveringsklar form |
| `output/sideudvaelgelse.md` | Forslag til hvordan andensider udpeges i hans pipeline |
| `output/efterproevning.md` | Vores tal mod hans apps tal på de samme sider |

## Test Contract

Afprøvningen gennem hans app skal give samme tekst som vores egen kørsel på de
samme billeder med samme model og prompt. Afviger de, skal årsagen findes,
før leverancen regnes for færdig — en forskel betyder, at billedbehandling
eller promptversion ikke er den samme de to steder.

## Handoff

Sidste stage. Reviewed betyder, at kollegaen har taget metoden i brug, og at
efterprøvningen stemmer.
