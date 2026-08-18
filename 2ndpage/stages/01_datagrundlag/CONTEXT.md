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

1. Aflever billedanmodningen til kollegaen; modtag de 307 billeder.
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

## Test Contract

Registerbygningen er deterministisk og skal testes: en række i registret må
aldrig pege på en billedfil, der ikke findes, og aldrig knytte en facit-fil
til et opslag fra et andet patientforløb. Testen skal dække tilfældet, hvor et
patientforløb løber over en bindgrænse, da gruppe-id i masterlisten fortsætter
ind i næste binds indledningssider.

## Handoff

Næste stage er `02_facit`. Reviewed betyder, at du har set svaret på
opslagsspørgsmålet og accepteret dækningsopgørelsen.
