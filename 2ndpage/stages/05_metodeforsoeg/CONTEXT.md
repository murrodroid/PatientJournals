# Stage 05 — Metodeforsøg

## Formål

Finde ud af hvilken fremgangsmåde der læser andensiderne bedst, ved at variere
én ting ad gangen og måle mod facit.

## Inputs

| Type | Sti |
|---|---|
| Testregler | `../../_config/tdd.md` |
| Måleapparat | `../03_maaleapparat/output/rapportformat.md` |
| Beskæringer | `../04_billedforberedelse/output/snit.csv` |
| Facit og opdeling | `../02_facit/output/facit.jsonl`, `../02_facit/output/opdeling.csv` |
| Kollegaens kontrakt | `../../references/app_interface_upstream.md` |
| Anbefalinger | `../00_forundersoegelse/output/forundersoegelse.md` |

## Process

1. **Første forsøg varierer kun billedforberedelsen.** Model og prompt holdes
   fast: `gemini-3.1-pro` og kollegaens eksisterende `textpage`-prompt, let
   tilpasset til dansk hospitalsmateriale. Sammenlign helt opslag mod delte
   halvsider.
2. Alle kørsler sker på **øvemængden**. Den låste prøvemængde røres først,
   når en vinder skal bedømmes endeligt.
3. Gem altid det rå modelsvar sammen med model, promptversion, indstillinger
   og dato, så en kørsel kan genfindes og genkøres.
4. Efterfølgende akser, én ad gangen og kun hvis den foregående er afklaret:
   model, promptformulering, opløsning, og forbehandling med linjedetektion
   (baselines) enten som udklip eller som hjælp til at vide, hvilke linjer der
   findes på siden.
5. **Anden stemme**: lad en model fra en anden familie, `claude-opus-4-6`,
   transskribere igen, og marker de steder hvor de to er uenige. Mål hvor stor
   en del af de faktiske fejl, uenigheden fanger, og hvor stor en del af
   teksten der skal ses efter. Dette er et lag oven på teksten, ikke en del af
   den.
6. Ingen fuld kørsel på hele materialet uden dit udtrykkelige go.

## Outputs

| Fil | Beskrivelse |
|---|---|
| `output/koersler/` | Rå modelsvar pr. kørsel med fuld angivelse af opsætning |
| `output/resultater.csv` | Én række pr. kørsel: opsætning og alle fem måletal |
| `output/sammenligning.md` | Hvad hver akse viste, og hvad der blev valgt |

## Test Contract

Modelsvar kan ikke testes fast. Det der skal testes, er bogholderiet: at en
kørsel altid gemmes med den opsætning, den blev kørt med, og at to kørsler
med samme opsætning kan skelnes fra hinanden på dato. Desuden en test, der
sikrer, at kode som læser den låste prøvemængde fejler, medmindre den
udtrykkeligt får lov — så den ikke bruges ved et uheld.

## Handoff

Næste stage er `06_integration`. Reviewed betyder, at du har set
sammenligningen og udpeget den metode, der skal afleveres.
