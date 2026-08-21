# Stage 05 — Første transskription

## Formål

Lukke sløjfen på stage 04 med rigtige tal: køre den første egentlige
transskription på et lille udsnit, med model og prompt holdt fast, og kun
variere billedforberedelsen. Det svarer den akse, der er unik for netop
vores materiale, før der bruges tid på model- og promptvalg, som andre
allerede har målt på (Humphries m.fl.).

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

1. Model og prompt holdes fast: `gemini-3.1-pro` og kollegaens eksisterende
   `textpage`-prompt, let tilpasset til dansk hospitalsmateriale.
2. Sammenlign hele opslag mod stage 04's beskårne sider mod facit.
3. Alle kørsler sker på **øvemængden**. Den låste prøvemængde røres først,
   når en vinder skal bedømmes endeligt, langt senere.
4. Gem altid det rå modelsvar sammen med model, promptversion, indstillinger
   og dato, så en kørsel kan genfindes og genkøres.
5. Ingen fuld kørsel på hele materialet uden dit udtrykkelige go.

### Låst efter leads svar 2026-08-20/21

- **Prompten beder modellen om at læse hvad der står — punktum.** Den bliver
  IKKE bedt om at genkende overstreget tekst eller om at afgøre, hvad der
  skulle stå i stedet (beslutning 24). Lead har dårlige erfaringer med det:
  modellen forveksler understregninger med overstregninger, og der er 404
  understregninger i materialet mod 33 overstregninger. Er noget streget ud,
  prøver vi bare at læse det, og vi læser også det, der står efter.
- **Prompten beder heller ikke modellen om at markere ulæselige steder.**
  Måleapparatet håndterer dem (stage 03, punkt 6-7).
- **Kør på øvemængden.** 118 sider fra 15 bind ligger i
  `../01_datagrundlag/output/oeve_billeder/`. Prøvemængdens sider er bevidst
  ikke hentet.
- **Kig efter, om modellen laver sine egne linjeskift eller følger sidens.**
  Det er en ubevist antagelse i hele projektet (beslutning 35), og denne
  kørsel svarer på den gratis. Skriv svaret ned som et måleresultat.
- **Kig efter, hvad modellen skriver på de 422 svære linjer** — dem hvor
  transskribenten gav op. Det er projektets bedste prøve på, om modellen
  digter, hvor siden er ulæselig.

## Outputs

| Fil | Beskrivelse |
|---|---|
| `output/koersler/` | Rå modelsvar pr. kørsel med fuld angivelse af opsætning |
| `output/resultater.csv` | Én række pr. kørsel: opsætning og alle fem måletal |
| `output/foerste_tal.md` | Hvad det første forsøg viste: virker beskæringen godt nok til at gå videre? |

## Test Contract

Modelsvar kan ikke testes fast. Det der skal testes, er bogholderiet: at en
kørsel altid gemmes med den opsætning, den blev kørt med, og at to kørsler
med samme opsætning kan skelnes fra hinanden på dato. Desuden en test, der
sikrer, at kode som læser den låste prøvemængde fejler, medmindre den
udtrykkeligt får lov — så den ikke bruges ved et uheld.

## Handoff

Næste stage er `06_prompt_og_model`. Reviewed betyder, at du har set de
første tal og accepteret, at billedforberedelsen er god nok til at bygge
videre på (husk: den behøver ikke være perfekt — resten kan løses ved at
prompte sig ud af det i næste stage).
