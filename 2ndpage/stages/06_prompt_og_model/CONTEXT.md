# Stage 06 — Prompt og model

## Formål

Selve læse-implementeringen: finde den bedste kombination af model, prompt
og opløsning. Dette er projektets kerneleverance, adskilt fra stage 05's
engangs-check af billedforberedelsen og fra stage 07's uenighedslag.

## Inputs

| Type | Sti |
|---|---|
| Testregler | `../../_config/tdd.md` |
| Første tal | `../05_foerste_transskription/output/foerste_tal.md` |
| Måleapparat | `../03_maaleapparat/output/rapportformat.md` |
| Anbefalinger | `../00_forundersoegelse/output/forundersoegelse.md` |

## Process

1. Én akse ad gangen, kun hvis den foregående er afklaret: model,
   promptformulering, **billedskarphed** (se stage 00's fund — dette er
   ikke en mindre detalje; litteraturen finder markant dårligere
   maskinlæsning under en vis skarphedsgrænse, og vores kildebilleder
   ligger sandsynligvis dér eller under), og forbehandling med
   linjedetektion (baselines) enten som udklip eller som hjælp til at
   vide, hvilke linjer der findes på siden.
1a. **Hold Transkribus/specialiseret håndskriftsgenkendelse som reel
    sammenligning, ikke kun baggrundsviden** (stage 00-anbefaling): et
    dansk forsøg fandt Transkribus foran både ChatGPT og Gemini på dansk
    håndskrift, og et uafhængigt studie fandt Claude/GPT-4o på 41-60 %
    tegnfejl på et andet historisk materiale — langt fra Humphries'
    5-7 %. Antag ikke Gemini vinder, før stage 05's egne tal viser det.
2. **Prompt som backup for ufuldkommen beskæring**: stage 04's snit behøver
   ikke være perfekt. Test eksplicit, om en instruktion om at ignorere
   delvis synlig nabotekst i kanten løser resten af problemet billigere end
   at forfine beskæringen yderligere.
3. **Vær opmærksom på fremmed bogstavbleed fra HELT andre opslag**, ikke kun
   det umiddelbare naboopslag. Lead: på en verso-side, fx side 101, kan man
   i affotograferingen nogle gange se stumper af en helt anden, langt
   tidligere verso-side (fx side 51) stikke ind i billedet — formentlig fordi
   bogen ikke ligger helt fladt, og en bagvedliggende side skinner igennem
   eller er synlig i en anden vinkel. Det er en ANDEN forureningskilde end
   den nære nabo-strimmel, stage 04 håndterer, og kan ende forkert i
   transskriptionen, hvis intet fanger det. Undersøg om det sker konsekvent
   på samme kant, og om en prompt-instruktion kan bede modellen ignorere
   løsrevne, ude-af-kontekst bogstaver/ord, der ikke hænger sammen med den
   omkringliggende tekst.
4. Alle kørsler på øvemængden. Den låste prøvemængde røres ikke.
5. Gem altid rå modelsvar + fuld opsætning + dato.
6. Ingen fuld kørsel uden dit udtrykkelige go.

## Outputs

| Fil | Beskrivelse |
|---|---|
| `output/koersler/` | Rå modelsvar pr. kørsel med fuld angivelse af opsætning |
| `output/resultater.csv` | Én række pr. kørsel: opsætning og alle fem måletal |
| `output/sammenligning.md` | Hvad hver akse viste, og hvad der blev valgt |

## Test Contract

Modelsvar kan ikke testes fast. Bogholderiet skal testes: en kørsel gemmes
altid med sin opsætning, og to kørsler med samme opsætning kan skelnes på
dato. En test skal sikre, at kode som læser den låste prøvemængde fejler,
medmindre den udtrykkeligt får lov.

## Handoff

Næste stage er `07_anden_stemme`. Reviewed betyder, at du har set
sammenligningen og udpeget den bedste kombination af model, prompt og
opløsning.
