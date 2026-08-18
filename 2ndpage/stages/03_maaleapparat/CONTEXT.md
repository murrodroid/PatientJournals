# Stage 03 — Måleapparat

## Formål

Kunne sætte et tal på en transskription. Tegnfejl på den fladede tekst er
tallet, vi træffer beslutninger på; linjeopdelingen bevares og bruges til en
særskilt kontrol for, om modellen har digtet hen over et linjeskift.

## Inputs

| Type | Sti |
|---|---|
| Testregler | `../../_config/tdd.md` |
| Facit | `../02_facit/output/facit.jsonl` |
| Forarbejde | `../../references/stadscer.md` |

## Process

1. Overtag målekoden fra `J-Hoffi/StadsCER` frem for at skrive ny:
   `normalize`, `strip_diacritics`, `levenshtein`, `align` og de fem
   varianter `raa`, `uden_versaler`, `uden_diakritika`, `uden_tegnsaetning`,
   `lempeligst`. Alle fem rapporteres side om side; ingen af dem må vælges
   efter, hvilken der klæder resultatet bedst.
2. Byg det, StadsCER mangler: samling af orddeling hen over linjeskift, som
   dagbogen dér udpeger som det dominerende fejlmønster. Dette er projektets
   bidrag tilbage.
3. Byg hallucinationskontrollen: find steder hvor modellen har fuldført et ord
   hen over et linjeskift, som facit deler. Kontrollen må ikke kræve, at model
   og facit har samme linjeopdeling — det har de ikke her, i modsætning til i
   StadsCER, hvor linjerne er givet på forhånd.
4. Beslut og skriv ned, hvordan `[?]` i facit behandles. Et ulæseligt sted må
   ikke tælle som en fejl mod modellen, men må heller ikke bare fjernes, så
   modellen belønnes for at digte noget.
5. Byg rapportformatet: fem varianter i tabel, hallucinationstal opgjort for
   sig, og de værste enkeltopslag udpeget, så de kan ses efter med øjnene.

## Outputs

| Fil | Beskrivelse |
|---|---|
| `output/rapportformat.md` | Eksempel på en færdig måling, så formatet er aftalt før første kørsel |
| `output/selvtest.md` | Måleapparatet kørt mod facit selv og mod bevidst forvanskede udgaver |

## Test Contract

Måleapparatet skal give nul fejl, når facit sammenlignes med sig selv, og et
kendt, forud udregnet tal på konstruerede forvanskninger. Der skal være en
test for hver af de fem varianter, en test for orddelingssamlingen, og en test
for at to kørsler på samme data giver nøjagtig samme rapport — rækkefølge fra
mængder og ordbøger har tidligere givet ikke-reproducerbare resultater i andre
projekter.

## Handoff

Næste stage er `04_billedforberedelse`. Reviewed betyder, at du har set
rapportformatet og er enig i, at det er de tal, du vil træffe valg ud fra.
