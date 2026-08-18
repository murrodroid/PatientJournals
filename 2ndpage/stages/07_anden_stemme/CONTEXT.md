# Stage 07 — Anden stemme

## Formål

Bygge uenighedslaget oven på den valgte metode fra stage 06: en anden
model, fra en anden familie, transskriberer igen, og steder hvor de to er
uenige markeres til menneskelig gennemgang. Dette er et kvalitetslag, ikke
en del af selve læsningen — derfor sin egen stage.

## Inputs

| Type | Sti |
|---|---|
| Testregler | `../../_config/tdd.md` |
| Valgt metode | `../06_prompt_og_model/output/sammenligning.md` |
| Måleapparat | `../03_maaleapparat/output/rapportformat.md` |
| Humphries-forarbejde | `../../references/humphries_generative_history.md` |

## Process

1. Lad `claude-opus-4-6` (anden modelfamilie end den valgte hovedmodel)
   transskribere de samme sider igen med samme prompt.
2. Marker steder hvor de to er uenige — det er signalet, ikke et forsøg på
   at afgøre hvem der har ret.
3. Mål, jf. Humphries' maj 2026-fund: hvor stor en del af de faktiske fejl
   fanger uenigheden, og hvor stor en del af teksten skal ses efter for at
   fange dem. Målet er høj fejlfangst for lav gennemsynsbyrde.
4. Alle kørsler på øvemængden. Den låste prøvemængde røres ikke.
5. Ingen fuld kørsel uden dit udtrykkelige go.

## Outputs

| Fil | Beskrivelse |
|---|---|
| `output/uenigheder.csv` | Pr. side: hvor de to modeller er uenige |
| `output/traeffik.md` | Hvor stor en andel af fejlene uenigheden fangede, og til hvilken pris i gennemsynstid |

## Test Contract

Modelsvar kan ikke testes fast. Selve sammenligningslogikken (find
uenigheder mellem to tekststrømme) er ren regnekode og skal testes med
konstruerede eksempler: identisk tekst (ingen uenighed), én ords forskel,
og forskellig linjeopdeling der ikke må give falske uenigheder.

## Handoff

Sidste stage før `08_integration`. Reviewed betyder, at du har set
træfsikkerheden og besluttet, om anden stemme er værd at bruge i den
endelige leverance.
