# Stage 00 — Forundersøgelse

## Formål

Én grundig gennemgang af, hvad der allerede vides om maskinlæsning af
håndskrevne historiske dokumenter, så metodevalgene i stage 04 og 05 hviler på
andres målinger frem for på vores gæt. Undersøgelsen laves som et samlet
arbejde nu, og kan sættes i gang igen senere, hvis vi står med et valg, vores
egne tal ikke kan afgøre.

## Inputs

| Type | Sti |
|---|---|
| Testregler | `../../_config/tdd.md` |
| Kendt forarbejde | `../../references/humphries_generative_history.md` |
| Kendt forarbejde | `../../references/stadscer.md` |
| Materialebeskrivelse | `../../references/manuelle_transskriptioner.md` |

## Process

1. Afgræns spørgsmålene, undersøgelsen skal besvare. Udgangspunktet er:
   hvilke metoder er målt bedst på håndskrift fra 1800-tallet; hvad gør man
   ved dobbeltopslag og sideafgrænsning; hvordan håndteres overstreget tekst;
   hvad findes der af arbejde på dansk og på nordisk arkivmateriale; hvordan
   måles kvalitet i faglitteraturen.
2. Send agenter ud pr. spørgsmål, ikke pr. kilde. Hver agent skal returnere
   påstand, kilde med årstal, og hvor stærkt belægget er.
3. Marker udtrykkeligt, hvad der er målt, og hvad der er formodning.
4. Saml til ét notat med en kort liste af **anbefalinger, der kan handles på**
   i stage 04 og 05 — ikke et referat.
5. Notér modsigelser mellem kilder frem for at glatte dem ud.

## Outputs

| Fil | Beskrivelse |
|---|---|
| `output/forundersoegelse.md` | Samlet notat med anbefalinger og kildeliste |
| `output/aabne_spoergsmaal.md` | Hvad litteraturen ikke svarer på, og som vi selv skal måle |

## Test Contract

Ingen kode, derfor ingen enhedstest. Kravet er i stedet, at hver påstand i
notatet har en kilde med årstal, og at anbefalinger er adskilt fra referat.
En påstand uden kilde skal fjernes eller mærkes som formodning.

## Handoff

Næste stage er `01_datagrundlag`. Reviewed betyder, at du har læst
anbefalingerne igennem og er enig i, at de er værd at afprøve — ikke at de er
sande.
