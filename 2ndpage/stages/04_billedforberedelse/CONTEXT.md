# Stage 04 — Billedforberedelse

## Formål

Gøre et dobbeltopslag klar til læsning: finde bogryggen, dele opslaget i to
sider, og fjerne naboblade der stikker ind i billedet. Forberedelsen bedømmes
for sig selv, før noget sendes til en model, så en forkert beskæring aldrig
kan forveksles med en dårlig læsning.

## Inputs

| Type | Sti |
|---|---|
| Testregler | `../../_config/tdd.md` |
| Opslagsregister | `../01_datagrundlag/output/opslagsregister.csv` |
| Opslagsstruktur | `../01_datagrundlag/output/opslag_struktur.md` |
| Anbefalinger | `../00_forundersoegelse/output/forundersoegelse.md` |
| Kendt forarbejde | `../../references/icm_metodik.md` |

## Process

1. Byg en kolonnevis blækmængde-profil over billedet. Fra magresprots
   separator-research vides, at en bogryg optræder som en mørk **top** i
   profilen, mens en blækstreg optræder som en **dal** — og at ryggen ligger
   på samme vandrette position hele siden igennem, hvor blæk flytter sig.
   Koden dér findes ikke, kun beskrivelsen; den skal skrives ny.
2. Del opslaget ved ryggen i venstre og højre side, og fastlæg
   læserækkefølgen. Kontrollér antagelsen om venstre før højre mod et faktisk
   opslag i stedet for at forudsætte den.
3. Find naboblade, der stikker ind i rammen, og beskær dem væk. Et blad, der
   rager ind, ligner en tekstsøjle for en model.
4. Detektér blanke halvsider, så en tom venstre side ikke sendes af sted.
5. Lav en visuel gennemgang: et kontaktark med indtegnet snitlinje for et
   antal opslag, så snittene kan bedømmes med øjnene i ét blik.
6. Registrér usikre tilfælde frem for at gætte. Et opslag, hvor ryggen ikke
   kan findes sikkert, skal mærkes og lægges til side, ikke skæres på slump.

## Outputs

| Fil | Beskrivelse |
|---|---|
| `output/snit.csv` | Én række pr. opslag: fundet rygposition, sikkerhed, beskæringsfelter, mærkning |
| `output/kontaktark/` | Billeder med indtegnet snit og beskæring til visuel gennemgang |
| `output/usikre.md` | Opslag hvor snittet ikke kunne fastlægges sikkert, med begrundelse |

## Test Contract

Snitfindingen er ren regnekode og skal testes mod et sæt opslag, hvor den
rigtige rygposition er fastlagt i hånden. Testen skal indeholde de svære
tilfælde: opslag med blank venstre side, opslag hvor et naboblad rager ind,
og opslag hvor teksten står tæt på ryggen. Der skal være en test, der sikrer,
at et usikkert fund bliver mærket og ikke skåret. Ingen model kaldes i denne
stage.

## Handoff

Næste stage er `05_metodeforsoeg`. Reviewed betyder, at du har set
kontaktarket igennem og godkendt snittene — først derefter må forberedelsen
låses som fast forudsætning for forsøgene.
