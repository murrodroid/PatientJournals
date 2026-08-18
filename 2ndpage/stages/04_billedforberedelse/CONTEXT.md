# Stage 04 — Billedforberedelse

## Formål

Gøre et opslag klar til læsning: beskære den forurenende strimmel af
naboopslaget væk, så kun målsiden (andenside ELLER tredjeside) sendes til en
model. Forberedelsen bedømmes for sig selv, før noget sendes til en model,
så en forkert beskæring aldrig kan forveksles med en dårlig læsning.

## Inputs

| Type | Sti |
|---|---|
| Testregler | `../../_config/tdd.md` |
| Opslagsregister | `../01_datagrundlag/output/opslagsregister.csv` |
| Opslagsstruktur | `../01_datagrundlag/output/opslag_struktur.md` |
| Anbefalinger | `../00_forundersoegelse/output/forundersoegelse.md` |
| Prøveopslag + fund | `../01_datagrundlag/output/proeve_opslag/` |

## Kendt regel (låst 2026-08-18, forklaret af lead)

Journalerne var oprindeligt løse blade, senere indbundet — og indbinding af
foldede blade starter altid på en enkelt recto. Derfor er **forsiden altid
recto, andensiden altid verso, tredjesiden altid recto**, og fremdeles
(lige `patient_page_counter` = recto, ulige = verso). Det afgør ENTYDIGT,
hvilken kant af billedet der bærer målsidens indhold, og hvilken kant der
bærer en strimmel af naboopslaget:

| Side | Recto/verso | Hovedindhold | Strimmel af nabo |
|---|---|---|---|
| Andenside (verso) | verso | venstre | højre |
| Tredjeside (recto) | recto | højre | venstre |

Verificeret mod fem virkelige billeder i `01_datagrundlag/output/proeve_opslag/`
— se `CONTEXT.md` (roden) for detaljerne. **Sidevalget kræver derfor ingen
CV-gætteri**: det kan udledes direkte af `patient_page_counter`s paritet.

## Process

1. Beregn kant (venstre/højre) direkte af `patient_page_counter`s paritet
   ud fra tabellen ovenfor — ikke ved billedanalyse.
2. Byg en metode til at finde selve SNITPUNKTET inden for den kendte kant
   (hvor strimlen holder op og hovedsiden begynder). Dette er en lettere
   opgave end oprindeligt antaget, da kanten allerede er kendt — men
   snitPUNKTET er stadig ukendt og skal findes. To naive kolonneprofil-forsøg
   (blækmængde-top/dal, `scripts/bogryg_profil.py`) fejlede tidligere på det
   forkerte problem (at finde en midterrygl); de kan genbruges som
   udgangspunkt for at finde selve overgangen inden for den nu kendte kant.
3. Find naboblade, der stikker ind i rammen ud over den forventede strimmel,
   og beskær dem væk.
4. Detektér blanke halvsider (typisk kun relevant for forsideopslag, som
   IKKE er i scope her, men kan påvirke nabosidens strimmel).
5. Lav en visuel gennemgang: et kontaktark med indtegnet snitlinje for et
   antal opslag, så snittene kan bedømmes med øjnene i ét blik.
6. Registrér usikre tilfælde frem for at gætte. Et opslag, hvor snitpunktet
   ikke kan findes sikkert, skal mærkes og lægges til side, ikke skæres på
   slump.

## Outputs

| Fil | Beskrivelse |
|---|---|
| `output/snit.csv` | Én række pr. opslag: kant (afledt af paritet), fundet snitpunkt, sikkerhed, beskæringsfelt, mærkning |
| `output/kontaktark/` | Billeder med indtegnet snit og beskæring til visuel gennemgang |
| `output/usikre.md` | Opslag hvor snittet ikke kunne fastlægges sikkert, med begrundelse |

## Test Contract

Kant-udledningen af paritet er triviel og skal testes direkte (lige/ulige
→ korrekt kant), inklusive kanttilfældet forside (`patient_page_counter=0`,
IKKE i scope, skal afvises tydeligt frem for fejlbehandles). Selve
snitpunkt-findingen er ren regnekode og skal testes mod et sæt opslag, hvor
det rigtige snitpunkt er fastlagt i hånden — testen skal dække tilfælde hvor
et naboblad rager usædvanligt langt ind, og hvor teksten står tæt på
snittet. Der skal være en test, der sikrer, at et usikkert fund bliver
mærket og ikke skåret. Ingen model kaldes i denne stage.

## Handoff

Næste stage er `05_metodeforsoeg`. Reviewed betyder, at du har set
kontaktarket igennem og godkendt snittene — først derefter må forberedelsen
låses som fast forudsætning for forsøgene.
