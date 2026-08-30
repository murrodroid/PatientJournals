# Billedlevering, august 2026

Kollegaens svar på vores egen [billedanmodning](../../../../billedanmodning/)
fra 18. august 2026. Modtaget på ekstern harddisk 2026-08-30 fra
`D:\notatsider_til_jonas_august2026` og hentet ind med
`scripts/hent_levering.py`.

**307 PNG-filer i alt, 1,4 GB**, fladt navngivet `<bind>_<id>.png` med de
samme `image_name`-værdier som i masterlisten. Anmodningens eget dokument
ligger her som `billedanmodning_2026-08-18.md`.

## Hvad der ligger her

| Mappe | Antal | Indhold |
|---|---:|---|
| `oeve/` | 173 | Alle journalsider fra øvemængdens patienter |
| `ekstra_uden_facit/` | 50 | Andensiden fra 50 andre patienter, spredt 1889-1897 |

**Prøvemængdens 84 billeder er bevidst IKKE hentet.** Beslutningen fra
stage 02 er, at de sider først må røres ved den endelige bedømmelse, og den
letteste måde at bryde den på ville være at have filerne liggende, hvor et
glob kan samle dem op. De ligger på harddisken og hentes den dag,
bedømmelsen skal køre — kør `scripts/hent_levering.py` og udvid den til at
tage `proeve` med.

## Om kvaliteten — målt, ikke formodet

PNG'erne har **samme opløsning** som de webp-filer, vi selv hentede fra
kbharkiv. De er ikke skarpere, kun ukomprimerede. Målt på to sider:

| | middelafvigelse | PSNR | pixels der afviger > 5 |
|---|---|---|---|
| `273098_001496` | 1,64 gråtoner | 40,9 dB | 3,0 % |
| `37554_001492` | 1,31 gråtoner | 42,5 dB | 2,0 % |

Gevinsten ved at skifte er altså reel, men lille. Et skifte ville
ugyldiggøre stage 04's snit og kræve en ny kørsel, så leveringen er lagt
**ved siden af** øvebillederne i stedet for at erstatte dem. Om skiftet er
umagen værd, kan afgøres med et forsøg i stage 06 (opløsnings-aksen), hvor
netop den slags spørgsmål hører hjemme.

## Hvad leveringen tilføjer

- **65 øvesider vi ikke havde.** Vores 118 selvhentede dækker de sider,
  facit har tekst for; leveringen dækker patienternes journalsider i det
  hele taget, inklusive dem uden transskription.
- **50 andensider uden facit, spredt over 1889-1897.** De kan ikke måles
  mod facit, men de kan svare på, om billedforberedelsen holder uden for de
  15 måneder, vi har facit for — det åbne punkt i stage 04.
- De 10 sider `273104_001637`-`001646` er **ikke** i leveringen; de blev
  hentet selv via kbharkiv, fordi anmodningen ikke dækkede dem.
