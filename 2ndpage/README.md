# 2ndpage

Metodetest for transskription af **andensiderne og tredjesiderne** i
håndskrevne patientjournaler fra Blegdamshospitalet (fuldt korpus
1880-1910). Siderne rummer modtagenotater skrevet fortløbende linje for
linje, i modsætning til forsiderne, der er felt-opdelte og allerede
transskriberet i
[murrodroid/PatientJournals](https://github.com/murrodroid/PatientJournals).
Projektet finder den bedste metode ved systematisk måling mod håndlavet facit
og afleverer vinderen som prompt, skema og måletal til kollegaens app.

## Sådan er projektet bygget

Projektet følger **ICM (Interpretable Context Methodology)**: mappestrukturen
er arkitekturen. Hver stage har sin egen kontekstfil med en fast seks-afsnits
kontrakt, sine egne outputfiler, og en menneskelig gennemgang før næste stage.
Strukturen håndhæves af `tests/test_icm_structure.py`.

```
AGENTS.md              Arbejdsregler (CLAUDE.md peger blot herpå)
CONTEXT.md              Beslutningslog — hvorfor
PROGRESS.md              Tjekliste — status
_config/tdd.md          Testregler
stages/00..08/          Ni stages, hver med CONTEXT.md og output/
references/              Kortlægninger af omverdenen
billedanmodning/         Liste over billeder bestilt hos kollegaen
scripts/                 Midlertidige værktøjer, uden for ICM-stagestrukturen
src/andenside/           Python-pakken: masterliste, opslagsregister, bogryg-snit
tests/                   Herunder strukturtesten
```

| Stage | Indhold |
|---|---|
| 00 forundersoegelse | Litteratur og state of the art, engangs |
| 01 datagrundlag | Billeder, opslagsregister, dækning |
| 02 facit | RTF-læser, klammekonventioner, øve- og prøvemængde |
| 03 maaleapparat | Tegnfejl, orddeling, hallucinationskontrol |
| 04 billedforberedelse | Snit af naboopslagets strimmel, kontaktark |
| 05 foerste_transskription | Første forsøg: fast model+prompt, kun beskæring varieres |
| 06 prompt_og_model | Selve læse-implementeringen: model, prompt, opløsning |
| 07 anden_stemme | Claude som anden stemme, uenighedsmarkering |
| 08 integration | Aflevering til kollegaens app |

## Status

Kortlægning og planlægning er færdig, og 38 beslutninger er låst i
`CONTEXT.md`.

**Billeder (stage 01):** kbharkiv.dk's kildeviser har et åbent API, så vi
kan hente selv (`scripts/kbharkiv_hent.py`). Hele øvemængden er hentet —
118 sider fra 15 bind i `stages/01_datagrundlag/output/oeve_billeder/`,
plus 8 tidligere pilotbilleder. Prøvemængdens sider er bevidst **ikke**
hentet. Forskydningen mellem kildeviserens sidetal og masterlistens
billed-id er efterprøvet på alle 15 bind. Billedanmodningen til kollegaen
(307 billeder) er sendt og afventer levering.

**Facit (stage 02):** færdigt, godkendt og låst 2026-08-21.
`src/andenside/facit.py` læser de 39 håndlavede RTF-filer og skriver fire
filer i `stages/02_facit/output/`: 168 sider med facit fra 39 patientforløb
i to udgaver (alt hvad der står / den rettede læsning), en udtømmende
optælling af klammeopmærkningen, opdelingen i øve- og prøvemængde, og de
blokke og steder, der er lagt til side. Kildefilerne på OneDrive læses kun
— en test håndhæver, at de aldrig røres.

**Billedforberedelse (stage 04):** godkendt og låst. Snitpunkt-detektionen
(`src/andenside/bogryg.py`) er verificeret på alle 8 pilotbilleder — kendt
begrænsning: kun afprøvet på to bind fra samme fotograferingssession.

**Næste:** stage 03, måleapparatet. Det er det sidste, der mangler, før et
forsøg overhovedet kan måles.

## Nøglefakta

- Andensider: `patient_page_counter == 1` (verso). Tredjesider:
  `patient_page_counter == 2` (recto). Tælleren starter på 0 ved forsiden.
  ~71.400 sider af hver.
- **Recto/verso-reglen** (låst): journalerne var løse blade, senere
  indbundet — indbinding starter altid på en enkelt recto. Forside=recto,
  andenside=verso, tredjeside=recto, fremdeles. Det afgør entydigt hvilken
  kant af et opslag der bærer naboopslagets strimmel, uden CV-gætteri.
- Ét billede er asymmetrisk beskåret om én målside (ikke et symmetrisk
  dobbeltopslag) — kun en smal strimmel af naboopslaget er synlig i én kant.
- Facit: 39 patientforløb fra dødsfald maj 1896 til august 1897. De har 268
  journalsider ud over forsiderne, men kun **168 af dem er faktisk
  transskriberet** — 40 sidemærker står uden tekst, og for en del patienter
  stopper transskriptionen før indlæggelsen gør. Notationen er rig, men
  ustandardiseret: 194 forskellige skrivemåder fordelt på 8 slags mærker.
- Målingen bygger på `J-Hoffi/StadsCER`s fem varianter og tilføjer samling af
  orddeling hen over linjeskift, som mangler dér.
