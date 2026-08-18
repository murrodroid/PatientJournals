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
research/                Litteraturarbejde
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

Kortlægning og planlægning er færdig, 16 beslutninger er låst i `CONTEXT.md`.
Stage 01 er ikke længere blokeret — et selvbetjent hente-script mod
kbharkiv.dk's åbne API (se `references/kbharkiv-api.md` i memory,
`scripts/kbharkiv_hent.py` her) har skaffet 8 rigtige anden-/tredjesider som
pilotmateriale, mens den formelle billedanmodning til kollegaen
(`billedanmodning/billedanmodning_2026-08-18.md`, 307 billeder) stadig
afventer svar.

Stage 04's snitpunkt-detektion er bygget og verificeret på alle 8
pilotbilleder (`src/andenside/bogryg.py`) — kendt begrænsning: kun afprøvet
på to bind fra samme fotograferingssession, bredere test mangler.

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
- Facit: 38 patientforløb fra dødsfald maj 1896 til august 1897, i alt 257
  journalsider ud over forsiderne, med rig men ustandardiseret klammenotation.
- Målingen bygger på `J-Hoffi/StadsCER`s fem varianter og tilføjer samling af
  orddeling hen over linjeskift, som mangler dér.
