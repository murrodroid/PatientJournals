# 2ndpage

Metodetest for transskription af **andensiderne** i håndskrevne
patientjournaler fra Blegdamshospitalet. Andensiderne rummer modtagenotater
skrevet fortløbende linje for linje, i modsætning til forsiderne, der er
felt-opdelte og allerede transskriberet i
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
CONTEXT.md             Beslutningslog — hvorfor
PROGRESS.md            Tjekliste — status
_config/tdd.md         Testregler
stages/00..06/         Syv stages, hver med CONTEXT.md og output/
references/            Kortlægninger af omverdenen
research/              Litteraturarbejde
billedanmodning/       Liste over billeder bestilt hos kollegaen
tests/                 Herunder strukturtesten
```

| Stage | Indhold |
|---|---|
| 00 forundersoegelse | Litteratur og state of the art, engangs |
| 01 datagrundlag | Billeder, opslagsregister, dækning |
| 02 facit | RTF-læser, klammekonventioner, øve- og prøvemængde |
| 03 maaleapparat | Tegnfejl, orddeling, hallucinationskontrol |
| 04 billedforberedelse | Bogryg-snit, naboblade, egne tests |
| 05 metodeforsoeg | Forsøgsmatrix, én akse ad gangen |
| 06 integration | Aflevering til kollegaens app |

## Status

Kortlægning og planlægning er færdig; 16 beslutninger er låst i `CONTEXT.md`.
Der er endnu ingen kode. **Stage 01 er blokeret**, fordi der ikke findes ét
eneste andenside-billede på maskinen — de 307 billeder er bestilt hos
kollegaen i `billedanmodning/billedanmodning_2026-08-18.md`. Stage 00 kan køre
imens.

## Nøglefakta

- Andensider udpeges i `Blegdam_master_list.csv` ved
  `patient_page_counter == 1` (tælleren starter på 0 ved forsiden): 71.391 sider.
- Ét billede er ét **dobbeltopslag** med to sider, ~900-1.000 pixels bredde
  pr. tekstside.
- Facit: 38 patientforløb fra dødsfald maj 1896 til august 1897, i alt 257
  journalsider ud over forsiderne, med rig men ustandardiseret klammenotation.
- Målingen bygger på `J-Hoffi/StadsCER`s fem varianter og tilføjer samling af
  orddeling hen over linjeskift, som mangler dér.
