# Dækning — stage 01 (2026-08-18)

## Hvad vi har lige nu

**8 sider** i `opslagsregister.csv`, hentet selvbetjent via kbharkiv.dk
(`scripts/kbharkiv_hent.py`) som pilotmateriale — ikke den formelle
billedanmodning, som stadig afventer kollegaen.

| Rolle | Antal |
|---|---|
| Andenside (verso) | 4 |
| Tredjeside (recto) | 4 |

Fordelt på 3 patientforløb, 2 bind (273098, 273099), alle med facit fundet
og korrekt matchet (se `../src/andenside/opslagsregister.py`s
`find_facit_file`, testet mod en regression hvor et bind har flere
patienter).

## Hvad vi mangler

- **De resterende 299 billeder** fra den formelle billedanmodning
  (`billedanmodning/billedanmodning_2026-08-18.md`, 307 i alt: 257 med
  facit + 50 uden). Kun 8 af de 257 facit-sider er hentet.
- Alle andensider (`patient_page_counter=1`) for de resterende 35 GT-patienter.
- Alle tredjesider (`patient_page_counter=2`) for samme.
- De 50 ekstra andensider fra andre årgange (diversitetsstikprøven).

## Kendte huller i pilotmaterialet

- Kun to bind repræsenteret (273098, 273099) — begge maj/juni 1896.
  Ingen spredning over årgange endnu; det dækker den formelle anmodnings
  50-ekstra-gruppe.
- Ingen forsider i registeret (bevidst — forsiderne er ude af scope, de er
  allerede transskriberet af kollegaens pipeline og bruges kun som
  facit-anker, ikke som eget datapunkt).
