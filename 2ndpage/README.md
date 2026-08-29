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
src/andenside/           Python-pakken: facit-læser, måleapparat, bogryg-snit
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

Kortlægning og planlægning er færdig, og 44 beslutninger er låst i
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

**Måleapparat (stage 03):** færdigt, godkendt og låst 2026-08-23.
`src/andenside/cer.py` er StadsCERs målekode overtaget direkte (fem
varianter, dansk tegnfoldning). `src/andenside/maal.py` er projektets eget
bidrag: **én** funktion, `forankr()`, finder facits kendte tekststumper i
modellens tekst. Ud af den ene handling falder alt det øvrige — tegnfejlen
måles på de fundne stumper, mellemrummet mellem to stumper er både
hallucinations-signal og modellens bud på et ulæseligt sted, og
linjeparringen sker gratis, fordi der søges i modellens rå tekst uden hensyn
til dens linjeskift. `src/andenside/rapport.py` skriver måletallene ud med
dækningen ved hvert eneste tal — og lægger det, modellen skrev på de
ulæselige steder, i en CSV for sig, så det kan forelægges senere uden
nogensinde at havne i facit. **Hvert tal rapporteres i to udgaver**: én der
tager de kendte stumper med fra linjer med et ulæseligt sted, og én streng,
hvor de linjer slet ikke er med. Forskellen mellem de to er skævheden, og den
skrives ud. Selvtesten
(`scripts/selvtest_maaleapparat.py`) kører apparatet mod facit selv og mod
ti konstruerede forvanskninger, hvor svaret er kendt på forhånd, og opgør
hvor stor en del af de indlagte fejl målingen faktisk finder igen.

**Billedforberedelse (stage 04):** to snit, ét i hver kant af siden.

*Falssiden* beskæres bånd for bånd (`src/andenside/skraa.py`): snitgrænsen
følger falsen ned gennem siden i stedet for at være lodret, fordi siden
krummer ind mod bindet og skriveren skrev helt ud. Alle 118 øvesider er
beskåret, ingen usikre.

*Yderkanten* — den modsatte — renses af `src/andenside/yderkant.py`. Uden
for vores side ligger enten bogsnittet (bogblokkens sammenpressede
sidekanter, harmløst) eller et blad længere inde i bindet, som er faldet
fladt ud og fotograferet med, så der står fremmed håndskrift langs kanten.
Det gælder 8 af de 118 øvesider. Detektionen måler papirets grundlyshed
pr. kolonne bånd for bånd og vælger den **inderste rette kant**, mindst 6
bånd kan enes om, og som kan bevise, at den ligger i en **søm** — en
fordybning i papiret. Snittet lægges på kantens ydre side, så ordender
ikke klippes. Findes ingen kant med en rigtig søm, skæres siden ikke.
Kontaktark: `scripts/yderkant_ark.py` (tilføj `--snit` for at se, hvad der
fjernes); tal: `scripts/yderkant_maal.py`. **Ingen beskårne billeder er
skrevet — det kræver go.**

**Næste:** stage 05, første transskription. Alt er nu på plads til at måle
et forsøg — facit, billeder og måleapparat. Der køres ingen modelkald, før
stage 03's rapportformat er gennemgået og der er givet go.

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
  orddeling hen over linjeskift, som mangler dér. En sjette variant,
  `arbejdstal` (uden versaler og tegnsætning), er det tal, valg træffes ud
  fra; `raa` er det, leverancen står ved.
- **Dækningen står ved hvert måletal.** De 11,5 % af linjerne, der rummer et
  ulæseligt sted, er de sværeste på siden. Forankringen henter de fleste af
  dem tilbage, men et måletal uden sin dækning er systematisk for pænt, og
  rapportformatet tillader ikke at udelade den.
