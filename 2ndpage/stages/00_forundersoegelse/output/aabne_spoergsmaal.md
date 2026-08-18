# Åbne spørgsmål — stage 00 (2026-08-18)

Det litteraturen IKKE svarer på, og som derfor kræver vores egen måling.
Se `forundersoegelse.md` for kilder og anbefalinger.

## 1. Hvor godt læser Gemini/Claude rent faktisk dansk gotisk/kursiv håndskrift?

Ingen kilde giver et hårdt tal for dansk. Det ene danske hands-on-fund
(ChatGPT hallucinerede, Gemini bedre men fejlede på navne, Transkribus
slog begge) er anekdotisk — én bruger, få dokumenter, ikke et formelt
benchmark. Vi må selv måle dette i stage 05, med et åbent sind om, at
resultatet kan være markant dårligere end Humphries' engelske tal.

## 2. Er vores billeders opløsning en reel begrænsning?

Litteraturen siger, at LLM-OCR forringes markant under ~150 ppi, og vores
sider (~900-1.600 px brede) ligger sandsynligvis i eller under den zone —
men ingen kilde har målt præcis VORES materiale. Skal afklares empirisk:
mål et rigtigt DPI-tal, og test om højere opløsning (hvis kollegaen kan
skaffe det, eller hvis kbharkiv har bedre originaler) giver en målbar
forskel.

## 3. Virker overstregnings-instruktionen, når den rent faktisk prøves på en LLM?

Ingen kilde har testet dette for LLM-transskription. Kollegaens
`frontpage`-prompt bruger en fungerende formulering for et andet skema,
men det er ubekræftet, om den overføres til linje-for-linje-transskription.
Ren empiri i stage 06.

## 4. Er "fjernt bleed" et systematisk problem, eller et enkeltstående fund?

Lead har set det én gang (side 51-i-side-101-eksemplet). Litteraturen
antyder, at det formentlig hænger sammen med bindets fysiske tilstand
(omindbinding, løse blade) snarere end en generel fotograferingsfejl —
hvilket betyder, at det kan variere fra bind til bind. Skal undersøges
systematisk i stage 06, ikke antages at være enten alle steder eller
ingen steder.

## 5. Hjælper et klinisk nøgleords-overlevelsesmål reelt med at forudsige, om metoden er god nok?

Litteraturen argumenterer generelt for entitets-/nøgleordsmål som
supplement til CER/WER, men ingen kilde har bygget det specifikt til
historiske kliniske journaler. Skal designes og afprøves som en let,
sekundær metrik i stage 03 — ikke som en garanti for, at den rent faktisk
korrelerer med det, vi vil vide, før vi har set den i brug.

## 6. Bør Transkribus eller anden specialiseret HTR indgå som reel konkurrent, ikke kun baggrundsviden?

Riksarkivets "Swedish Lion" og lignende specialiserede modeller opnår høj
nøjagtighed på nordisk håndskrift — men er trænet på specifikt materiale,
og intet tyder på, at der findes en færdigtrænet model til netop danske
lægejournaler fra 1880-1910. Om det er værd at bygge/finjustere en sådan,
eller om det er for dyrt i forhold til gevinsten, er uafklaret og bør
genovervejes, hvis stage 05's første Gemini-tal er dårlige.
