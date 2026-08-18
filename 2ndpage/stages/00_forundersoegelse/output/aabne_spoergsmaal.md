# Åbne spørgsmål — stage 00 (2026-08-18)

Det litteraturen IKKE svarer på, og som derfor kræver vores egen måling.
Se `forundersoegelse.md` for kilder og anbefalinger.

## 1. Hvor godt læser Gemini/Claude rent faktisk dansk håndskrift?

Ingen kilde giver et hårdt tal for dansk. Det ene danske forsøg (ChatGPT
digtede indhold, Gemini bedre men fejlede på navne, Transkribus slog
begge) er én persons erfaring, ikke en videnskabelig undersøgelse. Vi må
selv måle dette i stage 05, med et åbent sind om, at resultatet kan blive
markant dårligere end Humphries' engelske tal.

## 2. Er vores billeder skarpe nok?

Litteraturen siger, at maskinlæsning bliver markant dårligere under en
vis skarphedsgrænse, og vores sider (~900-1.600 billedpunkter brede)
ligger sandsynligvis i eller under den grænse — men ingen kilde har målt
netop vores materiale. Skal afklares selv: find ud af, hvor skarpe de
rigtige billeder faktisk er, og test om en skarpere udgave (hvis
kollegaen kan skaffe den, eller hvis kbharkiv har bedre originaler) giver
en mærkbar forskel.

## 3. Virker overstregnings-instruktionen, når den rent faktisk prøves på en sprogmodel?

Ingen kilde har testet dette for sprogmodeller. Kollegaens prompt til
forsiderne bruger en formulering, der virker på den type opgave, men det
er ubekræftet, om den samme formulering virker, når modellen skal
transskribere linje for linje. Ren afprøvning i stage 06.

## 4. Er det "fjerne bogstav-gennemskin" et gennemgående problem, eller et enkeltstående fund?

Lead har set det én gang (eksemplet med at side 51 kan skinne igennem
på en optagelse af side 101). Litteraturen antyder, at det formentlig
hænger sammen med den enkelte bogs fysiske tilstand snarere end en
generel fejl i selve fotograferingen — hvilket betyder, at det kan
variere meget fra bind til bind. Skal undersøges systematisk i stage 06,
ikke antages at gælde enten alle steder eller slet ingen steder.

## 5. Hjælper det reelt at tælle, om de vigtige kliniske ord overlever?

Litteraturen argumenterer generelt for den slags mål som supplement til
tegnfejl-procenten, men ingen kilde har bygget det specifikt til
historiske lægejournaler. Skal designes og afprøves som et let,
sekundært mål i stage 03 — vi ved ikke på forhånd, om det rent faktisk
viser noget nyttigt, før vi har set det i brug.

## 6a. Virker "lad modellen rette sin egen transskription" — eller ej?

To kilder modsiger hinanden. Humphries fandt en gevinst ved at lade en
model rette en første transskription igennem. Crosilla/Klic/Colavizza
testede en lignende metode og fandt ingen pålidelig forbedring — for de
mindre, gratis modeller blev det ligefrem værre. Kun vores egen test i
stage 06 kan afgøre, hvem der har ret for vores materiale.

## 6b. Hvordan klarer Gemini sig på svært, ikke-engelsk historisk materiale?

Crosilla/Klic/Colavizza testede slet ikke Gemini — kun GPT-4o, GPT-4o-mini
og Claude. Vi har altså mere ekstern dokumentation for Claudes evner på
svært materiale end for Gemini, som er vores planlagte hovedmodel. Endnu
en grund til ikke at antage, Gemini er det bedste valg, før vi selv har
målt det.

## 7. Bør Transkribus eller lignende specialiserede programmer være en reel konkurrent, ikke kun noget vi nævner?

Sveriges Riksarkiv har opnået høj træfsikkerhed på nordisk håndskrift med
deres eget, specialtrænede program — men det er trænet på deres eget
materiale, og intet tyder på, at der findes et færdigtrænet program til
netop danske lægejournaler fra 1880-1910. Om det er umagen værd selv at
bygge eller tilpasse et sådant program, eller om det er for dyrt i
forhold til gevinsten, er uafklaret og bør tages op igen, hvis stage 05's
første tal med Gemini er dårlige.
