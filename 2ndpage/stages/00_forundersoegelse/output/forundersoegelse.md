# Forundersøgelse — samlet notat (2026-08-18)

Fem spørgsmål, hver undersøgt af en selvstændig agent med webadgang. Hver
påstand herunder har en kilde med årstal; kildelisten står nederst, så
selve teksten kan læses uden at drukne i afhandlingstitler. Se
`aabne_spoergsmaal.md` for det, ingen kilde svarer på.

To ord, der går igen: **tegnfejl-procent** (hvor stor en andel af
bogstaverne/tegnene en maskine læser forkert — det, der ellers hedder CER
i litteraturen) og **billedskarphed** (hvor mange billedpunkter der er at
tegne et bogstav med — for lav skarphed betyder, at selv et menneske ville
få svært ved at læse teksten på skærmen).

## Anbefalinger, der kan handles på

1. **Undersøg om vores billeder er skarpe nok, tidligt og for sig selv.**
   Et studie fra 2025 [5] målte, at maskiner læser håndskrift markant
   dårligere, når billedet er for groft — under en vis skarphedsgrænse
   falder træfsikkerheden brat. Vores billeder er kun 900-1.600
   billedpunkter brede for en hel side, hvilket sandsynligvis ligger i
   eller under den grænse. Skaf et konkret tal for, hvor skarpe de rigtige
   billeder er, og test direkte om skarphed er en begrænsning — før vi
   konkluderer noget som helst om hvilken model eller prompt der er bedst.
2. **Stol ikke på, at vores egne tal bliver lige så gode som Humphries'.**
   Humphries (vores hovedkilde til gode resultater) fandt kun 5-7 %
   tegnfejl på historisk engelsk tekst. Men et helt andet, uafhængigt
   studie [2] fandt 41-60 % tegnfejl med de SAMME modeller (Claude, GPT-4o)
   på et andet historisk materiale. Modellerne er altså ikke jævnt gode —
   det afhænger meget af, hvilket materiale de bliver testet på. Vores
   egne første tal (stage 05) skal derfor tages meget alvorligt, uanset
   hvor godt andre har klaret sig andre steder.
3. **Antag ikke, at Gemini eller Claude er gode til dansk håndskrift —
   det er ikke bevist endnu.** En dansk person, der selv prøvede det [3],
   fandt at ChatGPT direkte digtede indhold, der slet ikke stod i
   originalen, på et dansk dokument fra 1844. Gemini klarede sig bedre,
   men læste stadig navne forkert. Det specialiserede program Transkribus
   slog begge. Det er kun ét menneskes forsøg, ikke en videnskabelig
   undersøgelse, men et andet studie [4] bekræfter mere generelt, at disse
   modeller er dårligere til andre sprog end engelsk. **Konsekvens**: lad
   Transkribus eller lignende specialiserede programmer blive ved med at
   være en reel mulighed i stage 06 — ikke kun noget vi nævner i forbifarten.
4. **Tilføj en instruktion om overstregninger til kollegaens prompt — men
   vi er de første til at afprøve, om det virker på en sprogmodel.**
   Der findes stort set ingen forskning i, hvordan disse modeller
   håndterer overstreget/udstreget tekst. Kollegaens EGEN prompt til
   forsiderne har allerede en formulering, der beder modellen om at
   springe overstreget tekst over — men den formulering findes ikke i den
   prompt, vi selv skal bygge videre på. Billigt at kopiere ind, men vi
   ved ikke om det faktisk virker, før vi selv har testet det.
5. **Overvej et simpelt supplement til tegnfejl-målingen: tæl om de
   vigtige kliniske ord overlever.** Flere kilder [6] viser, at et
   dokument kan have lav tegnfejl-procent, mens netop de vigtige ord
   (sygdomsnavne, personnavne) alligevel forsvinder — og det er jo dem,
   der betyder noget for at kunne finde fx "mæslinger i hjemmet" senere.
   Dette er IKKE fuld genkendelse af udsagn (det er stadig ude af scope,
   som lead har bestemt) — bare en billig kontrol: tag en liste af
   hyppige, vigtige ord fra facit, og tæl hvor mange af dem der er med i
   modellens svar. Kan være én ekstra kolonne i stage 03's måleværktøj.
6. **Det "fjerne bogstav-gennemskin" er nok et spørgsmål om selve bogens
   tilstand, ikke kun noget vi kan rette med billedbehandling.** De
   kendte forklaringer på, at man kan se lidt af en anden side gennem
   papiret, dækker kun forsiden og bagsiden af DET SAMME blad — ikke en
   side der ligger 50 sider væk. Den mest sandsynlige forklaring er, at
   bogen på et tidspunkt er blevet indbundet om, eller at løse blade er
   havnet forkert — eller at der er en gammel afsmitning fra dengang
   siderne engang lå ved siden af hinanden. Værd at spørge arkivet direkte
   om netop det binds historie, hvis det viser sig at ske ofte. Det er
   ikke noget, vi kan garantere en instruktion i prompten løser alene.

## Detaljeret grundlag pr. spørgsmål

### 1. Overstregninger

Ingen kilde giver en færdigtestet opskrift for sprogmodeller specifikt.
Tættest på emnet er et studie fra 2026 [1], som handler om klassiske,
ikke-sprogmodel-baserede genkendelsesprogrammer — det bekræfter, at
overstregninger er en reel og ikke triviel fejlkilde, og at både at vise
modellen flere eksempler og at rense billedet for overstregningen på
forhånd hjælper. Transkribus' egen vejledning advarer mod at markere
overstregninger inkonsistent, fordi det forvirrer træningen. Et andet
studie [7] noterer, at 23,6 % af siderne i deres testmateriale havde
overstregninger, men måler ikke selv, hvor godt de blev håndteret.

### 2. Nordisk/dansk arbejde med sprogmodeller på håndskrift

Dækningen er tynd. Sveriges Riksarkiv har det mest udviklede program
("Transkriberingsnod Sverige"), men det bygger på specialtrænede
programmer ("Swedish Lion", ca. 95 % træfsikkerhed på 1600-1900-tals
håndskrift) — IKKE på almindelige sprogmodeller som Gemini/Claude/GPT.
Intet fundet for det danske Rigsarkiv ud over frivillig
afskriverhjælp. Et studie med medforfatter fra Københavns Universitet [4]
dækker engelsk, fransk, tysk og italiensk — IKKE dansk — og finder
generelt, at sprogmodeller er dårligere til andre sprog end engelsk. Den
mest direkte relevante kilde er dansk-forsøget nævnt i anbefaling 3 —
men det er én persons erfaring, ikke en videnskabelig undersøgelse.

### 3. Andre kvalitetsmål end tegnfejl

Dette er reelt nyt stof, som hverken Humphries eller vores eget StadsCER
dækker. Et studie [8] viser, at selv små fejl i navne/vigtige ord giver
store fald i, om man senere kan FINDE dokumentet ved søgning — mens den
almindelige tegnfejl-procent knap rører sig. Flere studier om
navnegenkendelse viser samme mønster og anbefaler at måle det for sig.
Et studie fra oktober 2025 [9] bygger sine egne, opgave-specifikke mål i
stedet for kun at bruge tegnfejl — samme princip, som et klinisk
ord-mål ville være for os. Intet fundet specifikt for historiske
lægejournaler — det ville være et reelt, nyt bidrag fra vores side.

### 4. Fjernt bogstav-gennemskin fra andre opslag

Se anbefaling 6. De kendte fagudtryk for "man kan se lidt igennem
papiret" dækker udelukkende forsiden og bagsiden af SAMME blad — aldrig
en helt anden side. Den mørke skygge nær selve bogryggen (et andet kendt
fænomen) skyldes lysforhold, ikke indhold, der skinner igennem. Ingen
kilde beskriver eller navngiver det præcise fænomen, lead så. Den mest
sandsynlige forklaring er ikke fra en kilde, men agentens egen tekniske
overvejelse: bogen er formentlig blevet fysisk forstyrret på et tidspunkt
(indbundet om, blade taget ud og sat forkert i igen), eller der er tale
om en gammel afsmitning fra dengang to sider engang lå op ad hinanden.

### 5. Andre benchmarks end Humphries, og billedskarphed

Se anbefaling 1-2. Konkrete tal: Claude fik 41 % og GPT-4o omkring 60 %
tegnfejl på et andet historisk testsæt [2] — et stort spring fra
Humphries' 5-7 %. Et stort studie fra september 2025 [10] (155 historiske
samlinger, tekster fra 22 århundreder) fandt Gemini som den bedste af de
almindeligt tilgængelige modeller, men håndskrift klarer sig markant
dårligere end trykt tekst på tværs af alle sprog i den undersøgelse.
Skarphedsfundet [5]: sprogmodeller læser lige så godt som klassisk
tekstgenkendelse ved høj skarphed, men klart dårligere under en vis
grænse. Ingen af kilderne har testet dansk eller andet skandinavisk med
konkrete tal.

## Kilder

1. "A study of handwritten text recognition with cross-out words", IJDAR/Springer, 2026
2. Crosilla, Klic & Colavizza, "Benchmarking Large Language Models for Handwritten Text Recognition", arXiv 2503.15195, 2025
3. Per Hundevad Andersen, dansk hands-on-sammenligning af ChatGPT/Transkribus/Gemini, blog.slaegtsbibliotek.dk, februar 2026
4. Samme studie som kilde 2 (Crosilla, Klic & Colavizza, medforfatter fra Københavns Universitet)
5. "Context-Independent OCR with Multimodal LLMs: Effects of Image Resolution and Visual Complexity", arXiv 2503.23667, 2025
6. Se kilde 8 og tilhørende navnegenkendelses-studier
7. "Judge a Book by Its Cover", arXiv 2502.20295
8. "Beyond CER and WER: How Does OCR Really Impact Information Retrieval?", JCDL, 2025
9. "Evaluating LLMs for Historical Document OCR: A Methodological Framework for Digital Humanities", arXiv 2510.06743, oktober 2025
10. CHURRO, arXiv 2509.19768, september 2025
