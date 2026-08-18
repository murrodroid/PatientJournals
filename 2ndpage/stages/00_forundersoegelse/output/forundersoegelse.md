# Forundersøgelse — samlet notat (2026-08-18)

Fem spørgsmål, hver undersøgt af en selvstændig agent med webadgang. Hver
påstand herunder har kilde + år; se `aabne_spoergsmaal.md` for det, ingen
kilde svarer på. Anbefalinger står for sig, adskilt fra referat.

## Anbefalinger, der kan handles på

1. **Behandl opløsning som en førsteklasses eksperimentakse i stage 05/06,
   ikke en bagkant-detalje.** Et 2025-studie (arXiv 2503.23667) måler, at
   LLM-OCR forringes markant under ~150 ppi. Vores billeder er ~900-1.600
   px brede for en fysisk side, hvilket sandsynligvis lander i eller under
   den zone. Skaf om muligt et konkret pixel/DPI-tal for de rigtige
   billeder tidligt, og test eksplicit om opløsning er en begrænsende
   faktor, før noget konkluderes om model- eller promptvalg.
2. **Stol ikke på Humphries' optimistiske CER-tal (5-7 % for historisk
   engelsk) som en generel baseline for vores materiale.** Et uafhængigt
   studie (arXiv 2503.15195) fandt Claude 3.5 Sonnet på 41 % CER og GPT-4o
   på ~60 % CER på et andet historisk datasæt (ICDAR2017) — samme
   modelfamilier, vidt forskellige tal. Forskellen skyldes formentlig
   datasættets sværhedsgrad/skriftvariation, men gabet er stort nok til, at
   vores egen første måling (stage 05) skal tages alvorligt, uanset hvad
   litteraturen ellers siger.
3. **Behandl Gemini/Claudes evne på dansk gotisk/kursiv håndskrift som
   uverificeret, ikke som en selvfølge.** En dansk hands-on test (Per
   Hundevad Andersen, feb. 2026) fandt, at ChatGPT direkte hallucinerede
   indhold på et dansk 1844-dokument, Gemini klarede sig bedre men fejlede
   stadig på egennavne, og Transkribus' specialiserede modeller slog begge
   LLM'er. Et KU-medforfattet studie (arXiv 2503.15195) bekræfter generelt,
   at LLM'er underpræsterer på ikke-engelsk historisk tekst. Konsekvens:
   hold Transkribus/specialiserede HTR-modeller som reel sammenligning i
   stage 06, ikke kun som en fjern mulighed.
4. **Tilføj en overstregnings-instruktion til `textpage`-prompten, men vid
   at vi er de første til at teste den på en LLM.** Litteraturen om
   LLM-håndtering af overstreget tekst er tynd til ikke-eksisterende — intet
   fundet studie isolerer dette for GPT/Claude/Gemini. Kollegaens egen
   søsterprompt (`frontpage`) har allerede en fungerende formulering for
   samme korpus ("if a line is crossed out, it should not be included") —
   billigt at genbruge, men skal måles empirisk af os, ikke antages at virke.
5. **Overvej et let, sekundært nøgleords-overlevelsesmål ved siden af
   CER/WER.** Flere 2025-kilder (JCDR 2025, NER-studier) viser, at CER/WER
   kan stå stille, mens navngivne enheder/nøgleord forsvinder — præcis det,
   der ville skjule om "mæslinger i hjemmet" overlever. Dette er IKKE fuld
   NLP-detektion (som er ude af scope, jf. Lead), men et billigt,
   diagnostisk supplement: en liste af hyppige kliniske ord fra facit, talt
   op i modelsvar. Kan ligge som en ekstra kolonne i stage 03's
   måleapparat, ikke som en ny stor komponent.
6. **Genovervej "fjernt bleed"-fænomenet som et arkivspørgsmål, ikke kun et
   billedbehandlingsproblem.** Standardbegreberne (show-through,
   bleed-through, gutter shadow) dækker alle KUN samme blads for-/bagside —
   ingen dokumenteret betegnelse findes for indhold fra en side ~50
   positioner væk. Mest sandsynlige forklaring: bindet er omindbundet eller
   har løse/omrokerede blade, eller historisk affarvning ("offsetting") fra
   dengang siderne engang var nabosider. Værd at spørge arkivet direkte om
   det pågældende binds fysiske tilstand, hvis fænomenet viser sig
   systematisk — det er ikke noget, en prompt alene kan garanteres at løse.

## Detaljeret grundlag pr. spørgsmål

### 1. Overstregninger (kilde: research-agent, overstregningshåndtering)

Ingen kilde giver en testet opskrift for LLM'er specifikt. Tættest på:
"A study of handwritten text recognition with cross-out words" (IJDAR,
Springer, 2026) — klassisk CV/HTR, ikke LLM, men bekræfter at
overstregninger er en reel, ikke-triviel fejlkilde, og at både
træningsdata-udvidelse og fjernelses-forbehandling hjælper klassiske
modeller. Transkribus' egen vejledning advarer mod inkonsistent
overstregnings-opmærkning som en kontamineringsrisiko i træningsdata.
"Judge a Book by Its Cover" (arXiv 2502.20295) noterer overstregning som
tilstede i 23,6 % af sider i deres benchmark, men måler det ikke separat.

### 2. Nordisk/dansk LLM-HTR-arbejde (kilde: research-agent, nordisk arbejde)

Dækningen er tynd. Sveriges Riksarkivet har det mest udviklede program
("Transkriberingsnod Sverige"), men bygger på specialiserede
Transkribus-lignende modeller ("Swedish Lion", ~95 % nøjagtighed på
1600-1900-tals håndskrift), IKKE generelle LLM'er. Intet fundet for
Rigsarkivet Danmark ud over frivillig crowdsourcing. Et
KU-medforfattet studie (Crosilla, Klic & Colavizza, arXiv 2503.15195,
2025) dækker engelsk/fransk/tysk/italiensk — IKKE dansk — og finder
generelt at LLM'er underpræsterer på ikke-engelsk historisk tekst. Den
mest direkte relevante kilde er en dansk hands-on-test (se anbefaling 3
ovenfor), som er anekdotisk, ikke et formelt benchmark.

### 3. Kvalitetsmål ud over CER/WER (kilde: research-agent, kvalitetsmål)

Genuint nyt materiale, ikke dækket af Humphries/StadsCER. "Beyond CER and
WER" (JCDL 2025) viser, at selv mindre navngivne-enheder-fejl giver store
fald i informationsgenfinding, mens CER/WER knap rykker. Flere
NER-studier viser F1 falde skarpt med OCR-forringelse (fx 87 %→63 %) og
anbefaler at måle det separat. "Evaluating LLMs for Historical Document
OCR" (arXiv 2510.06743, okt. 2025) bygger problem-specifikke mål
(bevarelsesrate for historiske tegn m.m.) frem for generisk CER — samme
princip som et klinisk-nøgleords-mål ville være for os. Intet fundet
specifikt for historiske kliniske journaler — det ville være et reelt,
nyt bidrag.

### 4. Fjernt bleed fra andre opslag (kilde: research-agent, bleed-through)

Se anbefaling 6. Standardbegreberne "show-through" og "bleed-through" er
veldokumenterede i scanner-/bevaringslitteraturen, men beskriver
udelukkende samme blads for- og bagside — aldrig en ikke-nabo-side.
"Gutter shadow" er et lys-/geometriartefakt, ikke et indholdsartefakt.
Ingen kilde beskriver eller navngiver det specifikke fænomen, lead så.
Mest sandsynlige forklaring (agentens egen fysiske ræsonnement, ikke en
kilde): fysisk forstyrrelse af bindet (omindbinding, løse blade) eller
historisk affarvning/"offsetting" mellem sider, der engang var nabosider.

### 5. Benchmarks ud over Humphries + opløsning (kilde: research-agent, benchmarks)

Se anbefaling 1-2. Nøgletal: Claude 3.5 Sonnet 41,19 % CER / GPT-4o ~60 %
CER på ICDAR2017 (arXiv 2503.15195) — stærk kontrast til Humphries' 5-7 %.
CHURRO (arXiv 2509.19768, sept. 2025, 155 historiske korpusser, 22
århundreder) fandt Gemini 2.5 Pro som bedste proprietære baseline, men
håndskrift halter markant efter trykt tekst på tværs af sprog (70,1 % vs.
82,3 % normaliseret similaritet). Opløsningsfundet (arXiv 2503.23667):
LLM-OCR matcher klassisk OCR ved ~300 ppi, forringes markant under ~150
ppi. Ingen af kilderne dækker dansk eller skandinavisk håndskrift
specifikt med hårde tal.
