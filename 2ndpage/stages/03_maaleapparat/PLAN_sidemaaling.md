# Plan: forankringen erstattes af én sidemåling

**Status: UDKAST. Ikke besluttet i alle detaljer, ikke påbegyndt.**
Skrevet 2026-08-30 efter en grill-session. Der grilles videre, før noget køres.

## Hvorfor

Stage 03 blev låst 2026-08-23 med den udtrykkelige betingelse: *"Genåbnes uden
tøven, hvis der viser sig et hul ved de første rigtige tal."* Hullet er der.

Målingen parrer facits linjer med modellens tekst forfra og nedefter, og
søger fremad fra sidste træf. Rammer den forkert og lander for langt nede,
rykker søgepunktet med, og alt før det punkt er derefter uden for rækkevidde.

Det skete på `273107_001864` i variant V4:

```
facit linje  1: 'Ingen Snue.'                       -> forankret ved position 564 af 617
facit linje 26: 'Tg. ikke [?] suspect ingen Snue.'
```

Modellen læste linje 1 som `Ingen Hoste.` (en almindelig læsefejl) og linje 26
som `G. ikke Rødt. ingen Snue.` (en rimelig læsning). Sætningen "ingen Snue"
står altså **to gange på siden i facit selv** — det er ikke noget modellen fandt
på. Søgningen foretrak det ordrette match i linje 26 frem for det halvdårlige i
linje 1 og smed linje 2-25 væk. **26 af 29 linjer tabt.**

Målt på alle 71 sider i de seks varianter: ét tilfælde. Men det kostede hele
siden og vendte forsøgets rangorden om — V4 gik fra dårligst til bedst, da den
side blev udeladt.

Leads holdning (2026-08-30): *"jeg vil ikke have en model der søger noget frem
længere nede, orden vs orden måling"* og *"stage 03 skal være mere konservativ
og ikke inviterende til at hoppe rundt"*.

## Hvad der skal bygges

**Én global sammenligning af hele siden, i rækkefølge, uden søgning.**
Facits fulde tekst mod modellens fulde tekst, fra top til bund, i ét stræk.

Den eneste afvigelse fra en almindelig editeringsafstand: **hvor facit siger
`[?]`, må modellen skrive hvad som helst uden at det koster.** Det var den
eneste grund til at forankre — 11,5 % af facits linjer har et sted,
transskribenten ikke kunne læse, og modellen må ikke straffes for at gætte dér.

Følger af det:
- Der er kun én vej gennem siden. Et gentaget ord kan ikke flytte noget.
- **"Dækning" og "uden rabat" forsvinder som begreber.** Hele siden er altid
  målt. Dermed forsvinder også den fælde, der fik den forkerte konklusion
  igennem to gange på én dag.
- Gabene (hvad modellen skrev på et `[?]`) falder stadig ud af målingen, og
  mere direkte end før: det er præcis den modeltekst, der blev stillet op mod
  et jokertegn.

## Trin

Hvert trin kan kommenteres for sig. Rækkefølgen er bindende — trin 2 kan ikke
prøves af uden trin 1.

### Trin 1 — måleoperationen alene, uden at røre noget bestående

Ny funktion i `cer.py` eller et nyt modul: editeringsafstand mellem to
strenge, hvor facit-siden kan indeholde jokermærker der matcher vilkårligt
mange tegn til pris nul. Returnerer afstand, facit-længde, og hvad der lå i
hvert joker-felt.

Testes for sig mod konstruerede tilfælde, herunder netop det gentagne ord fra
`273107_001864`. **Testen på det tilfælde skal ses fejle mod den nuværende
`forankr()`.**

Rører intet bestående. Kan kasseres uden følger.

### Trin 2 — kør begge målinger på dagens seks gemte kørsler

Ingen modelkald; svarene ligger gemt. Sammenlign de to måleapparaters tal og
rangorden.

Formålet er ikke at vælge — valget er truffet — men at vide **hvor meget** de er
uenige, før de gamle tal kasseres. Er de enige overalt undtagen på den ene
side, er det et resultat i sig selv.

### Trin 3 — indsæt den nye måling i `maal.py`

`forankr()` fjernes. `SideMaaling` og `SaetMaaling` mister felterne om
dækning og forankring. Det er en ægte forenkling — de fleste af de nuværende
felter findes kun for at beskrive, hvor meget forankringen tabte.

Skal bevares, fordi de er kontraktbundne eller efterspurgte:
- **Gabene** (`skriv_gab`) — kontrakt fra rod-CONTEXT 2026-08-21.
- **Linjetrofastheden** — svaret på beslutning 35, og forudsætning for at
  aflevere `PageLine`-poster videre.
- **De seks varianter** (`raa` … `lempeligst`) — beslutning 26, alle
  rapporteres side om side.

### Trin 4 — rapporten skrives om

Afsnittene om dækning og om den strenge måling skal enten udgå eller omskrives
(se åbent spørgsmål 2). Afsnittet "Sådan er der målt" skal beskrive den nye
metode i almindeligt sprog.

### Trin 5 — selvtesten køres om

`scripts/selvtest_maaleapparat.py` ødelægger facit med vilje og måler, hvor
meget apparatet finder igen. Skævheden var 93,1 % med forankring. Den skal
måles på ny, og forvanskningerne skal udvides med **gentagne ord** — det er
den fejltype, der væltede den gamle måling, og selvtesten indeholdt den ikke.

Tager omkring ni minutter. Ingen modelkald.

### Trin 6 — dagens seks varianter måles forfra

Gratis; svarene ligger gemt. Først dér ved vi, hvad wordpicking-forsøget
faktisk viste.

## Åbne spørgsmål — skal afklares FØR trin 1

1. **Hvor meget må et joker-felt sluge?** Et `[?]` står typisk for ét ord. Må
   modellen skrive 200 tegn dér uden at det koster, har den frit lejde til at
   digte. Skal der være et loft, og hvad skal det være?

2. **Overlever den strenge måling (beslutning 44)?** Lead bad 2026-08-23 om et
   tal, hvor linjer med `[?]` slet ikke er med. Under en sidemåling kan det
   stadig laves — men det var indført som et værn mod netop forankringens
   rabat, og den findes ikke længere.

3. **Tegn eller ord som hovedtal?** Leads ord: *"ord til ord eller
   linjesammensætning til linjesammensætning"*. Hovedtallet har hidtil været
   tegnfejl (CER). Skal det forblive det?

4. **Hvad med omrokerede linjer?** Lead: *"vi skal tillade omrokering af hele
   linjer eller rækker af ord hvis de tydeligt bare er blevet sat i forkert
   rækkefølge, men det er svært at implementere ordentligt."*
   Foreløbig anbefaling: **byg det ikke ind i tallet.** Rækkefølgen bærer
   betydning i en journal og skal videre til `PageLine`. Opgør i stedet
   omrokering som sit eget tal ved siden af, så det kan ses frem for at blive
   tilgivet stiltiende.

5. **Hvad sker der med de eksisterende tal i `output/`?** Selvtestens tal og
   rapportformatet i stage 03's output bygger på den gamle måling. Skal de
   genberegnes, arkiveres med en note, eller begge dele?

## Hvad der IKKE er i planen

- Ingen ændring i facit (stage 02 er låst og urørt).
- Ingen modelkald. Alle trin kan køres på gemte svar.
- Ingen ændring i prøvemængdens status. Den er stadig låst.
