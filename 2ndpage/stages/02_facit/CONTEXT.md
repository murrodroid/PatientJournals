# Stage 02 — Facit

## Formål

Omsætte de håndlavede transskriptioner til et rent, entydigt facit pr. opslag,
og dele materialet i en øvemængde og en låst prøvemængde. Dette er det sted,
hvor en misforståelse er dyrest: bygges måleapparatet oven på en forkert
læsning af klammenotationen, måler alt bagefter forkert uden at nogen opdager
det.

## Inputs

| Type | Sti |
|---|---|
| Testregler | `../../_config/tdd.md` |
| Opslagsregister | `../01_datagrundlag/output/opslagsregister.csv` |
| Konventionsbeskrivelse | `../../references/manuelle_transskriptioner.md` |
| Facit-filer | `...\Patient journals\Manual transcriptions` |

## Process

1. Læs RTF-filerne og afkod dansk tegnsætning korrekt (filerne er
   cp1252-baserede med `\'e6`-escapes).
2. Del hver fil i `[page]`-blokke, og knyt hver blok til et billed-id.
3. Kortlæg klammekonventionerne **udtømmende** ved at udtrække alle
   forekomster af klammer i hele materialet og gruppere dem. Kortlægningen
   skrives ned, før parseren skrives færdig — vi bygger ikke på de otte filer,
   der er stikprøvelæst.
4. Udled en ren læsetekst pr. opslag efter disse regler:
   overstreget tekst fjernes, den erstattende tekst beholdes; noter om
   understregning og placering fjernes; `[?]` bevares som et
   ulæselighedsmærke; et gæt som `[dygtig?]` reduceres til ordet selv.
5. Bevar linjeskiftene i facit, men gem også en fladet udgave, hvor orddeling
   hen over linjeskift er samlet.
6. Undersøg efterstillede `[page]`-mærker uden tekst: blank side eller
   uskrevet transskription. Mærk dem, brug dem ikke som facit.
7. Del i øvemængde og prøvemængde **pr. patient**, aldrig pr. opslag, så
   sider fra samme forløb ikke havner på begge sider. Prøvemængden låses og
   røres ikke, før en metode skal endeligt bedømmes.

## Outputs

| Fil | Beskrivelse |
|---|---|
| `output/facit.jsonl` | Én række pr. opslag: billed-id, rå blok, ren læsetekst med linjer, fladet tekst |
| `output/klammekonventioner.md` | Fuld optælling af alle klammeformer i materialet, med eksempler |
| `output/opdeling.csv` | Hvilke patienter der hører til øvemængde og til den låste prøvemængde |
| `output/udeladte.md` | Blokke uden brugbar tekst, og hvorfor de er udeladt |

## Test Contract

Parseren skal testes på konstruerede eksempler, der hver fremkalder én bestemt
konvention: overstreget med erstatning, overstreget uden erstatning, stablede
`[?]`, gæt med spørgsmålstegn inde i klammen, understregning af hel linje og
af et citat, positionsmærke med håndskrevet linjeskift indeni, orddeling hen
over linjeskift, og en tastefejl i et mærke. Hver test skal være set fejle,
før den regnes som gyldig. Desuden en kontrakttest på, at hvert billed-id i
facit findes i opslagsregistret.

## Handoff

Næste stage er `03_maaleapparat`. Reviewed betyder, at du som historiker har
læst klammekortlægningen igennem og bekræftet, at den rene læsetekst er den
tekst, du ville regne for korrekt.
