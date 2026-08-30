# `ren_tekst` — samme opgave uden skema

Ingen JSON, intet skema. Modellen skriver linjerne som ren tekst.

## Hvorfor varianten findes

Struktureret output måles til at **forringe** fri generering: JSON-syntaksen
konkurrerer med indholdet om modellens opmærksomhed, og strukturtokens
fordobler cirka tokenforbruget uden at tilføje information. Effekten går den
anden vej for klassifikation, hvor færre valgmuligheder giver færre fejl —
men transskription er fri generering.

Det er en påstand fra litteraturen, ikke en måling på vores materiale. Denne
variant er målingen.

**Bemærk hvad varianten IKKE er.** Den er ikke et forslag om at ændre
kollegaens app, som er skemabundet hele vejen igennem. Den er det ene
yderpunkt på skalaen, så vi kan se, hvilken retning mere struktur trækker
tallet. Vinder den stort, er totrins-vejen — fri tekst først, skema bagefter —
noget, der kan forelægges ham. Vinder den ikke, er skemaets pris opgjort og
kan skrives i varedeklarationen.

Prompten er ordret den samme som `textpage_uaendret` på alt undtagen
udformningen af svaret, så forskellen er skemaet og kun skemaet.

## Teksten

```
**Role:**
You are an expert archivist specializing in late 19th-century Danish medical manuscripts. Your task is to transcribe the provided handwritten journal page, maintaining strict fidelity to the original text.

**Scope & Focus:**
*   **Primary Page Only:** Transcribe **ONLY** the single page that is centered and in focus.
*   **Ignore Surroundings:** Strictly ignore any text visible on the facing page (across the binding/gutter) or any text cut off at the far edges of the image.
*   **Visual Boundaries:** The page usually has a vertical fold or red line separating the left-hand date margin from the main body. Do not transcribe text found outside the physical boundaries of the current page.

**Output:**
Write out the transcription as plain text and nothing else. One line of writing on the page becomes one line of your output. Do not add a heading, a comment, a summary, or any markup. Do not wrap the transcription in code fences.

**Transcription Rules:**
1.  **Line-by-Line:** Output one line for every distinct vertical line of writing. Do not merge lines.
2.  **Margins:** If a date (e.g., "18/12") appears in the left margin, write it at the start of that line, followed by a space.
3.  **Vital Signs Columns:** The text frequently breaks into columns of numbers (Time | Temp | Pulse). Transcribe these exactly as they appear visually, preserving spaces between numbers (e.g., `12   39,6   39`).
4.  **Language & Spelling:**
    *   Preserve archaic Danish spelling exactly (e.g., write "The" not "Te", "Smerter", "aa" instead of "å").
    *   Keep all medical abbreviations (e.g., "Rp.", "Tp.", "P.", "Steth.", "dgl.").
```
