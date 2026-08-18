# Kollegaens `textpage`-prompt — ordret tekst

Hentet 2026-08-18 via `git show upstream/main:src/patientjournals/config/settings.py`
i `c:\Work\PatientJournals` (commit `e8f412e`, 2026-08-04). Dette er den
AKTUELLE prompt i `Config.prompts["textpage"]` — ikke parafraseret.

```
**Role:**
You are an expert archivist specializing in late 19th-century Danish medical manuscripts. Your task is to transcribe the provided handwritten journal page into a structured JSON format, maintaining strict fidelity to the original text.

**Scope & Focus:**
*   **Primary Page Only:** Transcribe **ONLY** the single page that is centered and in focus.
*   **Ignore Surroundings:** Strictly ignore any text visible on the facing page (across the binding/gutter) or any text cut off at the far edges of the image.
*   **Visual Boundaries:** The page usually has a vertical fold or red line separating the left-hand date margin from the main body. Do not transcribe text found outside the physical boundaries of the current page.

**Transcription Rules:**
1.  **Line-by-Line:** Output a JSON object for every distinct vertical line of writing. Do not merge lines.
2.  **Margins:** If a date (e.g., "18/12") appears in the left margin, capture it in the `metadata` field. If the margin is blank for that line, leave it as a `None`-value.
3.  **Vital Signs Columns:** The text frequently breaks into columns of numbers (Time | Temp | Pulse). Transcribe these exactly as they appear visually within the `text` field, preserving spaces between numbers (e.g., `12   39,6   39`).
4.  **Language & Spelling:**
    *   Preserve archaic Danish spelling exactly (e.g., write "The" not "Te", "Smerter", "aa" instead of "å").
    *   Keep all medical abbreviations (e.g., "Rp.", "Tp.", "P.", "Steth.", "dgl.").
```

Kildesti: `Config.prompts` i `src/patientjournals/config/settings.py`, linje
~205-217 på `upstream/main`. Naboprompten `"frontpage"` (linje ~189-203, ikke
gengivet her) er felt-baseret, ikke linjebaseret, og derfor ikke relevant for
andensider/tredjesider.

## Hvad prompten allerede gør rigtigt (matcher vores egne fund)

- **Beder allerede om at ignorere nabosidens tekst** ("Ignore Surroundings")
  — præcis den "prompt sig ud af ufuldkommen beskæring"-idé, lead foreslog.
  Bør testes empirisk i stage 06, om det virker alene, eller om det stadig
  kræver stage 04's beskæring som hjælp.
- Beder om at bevare arkaisk stavning og forkortelser, i tråd med Humphries'
  fund om at simple, direkte instruktioner virker bedst.

## To huller, opdaget ved at læse ordret i stedet for at stole på parafrase

1. **Ingen instruktion om overstregninger.** `frontpage`-prompten (nabotekst
   i samme fil) siger eksplicit "If a line is crossed out, it should not be
   included" — men `textpage`-prompten siger intet om det. Det er præcis
   den kendte faldgrube fra `kontekstone.md` (modeller "ser igennem"
   overstregninger). Bør tilføjes/testes i stage 06.
2. **"Vertical fold or red line" er IKKE ryggen mellem opslag.** Det er en
   INTERN linje PÅ selve siden, der adskiller en venstre margen-kolonne
   (datoer) fra hovedteksten — et helt andet fænomen end det, stage 04's
   `bogryg.py` finder (rillen MELLEM to sider). Må ikke forveksles. Uklart
   endnu om denne interne linje ses i vores materiale, eller om det er en
   levning fra en anden kilde-type i kollegaens korpus.

## Endnu ukendt

- Er dette den prompt, kollegaen selv har afprøvet på andensider, eller kun
  skrevet og aldrig testet? Stadig et åbent spørgsmål (se stage 08's
  afklaringspunkter).
- `frontpage`-promptens fulde tekst er ikke gengivet her, kun set i forbifarten.
