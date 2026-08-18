# Manuelle transskriptioner (ground truth)

Kortlagt 2026-08-18. Rod:
`<kilderod>\PID-scapes and Blegdam Patient journals\Patient journals\Manual transcriptions`

41 filer i alt (~235 KB), alle læsbare trods OneDrive-offline-attribut:
**39 RTF** (Apple TextEdit/Cocoa-RTF, cp1252, danske tegn som `\'e6`-escapes)
+ **2 xlsx**.

```
Manual transcriptions\
├── patient_journals_89-97_indtake_death_total.xlsx   (måneds-oversigt 1889-97:
│     kolonner png_file, month, intake, death, sum, kommentar — 108 rækker)
├── 273104_001636 <patientnavn>\273104_001636 - full journal.rtf
└── Deaths 1896-97\
    ├── trial_frontpages_death_22.12.25.xlsx  (108 rækker struktureret
    │     forside-metadata pr. dødsfald: afdeling, navn, alder, erhverv,
    │     civilstand, adresse, datoer, diagnoser, serum, obduktion m.m.;
    │     nøgle = side-id, samme som RTF-filnavne)
    └── 16 månedsmapper "01 May 1896" … "16 August 1897"
        └── 38 × "<id>_<side>_full_journal.rtf" (1,7-10,2 KB)
```

## Kritisk strukturelt fund: filerne er HELE indlæggelser, ikke enkeltsider

Navnemønster `{bind-id}_{side-id}_full_journal.rtf`, hvor side-id'et er
**forsiden**. Hver fil starter med
`[transcription of frontpage 273098_001471 - full journal]` og indeholder
derefter alle fortsættelsessider markeret `[page 273098_001472]`,
`[page 273098_001473]` … **Andensiden = første `[page]`-blok efter
forside-markøren** (typisk forside-id + 1). Nogle afsluttende side-id'er står
uden tekst efter sig — ambiguøst om siden var blank eller bare ikke
transskriberet.

## Opmærkningskonventioner (verbatim-eksempler)

| Fænomen | Notation | Eksempel |
|---|---|---|
| Ulæseligt | bare `[?]`, kan stables | "og Canylen [?][?] i Trachea." |
| Usikkert gæt | ord + `?` i klamme | "Stemmen [dygtig?] hæs." |
| Understregning, hel linje | efterstillet note | "Rask indtil for 8 Dage siden [this line is underlined]" |
| Understregning, del af linje | citat i anførselstegn + note | "[»gullig Belægning« is underlined]" |
| Indskud over/under linjen | `[added over line]…[continued on line]` / `[added under line]` | "[added over line]under affekt [continued on line]<middelstærke Ind-" |
| Overstreget | `[crossed out]…[written instead]…` | "[crossed out]En Del [written instead]Lidt at Drikke" |
| Marginaltekst | positions-tag + evt. håndskrevet `\n` | "[right side of page]Rp. Damp\nLincet. expect." |
| Daglige notater | dato + temperaturer + narrativ | "9/5 [top page left]39,5-38,2 [top page right]Kultur 8/5 fauces: DB" |
| Fortsættelses-header | patientnavn + Cont-nr (inkonsistent) | "<patientnavn> Cont II", "<patientnavn> Cmt. I" |

## Konsistens-problemer (bestemmer normaliseringslaget i eval)

- Positions-tags er fritekst: mindst 8 varianter (right side of page, bottom
  right side, top page left, top left corner, top mid page, mid page, …).
- Tastefejl i tags forekommer ("is underline]", "journall").
- `[?]` er overloadet: ulæseligt vs. `gæt?` er to forskellige signaler.
- To RTF-afsnitsformater: "01 May 1896" bruger ét `\pard`-blok for hele filen;
  fra "02 June 1896" og frem gentages `\pard\tx566…` pr. afsnit (kun kosmetisk,
  men rammer RTF-stripning).
- Kun 8 af 39 RTF'er blev stikprøvelæst — flere tag-varianter kan findes.

## Kobling til scanninger + nabo-mapper

Ingen billeder i mappen; kobling udelukkende via de numeriske id'er (matcher
masterlistens `image_name` og xlsx-kolonnen `png_file` — originalscanningerne
er PNG et andet sted). Nabo-mapper på samme niveau, ikke undersøgt endnu:
`1895`, `Diphtheria and Croup`, `Automatic transcription versions`, `Meta data`,
`Geocoding`, `Transcription codebooks` (indeholder `Front page codebook
v1/v2.docx` — mulig formel opmærknings-spec), `validation data`.
