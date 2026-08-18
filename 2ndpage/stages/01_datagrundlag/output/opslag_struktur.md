# Opslagsstruktur — stage 01 (2026-08-18)

## Konklusion

Hvert billede er **asymmetrisk beskåret om én målside**, som fylder ca.
75-85% af rammen, med kun en smal strimmel af naboopslaget synlig i den
ene kant. Det er IKKE et symmetrisk dobbeltopslag, som det først lignede
ud fra det allerførste forsidebillede.

## Recto/verso-reglen (bekræftet af lead, låst)

Journalerne var oprindeligt løse blade, senere indbundet — og indbinding
af foldede blade starter altid på en enkelt recto. Derfor:

| `patient_page_counter` | Rolle | Recto/verso | Hovedindhold i | Strimmel i |
|---|---|---|---|---|
| 0 | Forside | recto | højre | venstre |
| 1 | Andenside | verso | venstre | højre |
| 2 | Tredjeside | recto | højre | venstre |
| 3 | (fjerdeside, ude af scope) | verso | venstre | højre |

Reglen er implementeret i `src/andenside/masterlist.py` (`Side.recto_verso`,
`Side.rolle`) og testet i `tests/test_masterlist.py`, inkl. en
regressionstest der låser den strengt alternerende paritet.

## Verifikation mod facit

Krydstjekket ord for ord mod facit-RTF'erne på fem billeder, alle stemte:
`273098_001471` (forside), `273098_001496`/`_001508` (andensider),
`273099_001361`/`_001363` (tredjesider), `273099_001362` (andenside).
Se rod-`CONTEXT.md` for de fulde citater.

## Konsekvens for stage 04

Sidevalget (hvilken kant der skal beskæres) kræver ingen billedanalyse —
det er entydigt afledt af `patient_page_counter`s paritet. Det der
mangler at blive løst, er selve SNITPUNKTET inden for den kendte kant
(hvor strimlen holder op og hovedsiden begynder), ikke hvilken kant.

## Forsideopslag — anden billedtype, ude af scope

Forsideopslag ser bredere og mere symmetriske ud (synlig blank venstre
side = forrige patients ubrugte rest). Da forsider er ude af scope for
dette projekt, er dette ikke undersøgt yderligere.
