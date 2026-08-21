# StadsCER — eget forarbejde på måling af transskriptionskvalitet

Repo: `J-Hoffi/StadsCER` (privat, tilgået via `gh` 2026-08-18; ikke klonet
lokalt). Aktivt værktøj, ikke en skitse: Python med uv, pakken `stadscer/`,
tests, `CONTEXT.md` som beslutningslog, `docs/CER-forklaring.md` til
ikke-tekniske læsere, og en `diary/` med daterede sessionsnotater. Sidste
commit 2026-08-17. Måler Transkribus-genkendelse mod 32 korrekturlæste
hospitalsprotokolsider fra Stadsarkivet.

## Målekoden — genbruges direkte

`stadscer/cer.py`:

```python
def normalize(text, ignore_case=False, ignore_diacritics=False,
              ignore_punctuation=False, collapse_whitespace=True) -> str:
```

- Mellemrum slås altid sammen.
- `strip_diacritics()` **bevarer æ/ø/å** som rigtige danske bogstaver, men
  folder de tyske varianter skriverne og modellen bruger i flæng:
  **ö→ø, ä→æ, ü→y**. Dokumenteret som den hyppigste tegnforveksling
  overhovedet (131 tilfælde på 50 sider).
- Tegnsætning er en fast bogstavelig liste, bevidst ikke Unicode-kategori,
  "så resultatet ikke flytter sig med Python-versionen".

Fem varianter rapporteres altid side om side, aldrig kirsebærplukket:

```python
VARIANTS = {"raa": {}, "uden_versaler": {...}, "uden_diakritika": {...},
            "uden_tegnsaetning": {...}, "lempeligst": {...}}
```

Også genbrugeligt: `levenshtein`, `align` med deterministisk valg ved
uafgjort (diagonal → slet → indsæt), `Metrics`, `compare_all_variants`, og
rapportlayoutet.

## Kendt mangel, som 2ndpage skal bygge

**Orddeling hen over linjeskift samles ikke.** `diary/2026-08-17.md` udpeger
netop det (`Indqvarte-` → `Indqvarte.`) som det dominerende systematiske
fejlmønster, markeret som løsbart men uimplementeret. Vi bygger det i stage
03 og kan give det tilbage.

Heller ikke håndteret: `[?]`, overstreget tekst med erstatning,
understregningsnoter, positionsmærker. StadsCERs facit bruger dem ikke.
Kun `[underskrift]` er særbehandlet: linjer med det mærke droppes helt fra
målingen, fordi mærket dækker en ukendt mængde ulæst tekst.

## Linjeforankring — hvorfor StadsCER gør det modsatte af os

`diary/2026-08-17.md` argumenterer:

> "Linjeforankret sammenligning er derfor ikke en tilnærmelse, vi lever med;
> det er den rigtige måling... Står der `tilste` i linjebilledet, og skriver
> modellen `tilstedes`, mens `des` står på næste linje, ville sammenskrevet
> tekst udligne det til nul fejl — og dermed belønne modellen for at
> hallucinere præcis rigtigt."

**Forudsætningen holder muligvis ikke hos os — men det er UBEVIST.**
StadsCER fodrer modellen med facits egne baselines, så linjeopdelingen er ens
per konstruktion. Hos os får modellen hele siden, og *hvis* den laver sine egne
linjeskift, falder linjeforankringen fra hinanden.

**Vi har ikke målt det.** Sætningen "vores model laver sine egne linjeskift"
stod her oprindeligt som en kendsgerning; den var en formodning (påpeget af
Lead 2026-08-21). Der findes ét modsatrettet holdepunkt: kollegaens app har
allerede et linje-for-linje-skema (`TextPage`/`PageLine` med nummererede
linjer, se `patientjournals_repo.md`), altså har nogen bygget ud fra, at det
kan lade sig gøre — men om det nogensinde er afprøvet, står som et åbent
punkt i stage 08.

Konsekvensen for stage 03: måleapparatet må **ikke afhænge af svaret**. Vi
måler på den fladede strøm OG pr. linje via parring. Er modellen linjetro,
er parringen et nulled og koster intet; er den ikke, redder den målingen.
Hallucinationskontrollen bygges i begge tilfælde, så den ikke kræver
identisk linjeopdeling — netop for at fange den fejl, citatet advarer imod.

`errors.py` har allerede hallucinationsdetektion (fuldførelser, forskudte
brud, tabte begyndelser, opfindelser hen over linjegrænser), men forudsætter
matchende linje-id'er på begge sider og skal tilpasses.

## Målte fund derfra

- Underskriftslinjer med/uden: CER 2,31 % mod 1,34 % på én side (0,97
  procentpoints udsving fra fem linjer) — derfor rapporteres begge tal.
- ø/ö er den hyppigste enkeltforveksling.
- Tegnsætning er den dominerende fejlkategori, med orddeling som førende
  undermønster.
- `tests/test_determinisme.py` vogter mod, at mængde-iteration flytter
  rapportens rækkefølge mellem kørsler — samme fejlklasse som noteret i
  brugerens egen memory om reproducerbarhed.
