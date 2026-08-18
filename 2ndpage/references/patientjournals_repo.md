# PatientJournals-repoet (forside-pipelinen + kollegaens app)

Kortlagt 2026-08-18 i `c:\Work\PatientJournals`. Vigtig tidslinje:
- Lokal klon: `main` @ `4102f46` (2026-03-11) — **stærkt forældet**.
- `origin` = `J-Hoffi/PatientJournals` (leads fork; har også `origin/patch-1`
  med stor omskrivning). `upstream` = `murrodroid/PatientJournals`.
- `upstream/main` @ `e8f412e` (2026-08-04, efter `git fetch` 2026-08-18):
  fuld pakke-omskrivning `src/patientjournals/` inkl. **web-appen** under
  `src/patientjournals/app/` (dashboard.py, web.py, ui.py, jobs.py, leaderboard
  m.m., bygget maj-aug 2026). Appens plug-in-kontrakt: se
  `app_interface_upstream.md`.

## Arkitektur i den lokale `main` (marts-versionen — historisk reference)

- `config.py` — én `Config`-dataclass; `model = "gemini-3-pro-preview"` (l. 9);
  `prompts`-dict med navngivne prompts (l. 44-80); `output_model`/
  `input_prompt_name` vælger aktiv skema+prompt pr. kørsel; `image_settings`
  (resize/crop/kontrast); `fp_mode` filtrerer `_fp`-suffiksede mapper.
- `main.py` — async entrypoint; én asyncio-task pr. billede
  (semafor-begrænset), streamer rækker til `runs/<ts>/`, `--continue-dataset`
  til genoptag.
- `generate.py` — `client.aio.models.generate_content(...)` med
  `response_mime_type: application/json` + `response_json_schema`; retry med
  klassifikation af retryable/fatale API-fejl.
- `preprocess.py` — Pillow: RGB → resize til `max_dim` → margin-crop →
  kontrast → bytes+mime.
- `batch_submit.py`/`batch_retrieve.py` — ægte Gemini Batch API-spor (File API
  upload + `client.aio.batches.create`; retrieve poller og parser som generate).
- `schemas.py` — Pydantic. `Journal` = forside-skemaet (felt-for-felt, lange
  naturligt-sprogs `description=` som prompt-vejledning). **`TextPage`/`PageLine`
  (l. 134-151) = linje-for-linje-skemaet**: `page_lines: List[PageLine]`, hver
  med `text`, `metadata` (margindatoer/-tal), autonummereret `page_line_number`.
- `output_handler.py` — dispatch-tabel skema-type → række-formning
  (`Journal` → én flad række pr. side; `TextPage` → én række pr. linje).
- `tools.py` — fil-opdagelse, `flush_rows` (CSV `$`-separeret eller JSONL),
  datasæt-bogholderi, run-mappe med config-snapshot.
- `validate.py` — **Tkinter**-valideringsapp (ét felt ad gangen, random
  sampling; Accept/Somewhat/Reject/Unsure/Correction → CSV). Hardcodet til
  `Journal`-skemaet (l. 15, 84).
- `validation_analysis.py` — felt-accept-rate + plots. **Ingen CER/edit-distance
  findes i marts-versionen.**

## Nøglefund

1. **Linje-for-linje-tilgangen findes allerede**: `TextPage`-skemaet + den
   aktive default-prompt `textpage` (`config.py:62-78`) er netop kontinuerlig
   linjetransskription: "expert archivist, late 19th-century Danish medical
   manuscripts", Primary Page Only (ignorér modstående side), margindatoer i
   `metadata`, vitalkolonner (Time|Temp|Puls) med bevaret spacing, bevar
   arkaisk dansk stavning. Godt udgangspunkt for andenside-prompten.
2. **En "metode" i marts-arkitekturen** = Pydantic-skema i `schemas.py` +
   navngiven prompt i `config.prompts` + handler i `output_handler._HANDLERS`.
   (Upstream-pakkens nye kontrakt: se `app_interface_upstream.md`.)
3. Billedudvælgelse sker pr. mappe/konfiguration, ikke pr. filnavn — intet i
   koden skelner for-/andensider; det gør masterlisten (se
   `billeder_og_masterliste.md`).
4. API-nøgle via git-ignoreret `api_keys.py`.

## Åbne punkter

- `origin/patch-1` (stor omskrivning på leads egen fork) er aldrig merget —
  afklar om noget derfra stadig har værdi, eller om upstream har overhalet den.
- Lokal gren `Severity_prompt` (+11/-2 i schemas.py) ligeledes uafklaret.
