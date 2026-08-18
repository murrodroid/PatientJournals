# Kollegaens app: plug-in-kontrakt (upstream/main @ e8f412e, 2026-08-04)

Kortlagt 2026-08-18 via `git show upstream/main:<sti>` i `c:\Work\PatientJournals`
(arbejdstræet urørt). Dette er den AKTUELLE arkitektur — marts-versionen i den
lokale klon (se `patientjournals_repo.md`) er historisk.

## Appen

Ren stdlib-webapp — ingen Flask/FastAPI: `src/patientjournals/app/web.py`
bruger `http.server.ThreadingHTTPServer` med én indlejret HTML/JS-side
(`APP_HTML`) og JSON-API i samme proces. Startes via script-entrypoint
`app = "patientjournals.app.web:main"` (også `patientjournals-app`) eller
`python -m patientjournals.app.web`. Faner (web.py:137-144): **Dashboard,
Validate, Jobs, Datasets, Schemas, Submit, Cloud, Tasks**. Dashboard viser
datasæt/rækketal, processing-status, valideringsetiketter og en
validator-leaderboard ("race").

## Hvad en "metode" ER i denne arkitektur

**Pydantic-outputskema + prompt-nøgle + modelnavn**, bundet sammen af `Config`
(`src/patientjournals/config/settings.py`):

- **Skemaer**: `src/patientjournals/config/schemas.py` med
  `OUTPUT_SCHEMA_REGISTRY = {"FrontPage": ..., "TextPage": ...}` +
  `resolve_output_schema(name)`. **`TextPage`/`PageLine` findes allerede**:
  `page_lines: List[PageLine]`, hver med `text`, `metadata`,
  autonummereret `page_line_number`.
- **Prompts**: nøglede strenge i `Config.prompts` — `"frontpage"` og
  **`"textpage"`** (linje-for-linje, ignorér modstående side, margindatoer i
  metadata) er allerede skrevet. `Config.input_prompt_name` +
  `output_schema_name` vælger metoden pr. job.
- **Modeller**: `config/models.py` med
  `ModelSpec(name, provider, supports_batch, supports_confidence_scores,
  supports_thoughts)` i `_REGISTERED_MODELS`; ukendte navne får provider
  gættet på præfiks.
- **Skema-redigering i UI**: `app/schemas.py` (`SchemaService`, commit
  e8f412e) — versioneret skema-registry med `create_version`/`set_active`,
  GCS-sync, og UI-redigering af felter der regenererer Pydantic-modellen på
  runtime (`model_from_json_schema`). Version-id = `sv_<sha256[:20]>`.

**En andenside-metode kræver altså konkret:**
1. Skema: `TextPage` findes allerede (evt. UI-klonet version).
2. Prompt: `textpage`-prompten findes allerede — vores arbejde er at forbedre/
   specialisere den ud fra testresultater.
3. Model: registrér evt. i `config/models.py`.
4. **Billedudvælgelse — det reelle hul**: der findes intet `page_type`-felt.
   Udvælgelse sker via `fp_mode` (`"all"|"only_fp"|"exclude_fp"`,
   `fp_suffix="_fp"`) på fil-/blobnavne i `batch/submit_inputs.py` +
   `batch/upload.py`. En andenside-kørsel kræver `fp_mode="exclude_fp"` eller
   (bedre) en ny konvention/kobling til masterlistens
   `patient_page_counter == 2`. Dette er hook-punktet.
5. Jobbet: `SubmitJobDraft` (app/models.py) med run_mode
   `local_api`/`cloud_batch`; kørsel via `app/task_runner.py`/`workflows.py` →
   `local/service.py` (live) eller `batch/*` (Vertex/Anthropic batch).

## Providers

`pyproject.toml`: `google-genai`, `openai`, `anthropic`, `google-cloud-storage`.
`ProviderName = Literal["gemini", "openai", "anthropic"]`. Gemini + Anthropic
har batch-støtte; OpenAI kun live. Live-klienter i `local/model_client.py`
(`genai.Client`, `AsyncOpenAI`, `AsyncAnthropic`); Gemini-batch via GCS +
Vertex i `batch/client.py`.

## Data/jobs

Billeder i GCS (`data/bucket.py`; `gcs_bucket_name`/`gcs_pages_prefix` i
Config) eller lokal mappe. `app/image_access.py` udsteder signerede URL'er til
UI-preview. Jobs trackes i SQLite (`app/job_store.py`). Output = JSONL/CSV
under `runs/<job>/datasets/` + spejlet til GCS `datasets/`-prefix.
`date_mapping.csv` mapper kilde-bog-id → år, bruges kun som årsfilter ved
batch-submit (`batch_year_filter`), ikke som sidetype-skelnen.

## Validering — og hullet vi skal udfylde

**Ingen CER/edit-distance findes på upstream/main** (grep bekræftet).
Validering er menneske-etiketter pr. felt/række (`accept`, `somewhat_accept`,
`reject`, `corrected`, `unsure` → `*_validations.csv`, evt. GCS-synk).
Leaderboardet rangerer VALIDATORER (throughput/enighed), ikke modeller.
`validation/cli.py` har uncertainty-sampling til at udvælge rækker til review.
⇒ 2ndpage-projektets CER/WER-eval mod manuel ground truth bliver et reelt nyt
bidrag, ikke en dublet.

## Nøglefiler til plug-in-arbejdet

`config/schemas.py` (registry, TextPage), `config/settings.py` (prompts,
Config), `config/models.py` (modelregistry), `app/schemas.py` (SchemaService),
`batch/submit_inputs.py` (fp_mode-filtrering = hook-punktet for sidetype).
