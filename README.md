PatientJournals creates structured transcriptions of Blegdams patient journals with LLMs.

## Setup

Install uv, then sync the project:

```bash
pip install uv
uv sync
```

All project code lives under `src/patientjournals`. Runtime configuration is in:

```bash
src/patientjournals/config/settings.py
```

All non-schema model instructions are maintained in
`src/patientjournals/config/prompts.py`: page prompts, sub-agent context, and the
OCR evidence header. JSON Schema field descriptions remain co-located with their
Pydantic fields in `src/patientjournals/config/schemas.py`.

You can print that path with:

```bash
uv run invoke config.path
```

## Task Usage

Use Invoke for operational commands:

```bash
uv run invoke --list
```

Common tasks:

```bash
uv run app
uv run invoke local.run
uv run invoke batch.upload
uv run invoke batch.ocr
uv run invoke batch.submit
uv run invoke batch.status --watch
uv run invoke batch.retrieve --wait
uv run invoke batch.verify --source-run-dir runs/submits/...
uv run invoke batch.verify --retrieve --run-dir runs/verifications/... --wait
uv run invoke data.batch --summary
uv run invoke data.batch --validate
uv run invoke data.batch --summary --bucket
uv run invoke validation.validate --user lucas --images data --results runs/.../dataset.jsonl --corrections
uv run invoke validation.report --input-path validations --out validation_reports --min-n 1
```

Extra underlying CLI arguments can be passed with `--extra`, for example:

```bash
uv run invoke batch.submit --extra='--downscale 0.1'
```

## Web App

Run the app with:

```bash
uv run app
```

(`uv run patientjournals-app` is an equivalent alias, and `uv run invoke app.run` still works.)

For local development without opening a browser automatically:

```bash
uv run python -m patientjournals.app.web --no-open
```

The app stores UI settings in `~/.patientjournals/app_config.json`. Runtime job state is stored in SQLite at `runs/app_state.sqlite3`; old per-folder `job.json` files are treated as legacy import artifacts, not the authoritative app state. Current datasets for retrieved jobs are copied under `runs/jobs/<job_id>/datasets/current.*`, with previous versions kept under `runs/jobs/<job_id>/datasets/versions/`.

The current app architecture is split into small service modules:

- `patientjournals.app.access`: Google Cloud CLI, auth, bucket, prefix, and write/read/delete access checks.
- `patientjournals.app.catalog`: schema and Google model choices.
- `patientjournals.app.dashboard`: dataset, validation, and processing metric summaries.
- `patientjournals.app.datasets`: local/cloud dataset inspection and image-name matching.
- `patientjournals.app.image_access`: short-lived signed GCS links and local image previews.
- `patientjournals.app.job_store`: SQLite-backed jobs, events, dataset versions, and background task state.
- `patientjournals.app.jobs`: batch/local run helpers, retrieval/finalization/retry helpers, and validation command construction.
- `patientjournals.app.schemas`: immutable schema versions, active-schema state, and cloud synchronization.
- `patientjournals.app.settings_store`: app settings persistence.
- `patientjournals.app.task_runner`: lightweight background task execution persisted to the job store.
- `patientjournals.app.web`: the web UI and JSON API.
- `patientjournals.app.workflows`: the backend workflow service used by the UI.

The normal colleague workflow is:

1. Open **Schemas** to review the shared versions and star the version that should be selected by default.
2. Open **Submit**, select local or cloud folders, preview a random sample of the exact selected pages, then choose the schema version and model.
   Advanced job options independently control positional OCR, schema-specialist
   subagents, and second-pass verification; each job records the exact choices.
3. Follow the grouped job in **Jobs** and retrieve its chunks as one entity.
   For a job with second-pass validation enabled, retrieval creates a candidate;
   use the configurable verifier controls in **Jobs** to submit and retrieve the
   candidate-aware validation batch.
4. Open **Datasets** to inspect published rows. Double-click a dataset to open the inspector, then double-click a row to open its scanned page.
5. Use **Dashboard** for leaf-column completeness and model-specific validation results, or launch **Validate** directly.

Local API jobs are executed through `patientjournals.local.service.run_local_job`, so the app does not need to shell out for local runs. Cloud batch submission and retrieval are routed through `patientjournals.app.workflows.WorkflowService`. The Jobs page can retrieve one or many selected jobs; grouped batch chunks are handled as one job, retrieval reuses cached results when possible, and selected-job retrieval is parallelized by the workflow service. The Dashboard page summarizes run measurements, validates completeness/failure distributions for a selected dataset, and can launch the validation loop.

The Cloud page stores project/bucket/prefix settings, can start browser-based Google auth with `gcloud auth application-default login`, and runs the same bucket/prefix/write access checks used by the previous setup flow.

Google Cloud authentication supports either a service account JSON path or Application Default Credentials (`gcloud auth application-default login`) through the app setting `auth_mode`.

### Schema Versions

Schemas are immutable versions with IDs such as `sv_...`. Editing never changes the original: the app presents the original and proposed version side by side, then saves a new version with its parent, author, timestamp, and complete JSON Schema snapshot. Two selected versions can also be compared without editing. The starred active version is the default on Submit, but any version can be selected explicitly.

Schema versions and the shared active choice are synchronized through `<schemas_gcs_prefix>/index.json` and immutable objects under `<schemas_gcs_prefix>/versions/`. The default prefix is `schemas`. Cloud write/read checks include this prefix. Every app-submitted job records the selected version ID and full schema snapshot, so later retrieval still validates against the original version even if another schema becomes active.

## Project Pipeline

![Pipeline](/visualizations/PatientJournals.png)

Cloud batch jobs are the primary production workflow. Local generation remains available for development and recovery, but production features are expected to support the batch path first.

## Local Generation

Local generation supports Gemini, OpenAI, and Anthropic. Set `config.model` in `src/patientjournals/config/settings.py`; provider resolution happens through `src/patientjournals/config/models.py`.

Examples:

```bash
uv run invoke local.run
uv run invoke local.run --continue-dataset newest
uv run invoke local.run --verbose
uv run invoke local.run --data-folder data/8dec96
```

If `--data-folder` is omitted, the configured `target_folder` is used. `fp_mode` controls whether `_fp` folders are included, excluded, or exclusively selected.

Each local run writes research measurement files next to the dataset:

- `image_processing_manifest.jsonl`: one record per processed image, including source image name/path, preprocessing settings and dimensions, model/provider, attempts, timings, status, rows written, and failure reason if any.
- `image_processing_summary.json`: aggregate status/source counts and numeric distributions for attempts, generation seconds, total seconds, and rows written.

## Data Inspection

Use the data module to inspect local batch image folders before upload or model runs:

```bash
uv run invoke data.batch --summary
uv run invoke data.batch --validate
```

By default, this reads the configured batch image folder and glob. Override the input folder when needed:

```bash
uv run invoke data.batch --summary --validate --root data --glob '*.png'
```

Local validation runs single-core by default. Use more CPU cores for large folders:

```bash
uv run invoke data.batch --validate --root data --cores 8
uv run invoke data.batch --validate --root data --cores 0  # auto-detect cores
```

To inspect the online GCS bucket instead of local files:

```bash
uv run invoke data.batch --summary --bucket
uv run invoke data.batch --validate --bucket
uv run invoke data.batch --summary --bucket --bucket-name data-blegdamsjournaler --prefix pages
```

Bucket summary lists object metadata only. Bucket validation downloads each matching image object and verifies it with Pillow, so it can take time on large buckets.

Summary JSON reports are written to `summaries/`. Validation JSON and CSV reports are written to `validations/`. Validation checks include empty files, unreadable/corrupt images, invalid dimensions, extension/format mismatches, and duplicate basenames.

## Dataset Format

Generated datasets use `image_name` as the primary image identity. Image names are expected to be unique across the complete dataset. The legacy `file_name` column is still written as provenance and may contain the original local path, GCS object name, or older source reference.

Every newly processed row, including explicit failed placeholders, records `model`, `provider` when available, `schema_name`, and `schema_version_id`. Dataset objects uploaded to GCS carry the same provenance as object metadata. Combined datasets preserve row-level provenance and report single or mixed model/schema counts in their manifest.

Continuation, collection, and coverage checks compare rows by `image_name`. Older datasets that only contain `file_name` are still accepted; copied rows are upgraded with `image_name` when possible.

## Batch Flow

Batch jobs support Gemini batch jobs through Vertex/mldev and Anthropic Message Batches.

Typical flow:

```bash
uv run invoke batch.upload
uv run invoke batch.ocr  # only when OCR context is enabled for the job
uv run invoke batch.submit
uv run invoke batch.status --watch
uv run invoke batch.retrieve --wait
uv run invoke batch.collect-outputs --continue-dataset runs/.../dataset.jsonl
```

Optional candidate-aware model validation adds a second batch after retrieval:

```bash
uv run invoke batch.verify --source-run-dir runs/submits/submit_YYYYMMDD_HHMMSS
uv run invoke batch.status --run-dir runs/verifications/YYYYMMDD_HHMMSS --watch
uv run invoke batch.verify --retrieve --run-dir runs/verifications/YYYYMMDD_HHMMSS --wait
```

The three optional stages are independent and are selectable for every app job:

| Stage | Setting | Behavior when disabled |
|---|---|---|
| Positional OCR | `ocr_enabled` | No sidecar is required and no OCR tokens are sent. |
| Schema specialists | `subagents` | Uses the existing single request per page. |
| Model verification | `model_validation_enabled` | Publishes directly after deterministic extraction checks, as before. |

Persistent app defaults only prefill these controls; submission snapshots the
selected values in the run metadata. Verification is Cloud-batch-only, but it
can run with OCR either on or off.

Enable it with `model_validation_enabled=True` or the app's **Second-pass model
validation** switch. The app exposes the verifier model, thinking level, maximum
output tokens, and validation batch chunk count as persistent defaults and
per-job controls. Final corrections use a fixed automatic-acceptance policy; new
jobs do not expose a report/apply choice. The verifier validates all pages. It sends
the exact page evidence, optional positional OCR, original extraction schema,
and the unflattened extraction candidate to a candidate-aware verifier. An
inaccurate field must be returned as the smallest page-supported correction,
encoded as a sparse RFC 6902 patch with short evidence.

With validation enabled, `batch.retrieve` performs the normal deterministic and
schema checks but records the result as a pre-version candidate; it does not
create `v001`. The final verifier automatically accepts sparse corrections only
after they apply cleanly and the corrected page revalidates against the original
schema. It publishes the first dataset version only when every expected page has
a valid result and there are no missing, failed, or `unverifiable` pages. Image
generation, candidate, schema, request, and enabled OCR bindings are persisted so
a changed input blocks validation instead of silently comparing different bytes.
This pinning starts with extraction itself: Gemini reads a write-once staged copy,
Anthropic receives a generation-qualified signed URL, and retrieval rejects
truncated or refused provider terminations before a candidate is created.

Every verifier retrieval writes and uploads an immutable, content-addressed
`field_corrections.json`. It contains one record per expected page and one entry
per proposed field patch, including the JSON Pointer, operation, issue, original
and proposed values, evidence, OCR box references when available, and separate
accepted/applied/complete-dataset-inclusion states. A patch may be accepted on its page
while `corrected=false` if another page blocks publication of the complete
dataset. The artifact URI, GCS generation, SHA-256, corrected-dataset digest, and
field counts are propagated to the job result and dataset-version ledger. The
ledger is the authority for whether those exact corrected bytes became a final
`vNNN`.

Each completed verifier run keeps its model/configuration, counts, hashes,
report, and immutable dataset-version pointer. The CLI and app share one
`dataset_versions.json` ledger; each entry records its local immutable file, GCS
URI, GCS generation, and SHA-256. Replaying the same verifier run reuses its
version. Re-running verification with a different configuration therefore
publishes `v002`, `v003`, and so on instead of overwriting `v001`. Historical
unanchored/report-only runs remain report-only for reproducibility; submit a new
verifier run to use automatic correction acceptance. New runs freeze this policy
and their storage prefixes in a create-only, generation-bound
`final_validation_policy.json`, so editing operational job metadata cannot
promote an old run or redirect publication.

See [PIPELINE.md](PIPELINE.md) for the researcher-facing stage, setting,
artifact, gate, and provenance specification.

Useful batch commands:

```bash
uv run invoke batch.submit --num-batches 10
uv run invoke batch.submit --continue-dataset runs/.../dataset.jsonl
uv run invoke batch.submit --rerun --run-dir runs/submit_YYYYMMDD_HHMMSS
uv run invoke batch.retrieve --run-dir runs/submit_YYYYMMDD_HHMMSS --allow-partial
uv run invoke batch.retrieve --run-dir runs/submit_YYYYMMDD_HHMMSS --batch-names BATCH_A,BATCH_B
uv run invoke batch.retrieve --run-dir runs/submit_YYYYMMDD_HHMMSS --allow-partial --recover-missing-with-api
uv run invoke batch.retrieve --run-dir runs/submit_YYYYMMDD_HHMMSS --duplicate-strategy first_successful
uv run invoke batch.retrieve --wait --submit-failed
uv run invoke batch.collect-outputs --continue-dataset newest
uv run invoke batch.status --simple --watch
uv run invoke batch.check-models --contains gemini
```

`batch.ocr` enumerates the configured GCS input images, downloads each exact object generation, creates compact OCR line metadata, and uploads an adjacent `.ocr.json` sidecar. Google Vision requests contain up to 16 images per API call and multiple API batches run concurrently; tune these with `--api-batch-size` and `--workers`. The request-byte cap in `batch_ocr_api_batch_max_bytes` can split batches earlier for large images. The command also writes the cloud manifest configured by `batch_ocr_manifest_object`. Existing sidecars are reused when their bucket, object name, generation, size, and checksums still match; use `batch.ocr --force` to rebuild them. Batch submission only reads these sidecars and does not run OCR or download image content. When `ocr_enabled=True`, missing or stale sidecars block submission if `batch_ocr_metadata_required=True` (the default); when OCR is disabled, no sidecar is required.

Set `subagents=True` in `settings.py`, or enable **Schema specialists** for an app job, to decompose each page into one batch request per top-level schema field. Every schema specialist searches the full processed page, plus OCR context when enabled, but receives only its one-field response schema and a compact prompt. Retrieval validates each specialist response, joins only pages with every specialist present, validates the full page schema, and then uses the normal dataset writer. The submit run records the mode and field list in `batch_job.json`; retrieve runs keep `subagent_combined.jsonl` and `subagent_failures.jsonl` as durable audit artifacts. This mode increases image/request count by the number of top-level schema fields; provider batch execution supplies the parallelism.

`batch.collect-outputs` scans `batch/outputs` for `*predictions.jsonl`, keeps the first schema-valid non-empty response per unique key, writes a recovered dataset, and reports coverage against `pages/`.
For Vertex retrieval, `batch.retrieve --allow-partial` also parses available output files from non-succeeded jobs instead of dropping the whole chunk. Batch retrieval resolves chunk states and downloads outputs in parallel using `api_concurrent_tasks`; live API recovery is also parallelized.
Repeated raw `--batch-name` values retrieve selected chunks from a grouped submit run; the Invoke wrapper also accepts comma-separated `--batch-names`. `--recover-missing-with-api` sends missing expected Gemini pages through the live API after partial retrieval, which is useful when some chunks succeeded and others failed. Duplicate successful keys are controlled by `--duplicate-strategy`: `first_successful` keeps the first valid page output, while `provide_all` writes every valid duplicate.
Use `batch.submit --continue-dataset DATASET` to submit only pages whose GCS key is not present in the dataset `file_name` column, then use `batch.collect-outputs --continue-dataset DATASET` to produce a total dataset with the existing rows plus newly collected outputs.
Set `include_response_avg_logprobs` in `settings.py` to control whether Gemini `avgLogprobs` is written as `avg_logprobs` in batch datasets.

Each retrieve run writes `image_processing_manifest.jsonl` and `image_processing_summary.json`. These files document batch output parsing, duplicate decisions, API recovery attempts, retry counts, timings, downloaded bytes, row counts, and failure reasons.

For Vertex, configure `service_account_file`, `gcp_project_id`, `gcp_location`, `gcs_bucket_name`, and related GCS prefixes in `settings.py`.

For Anthropic batch runs, inputs still come from GCS. Images are referenced through signed HTTPS URLs generated from GCS object keys.

## Validation

Run the validation UI:

```bash
uv run invoke validation.validate --user lucas --images data --results runs/20260127_103351/20260127_103351_dataset.jsonl --corrections --sampling-mode balanced_ucb
```

Generate validation plots:

```bash
uv run invoke validation.report --input-path validations --out validation_reports --min-n 1
```

The web app Dashboard reads job-store datasets, `runs/**/image_processing_manifest.jsonl`, and `validations/**/*_validations.csv` to display status, source, attempts, timing, failure, leaf-field completeness, and validation label distributions. Parent objects such as `diagnoses.sektion` are not treated as empty columns; only their transcribed leaf fields are measured. The same page can launch browser validation with corrections enabled.

Browser validation derives the validator username from the active Google account (for example `name@gmail.com` becomes `name`) and records the account in each validation run's CSV and metadata. Each decision also records the source model, schema version, original extracted value, and any correction. For cloud images, validation matches dataset `image_name` values directly against the configured GCS pages prefix and resolves signed URLs lazily in small batches while you validate; there is no separate cloud image-folder selection step. Validation runs sync to the shared cloud bucket by default; Advanced includes an offline mode for debugging that saves locally without uploading.

Validation sampling supports `random` and `balanced_ucb`. Both sample only the true leaf columns from the schema version recorded on each row and exclude processing metadata such as `thoughts`, `failure_reason`, `avg_logprobs`, and `crossed_out`. Balanced UCB keeps separate strata for missing/present values, model, schema version, and leaf column, so results from different extraction models or schema definitions are not conflated.

## Tests

Run the test suite with:

```bash
uv run pytest
```
