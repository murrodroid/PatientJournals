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
uv run invoke batch.submit
uv run invoke batch.status --watch
uv run invoke batch.retrieve --wait
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
- `patientjournals.app.job_store`: SQLite-backed jobs, events, dataset versions, and background task state.
- `patientjournals.app.jobs`: batch/local run helpers, retrieval/finalization/retry helpers, and validation command construction.
- `patientjournals.app.settings_store`: app settings persistence.
- `patientjournals.app.task_runner`: lightweight background task execution persisted to the job store.
- `patientjournals.app.web`: the web UI and JSON API.
- `patientjournals.app.workflows`: the backend workflow service used by the UI.

Local API jobs are executed through `patientjournals.local.service.run_local_job`, so the app does not need to shell out for local runs. Cloud batch submission and retrieval are routed through `patientjournals.app.workflows.WorkflowService`. The Jobs page can retrieve one or many selected jobs; grouped batch chunks are handled as one job, retrieval reuses cached results when possible, and selected-job retrieval is parallelized by the workflow service. The Dashboard page summarizes run measurements, validates completeness/failure distributions for a selected dataset, and can launch the validation loop for a selected dataset and image folder.

The Cloud page stores project/bucket/prefix settings, can start browser-based Google auth with `gcloud auth application-default login`, and runs the same bucket/prefix/write access checks used by the previous setup flow.

Google Cloud authentication supports either a service account JSON path or Application Default Credentials (`gcloud auth application-default login`) through the app setting `auth_mode`.

## Project Pipeline

![Pipeline](/visualizations/PatientJournals.png)

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

Continuation, collection, and coverage checks compare rows by `image_name`. Older datasets that only contain `file_name` are still accepted; copied rows are upgraded with `image_name` when possible.

## Batch Flow

Batch jobs support Gemini batch jobs through Vertex/mldev and Anthropic Message Batches.

Typical flow:

```bash
uv run invoke batch.upload
uv run invoke batch.submit
uv run invoke batch.status --watch
uv run invoke batch.retrieve --wait
uv run invoke batch.collect-outputs --continue-dataset runs/.../dataset.jsonl
```

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

The web app Dashboard reads job-store datasets, `runs/**/image_processing_manifest.jsonl`, and `validations/**/*_validations.csv` to display status, source, attempts, timing, failure, field-completeness, and validation label distributions. The same page can launch `validation.validate` with corrections enabled. Validation sampling supports `random` and `balanced_ucb`; both sample only true schema fields and exclude processing metadata such as `thoughts`, `failure_reason`, `avg_logprobs`, and `crossed_out`.

## Tests

Run the test suite with:

```bash
uv run pytest
```
