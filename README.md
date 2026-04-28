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

To inspect the online GCS bucket instead of local files:

```bash
uv run invoke data.batch --summary --bucket
uv run invoke data.batch --validate --bucket
uv run invoke data.batch --summary --bucket --bucket-name data-blegdamsjournaler --prefix pages
```

Bucket summary lists object metadata only. Bucket validation downloads each matching image object and verifies it with Pillow, so it can take time on large buckets.

Summary JSON reports are written to `summaries/`. Validation JSON and CSV reports are written to `validations/`. Validation checks include empty files, unreadable/corrupt images, invalid dimensions, extension/format mismatches, and duplicate basenames.

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
uv run invoke batch.retrieve --wait --submit-failed
uv run invoke batch.collect-outputs --continue-dataset newest
uv run invoke batch.status --simple --watch
uv run invoke batch.check-models --contains gemini
```

`batch.collect-outputs` scans `batch/outputs` for `*predictions.jsonl`, keeps the first schema-valid non-empty response per unique key, writes a recovered dataset, and reports coverage against `pages/`.
For Vertex retrieval, `batch.retrieve --allow-partial` also parses available output files from non-succeeded jobs instead of dropping the whole chunk.
Use `batch.submit --continue-dataset DATASET` to submit only pages whose GCS key is not present in the dataset `file_name` column, then use `batch.collect-outputs --continue-dataset DATASET` to produce a total dataset with the existing rows plus newly collected outputs.
Set `include_response_avg_logprobs` in `settings.py` to control whether Gemini `avgLogprobs` is written as `avg_logprobs` in batch datasets.

For Vertex, configure `service_account_file`, `gcp_project_id`, `gcp_location`, `gcs_bucket_name`, and related GCS prefixes in `settings.py`.

For Anthropic batch runs, inputs still come from GCS. Images are referenced through signed HTTPS URLs generated from GCS object keys.

## Validation

Run the validation UI:

```bash
uv run invoke validation.validate --user lucas --images data --results runs/20260127_103351/20260127_103351_dataset.jsonl --corrections
```

Generate validation plots:

```bash
uv run invoke validation.report --input-path validations --out validation_reports --min-n 1
```

## Tests

Run the test suite with:

```bash
uv run pytest
```
