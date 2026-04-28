This repo is for creating transcriptions of Blegdams Journals, using LLM models with structured output.



To get started, install uv:
```bash
pip install uv
```

Then, run the following to sync dependencies:
```bash
uv sync
```

You're now ready to go.


**Project Pipeline**

![Pipeline](/visualizations/PatientJournals.png)

**Validator usage**

Usage (with uv):
```bash
uv run validate.py --user lucas --images data --results runs/20260127_103351/20260127_103351_dataset.jsonl --corrections
```

(without uv):
```bash
python validate.py --user lucas --images data --results runs/20260127_103351/20260127_103351_dataset.jsonl
```
- --user: saves who did what; keep the same username each time.

- --images: root folder of all images (subfolders are fine). Place in /data folder.

- --results: dataset location.

- --corrections: enables corrections to the field.

**Validation Report**

With enough validations, a user can generate plots which display the accuracy of the model.

Usage (with uv):
```bash
uv run validation_analysis.py --input validations --out validation_reports --min-n 1
```

(without uv):
```bash
python validation_analysis.py --input validations --out validation_reports --min-n 1
```

- --input: specifies the input of validation dataset(s)

- --out: specifies where the use wants the validation report

- --min-n: limits the minimum amount of validations for a column's data to be included.

**Local Transcription Generation (`main.py`)**

`main.py` supports multiple providers for local requests:
- Gemini
- OpenAI
- Anthropic

Set `config.model` to the model you want to run. The provider is resolved dynamically from `models.py`.

Field confidence note:
- `include_confidence_scores` currently requires token logprobs support.
- In this pipeline, field-level confidence is available for Gemini responses.
- Anthropic responses do not include token logprobs in the native Messages API, so `field_confidence` remains empty when using Claude.

Create/update `api_keys.py` with provider keys. Supported key names:
- Gemini: `gemini_maarten`, `gemini`, `google`, or `google_gemini`
- OpenAI: `openai`, `openai_api_key`, or `gpt`
- Anthropic: `anthropic`, `anthropic_api_key`, or `claude`

In `config.py`, you can specify default generation settings. Basic run:
```bash
uv run main.py
```

Useful `main.py` features:

- Continue from an existing dataset and skip already-transcribed files:
```bash
uv run main.py --continue-dataset runs/20260301_101010/20260301_101010_dataset.jsonl
```

- Continue from the most recent run automatically:
```bash
uv run main.py --continue-dataset newest
```

- Show dataset coverage before/after the run:
```bash
uv run main.py --verbose
```

- Run against a specific folder instead of the default data root:
```bash
uv run main.py --data-folder data/8dec96
```
You can also pass a folder name relative to the configured `target_folder`, e.g. `--data-folder 8dec96`.
For a local folder outside the project data root, pass an absolute path, e.g. `--data-folder /Users/you/local_journal_images`.

Behavior notes:

- If `--data-folder` is not set, `main.py` uses the configured `target_folder` (default: `data`).
- You can also set `target_folder` in `config.py` to an absolute local path if you want that location as the default input root.
- When `--continue-dataset` is used, existing rows are carried into the new run output and only missing files are processed.
- `fp_mode` in `config.py` controls whether input selection uses `_fp` folders (`only_fp`), non-`_fp` folders (`exclude_fp`), or both (`all`).

**Batch Jobs On Cloud (`batch_*.py`)**

Batch scripts (`batch_submit.py`, `batch_status.py`, `batch_retrieve.py`) are configured through `config.py` and support:
- Gemini batch jobs (Vertex/mldev via `google-genai`)
- Anthropic Message Batches (`messages.batches.*`)

For Gemini on Vertex, activate "Vertex AI API" in your GCP project.
 
1. Create or choose a GCP project and a GCS bucket.

2. Create a service account and key (example with `gcloud`):
```bash
gcloud iam service-accounts create patientjournals-batch \
  --display-name="PatientJournals Batch"

gcloud iam service-accounts keys create ./service-account.json \
  --iam-account=patientjournals-batch@PROJECT_ID.iam.gserviceaccount.com
```

3. Grant roles to that service account:
```bash
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:patientjournals-batch@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:patientjournals-batch@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```
For Anthropic-only batch runs, Vertex role (`roles/aiplatform.user`) is not required.

4. Set required `config.py` values:
- `batch_backend = "vertex"`
- `service_account_file = "service-account.json"` (or absolute path)
- `gcp_project_id = "your-project-id"`
- `gcp_location = "europe-north1"` (or your Vertex region)
- `gcs_bucket_name = "your-bucket-name"`
- `input_prompt_name = "textpage"` (or another prompt key in `prompts`)

5. Common optional `config.py` values:
- `gcs_pages_prefix`, `batch_requests_gcs_prefix`, `batch_outputs_gcs_prefix`, `datasets_gcs_prefix`
- `batch_input_prefix` (limit batch input to a specific prefix in bucket)
- `batch_date_mapping_file` + `batch_year_filter` (optional year-based folder filtering using a CSV mapping with `sbid` and `year` columns; supports 2-digit tokens like `91` -> `1891` when unambiguous)
- `batch_num_chunks` (split one logical submit into multiple smaller batch jobs)
- `batch_use_local_pdf_folders` and `batch_auto_upload_missing`
- `upload_source` (`"pdf"`, `"images"`, or `"auto"`)
- `upload_images_folder`, `upload_images_recursive`, `upload_images_glob`
- `upload_auto_tune`, `upload_profile` (`"light"`, `"normal"`, `"aggressive"`), `upload_max_workers`
- `upload_timeout_seconds`, `upload_retry_attempts`, `upload_retry_initial_delay_seconds`, `upload_retry_max_delay_seconds`
- `batch_job_display_name`, `batch_poll_interval_seconds`
- `upload_dataset_to_gcs`
- `api_recovery_enabled`, `api_recovery_max_missing_pages`, `api_recovery_model`, `api_key` (only if recovery is enabled)
- `batch_submit_failed_pages` (if `True`, `batch_retrieve.py` auto-submits a separate retry batch for errored/schema-invalid keys)
- `anthropic_signed_url_ttl_hours` (Anthropic batch only; signed URL lifetime for GCS image URLs)


Typical batch flow:
```bash
uv run upload.py
uv run batch_submit.py
uv run batch_status.py --watch
uv run batch_retrieve.py --wait
```

Anthropic + GCS note:

- When `config.model` resolves to Anthropic (`claude-*`), submit/retrieve/status use Anthropic batch APIs.
- Inputs still come from your GCS bucket; images are referenced via signed HTTPS URLs generated from GCS object keys.
- Image bytes are not downloaded to your laptop for submission.

Batch chunking:

- Set `batch_num_chunks = 10` in `config.py`, or pass `-n 10` to `batch_submit.py`.
- `uv run batch_submit.py -n 10` submits 10 smaller batch jobs in one submit run.
- `batch_status.py --watch` tracks all chunk jobs from that run.
- `batch_retrieve.py --wait` automatically collects and merges all chunk outputs into one dataset.

Chunk rerun (only incomplete/failed chunks):

- `uv run batch_submit.py --rerun` reuses the latest submit run and only resubmits chunk jobs that are not `JOB_STATE_SUCCEEDED`.
- You can target a specific run with `uv run batch_submit.py --rerun --run-dir runs/submit_YYYYMMDD_HHMMSS`.

Partial retrieval (acknowledging incomplete jobs):

- `uv run batch_retrieve.py --run-dir runs/submit_YYYYMMDD_HHMMSS --allow-partial`
- This retrieves only succeeded chunk jobs and skips running/failed/cancelled chunks.
- If you omit `--allow-partial`, retrieval stays strict and fails when not all jobs have succeeded.

Submit failed keys as a separate retry batch:

- `uv run batch_retrieve.py --wait --submit-failed`
- This compiles keys that failed (batch errors, invalid JSON, schema validation failures, or otherwise missing successful output) and submits them as a new one-chunk batch job in a new `runs/submit_*` directory.
- You can enable the same behavior by default with `batch_submit_failed_pages = True` in `config.py`.

PNG folder upload mode:

- Set `upload_source = "images"` in `config.py`.
- Set `upload_images_folder` to the local PNG root folder.
- Set `upload_images_recursive = True` (or `False`) depending on whether subfolders should be scanned.
- Keep `upload_images_glob = "*.png"` (or adjust if needed).
- Uploaded object names keep local folder/file names under `gcs_pages_prefix`; preprocessing still uses `config.image_settings`.
- If `upload_auto_tune = True`, uploader adjusts `upload_workers` and effective batch size during the run based on measured throughput.
- Use `upload_profile = "light"` for conservative load, `"normal"` for balanced, `"aggressive"` for max throughput attempts.
