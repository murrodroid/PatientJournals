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

**Batch Jobs On GCP (`batch_*.py`)**

Batch scripts (`batch_submit.py`, `batch_status.py`, `batch_retrieve.py`) are configured through `config.py` and use Vertex AI + GCS when `batch_backend="vertex"`. You should use vertex, and activate "Vertex AI API" in your GCS project.
 
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
- `batch_use_local_pdf_folders` and `batch_auto_upload_missing`
- `batch_job_display_name`, `batch_poll_interval_seconds`
- `upload_dataset_to_gcs`
- `api_recovery_enabled`, `api_recovery_max_missing_pages`, `api_recovery_model`, `api_key` (only if recovery is enabled)


Typical batch flow:
```bash
uv run upload.py
uv run batch_submit.py
uv run batch_status.py --watch
uv run batch_retrieve.py --wait
```
