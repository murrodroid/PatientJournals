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

Currently, the project is only set up for using Gemini. This will be updated in the future.

Create a file `api_keys.py` and write your API key in it. This should then be the one imported in `main.py`.

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
