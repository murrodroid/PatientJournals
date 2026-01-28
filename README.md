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

**Model Generation**

Currently, the project is only set up for using Gemini. This will be updated in the future.

Create a file `api_keys.py` and write your API key in it. This should then be the one imported in `main.py`.

In `config.py`, you can specify all details of generation, and then simply run:
```bash
uv run main.py
```