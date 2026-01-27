This repo is for creating transcriptions of Blegdams Journals, using LLM models with structured output.



To get started, install uv:
https://docs.astral.sh/uv/getting-started/installation/

Then, run the following to sync dependencies:
```bash
uv sync
```

Validator usage

With uv:
```bash
uv run validate.py --user lucas --images data --results runs/20260127_103351/20260127_103351_dataset.jsonl --corrections
```

Without uv:
```bash
python validate.py --user lucas --images data --results runs/20260127_103351/20260127_103351_dataset.jsonl --corrections
```
- --user: saves who did what; keep the same username each time.

- --images: root folder of all images (subfolders are fine).

- --results: dataset location.

- --corrections: enables corrections to the field.