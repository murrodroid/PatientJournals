"""Håndhæver ICM-strukturen: uden denne test skrider konventionen stille."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

STAGES = [
    "00_forundersoegelse",
    "01_datagrundlag",
    "02_facit",
    "03_maaleapparat",
    "04_billedforberedelse",
    "05_foerste_transskription",
    "06_prompt_og_model",
    "07_anden_stemme",
    "08_integration",
]

REQUIRED_SECTIONS = [
    "## Formål",
    "## Inputs",
    "## Process",
    "## Outputs",
    "## Test Contract",
    "## Handoff",
]


def test_icm_root_files_exist():
    for name in ("CLAUDE.md", "AGENTS.md", "CONTEXT.md", "PROGRESS.md", "_config/tdd.md"):
        assert (ROOT / name).is_file(), f"mangler {name}"


def test_claude_md_delegates_to_agents_md():
    # AGENTS.md skal være eneste kilde; CLAUDE.md må ikke føre sit eget liv.
    assert (ROOT / "CLAUDE.md").read_text(encoding="utf-8").strip() == "@AGENTS.md"


@pytest.mark.parametrize("stage", STAGES)
def test_stage_has_context_and_output(stage):
    stage_dir = ROOT / "stages" / stage
    assert stage_dir.is_dir(), f"mangler mappe {stage}"
    assert (stage_dir / "output").is_dir(), f"mangler {stage}/output"
    text = (stage_dir / "CONTEXT.md").read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        assert section in text, f"{stage}/CONTEXT.md mangler afsnittet {section!r}"


def test_stage_dirs_match_declared_list():
    found = sorted(p.name for p in (ROOT / "stages").iterdir() if p.is_dir())
    assert found == sorted(STAGES), "stages/ og STAGES-listen er uenige"


def test_stage_numbers_are_unique_and_ordered():
    numbers = [name.split("_", 1)[0] for name in STAGES]
    assert numbers == sorted(numbers), "stage-numre står ikke i rækkefølge"
    assert len(set(numbers)) == len(numbers), "to stages deler nummer"
