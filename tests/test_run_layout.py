import json

from patientjournals.shared import run_layout as rl


def _mk(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_classify_legacy_dir() -> None:
    assert rl.classify_legacy_dir("submit_20260101_000000") == "submit"
    assert rl.classify_legacy_dir("retrieve_20260101_000000") == "retrieve"
    assert rl.classify_legacy_dir("collect_outputs_20260101_000000") == "collect_outputs"
    assert rl.classify_legacy_dir("20260101_000000") == "local"
    assert rl.classify_legacy_dir("something_else") is None


def test_iter_run_dirs_reads_both_layouts(tmp_path) -> None:
    # New layout
    _mk(tmp_path / "submits" / "20260102_000000")
    # Legacy flat
    _mk(tmp_path / "submit_20260101_000000")
    # Unrelated
    _mk(tmp_path / "retrieves" / "20260101_120000")

    names = {p.name for p in rl.iter_run_dirs(tmp_path, "submit")}
    assert names == {"20260102_000000", "submit_20260101_000000"}


def test_iter_all_run_dirs(tmp_path) -> None:
    _mk(tmp_path / "submits" / "20260102_000000")
    _mk(tmp_path / "retrieves" / "20260101_120000")
    _mk(tmp_path / "20260101_030000")  # legacy bare/local
    found = {p.name for p in rl.iter_all_run_dirs(tmp_path)}
    assert found == {"20260102_000000", "20260101_120000", "20260101_030000"}


def test_reorganize_runs_moves_and_fixes_references(tmp_path) -> None:
    submit = _mk(tmp_path / "submit_20260101_000000")
    (submit / "batch_job.json").write_text('{"provider": "gemini"}', encoding="utf-8")
    retrieve = _mk(tmp_path / "retrieve_20260101_010000")
    dataset = retrieve / "retrieve_20260101_010000_dataset.jsonl"
    dataset.write_text('{"image_name": "a.png"}\n', encoding="utf-8")
    (submit / "batch_results.json").write_text(
        json.dumps({"dataset_path": str(dataset), "successful_pages": 1}),
        encoding="utf-8",
    )
    _mk(tmp_path / "collect_outputs_20260101_020000")
    _mk(tmp_path / "20260101_030000")  # local

    report = rl.reorganize_runs(tmp_path, apply=True)

    assert len(report["moved"]) == 4
    assert report["reference_fixes"] == 1
    assert (tmp_path / "submits" / "20260101_000000").is_dir()
    assert (tmp_path / "retrieves" / "20260101_010000").is_dir()
    assert (tmp_path / "collect_outputs" / "20260101_020000").is_dir()
    assert (tmp_path / "local" / "20260101_030000").is_dir()
    assert not (tmp_path / "submit_20260101_000000").exists()
    assert (tmp_path / rl.README_NAME).is_file()

    # The rewritten dataset reference still resolves.
    fixed = json.loads(
        (tmp_path / "submits" / "20260101_000000" / "batch_results.json").read_text()
    )
    from pathlib import Path

    assert Path(fixed["dataset_path"]).is_file()
    assert "retrieves/20260101_010000" in fixed["dataset_path"]


def test_reorganize_runs_dry_run_does_not_move(tmp_path) -> None:
    _mk(tmp_path / "submit_20260101_000000")
    report = rl.reorganize_runs(tmp_path, apply=False)
    assert report["moved"] and report["applied"] is False
    assert (tmp_path / "submit_20260101_000000").is_dir()
    assert not (tmp_path / "submits").exists()


def test_document_existing_runs_backfills_kind(tmp_path) -> None:
    run = _mk(tmp_path / "submits" / "20260101_000000")
    (run / "metadata.json").write_text('{"created_at": "x"}', encoding="utf-8")
    bare = _mk(tmp_path / "local" / "20260102_030405")  # no metadata.json

    documented = rl.document_existing_runs(tmp_path)

    assert documented == 2
    assert json.loads((run / "metadata.json").read_text())["kind"] == "submit"
    backfilled = json.loads((bare / "metadata.json").read_text())
    assert backfilled["kind"] == "local"
    assert backfilled["created_at"] == "2026-01-02T03:04:05"
    # Idempotent: a second pass writes nothing.
    assert rl.document_existing_runs(tmp_path) == 0
