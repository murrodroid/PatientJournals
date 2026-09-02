from __future__ import annotations

import inspect

import pytest

pytest.importorskip("tkinter")

from patientjournals.app.ui import (
    PatientJournalsApp,
    _verification_scope_label,
    _verification_scope_value,
    _verification_thinking_index,
    _verification_thinking_level,
)
from patientjournals.app.web import APP_HTML


def test_tk_verification_control_mappings_are_discrete_and_reversible() -> None:
    assert [_verification_thinking_level(index) for index in range(3)] == [
        "low",
        "medium",
        "high",
    ]
    assert _verification_thinking_index("high") == 2
    assert _verification_thinking_index("invalid") == 2
    assert _verification_scope_value(_verification_scope_label("flagged")) == "flagged"
    assert _verification_scope_value(_verification_scope_label("all")) == "all"


def test_web_verification_controls_cover_submit_settings_and_jobs() -> None:
    for control_id in (
        "mainThinking",
        "cloudMainThinking",
        "verificationThinking",
        "cloudVerificationThinking",
        "jobVerificationThinking",
    ):
        assert f"verificationThinkingSlider('{control_id}'" in APP_HTML
        assert f"verificationThinkingLevel('{control_id}')" in APP_HTML

    assert "Maximum (default)" in APP_HTML
    assert "Risk-routed + control sample" in APP_HTML
    assert "Routine-page control sample (%)" in APP_HTML
    assert "verification_scope: 'all'" not in APP_HTML
    assert "verification_control_sample_percent" in APP_HTML


def test_submission_ui_groups_pipeline_choices_and_maps_job_type() -> None:
    source = inspect.getsource(PatientJournalsApp.show_submit)

    for label in (
        "General setup",
        "Schema Version",
        "Job type",
        "Validation",
        "OCR",
        "Model setup",
        "Main Model",
        "Validation Model",
    ):
        assert label in APP_HTML
        assert label in source

    assert 'id="jobType"' in APP_HTML
    assert 'value="subagentic"' in APP_HTML
    assert "subagents: ($('#jobType')?.value || 'single') === 'subagentic'" in APP_HTML
    assert "thinking_level: verificationThinkingLevel('mainThinking')" in APP_HTML
    assert 'subagents=job_type_var.get() == "Subagentic"' in source
    assert "thinking_level=_verification_thinking_level(" in source
    assert "onOffControl('modelValidationEnabled'" in APP_HTML
    assert "onOffControl('ocrUsage'" in APP_HTML
    assert "_on_off_button(" in source


def test_submission_ui_prepares_masterlist_boundaries_and_sampling() -> None:
    source = inspect.getsource(PatientJournalsApp.show_submit)

    for label in (
        "Page population",
        "All images",
        "From year",
        "To year",
        "Submission Type",
        "Complete",
        "Sample",
        "Sample seed",
        "Sample size",
        "Images in current range",
        "Pages in this submission",
    ):
        assert label in APP_HTML
        assert label in source

    assert 'id="rangeStartYear"' in APP_HTML
    assert 'id="rangeEndYear"' in APP_HTML
    assert 'id="submissionType"' in APP_HTML
    assert 'id="sampleSeed"' in APP_HTML
    assert 'id="samplePercent"' in APP_HTML
    assert "submission_type: submissionType" in APP_HTML
    assert "sample_percent: submissionType === 'sample'" in APP_HTML
    assert "sample_seed: submissionType === 'sample'" in APP_HTML
    assert '"sample"' in source
    assert "sample_percent=sample_percent" in source
    assert "sample_seed=sample_seed" in source


def test_submission_ui_uses_fixed_cloud_batch_population() -> None:
    source = inspect.getsource(PatientJournalsApp.show_submit)

    assert '<label>Source</label><select id="source"' not in APP_HTML
    assert '<label>Run mode</label><select id="mode"' not in APP_HTML
    assert 'id="inputChoices"' not in APP_HTML
    assert 'dataset_source="cloud"' in source
    assert 'run_mode="cloud_batch"' in source
    assert 'self._section(page, "Run")' not in source
    assert 'text="Dataset source"' not in source
    assert 'text="Local folder"' not in source
    assert 'text="Cloud dataset"' not in source


def test_tk_live_status_uses_settings_aware_workflow_boundary() -> None:
    source = inspect.getsource(PatientJournalsApp)

    assert ".readiness(job.run_dir)" in source
    assert ".live_batch_status(run_dir)" in source
    assert "resolve_batch_run_readiness(" not in source
    assert "list_batch_chunks_with_state(" not in source
