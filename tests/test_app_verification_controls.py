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


def test_tk_live_status_uses_settings_aware_workflow_boundary() -> None:
    source = inspect.getsource(PatientJournalsApp)

    assert ".readiness(job.run_dir)" in source
    assert ".live_batch_status(run_dir)" in source
    assert "resolve_batch_run_readiness(" not in source
    assert "list_batch_chunks_with_state(" not in source
