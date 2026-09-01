from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import threading
from types import SimpleNamespace

import pytest

from patientjournals.app.dashboard import analyze_dataset_file
from patientjournals.app.job_store import JobStore
from patientjournals.app.models import AppSettings, BatchChunkSummary, SubmitJobDraft
from patientjournals.app.settings_store import load_app_settings
from patientjournals.app.access import AccessCheckReport, AccessCheckResult
from patientjournals.app import workflows as workflow_module
from patientjournals.app.workflows import WorkflowService
from patientjournals.config import config


def test_local_input_choices_lists_configured_image_folders(tmp_path) -> None:
    images_root = tmp_path / "images"
    folder = images_root / "folder-a"
    folder.mkdir(parents=True)
    (folder / "a.png").write_bytes(b"png")
    (folder / "b.jpg").write_bytes(b"jpg")

    service = WorkflowService(
        AppSettings(
            local_runs_root=str(tmp_path / "runs"),
            validation_images_root=str(images_root),
        )
    )

    choices = service.local_input_choices()

    assert any(item["path"] == str(folder) and item["image_count"] == 2 for item in choices)


def test_retrieve_many_deduplicates_and_reports_results(monkeypatch, tmp_path) -> None:
    service = WorkflowService(AppSettings(local_runs_root=str(tmp_path / "runs")))

    def fake_retrieve(run_dir: str, **kwargs):
        if run_dir == "bad":
            raise RuntimeError("failed")
        return {"run_dir": run_dir, "ignore_failed": kwargs["ignore_failed"]}

    monkeypatch.setattr(service, "retrieve_results", fake_retrieve)

    result = service.retrieve_many(["one", "bad", "one"], ignore_failed=True)

    assert result["requested"] == 2
    assert result["succeeded"] == 1
    assert result["failed"] == 1
    assert result["results"][0]["payload"]["ignore_failed"] is True
    assert result["failures"] == [{"run_dir": "bad", "error": "failed"}]


def test_live_batch_status_serializes_distinct_run_overrides(monkeypatch, tmp_path) -> None:
    service = WorkflowService(AppSettings(local_runs_root=str(tmp_path / "runs")))
    baseline_model = config.model
    models = {
        "run-one": "gemini-3.5-flash",
        "run-two": "gemini-3.1-pro-preview",
    }
    monkeypatch.setattr(
        workflow_module,
        "command_overrides_for_run",
        lambda _settings, run_dir: {"model": models[run_dir]},
    )

    first_entered = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    second_entered = threading.Event()
    observed: list[tuple[str, str]] = []

    def live_chunks(run_dir: str):
        observed.append((run_dir, config.model))
        if run_dir == "run-one":
            first_entered.set()
            assert release_first.wait(timeout=2)
        else:
            second_entered.set()
        return [
            BatchChunkSummary(
                chunk_index=1,
                total_chunks=1,
                chunk_label="chunk_001_of_001",
                batch_job_name=f"batch-{run_dir}",
                request_count=1,
                status="FAILED",
            )
        ]

    monkeypatch.setattr(workflow_module, "list_batch_chunks_with_state", live_chunks)

    def second_status():
        second_started.set()
        return service.live_batch_status("run-two")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(service.live_batch_status, "run-one")
        assert first_entered.wait(timeout=2)
        second = pool.submit(second_status)
        assert second_started.wait(timeout=2)
        try:
            assert not second_entered.wait(timeout=0.05)
        finally:
            release_first.set()
        first_result = first.result(timeout=2)
        second_result = second.result(timeout=2)

    assert observed == list(models.items())
    assert first_result["readiness"]["state"] == "failed"
    assert second_result["readiness"]["state"] == "failed"
    assert config.model == baseline_model


def test_retrieve_readiness_uses_source_run_overrides(monkeypatch, tmp_path) -> None:
    service = WorkflowService(AppSettings(local_runs_root=str(tmp_path / "runs")))
    baseline_model = config.model
    recorded_model = (
        "gemini-3.5-flash"
        if baseline_model != "gemini-3.5-flash"
        else "gemini-3.1-pro-preview"
    )
    monkeypatch.setattr(
        workflow_module,
        "command_overrides_for_run",
        lambda _settings, _run_dir: {"model": recorded_model},
    )
    observed: list[str] = []

    def readiness(_run_dir: str):
        observed.append(config.model)
        return SimpleNamespace(state="succeeded", detail="")

    monkeypatch.setattr(workflow_module, "resolve_batch_run_readiness", readiness)
    monkeypatch.setattr(
        workflow_module,
        "run_retrieve_direct",
        lambda *_args, **_kwargs: {"dataset_path": "candidate.jsonl"},
    )
    monkeypatch.setattr(
        workflow_module,
        "read_dataset_preview",
        lambda *_args, **_kwargs: ([], []),
    )

    service.retrieve_results("source-run")

    assert observed == [recorded_model]
    assert config.model == baseline_model


def test_dataset_analysis_handles_nested_json_values(tmp_path) -> None:
    dataset = tmp_path / "current.jsonl"
    rows = [
        {
            "image_name": "a.png",
            "fk_info": "FK",
            "patient": {"name": "A", "age": {"number": 12}},
            "crossed_out": "ignored text",
            "names": ["one", "two"],
            "metadata": {"page": 1},
            "empty_list": [],
            "empty_dict": {},
            "attempts": "2",
            "failed": "false",
        },
        {
            "image_name": "b.png",
            "fk_info": "",
            "patient": {"name": "", "age": {"number": None}},
            "crossed_out": "",
            "names": [],
            "metadata": {},
            "failure_reason": ["api", "timeout"],
            "attempts": 3,
            "failed": "true",
        },
    ]
    dataset.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    analysis = analyze_dataset_file(dataset)
    metadata_completeness = {
        item.column: item for item in analysis.metadata_field_completeness
    }
    schema_completeness = {
        item.column: item for item in analysis.schema_field_completeness
    }

    assert analysis.row_count == 2
    assert analysis.failed_rows == 1
    assert analysis.failure_reasons == {'["api", "timeout"]': 1}
    assert analysis.attempts["count"] == 2
    assert schema_completeness["fk_info"].populated == 1
    assert schema_completeness["patient.name"].populated == 1
    assert schema_completeness["patient.age.number"].populated == 1
    assert "crossed_out" not in schema_completeness
    assert metadata_completeness["crossed_out"].populated == 1
    assert metadata_completeness["names"].populated == 1
    assert metadata_completeness["metadata.page"].populated == 1
    assert metadata_completeness["empty_list"].populated == 0
    assert metadata_completeness["empty_dict"].populated == 0


def test_job_store_persists_background_tasks(tmp_path) -> None:
    store = JobStore(tmp_path)

    store.upsert_task(
        "task-1",
        kind="retrieve_many",
        status="succeeded",
        metadata={"jobs": 2},
        result={"succeeded": 2},
    )

    tasks = store.list_tasks()

    assert tasks[0]["task_id"] == "task-1"
    assert tasks[0]["kind"] == "retrieve_many"
    assert tasks[0]["metadata"] == {"jobs": 2}
    assert tasks[0]["result"] == {"succeeded": 2}


def test_cloud_settings_are_saved_to_config_file(tmp_path) -> None:
    config_path = tmp_path / "app_config.json"
    service = WorkflowService(
        AppSettings(local_runs_root=str(tmp_path / "runs")),
        settings_path=config_path,
    )

    updated = service.save_cloud_settings(
        {
            "auth_mode": "adc",
            "gcp_project_id": "project-1",
            "gcs_bucket_name": "bucket-1",
        }
    )

    assert updated["auth_mode"] == "adc"
    assert updated["gcp_project_id"] == "project-1"
    assert updated["gcs_bucket_name"] == "bucket-1"
    assert config_path.is_file()


def test_cloud_settings_persist_model_validation_defaults(tmp_path) -> None:
    config_path = tmp_path / "app_config.json"
    service = WorkflowService(
        AppSettings(local_runs_root=str(tmp_path / "runs")),
        settings_path=config_path,
    )

    updated = service.save_cloud_settings(
        {
            "ocr_enabled": False,
            "subagents": True,
            "model_validation_enabled": True,
            "verification_model": "claude-sonnet-4-5",
            "verification_thinking_level": "medium",
            "verification_scope": "all",
            "verification_apply_mode": "apply_patches",
            "verification_max_output_tokens": 2048,
            "verification_num_chunks": 3,
        }
    )

    assert updated["ocr_enabled"] is False
    assert updated["subagents"] is True
    assert updated["model_validation_enabled"] is True
    assert updated["verification_model"] == "claude-sonnet-4-5"
    assert updated["verification_num_chunks"] == 3
    assert load_app_settings(config_path).verification_apply_mode == "apply_patches"


def test_cloud_settings_reject_zero_sample_for_enabled_flagged_scope(tmp_path) -> None:
    service = WorkflowService(
        AppSettings(local_runs_root=str(tmp_path / "runs")),
        settings_path=tmp_path / "app_config.json",
    )

    with pytest.raises(ValueError, match="positive routine-page control sample"):
        service.save_cloud_settings(
            {
                "model_validation_enabled": True,
                "verification_scope": "flagged",
                "verification_control_sample_percent": 0,
            }
        )


def test_model_validation_rejects_local_api_before_submission(tmp_path) -> None:
    service = WorkflowService(
        AppSettings(local_runs_root=str(tmp_path / "runs")),
        settings_path=tmp_path / "app_config.json",
    )
    draft = SubmitJobDraft(
        dataset_source="local",
        run_mode="local_api",
        schema_name="TextPage",
        model_name="gemini-3.1-pro",
        model_validation_enabled=True,
    )

    try:
        service.submit_batch(draft)
    except ValueError as exc:
        assert "requires Cloud batch" in str(exc)
    else:
        raise AssertionError("local model validation should have been rejected")


def test_apply_mode_does_not_publish_non_publishable_validation(
    monkeypatch, tmp_path
) -> None:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "submits" / "20260827_120000"
    run_dir.mkdir(parents=True)
    (run_dir / "batch_job.json").write_text(
        json.dumps(
            {
                "model": "gemini-3.7-flash",
                "provider": "gemini",
                "model_validation_enabled": True,
                "verification_apply_mode": "apply_patches",
                "batch_jobs": [],
            }
        ),
        encoding="utf-8",
    )
    store = JobStore(runs_root)
    store.sync_legacy_submit_run(
        run_dir,
        batch_meta=json.loads((run_dir / "batch_job.json").read_text()),
        image_count=1,
        status="submitted",
    )
    candidate = run_dir / "candidate.jsonl"
    candidate.write_text('{"image_name":"a.png"}\n', encoding="utf-8")
    store.record_candidate_retrieval(
        run_dir,
        {
            "dataset_path": str(candidate),
            "expected_pages": 1,
            "successful_pages": 1,
        },
        signature="candidate",
    )
    verification_run = runs_root / "verifications" / "20260827_130000"
    verification_run.mkdir(parents=True)
    store.mark_model_validation_submitted(
        run_dir,
        verification_run_dir=verification_run,
        model="gemini-3.1-pro-preview",
        apply_mode="apply_patches",
    )

    from patientjournals.batch import verify as verify_module

    monkeypatch.setattr(
        verify_module,
        "retrieve_model_validation",
        lambda _request: {
            "run_dir": str(verification_run),
            "model": "gemini-3.1-pro-preview",
            "status": "unverifiable",
            "publishable": False,
            "dataset_path": None,
            "dataset_rows": 0,
            "expected_pages": 1,
            "completed_pages": 1,
            "successful_pages": 1,
            "missing_pages": 0,
            "unverifiable_pages": 1,
        },
    )

    service = WorkflowService(
        AppSettings(local_runs_root=str(runs_root)),
        settings_path=tmp_path / "app_config.json",
    )
    result = service.retrieve_model_validation(str(run_dir), allow_partial=True)

    record = store.record_for_run_dir(run_dir)
    assert result["model_validation_status"] == "unverifiable"
    assert record["status"] == "validation_unverifiable"
    assert record["dataset"].get("versions", []) == []


def test_submit_model_validation_passes_configurable_max_output_tokens(
    monkeypatch, tmp_path
) -> None:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "submits" / "20260827_140000"
    run_dir.mkdir(parents=True)
    batch_meta = {
        "model": "gemini-3.1-pro-preview",
        "provider": "gemini",
        "model_validation_enabled": True,
        "verification_control_sample_percent": 0.0,
        "batch_jobs": [],
    }
    (run_dir / "batch_job.json").write_text(
        json.dumps(batch_meta), encoding="utf-8"
    )
    store = JobStore(runs_root)
    store.sync_legacy_submit_run(
        run_dir,
        batch_meta=batch_meta,
        image_count=1,
        status="validation_pending",
    )
    candidate = run_dir / "candidate.jsonl"
    candidate.write_text('{"image_name":"a.png"}\n', encoding="utf-8")
    store.record_candidate_retrieval(
        run_dir,
        {
            "dataset_path": str(candidate),
            "expected_pages": 1,
            "successful_pages": 1,
        },
        signature="candidate",
    )

    verification_run = runs_root / "verifications" / "20260827_150000"
    seen = {}
    from patientjournals.batch import verify as verify_module

    def fake_submit(request):
        seen["request"] = request
        return {"run_dir": str(verification_run)}

    monkeypatch.setattr(verify_module, "submit_model_validation", fake_submit)
    service = WorkflowService(
        AppSettings(
            local_runs_root=str(runs_root),
            verification_max_output_tokens=3333,
        ),
        settings_path=tmp_path / "app_config.json",
    )

    service.submit_model_validation(str(run_dir), max_output_tokens=1777)

    assert seen["request"].max_output_tokens == 1777
    validation = store.record_for_run_dir(run_dir)["model_validation"]
    assert validation["max_output_tokens"] == 1777
    assert validation["control_sample_percent"] == 0.0


def test_cloud_access_report_uses_saved_settings(monkeypatch, tmp_path) -> None:
    seen = {}

    def fake_checks(settings):
        seen["project"] = settings.gcp_project_id
        return AccessCheckReport(
            (
                AccessCheckResult("gcloud installed", "pass", "ok"),
                AccessCheckResult("Vertex role", "warn", "not verified"),
            )
        )

    monkeypatch.setattr(workflow_module, "run_access_checks", fake_checks)
    service = WorkflowService(
        AppSettings(local_runs_root=str(tmp_path / "runs")),
        settings_path=tmp_path / "app_config.json",
    )

    report = service.cloud_access_report({"gcp_project_id": "project-2"})

    assert seen == {"project": "project-2"}
    assert report["ready"] is True
    assert report["passed"] == 1
    assert report["warnings"] == 1
    assert report["results"][0]["name"] == "gcloud installed"


def test_start_cloud_browser_login_launches_gcloud_adc(monkeypatch, tmp_path) -> None:
    commands = []

    class FakeProcess:
        pid = 1234

    def fake_popen(command):
        commands.append(tuple(command))
        return FakeProcess()

    monkeypatch.setattr(workflow_module.subprocess, "Popen", fake_popen)
    service = WorkflowService(
        AppSettings(local_runs_root=str(tmp_path / "runs"), auth_mode="service_account"),
        settings_path=tmp_path / "app_config.json",
    )

    result = service.start_cloud_browser_login(
        mode="adc",
        payload={"gcp_project_id": "project-3"},
    )

    assert commands == [("gcloud", "auth", "application-default", "login")]
    assert result["pid"] == 1234
    assert service.settings.auth_mode == "adc"
    assert service.settings.gcp_project_id == "project-3"
