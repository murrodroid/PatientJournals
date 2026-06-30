from __future__ import annotations

import json

from patientjournals.app.dashboard import analyze_dataset_file
from patientjournals.app.job_store import JobStore
from patientjournals.app.models import AppSettings
from patientjournals.app.access import AccessCheckReport, AccessCheckResult
from patientjournals.app import workflows as workflow_module
from patientjournals.app.workflows import WorkflowService


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
