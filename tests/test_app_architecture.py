from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

from patientjournals.app import datasets as app_datasets
from patientjournals.app.access import run_access_checks
from patientjournals.app.catalog import (
    list_google_model_options,
    list_schema_options,
    resolve_schema_class,
)
from patientjournals.app.dashboard import latest_dataset_path, summarize_dashboard
from patientjournals.app.job_store import JobStore
from patientjournals.app.jobs import (
    JobRegistry,
    RegisteredJob,
    _is_success_state,
    batch_run_provider,
    build_retrieve_command,
    build_submit_command,
    finalize_dataset_with_failed_rows,
    find_dataset_near,
    list_batch_chunks,
    list_submit_jobs,
    read_dataset_preview,
    read_recorded_results,
    read_run_error,
    record_batch_chunk_statuses,
    resolve_batch_run_readiness,
    reusable_recorded_results,
    run_retrieve_direct,
)
from patientjournals.app.models import AppSettings, BatchChunkSummary, SubmitJobDraft
from patientjournals.app.settings_store import load_app_settings, save_app_settings
from patientjournals.app.workflows import WorkflowService
from patientjournals.config.schemas import FrontPage, TextPage, resolve_output_schema
from patientjournals.validation import sync as validation_sync


def test_schema_catalog_lists_top_level_output_schemas() -> None:
    names = {option.name for option in list_schema_options()}

    assert {"FrontPage", "TextPage"}.issubset(names)
    assert "Address" not in names


def test_schema_registry_resolves_aliases() -> None:
    assert resolve_output_schema("frontpage") is FrontPage
    assert resolve_schema_class("text_page") is TextPage


def test_google_model_catalog_is_google_only() -> None:
    models = list_google_model_options()

    assert models
    assert {model.provider for model in models} == {"gemini"}
    assert "gemini-3.1-pro" in {model.name for model in models}
    assert "gemini-3.5-flash" in {model.name for model in models}


def test_submit_command_carries_job_overrides() -> None:
    settings = AppSettings(gcs_bucket_name="bucket")
    draft = SubmitJobDraft(
        dataset_source="local",
        run_mode="local_api",
        schema_name="TextPage",
        model_name="gemini-2.5-flash",
        output_format="csv",
        local_path="/tmp/images",
    )

    command = build_submit_command(draft, settings)

    assert command.module == "patientjournals.local.cli"
    assert command.args == ("--data-folder", "/tmp/images")
    assert command.config_overrides["schema_name"] == "TextPage"
    assert command.config_overrides["model"] == "gemini-2.5-flash"
    assert command.config_overrides["output_format"] == "csv"


def test_submit_command_carries_selected_cloud_prefixes() -> None:
    settings = AppSettings(gcs_bucket_name="bucket")
    draft = SubmitJobDraft(
        dataset_source="cloud",
        run_mode="cloud_batch",
        schema_name="TextPage",
        model_name="gemini-2.5-flash",
        output_format="jsonl",
        cloud_prefix="pages/folder-a",
        cloud_prefixes=("pages/folder-a", "pages/folder-b"),
        num_batches=2,
    )

    command = build_submit_command(draft, settings)

    assert command.module == "patientjournals.batch.submit"
    assert command.args == ("--num-batches", "2")
    assert command.config_overrides["batch_input_prefix"] == "pages/folder-a"
    assert command.config_overrides["batch_input_prefixes"] == (
        "pages/folder-a",
        "pages/folder-b",
    )


def test_cloud_dataset_choices_are_newest_first(monkeypatch) -> None:
    class Blob:
        def __init__(self, name: str, updated: datetime) -> None:
            self.name = name
            self.updated = updated

    blobs = [
        Blob("pages/folder-a/page-1.png", datetime(2026, 5, 1, 8, tzinfo=timezone.utc)),
        Blob("pages/folder-b/page-1.png", datetime(2026, 5, 3, 9, 30, tzinfo=timezone.utc)),
        Blob("pages/folder-a/page-2.png", datetime(2026, 5, 2, 12, tzinfo=timezone.utc)),
    ]
    monkeypatch.setattr(app_datasets, "build_storage_bucket", lambda bucket_name: object())
    monkeypatch.setattr(app_datasets, "list_bucket_blobs", lambda bucket, prefix: blobs)
    monkeypatch.setattr(
        app_datasets,
        "select_bucket_image_blobs",
        lambda selected_blobs, glob_pattern=None: selected_blobs,
    )

    choices = app_datasets.list_cloud_dataset_choices(
        bucket_name="bucket",
        pages_prefix="pages",
    )

    assert [choice.prefix for choice in choices] == ["pages/folder-b", "pages/folder-a"]
    assert [choice.image_count for choice in choices] == [1, 2]
    assert choices[0].updated_at == "2026-05-03 09:30"


def test_retrieve_command_supports_selected_chunks_and_strategy() -> None:
    settings = AppSettings(
        gcs_bucket_name="bucket",
        batch_duplicate_strategy="provide_all",
    )

    command = build_retrieve_command(
        settings,
        run_dir="runs/submit_1",
        batch_names=("batch-a", "batch-b"),
        allow_partial=True,
        recover_missing_with_api=True,
        ignore_failed=True,
    )

    assert command.args == (
        "--run-dir",
        "runs/submit_1",
        "--batch-name",
        "batch-a",
        "--batch-name",
        "batch-b",
        "--allow-partial",
        "--recover-missing-with-api",
        "--ignore-failed",
        "--duplicate-strategy",
        "provide_all",
    )
    assert command.config_overrides["batch_duplicate_strategy"] == "provide_all"


def test_batch_chunks_are_grouped_from_submit_metadata(tmp_path) -> None:
    run_dir = tmp_path / "submit_1"
    run_dir.mkdir()
    (run_dir / "batch_job.json").write_text(
        json.dumps(
            {
                "provider": "gemini",
                "batch_jobs": [
                    {
                        "chunk_index": 2,
                        "total_chunks": 2,
                        "chunk_label": "chunk_002_of_002",
                        "batch_job_name": "batch-b",
                        "request_count": 4,
                        "requests_file": "b.jsonl",
                    },
                    {
                        "chunk_index": 1,
                        "total_chunks": 2,
                        "chunk_label": "chunk_001_of_002",
                        "batch_job_name": "batch-a",
                        "request_count": 5,
                        "requests_file": "a.jsonl",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    chunks = list_batch_chunks(run_dir)

    assert [chunk.batch_job_name for chunk in chunks] == ["batch-a", "batch-b"]
    assert [chunk.request_count for chunk in chunks] == [5, 4]
    assert [chunk.status for chunk in chunks] == ["submitted", "submitted"]


def test_partial_success_state_is_not_clean_success() -> None:
    assert _is_success_state("JOB_STATE_SUCCEEDED") is True
    assert _is_success_state("ended") is True
    assert _is_success_state("JOB_STATE_PARTIALLY_SUCCEEDED") is False
    partial = [
        BatchChunkSummary(
            chunk_index=1,
            total_chunks=1,
            chunk_label="chunk_001_of_001",
            batch_job_name="b1",
            request_count=10,
            status="JOB_STATE_PARTIALLY_SUCCEEDED",
        )
    ]
    assert resolve_batch_run_readiness("unused", chunks=partial).state == "failed"


def test_batch_readiness_waits_for_gemini_output_files(tmp_path, monkeypatch) -> None:
    from patientjournals.app import jobs as app_jobs

    run_dir = tmp_path / "submit_x"
    run_dir.mkdir()
    chunks = [
        BatchChunkSummary(
            chunk_index=1,
            total_chunks=1,
            chunk_label="chunk_001_of_001",
            batch_job_name="b1",
            request_count=10,
            status="JOB_STATE_SUCCEEDED",
        )
    ]
    monkeypatch.setattr(
        app_jobs,
        "_batch_model_progress",
        lambda _run_dir: SimpleNamespace(
            processed=3,
            total=10,
            detail="1 prediction file(s)",
        ),
    )

    readiness = resolve_batch_run_readiness(run_dir, chunks=chunks)

    assert readiness.state == "finalizing"
    assert readiness.output_rows == 3
    assert readiness.expected_rows == 10


def test_dashboard_summary_reads_metrics_and_validations(tmp_path) -> None:
    runs = tmp_path / "runs"
    run_dir = runs / "run_1"
    run_dir.mkdir(parents=True)
    (run_dir / "run_1_dataset.jsonl").write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    (run_dir / "image_processing_manifest.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"image_name": "a.png", "status": "success", "source": "local_api", "attempts": 1}),
                json.dumps({"image_name": "b.png", "status": "failed", "source": "api_recovery", "attempts": 3, "failure_reason": "schema"}),
                json.dumps({"image_name": "b.png", "status": "duplicate_skipped", "source": "batch_input_selection", "duplicate_action": "skipped_input_duplicate"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    validations = tmp_path / "validations" / "user_1"
    validations.mkdir(parents=True)
    (validations / "user_1_validations.csv").write_text(
        "label,column_name,image_name\naccept,x,a.png\nreject,y,b.png\n",
        encoding="utf-8",
    )

    summary = summarize_dashboard(run_root=runs, validations_root=tmp_path / "validations")

    assert latest_dataset_path(runs) == run_dir / "run_1_dataset.jsonl"
    assert summary.dataset_count == 1
    assert summary.dataset_rows == 2
    assert summary.processing_record_count == 3
    assert summary.processing_image_count == 2
    assert summary.status_counts == {"duplicate_skipped": 1, "failed": 1, "success": 1}
    assert summary.duplicate_actions == {"skipped_input_duplicate": 1}
    assert summary.validation_label_counts == {"accept": 1, "reject": 1}
    assert summary.validation_runs[0].accuracy == 50.0
    assert [metric.metric for metric in summary.validation_metrics] == ["x", "y"]


def test_dashboard_summary_reads_shared_validation_rows(tmp_path, monkeypatch) -> None:
    class Blob:
        name = "validations/alice_20260618/alice_20260618_validations.csv"
        updated = datetime(2026, 6, 18, 9, tzinfo=timezone.utc)

        def download_as_text(self, *, encoding="utf-8") -> str:
            return (
                "label,column_name,image_name,dataset_file,validator_id,decided_at\n"
                "accept,name,a.png,run_dataset.jsonl,alice,2026-06-18T09:00:00\n"
                "reject,date,b.png,run_dataset.jsonl,alice,2026-06-18T09:01:00\n"
            )

    class Bucket:
        name = "bucket"

    from patientjournals.app import dashboard as dashboard_module

    monkeypatch.setattr(dashboard_module, "build_storage_bucket", lambda _name: Bucket())
    monkeypatch.setattr(
        dashboard_module,
        "list_bucket_blobs",
        lambda _bucket, prefix=None: [Blob()],
    )

    summary = summarize_dashboard(
        run_root=tmp_path / "runs",
        validations_root=tmp_path / "validations",
        cloud_validations_bucket="bucket",
        cloud_validations_prefix="validations",
    )

    assert summary.validation_count == 2
    assert summary.validation_runs[0].validator_id == "alice"
    assert summary.validation_runs[0].accuracy == 50.0
    assert summary.validation_sync_error == ""


def test_dataset_library_lists_local_and_cloud_datasets(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "runs" / "submits" / "20260618_094819"
    run_dir.mkdir(parents=True)
    dataset = run_dir / "20260618_094819_dataset.jsonl"
    dataset.write_text('{"image_name": "a.png"}\n{"image_name": "b.png"}\n', encoding="utf-8")
    current = tmp_path / "runs" / "jobs" / "20260618_094819" / "datasets" / "current.jsonl"
    current.parent.mkdir(parents=True)
    current.write_text(
        '{"image_name": "a.png"}\n{"image_name": "b.png"}\n{"image_name": "c.png"}\n',
        encoding="utf-8",
    )

    local_items = app_datasets.list_local_dataset_library(tmp_path / "runs")

    assert len(local_items) == 1
    assert local_items[0].local_path == str(current)
    assert local_items[0].row_count == 3
    assert local_items[0].run_id == "20260618_094819"

    class Blob:
        name = "datasets/20260618_094819/20260618_094819_dataset.jsonl"
        size = 123
        updated = datetime(2026, 6, 18, 10, tzinfo=timezone.utc)
        metadata = {"rows": "1068"}

    class Bucket:
        name = "bucket"

    monkeypatch.setattr(app_datasets, "build_storage_bucket", lambda _name: Bucket())
    monkeypatch.setattr(
        app_datasets,
        "list_bucket_blobs",
        lambda _bucket, prefix=None: [Blob()],
    )

    cloud_items = app_datasets.list_cloud_dataset_library(
        bucket_name="bucket",
        datasets_prefix="datasets",
    )

    assert len(cloud_items) == 1
    assert cloud_items[0].source == "cloud"
    assert cloud_items[0].row_count == 1068
    assert cloud_items[0].gcs_uri.startswith("gs://bucket/datasets/")


def test_combine_dataset_files_prefers_first_successful_duplicate(tmp_path) -> None:
    first = tmp_path / "first_dataset.jsonl"
    first.write_text(
        '{"image_name":"a.png","value":"failed","failed":true}\n'
        '{"image_name":"b.png","value":"kept"}\n',
        encoding="utf-8",
    )
    second = tmp_path / "second_dataset.jsonl"
    second.write_text(
        '{"image_name":"a.png","value":"success","failed":false}\n'
        '{"file_name":"/images/c.png","value":"identity-filled"}\n',
        encoding="utf-8",
    )

    result = app_datasets.combine_dataset_files(
        [
            {"name": "first", "location": str(first), "local_path": str(first)},
            {"name": "second", "location": str(second), "local_path": str(second)},
        ],
        output_name="combined 0618",
        output_root=tmp_path / "runs",
        duplicate_strategy="first_successful",
    )

    rows = [
        json.loads(line)
        for line in Path(result["dataset_path"]).read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))

    assert result["row_count"] == 3
    assert result["duplicates_detected"] == 1
    assert result["duplicates_replaced"] == 1
    assert {row["image_name"] for row in rows} == {"a.png", "b.png", "c.png"}
    assert next(row for row in rows if row["image_name"] == "a.png")["value"] == "success"
    assert manifest["duplicate_rows"][0]["action"] == "replaced_failed_duplicate"
    assert manifest["duplicate_image_names"] == ["a.png"]


def test_combine_dataset_files_can_include_all_duplicates(tmp_path) -> None:
    first = tmp_path / "first_dataset.jsonl"
    first.write_text('{"image_name":"a.png","value":"one"}\n', encoding="utf-8")
    second = tmp_path / "second_dataset.jsonl"
    second.write_text('{"image_name":"a.png","value":"two"}\n', encoding="utf-8")

    result = app_datasets.combine_dataset_files(
        [
            {"name": "first", "location": str(first), "local_path": str(first)},
            {"name": "second", "location": str(second), "local_path": str(second)},
        ],
        output_name="all duplicates",
        output_root=tmp_path / "runs",
        duplicate_strategy="provide_all",
    )

    rows = [
        json.loads(line)
        for line in Path(result["dataset_path"]).read_text(encoding="utf-8").splitlines()
    ]

    assert result["row_count"] == 2
    assert result["duplicates_included"] == 1
    assert result["duplicates_skipped"] == 0
    assert [row["value"] for row in rows] == ["one", "two"]


def test_workflow_combine_datasets_uploads_dataset_and_manifest(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "source_dataset.jsonl"
    dataset.write_text('{"image_name":"a.png","value":"one"}\n', encoding="utf-8")
    uploaded: dict[str, dict[str, object]] = {}

    class Blob:
        def __init__(self, name: str) -> None:
            self.name = name
            self.metadata = {}

        def upload_from_filename(self, filename: str, *, content_type: str) -> None:
            uploaded[self.name] = {
                "filename": filename,
                "content_type": content_type,
                "metadata": self.metadata,
            }

    class Bucket:
        name = "bucket"

        def blob(self, name: str) -> Blob:
            return Blob(name)

    monkeypatch.setattr(app_datasets, "build_storage_bucket", lambda _name: Bucket())
    service = WorkflowService(
        AppSettings(
            local_runs_root=str(tmp_path / "runs"),
            gcs_bucket_name="bucket",
            datasets_gcs_prefix="shared/datasets",
        )
    )

    result = service.combine_datasets(
        [{"source": "local", "name": "source", "local_path": str(dataset)}],
        output_name="merged",
        duplicate_strategy="first_successful",
    )

    assert result["cloud_uri"] == "gs://bucket/shared/datasets/merged/merged_dataset.jsonl"
    assert result["manifest_cloud_uri"] == "gs://bucket/shared/datasets/merged/dataset_manifest.json"
    assert "shared/datasets/merged/merged_dataset.jsonl" in uploaded
    assert "shared/datasets/merged/dataset_manifest.json" in uploaded
    assert uploaded["shared/datasets/merged/merged_dataset.jsonl"]["content_type"] == "application/jsonl"
    assert uploaded["shared/datasets/merged/dataset_manifest.json"]["content_type"] == "application/json"


def test_validation_upload_writes_metadata_and_uploads_files(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "validations" / "alice_20260618"
    run_dir.mkdir(parents=True)
    csv_path = run_dir / "alice_20260618_validations.csv"
    csv_path.write_text("label,column_name\naccept,name\n", encoding="utf-8")
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"image_name": "a.png"}\n', encoding="utf-8")

    metadata_path = validation_sync.write_validation_metadata(
        run_dir=run_dir,
        csv_path=csv_path,
        dataset_path=dataset_path,
        validator_id="alice",
        decision_count=1,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["validator_id"] == "alice"
    assert metadata["dataset_file"] == "dataset.jsonl"
    assert metadata["decision_count"] == 1

    uploaded: list[tuple[str, str]] = []

    class Blob:
        def __init__(self, name: str) -> None:
            self.name = name

        def upload_from_filename(self, filename: str, *, content_type: str) -> None:
            uploaded.append((self.name, content_type))

    class Bucket:
        name = "bucket"

        def blob(self, name: str) -> Blob:
            return Blob(name)

    monkeypatch.setattr(validation_sync, "build_storage_bucket", lambda _name: Bucket())

    result = validation_sync.upload_validation_run(
        run_dir=run_dir,
        csv_path=csv_path,
        metadata_path=metadata_path,
        bucket_name="bucket",
        prefix="validations",
    )

    assert result["validation_csv_uri"].endswith("/alice_20260618_validations.csv")
    assert result["validation_metadata_uri"].endswith("/validation_metadata.json")
    assert uploaded == [
        ("validations/alice_20260618/alice_20260618_validations.csv", "text/csv"),
        ("validations/alice_20260618/validation_metadata.json", "application/json"),
    ]


def test_settings_store_roundtrip(tmp_path) -> None:
    path = tmp_path / "app_config.json"
    settings = AppSettings(
        auth_mode="service_account",
        service_account_file="service.json",
        gcp_project_id="project",
    )

    save_app_settings(settings, path)
    loaded = load_app_settings(path)

    assert loaded.service_account_file == "service.json"
    assert loaded.gcp_project_id == "project"


def test_access_checks_pass_with_gcloud_and_bucket_probe() -> None:
    import subprocess

    commands: list[tuple[str, ...]] = []

    def runner(command):
        commands.append(tuple(command))
        if command[:2] == ("gcloud", "--version"):
            return subprocess.CompletedProcess(command, 0, stdout="Google Cloud SDK 999\n", stderr="")
        if command[:3] == ("gcloud", "auth", "list"):
            return subprocess.CompletedProcess(command, 0, stdout="person@example.com\n", stderr="")
        if command[:3] == ("gcloud", "config", "get-value"):
            return subprocess.CompletedProcess(command, 0, stdout="project\n", stderr="")
        if command[:4] == ("gcloud", "auth", "application-default", "print-access-token"):
            return subprocess.CompletedProcess(command, 0, stdout="token\n", stderr="")
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="unexpected")

    class Blob:
        def __init__(self, name: str) -> None:
            self.name = name
            self.deleted = False

        def upload_from_string(self, _text: str, *, content_type: str) -> None:
            assert content_type == "text/plain"

        def download_as_text(self) -> str:
            return "patientjournals access check\n"

        def delete(self) -> None:
            self.deleted = True

    class Bucket:
        def exists(self) -> bool:
            return True

        def list_blobs(self, *, prefix=None, max_results=None):
            assert max_results == 1
            return iter([])

        def blob(self, name: str) -> Blob:
            return Blob(name)

    report = run_access_checks(
        AppSettings(
            auth_mode="adc",
            gcp_project_id="project",
            gcs_bucket_name="bucket",
            gcs_pages_prefix="pages",
            batch_requests_gcs_prefix="batch/requests",
            batch_outputs_gcs_prefix="batch/outputs",
        ),
        runner=runner,
        bucket_factory=lambda _name: Bucket(),
    )

    assert report.failed == 0
    assert any(result.name == "GCS write/read/delete" for result in report.results)
    assert ("gcloud", "--version") in commands


def test_access_checks_report_missing_gcloud() -> None:
    def runner(_command):
        raise FileNotFoundError("gcloud")

    report = run_access_checks(
        AppSettings(auth_mode="adc", gcs_bucket_name=""),
        runner=runner,
    )

    assert report.failed >= 1
    assert report.results[0].name == "gcloud installed"
    assert report.results[0].status == "fail"
    assert "Google Cloud CLI" in report.results[0].fix


def test_job_registry_roundtrip(tmp_path) -> None:
    registry = JobRegistry(tmp_path / "jobs.json")
    job = RegisteredJob(
        job_id="job1",
        created_at="2026-01-01T00:00:00",
        kind="local_api",
        status="running",
        command="python -m patientjournals.local.cli",
        config_path="job1.json",
        pid=123,
    )

    registry.add(job)

    assert registry.list() == [job]


def _write_submit_run(root, name, *, batch_meta, results=None, metadata=None):
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "batch_job.json").write_text(json.dumps(batch_meta), encoding="utf-8")
    if metadata is not None:
        (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    if results is not None:
        (run_dir / "batch_results.json").write_text(json.dumps(results), encoding="utf-8")
    return run_dir


def test_list_submit_jobs_only_returns_submits(tmp_path) -> None:
    _write_submit_run(
        tmp_path,
        "submit_20260101_000000",
        batch_meta={
            "model": "gemini-3.1-pro-preview",
            "request_count": 896,
            "batch_jobs": [
                {"chunk_index": 1, "total_chunks": 1, "batch_job_name": "b1", "request_count": 896}
            ],
        },
        metadata={
            "config_values": {
                "config": {
                    "gcs_bucket_name": "my-bucket",
                    "gcs_pages_prefix": "pages",
                    "batch_restrict_image_names": ["a.png", "b.png"],
                    "target_folder": "/data/folder-x",
                }
            }
        },
    )
    # Operational + incomplete dirs must be ignored.
    (tmp_path / "retrieve_20260101_010000").mkdir()
    (tmp_path / "collect_outputs_20260101_020000").mkdir()
    incomplete = tmp_path / "submit_20260101_030000"
    incomplete.mkdir()
    (incomplete / "metadata.json").write_text("{}", encoding="utf-8")

    jobs = list_submit_jobs(tmp_path)

    assert len(jobs) == 1
    job = jobs[0]
    assert job.kind == "batch"
    assert job.image_count == 896
    assert job.model == "gemini-3.1-pro-preview"
    assert job.retrieved is False
    assert job.succeeded is None and job.failed is None
    assert "folder-x" in job.input_location and "2 scoped" in job.input_location


def test_list_submit_jobs_folds_in_retrieval_results(tmp_path) -> None:
    _write_submit_run(
        tmp_path,
        "submit_20260101_000000",
        batch_meta={
            "model": "claude-opus-4-7",
            "request_count": 100,
            "batch_jobs": [
                {"chunk_index": 1, "total_chunks": 1, "batch_job_name": "b1", "request_count": 100}
            ],
        },
        results={"successful_pages": 90, "expected_pages": 100, "recovered_pages": 2},
    )

    job = list_submit_jobs(tmp_path)[0]

    assert job.retrieved is True
    assert job.status == "retrieved"
    assert job.succeeded == 90
    assert job.failed == 10
    assert job.recovered == 2


def test_job_store_migrates_legacy_submit_results(tmp_path) -> None:
    run_dir = _write_submit_run(
        tmp_path,
        "submit_20260101_000000",
        batch_meta={
            "model": "gemini-3.1-pro-preview",
            "request_count": 2,
            "batch_jobs": [
                {"chunk_index": 1, "total_chunks": 1, "batch_job_name": "b1", "request_count": 2}
            ],
        },
        results={
            "dataset_path": str(tmp_path / "submit_20260101_000000" / "dataset.jsonl"),
            "successful_pages": 2,
            "expected_pages": 2,
            "rows_written": 2,
        },
    )
    dataset = run_dir / "dataset.jsonl"
    dataset.write_text('{"image_name": "a.png"}\n{"image_name": "b.png"}\n', encoding="utf-8")

    jobs = list_submit_jobs(tmp_path)

    assert [job.job_id for job in jobs] == ["submit_20260101_000000"]
    record = JobStore(tmp_path).read("submit_20260101_000000")
    assert record["legacy"]["submit_run_dir"] == str(run_dir)
    assert record["metrics"]["successful_pages"] == 2
    current_path = record["dataset"]["current_path"]
    assert current_path.endswith("jobs/submit_20260101_000000/datasets/current.jsonl")
    assert Path(current_path).read_text(encoding="utf-8") == dataset.read_text(
        encoding="utf-8"
    )


def test_run_retrieve_direct_reuses_job_store_cache(tmp_path, monkeypatch) -> None:
    run_dir = _write_submit_run(
        tmp_path,
        "submit_20260101_000000",
        batch_meta={
            "model": "gemini-3.1-pro-preview",
            "provider": "gemini",
            "request_count": 1,
            "batch_jobs": [
                {"chunk_index": 1, "total_chunks": 1, "batch_job_name": "b1", "request_count": 1}
            ],
        },
    )
    dataset = run_dir / "dataset.jsonl"
    dataset.write_text('{"image_name": "a.png"}\n', encoding="utf-8")
    settings = AppSettings(batch_duplicate_strategy="first_successful")
    store = JobStore(tmp_path)
    signature = store.build_retrieval_signature(
        run_dir,
        allow_partial=True,
        duplicate_strategy="first_successful",
    )
    cached = store.record_retrieval(
        run_dir,
        {
            "retrieved_at": "2026-01-01T00:00:00",
            "dataset_path": str(dataset),
            "provider": "gemini",
            "batch_count": 1,
            "rows_written": 1,
            "error_rows": 0,
            "expected_pages": 1,
            "observed_pages": 1,
            "successful_pages": 1,
            "recovered_pages": 0,
            "missing_pages": 0,
        },
        signature=signature,
        operation="retrieve",
    )

    def fail_retrieve(*_args, **_kwargs):
        raise AssertionError("retrieval backend should not run for cached signature")

    from patientjournals.batch import service as batch_service

    monkeypatch.setattr(batch_service.BatchResultService, "retrieve", fail_retrieve)

    result = run_retrieve_direct(run_dir, settings, allow_partial=True)

    assert result["dataset_path"] == cached["dataset_path"]
    assert result["successful_pages"] == 1


def test_reusable_recorded_results_survives_retry_submitted_state(tmp_path) -> None:
    run_dir = _write_submit_run(
        tmp_path,
        "submit_20260101_000000",
        batch_meta={
            "model": "gemini-3.1-pro-preview",
            "provider": "gemini",
            "request_count": 2,
            "batch_jobs": [
                {"chunk_index": 1, "total_chunks": 1, "batch_job_name": "b1", "request_count": 2}
            ],
        },
    )
    dataset = run_dir / "dataset.jsonl"
    dataset.write_text('{"image_name": "a.png"}\n', encoding="utf-8")
    store = JobStore(tmp_path)
    signature = store.build_retrieval_signature(
        run_dir,
        allow_partial=True,
        duplicate_strategy="first_successful",
    )
    cached = store.record_retrieval(
        run_dir,
        {
            "retrieved_at": "2026-01-01T00:00:00",
            "dataset_path": str(dataset),
            "provider": "gemini",
            "batch_count": 1,
            "rows_written": 1,
            "error_rows": 1,
            "expected_pages": 2,
            "observed_pages": 2,
            "successful_pages": 1,
            "recovered_pages": 0,
            "missing_pages": 1,
        },
        signature=signature,
        operation="retrieve",
    )
    store.mark_retry_submitted(run_dir)

    result = reusable_recorded_results(run_dir)

    assert result["dataset_path"] == cached["dataset_path"]
    assert result["missing_pages"] == 1
    assert Path(result["dataset_path"]).is_file()


def test_find_dataset_near_prefers_job_store_current_dataset(tmp_path) -> None:
    run_dir = _write_submit_run(
        tmp_path,
        "submit_20260101_000000",
        batch_meta={
            "model": "gemini-3.1-pro-preview",
            "provider": "gemini",
            "request_count": 1,
            "batch_jobs": [
                {"chunk_index": 1, "total_chunks": 1, "batch_job_name": "b1", "request_count": 1}
            ],
        },
    )
    legacy = run_dir / "submit_20260101_000000_dataset.jsonl"
    legacy.write_text('{"image_name": "legacy.png"}\n', encoding="utf-8")
    source = run_dir / "dataset.jsonl"
    source.write_text('{"image_name": "current.png"}\n', encoding="utf-8")
    store = JobStore(tmp_path)
    current = store.record_retrieval(
        run_dir,
        {
            "retrieved_at": "2026-01-01T00:00:00",
            "dataset_path": str(source),
            "provider": "gemini",
            "batch_count": 1,
            "rows_written": 1,
            "error_rows": 0,
            "expected_pages": 1,
            "observed_pages": 1,
            "successful_pages": 1,
            "recovered_pages": 0,
            "missing_pages": 0,
        },
        signature=store.build_retrieval_signature(
            run_dir,
            allow_partial=True,
            duplicate_strategy="first_successful",
        ),
        operation="retrieve",
    )["dataset_path"]

    assert find_dataset_near(run_dir) == current
    assert find_dataset_near(run_dir / "gone.jsonl") == current
    assert find_dataset_near(legacy) == str(legacy)


def test_finalize_dataset_with_failed_rows_completes_current_dataset(tmp_path) -> None:
    run_dir = _write_submit_run(
        tmp_path,
        "submit_20260101_000000",
        batch_meta={
            "model": "gemini-3.1-pro-preview",
            "provider": "gemini",
            "request_count": 2,
            "batch_jobs": [
                {
                    "chunk_index": 1,
                    "total_chunks": 1,
                    "batch_job_name": "b1",
                    "requests_file": "batch_requests.jsonl",
                    "request_count": 2,
                }
            ],
        },
        results={
            "retrieved_at": "2026-01-01T00:00:00",
            "dataset_path": str(tmp_path / "submit_20260101_000000" / "dataset.jsonl"),
            "provider": "gemini",
            "batch_count": 1,
            "rows_written": 1,
            "error_rows": 1,
            "expected_pages": 2,
            "observed_pages": 2,
            "successful_pages": 1,
            "recovered_pages": 0,
            "failed_rows_included": 0,
            "missing_pages": 1,
            "ignore_failed": False,
        },
    )
    (run_dir / "batch_requests.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"key": "pages/a.png"}),
                json.dumps({"key": "pages/b.png"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "dataset.jsonl").write_text(
        json.dumps({"image_name": "a.png", "file_name": "pages/a.png"}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "image_processing_manifest.jsonl").write_text(
        json.dumps(
            {
                "image_reference": "pages/b.png",
                "image_name": "b.png",
                "status": "failed",
                "failure_reason": "schema_validation_failed",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = finalize_dataset_with_failed_rows(run_dir, AppSettings())

    assert result["rows_written"] == 2
    assert result["successful_pages"] == 1
    assert result["failed_rows_included"] == 1
    assert result["missing_pages"] == 0
    assert result["ignore_failed"] is True
    job = list_submit_jobs(tmp_path)[0]
    assert job.status == "retrieved"
    assert job.failed == 0
    assert job.failed_included == 1
    current_path = Path(result["dataset_path"])
    assert "jobs/submit_20260101_000000/datasets/current.jsonl" in str(current_path)
    rows = [
        json.loads(line)
        for line in current_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [
        {
            "image_name": "a.png",
            "file_name": "pages/a.png",
            "failed": False,
            "failure_reason": "",
        },
        {
            "image_name": "b.png",
            "file_name": "pages/b.png",
            "failed": True,
            "failure_reason": "schema_validation_failed",
        },
    ]
    recorded = json.loads((run_dir / "batch_results.json").read_text(encoding="utf-8"))
    assert recorded["missing_pages"] == 0
    assert JobStore(tmp_path).read("submit_20260101_000000")["status"] == "retrieved_complete"
    JobStore(tmp_path).mark_retry_submitted(run_dir)
    reopened = reusable_recorded_results(run_dir, ignore_failed=True)
    assert reopened["dataset_path"] == str(current_path)
    assert reopened["ignore_failed"] is True
    assert reopened["failed_rows_included"] == 1


def test_list_submit_jobs_groups_failed_retry_submissions(tmp_path) -> None:
    parent = _write_submit_run(
        tmp_path,
        "submit_20260101_000000",
        batch_meta={
            "model": "gemini-3.1-pro-preview",
            "request_count": 100,
            "batch_job_names": ["b1"],
            "batch_jobs": [
                {
                    "chunk_index": 1,
                    "total_chunks": 1,
                    "chunk_label": "chunk_001_of_001",
                    "batch_job_name": "b1",
                    "request_count": 100,
                }
            ],
        },
        results={"successful_pages": 90, "expected_pages": 100},
    )
    _write_submit_run(
        tmp_path,
        "submit_20260101_010000",
        batch_meta={
            "model": "gemini-3.1-pro-preview",
            "request_count": 10,
            "batch_job_names": ["b2"],
            "batch_jobs": [
                {
                    "chunk_index": 1,
                    "total_chunks": 1,
                    "chunk_label": "chunk_001_of_001",
                    "batch_job_name": "b2",
                    "request_count": 10,
                }
            ],
            "retry_source_run": str(parent),
            "retry_failed_keys_file": "failed_keys.jsonl",
        },
    )

    jobs = list_submit_jobs(tmp_path)

    assert [job.job_id for job in jobs] == ["submit_20260101_000000"]
    job = jobs[0]
    assert job.image_count == 100
    assert job.chunk_count == 2
    assert job.status == "retry_submitted"
    assert "1 retry batch" in job.detail

    chunks = list_batch_chunks(parent)
    assert [chunk.batch_job_name for chunk in chunks] == ["b1", "b2"]
    assert [chunk.status for chunk in chunks] == ["submitted", "submitted"]

    record_batch_chunk_statuses(parent, {"b2": "JOB_STATE_SUCCEEDED"})
    chunks = list_batch_chunks(parent)
    assert [chunk.status for chunk in chunks] == ["submitted", "JOB_STATE_SUCCEEDED"]

    repaired_parent = json.loads((parent / "batch_job.json").read_text(encoding="utf-8"))
    assert repaired_parent["job_group_id"] == parent.name
    assert repaired_parent["job_group_role"] == "root"
    assert repaired_parent["request_count"] == 100
    assert repaired_parent["batch_job_names"] == ["b1", "b2"]
    assert repaired_parent["batch_jobs"][1]["is_retry"] is True
    assert repaired_parent["batch_jobs"][1]["status"] == "JOB_STATE_SUCCEEDED"


def test_read_dataset_preview_jsonl(tmp_path) -> None:
    dataset = tmp_path / "run_dataset.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"image_name": f"p{i}.png", "text": "hi"}) for i in range(5)
        ),
        encoding="utf-8",
    )

    columns, rows = read_dataset_preview(dataset, limit=3)

    assert columns == ["image_name", "text"]
    assert len(rows) == 3
    assert rows[0]["image_name"] == "p0.png"


def test_read_dataset_preview_csv(tmp_path) -> None:
    dataset = tmp_path / "run_dataset.csv"
    dataset.write_text("image_name,text\na.png,one\nb.png,two\n", encoding="utf-8")

    columns, rows = read_dataset_preview(dataset, limit=10)

    assert columns == ["image_name", "text"]
    assert [row["image_name"] for row in rows] == ["a.png", "b.png"]


def test_read_run_error_returns_local_error_text(tmp_path) -> None:
    assert read_run_error(tmp_path) == ""
    (tmp_path / "error_001.txt").write_text("batch FAILED: quota", encoding="utf-8")
    assert "quota" in read_run_error(tmp_path)


def test_app_settings_cost_rate_roundtrip(tmp_path) -> None:
    path = tmp_path / "app_config.json"
    settings = AppSettings(estimated_cost_per_1k_images=12.5)

    save_app_settings(settings, path)
    loaded = load_app_settings(path)

    assert loaded.estimated_cost_per_1k_images == 12.5


def test_find_dataset_near_resolves_file_or_directory(tmp_path) -> None:
    run_dir = tmp_path / "retrieve_x"
    run_dir.mkdir()
    dataset = run_dir / "retrieve_x_dataset.jsonl"
    dataset.write_text('{"image_name": "a.png"}\n', encoding="utf-8")

    assert find_dataset_near(dataset) == str(dataset)
    # A missing exact path still resolves via the directory.
    assert find_dataset_near(run_dir / "gone.jsonl") == str(dataset)
    assert find_dataset_near(run_dir) == str(dataset)
    assert find_dataset_near(tmp_path / "nope") == ""
    assert find_dataset_near("") == ""


def test_read_recorded_results_reads_batch_results(tmp_path) -> None:
    assert read_recorded_results(tmp_path) == {}
    (tmp_path / "batch_results.json").write_text(
        json.dumps({"successful_pages": 5, "dataset_path": "/x/y.jsonl"}),
        encoding="utf-8",
    )
    recorded = read_recorded_results(tmp_path)
    assert recorded["successful_pages"] == 5
    assert recorded["dataset_path"] == "/x/y.jsonl"


def test_read_recorded_results_derives_missing_batch_results(tmp_path) -> None:
    run_dir = _write_submit_run(
        tmp_path / "submits",
        "20260618_094819",
        batch_meta={
            "provider": "gemini",
            "request_count": 3,
            "batch_jobs": [
                {
                    "chunk_index": 1,
                    "total_chunks": 1,
                    "batch_job_name": "b1",
                    "request_count": 3,
                    "requests_file": "batch_requests.jsonl",
                }
            ],
        },
    )
    (run_dir / "batch_requests.jsonl").write_text(
        "\n".join(
            json.dumps({"key": f"pages/folder/p{i}.png"}) for i in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "batch_output_001.jsonl").write_text("{}\n{}\n{}\n", encoding="utf-8")
    (run_dir / "20260618_094819_dataset.jsonl").write_text(
        "\n".join(
            json.dumps({"image_name": f"p{i}.png", "field": "ok"}) for i in range(2)
        )
        + "\n",
        encoding="utf-8",
    )

    recorded = read_recorded_results(run_dir)
    job = list_submit_jobs(tmp_path)[0]

    assert recorded["result_inferred"] is True
    assert recorded["expected_pages"] == 3
    assert recorded["successful_pages"] == 2
    assert recorded["missing_pages"] == 1
    assert job.succeeded == 2
    assert job.failed == 1


def test_read_run_error_falls_back_to_failure_diagnostics(tmp_path) -> None:
    manifest = tmp_path / "image_processing_manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "status": "failed",
                        "image_name": "a.png",
                        "failure_reason": "schema_validation_failed",
                    }
                ),
                json.dumps(
                    {
                        "status": "failed",
                        "image_name": "a.png",
                        "failure_reason": "schema_validation_failed",
                    }
                ),
                json.dumps(
                    {
                        "status": "failed",
                        "image_name": "b.png",
                        "failure_reason": "missing_response",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    text = read_run_error(tmp_path)

    assert "schema_validation_failed=1" in text
    assert "missing_response=1" in text


def test_batch_run_provider_reads_metadata(tmp_path) -> None:
    assert batch_run_provider(tmp_path) == ""
    (tmp_path / "batch_job.json").write_text(
        json.dumps({"provider": "Gemini"}), encoding="utf-8"
    )
    assert batch_run_provider(tmp_path) == "gemini"


def test_batch_run_provider_infers_legacy_gemini_metadata(tmp_path) -> None:
    (tmp_path / "batch_job.json").write_text(
        json.dumps(
            {
                "model": "gemini-3.1-pro-preview",
                "batch_job_name": "projects/p/locations/europe-north1/batchPredictionJobs/1",
            }
        ),
        encoding="utf-8",
    )

    assert batch_run_provider(tmp_path) == "gemini"


def test_app_settings_api_recovery_threshold_roundtrip(tmp_path) -> None:
    path = tmp_path / "app_config.json"
    save_app_settings(AppSettings(api_recovery_threshold=8), path)
    assert load_app_settings(path).api_recovery_threshold == 8


def test_recover_dataset_gaps_only_targets_missing_pages(tmp_path, monkeypatch) -> None:
    from patientjournals.app import jobs as app_jobs
    from patientjournals.batch import retrieve as retrieve_module
    from patientjournals.shared.identity import image_name_from_reference

    # Existing dataset already covers 3 of 5 expected pages.
    run_dir = tmp_path / "submit_x"
    run_dir.mkdir()
    (run_dir / "batch_job.json").write_text(
        json.dumps(
            {
                "provider": "gemini",
                "batch_jobs": [{"batch_job_name": "b1", "request_count": 5}],
            }
        ),
        encoding="utf-8",
    )
    dataset = run_dir / "submit_x_dataset.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"image_name": f"p{i}.png", "field": "ok"}) for i in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "batch_results.json").write_text(
        json.dumps(
            {
                "dataset_path": str(dataset),
                "recovered_pages": 1,
                "recovery_history": [{"recovered_pages": 1, "method": "api"}],
            }
        ),
        encoding="utf-8",
    )

    expected = {f"pages/dir/p{i}.png" for i in range(5)}  # p0..p4; p3,p4 missing
    monkeypatch.setattr(
        retrieve_module,
        "_resolve_expected_request_keys",
        lambda **kwargs: set(expected),
    )

    recovered_for = {}

    def fake_recover(*, missing_keys, rows_to_flush, **kwargs):
        recovered_for["keys"] = set(missing_keys)
        for key in missing_keys:
            rows_to_flush.append(
                {"image_name": image_name_from_reference(key), "field": "recovered"}
            )
        return len(missing_keys)

    monkeypatch.setattr(
        retrieve_module, "_recover_missing_pages_via_api_key", fake_recover
    )

    result = app_jobs.recover_dataset_gaps(run_dir, AppSettings())

    # Only the 2 genuinely missing pages are recovered, not all 5.
    assert recovered_for["keys"] == {"pages/dir/p3.png", "pages/dir/p4.png"}
    assert result["recovered_pages"] == 3
    assert len(result["recovery_history"]) == 2
    assert result["successful_pages"] == 5
    assert result["missing_pages"] == 0
    assert result["api_recovery_attempted"] is True
    assert result["api_recovery_completed"] is True
    assert result["api_recovery_failed"] is False
    assert result["api_recovery_errors"] == []
    assert result["api_recovered_row_count"] == 2
    assert {row["image_name"] for row in result["api_recovered_rows"]} == {
        "p3.png",
        "p4.png",
    }
    # The recovered rows were appended to the canonical job dataset (3 -> 5).
    current_dataset = Path(result["dataset_path"])
    assert current_dataset.read_text(encoding="utf-8").strip().count("\n") + 1 == 5
    assert "jobs/submit_x/datasets/current.jsonl" in str(current_dataset)


def test_recover_dataset_gaps_reports_zero_row_api_completion(
    tmp_path,
    monkeypatch,
) -> None:
    from patientjournals.app import jobs as app_jobs
    from patientjournals.batch import retrieve as retrieve_module

    run_dir = tmp_path / "submit_x"
    run_dir.mkdir()
    (run_dir / "batch_job.json").write_text(
        json.dumps(
            {
                "provider": "gemini",
                "batch_jobs": [{"batch_job_name": "b1", "request_count": 2}],
            }
        ),
        encoding="utf-8",
    )
    dataset = run_dir / "submit_x_dataset.jsonl"
    dataset.write_text('{"image_name": "p0.png", "field": "ok"}\n', encoding="utf-8")
    (run_dir / "batch_results.json").write_text(
        json.dumps({"dataset_path": str(dataset)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        retrieve_module,
        "_resolve_expected_request_keys",
        lambda **kwargs: {"pages/dir/p0.png", "pages/dir/p1.png"},
    )

    def fake_recover(*, failures, **kwargs):
        failures["pages/dir/p1.png"] = "api_key_recovery_failed:permission_denied"
        return 0

    monkeypatch.setattr(
        retrieve_module,
        "_recover_missing_pages_via_api_key",
        fake_recover,
    )

    result = app_jobs.recover_dataset_gaps(run_dir, AppSettings())

    assert result["api_recovery_attempted"] is True
    assert result["api_recovery_completed"] is True
    assert result["api_recovery_failed"] is True
    assert result["api_recovered_row_count"] == 0
    assert result["api_recovered_rows"] == []
    assert result["api_recovery_errors"] == [
        {
            "image_name": "p1.png",
            "key": "pages/dir/p1.png",
            "failure_reason": "api_key_recovery_failed:permission_denied",
        }
    ]
    assert "permission_denied" in result["api_recovery_error_summary"]
    assert result["missing_pages"] == 1
