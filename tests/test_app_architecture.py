from datetime import datetime, timezone
import json

from patientjournals.app import datasets as app_datasets
from patientjournals.app.catalog import (
    list_google_model_options,
    list_schema_options,
    resolve_schema_class,
)
from patientjournals.app.dashboard import latest_dataset_path, summarize_dashboard
from patientjournals.app.jobs import (
    JobRegistry,
    RegisteredJob,
    batch_run_provider,
    build_retrieve_command,
    build_submit_command,
    find_dataset_near,
    list_batch_chunks,
    list_submit_jobs,
    read_dataset_preview,
    read_recorded_results,
    read_run_error,
)
from patientjournals.app.models import AppSettings, SubmitJobDraft
from patientjournals.app.settings_store import load_app_settings, save_app_settings
from patientjournals.config.schemas import FrontPage, TextPage, resolve_output_schema


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
        results={"successful_pages": 90, "expected_pages": 100},
    )

    job = list_submit_jobs(tmp_path)[0]

    assert job.retrieved is True
    assert job.status == "retrieved"
    assert job.succeeded == 90
    assert job.failed == 10


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


def test_batch_run_provider_reads_metadata(tmp_path) -> None:
    assert batch_run_provider(tmp_path) == ""
    (tmp_path / "batch_job.json").write_text(
        json.dumps({"provider": "Gemini"}), encoding="utf-8"
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
        json.dumps({"dataset_path": str(dataset)}), encoding="utf-8"
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
    assert result["recovered_pages"] == 2
    assert result["successful_pages"] == 5
    assert result["missing_pages"] == 0
    # The recovered rows were appended to the existing dataset (3 -> 5).
    assert dataset.read_text(encoding="utf-8").strip().count("\n") + 1 == 5
