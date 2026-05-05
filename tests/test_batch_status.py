from types import SimpleNamespace

from patientjournals.batch import status


def test_request_count_from_payload_sums_chunk_counts() -> None:
    payload = {
        "batch_jobs": [
            {"request_count": 2},
            {"request_count": 3},
        ]
    }

    assert status._request_count_from_payload(payload) == 5


def test_anthropic_model_progress_counts_terminal_requests() -> None:
    jobs = [
        SimpleNamespace(
            request_counts=SimpleNamespace(
                succeeded=7,
                errored=2,
                canceled=1,
                expired=0,
                processing=5,
            )
        )
    ]

    progress = status._anthropic_model_progress(jobs, total=15)

    assert progress.processed == 10
    assert progress.total == 15


def test_model_progress_line_includes_ratio() -> None:
    line = status._model_progress_line(
        status.ModelProgress(processed=10, total=20, detail="test")
    )

    assert line == "Model outputs: 10/20 (50.00%) (test)"


def test_gemini_model_progress_counts_prediction_rows(monkeypatch) -> None:
    job = SimpleNamespace(dest=SimpleNamespace(gcs_uri="gs://bucket/output"))
    monkeypatch.setattr(status, "_count_gemini_prediction_rows", lambda uris: (4, 2))

    progress = status._gemini_model_progress(
        {"batch": job},
        run_dir=None,
        total=8,
    )

    assert progress.processed == 4
    assert progress.total == 8
    assert progress.detail == "2 prediction file(s)"
