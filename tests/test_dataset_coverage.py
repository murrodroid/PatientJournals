import json

import pandas as pd

from patientjournals.shared.dataset_coverage import (
    copy_dataset_rows_for_keys,
    load_dataset_key_coverage,
    normalize_gcs_file_key,
)


def test_normalize_gcs_file_key_accepts_object_keys_and_matching_uris() -> None:
    assert normalize_gcs_file_key("pages/a.png") == "pages/a.png"
    assert (
        normalize_gcs_file_key(
            "gs://bucket/pages/a.png",
            bucket_name="bucket",
        )
        == "pages/a.png"
    )
    assert (
        normalize_gcs_file_key(
            "gs://other/pages/a.png",
            bucket_name="bucket",
        )
        is None
    )


def test_load_dataset_key_coverage_jsonl_normalizes_gcs_uris(tmp_path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"file_name": "pages/a.png"}),
                json.dumps({"file_name": "gs://bucket/pages/b.png"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fmt, keys, row_count = load_dataset_key_coverage(path, bucket_name="bucket")

    assert fmt == "jsonl"
    assert keys == {"pages/a.png", "pages/b.png"}
    assert row_count == 2


def test_copy_dataset_rows_for_keys_filters_csv_by_normalized_key(tmp_path) -> None:
    src = tmp_path / "dataset.csv"
    dest = tmp_path / "filtered.csv"
    pd.DataFrame(
        [
            {"file_name": "pages/a.png", "value": 1},
            {"file_name": "gs://bucket/pages/b.png", "value": 2},
            {"file_name": "pages/c.png", "value": 3},
        ]
    ).to_csv(src, index=False, sep="$")

    kept = copy_dataset_rows_for_keys(
        src,
        dest,
        keys={"pages/b.png"},
        output_format="csv",
        csv_sep="$",
        bucket_name="bucket",
    )

    assert kept == 1
    out = pd.read_csv(dest, sep="$")
    assert out["file_name"].tolist() == ["gs://bucket/pages/b.png"]
