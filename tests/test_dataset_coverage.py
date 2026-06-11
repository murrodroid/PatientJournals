import json

import pandas as pd

from patientjournals.shared.dataset_coverage import (
    copy_dataset_rows_for_image_names,
    load_dataset_image_coverage,
    normalize_dataset_image_name,
)


def test_normalize_dataset_image_name_accepts_paths_and_gcs_uris() -> None:
    assert normalize_dataset_image_name("pages/a.png") == "a.png"
    assert normalize_dataset_image_name("gs://bucket/pages/a.png") == "a.png"
    assert normalize_dataset_image_name("/tmp/images/a.png") == "a.png"


def test_load_dataset_image_coverage_jsonl_prefers_image_name_with_legacy_fallback(
    tmp_path,
) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"image_name": "a.png", "file_name": "pages/a.png"}),
                json.dumps({"file_name": "gs://bucket/pages/b.png"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fmt, image_names, row_count = load_dataset_image_coverage(path, bucket_name="bucket")

    assert fmt == "jsonl"
    assert image_names == {"a.png", "b.png"}
    assert row_count == 2


def test_copy_dataset_rows_for_image_names_filters_csv_by_primary_key(tmp_path) -> None:
    src = tmp_path / "dataset.csv"
    dest = tmp_path / "filtered.csv"
    pd.DataFrame(
        [
            {"image_name": "a.png", "file_name": "pages/a.png", "value": 1},
            {"image_name": "b.png", "file_name": "gs://bucket/pages/b.png", "value": 2},
            {"image_name": "c.png", "file_name": "pages/c.png", "value": 3},
        ]
    ).to_csv(src, index=False, sep="$")

    kept = copy_dataset_rows_for_image_names(
        src,
        dest,
        image_names={"b.png"},
        output_format="csv",
        csv_sep="$",
        bucket_name="bucket",
    )

    assert kept == 1
    out = pd.read_csv(dest, sep="$")
    assert out["image_name"].tolist() == ["b.png"]
    assert out["file_name"].tolist() == ["gs://bucket/pages/b.png"]
