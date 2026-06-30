import json

import pandas as pd

from patientjournals.shared.tools import (
    build_image_name_id_set,
    filter_dataset_by_input_ids,
    flush_rows,
    list_input_files,
    load_existing_dataset,
)


def test_list_input_files_respects_fp_mode(tmp_path) -> None:
    root = tmp_path / "data"
    standard = root / "standard"
    frontpage = root / "front_fp"
    standard.mkdir(parents=True)
    frontpage.mkdir(parents=True)
    standard_file = standard / "page_0001.png"
    frontpage_file = frontpage / "page_0002.png"
    standard_file.write_bytes(b"")
    frontpage_file.write_bytes(b"")

    base_cfg = {
        "target_folder": str(root),
        "input_glob": "*.png",
        "recursive": True,
        "fp_suffix": "_fp",
    }

    assert list_input_files({**base_cfg, "fp_mode": "all"}) == [
        str(frontpage_file),
        str(standard_file),
    ]
    assert list_input_files({**base_cfg, "fp_mode": "only_fp"}) == [
        str(frontpage_file)
    ]
    assert list_input_files({**base_cfg, "fp_mode": "exclude_fp"}) == [
        str(standard_file)
    ]


def test_filter_dataset_by_input_ids_jsonl(tmp_path) -> None:
    src = tmp_path / "dataset.jsonl"
    keep_file = tmp_path / "data" / "page_0001.png"
    drop_file = tmp_path / "data" / "page_0002.png"
    keep_file.parent.mkdir()
    keep_file.write_bytes(b"")
    drop_file.write_bytes(b"")
    src.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "image_name": keep_file.name,
                        "file_name": str(keep_file),
                        "value": 1,
                    }
                ),
                json.dumps({"file_name": str(drop_file), "value": 2}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "filtered.jsonl"

    kept = filter_dataset_by_input_ids(
        src,
        dest,
        input_ids={keep_file.name},
        output_format="jsonl",
    )

    assert kept == 1
    assert [json.loads(line) for line in dest.read_text(encoding="utf-8").splitlines()] == [
        {"image_name": keep_file.name, "file_name": str(keep_file), "value": 1}
    ]


def test_load_existing_dataset_csv(tmp_path) -> None:
    path = tmp_path / "dataset.csv"
    pd.DataFrame(
        [
            {"image_name": "a.png", "file_name": "/tmp/a.png", "value": 1},
            {"file_name": "b.png", "value": 2},
        ]
    ).to_csv(path, index=False, sep="$")

    fmt, files, row_count = load_existing_dataset(path)

    assert fmt == "csv"
    assert files == {"a.png", "b.png"}
    assert row_count == 2


def test_flush_rows_aligns_appended_csv_rows_to_existing_header(tmp_path) -> None:
    path = tmp_path / "dataset.csv"

    header_written = flush_rows(
        [
            {
                "image_name": "a.png",
                "file_name": "pages/a.png",
                "value": "ok",
                "failed": False,
                "failure_reason": "",
            }
        ],
        str(path),
        header_written=False,
        output_format="csv",
    )
    assert header_written is True

    flush_rows(
        [
            {
                "image_name": "b.png",
                "file_name": "pages/b.png",
                "failed": True,
                "failure_reason": "schema_validation_failed",
            }
        ],
        str(path),
        header_written=True,
        output_format="csv",
    )

    frame = pd.read_csv(path, sep="$")
    assert list(frame.columns) == [
        "image_name",
        "file_name",
        "value",
        "failed",
        "failure_reason",
    ]
    assert frame.loc[1, "image_name"] == "b.png"
    assert bool(frame.loc[1, "failed"]) is True
    assert frame.loc[1, "failure_reason"] == "schema_validation_failed"


def test_build_image_name_id_set_uses_basename_identity(tmp_path) -> None:
    paths = [
        str(tmp_path / "one" / "a.png"),
        str(tmp_path / "two" / "b.png"),
    ]

    assert build_image_name_id_set(paths) == {"a.png", "b.png"}
