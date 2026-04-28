import json

import pandas as pd

from patientjournals.shared.tools import (
    filter_dataset_by_input_ids,
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
                json.dumps({"file_name": str(keep_file), "value": 1}),
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
        input_ids={str(keep_file.resolve())},
        output_format="jsonl",
    )

    assert kept == 1
    assert [json.loads(line) for line in dest.read_text(encoding="utf-8").splitlines()] == [
        {"file_name": str(keep_file), "value": 1}
    ]


def test_load_existing_dataset_csv(tmp_path) -> None:
    path = tmp_path / "dataset.csv"
    pd.DataFrame(
        [
            {"file_name": "a.png", "value": 1},
            {"file_name": "b.png", "value": 2},
        ]
    ).to_csv(path, index=False, sep="$")

    fmt, files, row_count = load_existing_dataset(path)

    assert fmt == "csv"
    assert files == {"a.png", "b.png"}
    assert row_count == 2

