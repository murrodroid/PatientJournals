import json

from PIL import Image

from patientjournals.data.inspection import (
    summarize_batch_data,
    validate_batch_data,
    write_json_report,
    write_validation_csv,
)


def test_summarize_batch_data_counts_files_and_folders(tmp_path) -> None:
    root = tmp_path / "images"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "page_0001.png").write_bytes(b"abc")
    (nested / "page_0002.jpg").write_bytes(b"abcd")
    (root / "notes.txt").write_text("ignore", encoding="utf-8")

    report = summarize_batch_data(
        root,
        glob_pattern="*",
        recursive=True,
        allowed_extensions={"png", "jpg"},
    )

    assert report["total_files"] == 3
    assert report["image_files"] == 2
    assert report["non_image_files"] == 1
    assert report["folder_count"] == 2
    assert report["files_by_extension"] == {"jpg": 1, "png": 1}
    assert report["image_size_bytes"]["total"] == 7


def test_summarize_batch_data_can_skip_nested_files(tmp_path) -> None:
    root = tmp_path / "images"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "page_0001.png").write_bytes(b"abc")
    (nested / "page_0002.png").write_bytes(b"abcd")

    report = summarize_batch_data(
        root,
        glob_pattern="*.png",
        recursive=False,
        allowed_extensions={"png"},
    )

    assert report["total_files"] == 1
    assert report["image_files"] == 1
    assert report["files_by_folder"] == {".": 1}


def test_validate_batch_data_detects_corrupt_images(tmp_path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    Image.new("RGB", (10, 12), "white").save(root / "valid.png")
    (root / "corrupt.png").write_bytes(b"not an image")

    report = validate_batch_data(
        root,
        glob_pattern="*.png",
        recursive=True,
        allowed_extensions={"png"},
    )
    records = {record["file_name"]: record for record in report["records"]}

    assert report["status"] == "error"
    assert report["total_images"] == 2
    assert report["ok_count"] == 1
    assert report["error_count"] == 1
    assert records["valid.png"]["width"] == 10
    assert records["valid.png"]["height"] == 12
    assert records["corrupt.png"]["status"] == "error"
    assert "image_open_failed" in records["corrupt.png"]["issues"]


def test_validation_reports_are_written(tmp_path) -> None:
    report = {
        "status": "ok",
        "records": [
            {
                "status": "ok",
                "issues": "",
                "warnings": "",
                "file_name": "page.png",
                "relative_path": "page.png",
                "parent_folder": ".",
                "extension": "png",
                "size_bytes": 10,
                "width": 1,
                "height": 1,
                "format": "PNG",
                "mode": "RGB",
            }
        ],
    }

    json_path = write_json_report(report, tmp_path, "report")
    csv_path = write_validation_csv(report, tmp_path, "report")

    assert json.loads(json_path.read_text(encoding="utf-8"))["status"] == "ok"
    assert "page.png" in csv_path.read_text(encoding="utf-8")
