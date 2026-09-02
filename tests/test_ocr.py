from __future__ import annotations

import json
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from patientjournals.batch import ocr_context as batch_ocr
from patientjournals.batch import prepare_ocr as cloud_ocr
from patientjournals.batch import submit as batch_submit
from patientjournals.batch import submit_requests
from patientjournals.batch import upload as batch_upload
from patientjournals.config import config
from patientjournals.shared.ocr import (
    GoogleVisionOcrBackend,
    OcrAttempt,
    OcrDocument,
    OcrImageInput,
    OcrLine,
    extract_google_vision_lines,
    render_ocr_context,
)
from patientjournals.shared.preprocess import prepare_page


def _png_bytes(size: tuple[int, int] = (200, 100), color: str = "white") -> bytes:
    output = BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


class FakeOcrBackend:
    name = "fake-prepared-page"

    def __init__(self) -> None:
        self.received: bytes | None = None

    def detect(
        self,
        *,
        image_bytes: bytes,
        width: int,
        height: int,
        image_sha256: str,
        language_hints,
    ) -> OcrDocument:
        self.received = image_bytes
        return OcrDocument(
            image_sha256=image_sha256,
            width=width,
            height=height,
            lines=(OcrLine("Patient Journal", (10, 20, 700, 100)),),
            backend=self.name,
        )


def _symbol(text: str, break_type: int = 0):
    return SimpleNamespace(
        text=text,
        property=SimpleNamespace(
            detected_break=SimpleNamespace(type_=break_type)
        ),
    )


def _word(text: str, box: tuple[int, int, int, int], break_type: int):
    left, top, right, bottom = box
    symbols = [_symbol(char) for char in text]
    symbols[-1] = _symbol(text[-1], break_type)
    return SimpleNamespace(
        symbols=symbols,
        bounding_box=SimpleNamespace(
            vertices=(
                SimpleNamespace(x=left, y=top),
                SimpleNamespace(x=right, y=top),
                SimpleNamespace(x=right, y=bottom),
                SimpleNamespace(x=left, y=bottom),
            )
        ),
    )


def test_google_vision_hierarchy_collapses_to_normalized_lines() -> None:
    response = SimpleNamespace(
        full_text_annotation=SimpleNamespace(
            pages=(
                SimpleNamespace(
                    blocks=(
                        SimpleNamespace(
                            paragraphs=(
                                SimpleNamespace(
                                    words=(
                                        _word("Journal", (10, 10, 60, 20), 1),
                                        _word("1889", (70, 10, 100, 20), 5),
                                        _word("Feber", (20, 30, 60, 40), 5),
                                    )
                                ),
                            )
                        ),
                    )
                ),
            )
        )
    )

    lines = extract_google_vision_lines(response, width=200, height=100)

    assert lines == (
        OcrLine("Journal 1889", (50, 100, 500, 200)),
        OcrLine("Feber", (100, 300, 300, 400)),
    )


def test_google_vision_backend_sends_multiple_images_in_one_rpc() -> None:
    captured_requests = []
    success = SimpleNamespace(
        error=SimpleNamespace(message=""),
        full_text_annotation=SimpleNamespace(pages=()),
    )
    failure = SimpleNamespace(
        error=SimpleNamespace(message="quota"),
        full_text_annotation=SimpleNamespace(pages=()),
    )

    class Client:
        def batch_annotate_images(self, *, requests):
            captured_requests.extend(requests)
            return SimpleNamespace(responses=(success, failure))

    inputs = tuple(
        OcrImageInput(
            image_bytes=value,
            width=10,
            height=20,
            image_sha256=sha256(value).hexdigest(),
        )
        for value in (b"first", b"second")
    )

    attempts = GoogleVisionOcrBackend(Client()).detect_batch(
        images=inputs,
        language_hints=("da",),
    )

    assert len(captured_requests) == 2
    assert captured_requests[0].image.content == b"first"
    assert attempts[0].document is not None
    assert attempts[0].document.image_sha256 == sha256(b"first").hexdigest()
    assert attempts[1].document is None
    assert attempts[1].error == "Google Vision OCR failed: quota"


def test_prepare_page_ocr_scans_the_exact_serialized_model_bytes(
    tmp_path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "page.png"
    source_path.write_bytes(_png_bytes())
    backend = FakeOcrBackend()
    monkeypatch.setattr(config, "ocr_enabled", True)
    monkeypatch.setattr(config, "ocr_required", True)

    prepared = prepare_page(
        source_path,
        max_dim=80,
        margins=(5, 2, 3, 4),
        contrast_factor=1.1,
        ocr_backend=backend,
    )

    assert backend.received == prepared.image_bytes
    assert prepared.image_sha256 == sha256(prepared.image_bytes).hexdigest()
    with Image.open(BytesIO(prepared.image_bytes)) as final_image:
        assert (prepared.width, prepared.height) == final_image.size
    assert prepared.preprocessing["image_sha256"] == prepared.image_sha256
    assert prepared.preprocessing["ocr_line_count"] == 1
    assert "10,20,700,100|Patient Journal" in prepared.ocr_context


def test_ocr_prompt_format_contains_all_text_without_json_field_overhead() -> None:
    document = OcrDocument(
        image_sha256="a" * 64,
        width=100,
        height=200,
        lines=(
            OcrLine("første linje", (1, 2, 3, 4)),
            OcrLine("anden linje", (5, 6, 7, 8)),
        ),
        backend="fake",
    )

    rendered = render_ocr_context(document)

    assert "1,2,3,4|første linje" in rendered
    assert "5,6,7,8|anden linje" in rendered
    assert '"text"' not in rendered
    assert document.image_sha256 not in rendered


class FakeBlob:
    def __init__(self, bucket: "FakeBucket", name: str, payload: bytes | None = None):
        self.bucket = bucket
        self.name = name
        self._payload: bytes | None = None
        self.generation = ""
        self.size: int | None = None
        self.crc32c: str | None = None
        self.md5_hash: str | None = None
        self.etag: str | None = None
        self.download_calls: list[dict[str, object]] = []
        self.content_type = "image/png" if name.endswith(".png") else None
        if payload is not None:
            self.payload = payload

    @property
    def payload(self) -> bytes | None:
        return self._payload

    @payload.setter
    def payload(self, value: bytes | None) -> None:
        self._payload = value
        if value is None:
            return
        self.generation = str(int(self.generation or "0") + 1)
        self.size = len(value)
        digest = sha256(value).hexdigest()
        self.crc32c = f"crc-{digest[:12]}"
        self.md5_hash = digest[:24]
        self.etag = f"etag-{self.generation}"

    def download_as_bytes(self, **kwargs) -> bytes:
        self.download_calls.append(kwargs)
        if self.payload is None:
            raise FileNotFoundError(self.name)
        expected_generation = kwargs.get("if_generation_match")
        if expected_generation is not None:
            assert int(expected_generation) == int(self.generation)
        return self.payload

    def upload_from_string(self, value: str | bytes, content_type: str) -> None:
        self.payload = value.encode("utf-8") if isinstance(value, str) else value

    def reload(self) -> None:
        return None


class FakeBucket:
    def __init__(self, name: str = "test-bucket") -> None:
        self.name = name
        self.objects: dict[str, FakeBlob] = {}

    def blob(self, name: str) -> FakeBlob:
        if name not in self.objects:
            self.objects[name] = FakeBlob(self, name)
        return self.objects[name]


def test_batch_ocr_preparation_creates_generation_bound_reusable_sidecar(
    monkeypatch,
) -> None:
    monkeypatch.setattr(config, "ocr_enabled", True)
    monkeypatch.setattr(config, "batch_ocr_metadata_required", True)
    batch_ocr._METADATA_CACHE.clear()
    bucket = FakeBucket()
    image_blob = bucket.blob("pages/page.png")
    image_blob.payload = _png_bytes()
    digest = sha256(image_blob.payload).hexdigest()
    calls = 0

    def fake_detect(image_bytes: bytes) -> OcrAttempt:
        nonlocal calls
        calls += 1
        width, height = Image.open(BytesIO(image_bytes)).size
        return OcrAttempt(
            OcrDocument(
                image_sha256=sha256(image_bytes).hexdigest(),
                width=width,
                height=height,
                lines=(OcrLine("cached", (1, 2, 3, 4)),),
                backend="fake-batch",
            )
        )

    monkeypatch.setattr(batch_ocr, "detect_configured_ocr", fake_detect)

    result = batch_ocr.prepare_ocr_metadata_for_blob(image_blob)
    batch_ocr._METADATA_CACHE.clear()
    first = batch_ocr.ocr_document_for_blob(image_blob)
    second = batch_ocr.ocr_document_for_blob(image_blob)

    assert result.status == "prepared"
    assert first == second
    assert first is not None and first.image_sha256 == digest
    assert calls == 1
    assert image_blob.download_calls == [{"if_generation_match": 1}]
    sidecar = bucket.blob("pages/page.png.ocr.json")
    assert sidecar.payload is not None
    metadata = json.loads(sidecar.payload)
    assert metadata["version"] == 2
    assert metadata["source"]["generation"] == "1"


def test_batch_submission_rejects_stale_sidecar_without_fetching_image(
    monkeypatch,
) -> None:
    monkeypatch.setattr(config, "ocr_enabled", True)
    monkeypatch.setattr(config, "batch_ocr_metadata_required", True)
    batch_ocr._METADATA_CACHE.clear()
    bucket = FakeBucket()
    image_blob = bucket.blob("pages/page.png")
    image_blob.payload = _png_bytes(color="white")

    def fake_detect(image_bytes: bytes) -> OcrAttempt:
        width, height = Image.open(BytesIO(image_bytes)).size
        return OcrAttempt(
            OcrDocument(
                image_sha256=sha256(image_bytes).hexdigest(),
                width=width,
                height=height,
                lines=(OcrLine("old generation", (1, 2, 3, 4)),),
                backend="fake-batch",
            )
        )

    monkeypatch.setattr(batch_ocr, "detect_configured_ocr", fake_detect)
    assert batch_ocr.prepare_ocr_metadata_for_blob(image_blob).status == "prepared"
    image_download_count = len(image_blob.download_calls)

    image_blob.payload = _png_bytes(color="black")
    batch_ocr._METADATA_CACHE.clear()
    monkeypatch.setattr(
        batch_ocr,
        "detect_configured_ocr",
        lambda _image_bytes: pytest.fail("submission must not run OCR"),
    )

    with pytest.raises(RuntimeError, match="batch.ocr"):
        batch_ocr.validate_ocr_metadata_for_blobs([image_blob])

    assert len(image_blob.download_calls) == image_download_count


def test_cloud_ocr_prepares_all_images_and_writes_manifest(monkeypatch) -> None:
    monkeypatch.setattr(config, "ocr_enabled", True)
    monkeypatch.setattr(config, "batch_ocr_workers", 2)
    monkeypatch.setattr(
        config,
        "batch_ocr_manifest_object",
        "batch/ocr/test-manifest.json",
    )
    batch_ocr._METADATA_CACHE.clear()
    bucket = FakeBucket()
    blobs = [bucket.blob("pages/1.png"), bucket.blob("pages/2.png")]
    for blob in blobs:
        blob.payload = _png_bytes()

    api_batch_sizes: list[int] = []

    def fake_detect_batch(image_payloads) -> tuple[OcrAttempt, ...]:
        api_batch_sizes.append(len(image_payloads))
        attempts: list[OcrAttempt] = []
        for image_bytes in image_payloads:
            width, height = Image.open(BytesIO(image_bytes)).size
            attempts.append(
                OcrAttempt(
                    OcrDocument(
                        image_sha256=sha256(image_bytes).hexdigest(),
                        width=width,
                        height=height,
                        lines=(OcrLine("journal", (10, 20, 30, 40)),),
                        backend="fake-batch",
                    )
                )
            )
        return tuple(attempts)

    monkeypatch.setattr(
        batch_ocr,
        "detect_configured_ocr_batch",
        fake_detect_batch,
    )

    first = cloud_ocr.prepare_cloud_ocr_metadata(
        bucket=bucket,
        blobs=blobs,
        workers=2,
        log=lambda _message: None,
    )
    second = cloud_ocr.prepare_cloud_ocr_metadata(
        bucket=bucket,
        blobs=blobs,
        workers=2,
        log=lambda _message: None,
    )

    assert (first.selected, first.prepared, first.cached, first.failed) == (2, 2, 0, 0)
    assert (second.selected, second.prepared, second.cached, second.failed) == (2, 0, 2, 0)
    assert api_batch_sizes == [2]
    assert all(
        blob.download_calls == [{"if_generation_match": 1}] for blob in blobs
    )
    manifest_payload = bucket.blob("batch/ocr/test-manifest.json").payload
    assert manifest_payload is not None
    manifest = json.loads(manifest_payload)
    assert manifest["selected"] == 2
    assert manifest["cached"] == 2
    assert {record["blob_name"] for record in manifest["records"]} == {
        "pages/1.png",
        "pages/2.png",
    }


def test_submission_ocr_preflight_scans_only_missing_selected_pages(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(config, "ocr_enabled", True)
    monkeypatch.setattr(config, "batch_ocr_metadata_required", True)
    monkeypatch.setattr(config, "batch_ocr_workers", 2)
    monkeypatch.setattr(config, "batch_ocr_api_batch_size", 16)
    monkeypatch.setattr(
        config,
        "batch_ocr_manifest_object",
        "batch/ocr/test-submission-manifest.json",
    )
    batch_ocr._METADATA_CACHE.clear()
    bucket = FakeBucket()
    first = bucket.blob("pages/1.png")
    second = bucket.blob("pages/2.png")
    first.payload = _png_bytes(color="white")
    second.payload = _png_bytes(color="black")
    batch_sizes: list[int] = []

    def fake_detect_batch(image_payloads) -> tuple[OcrAttempt, ...]:
        batch_sizes.append(len(image_payloads))
        return tuple(
            OcrAttempt(
                OcrDocument(
                    image_sha256=sha256(image_bytes).hexdigest(),
                    width=200,
                    height=100,
                    lines=(OcrLine("journal", (1, 2, 3, 4)),),
                    backend="fake-batch",
                )
            )
            for image_bytes in image_payloads
        )

    monkeypatch.setattr(batch_ocr, "detect_configured_ocr_batch", fake_detect_batch)
    monkeypatch.setattr(
        batch_submit,
        "_upload_submit_artifact",
        lambda *, bucket, run_dir, path: (
            f"gs://{bucket.name}/batch/requests/{run_dir.name}/{path.name}"
        ),
    )

    cloud_ocr.prepare_cloud_ocr_metadata(
        bucket=bucket,
        blobs=[first],
        log=lambda _message: None,
    )
    result = batch_submit._prepare_submission_ocr(
        bucket=bucket,
        blobs=[first, second],
        run_dir=tmp_path,
        submitted_object_names_sha256="cohort-digest",
        log=lambda _message: None,
    )

    assert batch_sizes == [1, 1]
    assert result["prepared"] == 1
    assert result["cached"] == 1
    assert result["failed"] == 0
    artifact = json.loads(
        (tmp_path / batch_submit.OCR_PREFLIGHT_FILE_NAME).read_text(encoding="utf-8")
    )
    assert artifact["submitted_object_names_sha256"] == "cohort-digest"
    assert {record["blob_name"] for record in artifact["records"]} == {
        "pages/1.png",
        "pages/2.png",
    }


def test_submission_ocr_preflight_blocks_batch_after_ocr_failure(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(config, "ocr_enabled", True)
    monkeypatch.setattr(config, "batch_ocr_metadata_required", True)
    monkeypatch.setattr(config, "batch_ocr_workers", 1)
    monkeypatch.setattr(config, "batch_ocr_api_batch_size", 16)
    monkeypatch.setattr(
        config,
        "batch_ocr_manifest_object",
        "batch/ocr/test-failed-submission-manifest.json",
    )
    batch_ocr._METADATA_CACHE.clear()
    bucket = FakeBucket()
    image_blob = bucket.blob("pages/failed.png")
    image_blob.payload = _png_bytes()
    monkeypatch.setattr(
        batch_ocr,
        "detect_configured_ocr_batch",
        lambda image_payloads: tuple(
            OcrAttempt(document=None, error="quota exhausted")
            for _ in image_payloads
        ),
    )
    monkeypatch.setattr(
        batch_submit,
        "_upload_submit_artifact",
        lambda *, bucket, run_dir, path: (
            f"gs://{bucket.name}/batch/requests/{run_dir.name}/{path.name}"
        ),
    )

    with pytest.raises(RuntimeError, match="No extraction batch was submitted"):
        batch_submit._prepare_submission_ocr(
            bucket=bucket,
            blobs=[image_blob],
            run_dir=tmp_path,
            submitted_object_names_sha256="failed-cohort",
            log=lambda _message: None,
        )

    artifact = json.loads(
        (tmp_path / batch_submit.OCR_PREFLIGHT_FILE_NAME).read_text(encoding="utf-8")
    )
    assert artifact["failed"] == 1
    assert artifact["records"][0]["error"] == "quota exhausted"


def test_cloud_ocr_splits_provider_calls_at_sixteen_images() -> None:
    bucket = FakeBucket()
    blobs = [bucket.blob(f"pages/{index}.png") for index in range(34)]
    for blob in blobs:
        blob.payload = b"x"

    batches = cloud_ocr._split_vision_batches(
        blobs,
        batch_size=16,
        max_bytes=1_000,
    )

    assert [len(batch) for batch in batches] == [16, 16, 2]


def test_batch_request_appends_ocr_after_the_task_prompt(monkeypatch) -> None:
    bucket = FakeBucket()
    blob = bucket.blob("pages/page.png")
    blob.payload = _png_bytes()
    monkeypatch.setattr(
        submit_requests,
        "ocr_context_for_blob",
        lambda _blob: "OCR evidence:\n1,2,3,4|Journal",
    )

    request = submit_requests._build_request_line(
        blob,
        "bucket",
        for_vertex=True,
    )

    prompt = request["request"]["contents"][0]["parts"][-1]["text"]
    assert prompt.startswith(config.input_prompt.rstrip())
    assert prompt.endswith("1,2,3,4|Journal")


def test_batch_page_uploads_are_create_only(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class UploadBlob:
        def upload_from_string(self, value: bytes, **kwargs) -> None:
            captured["value"] = value
            captured.update(kwargs)

    class UploadBucket:
        def blob(self, _name: str) -> UploadBlob:
            return UploadBlob()

    monkeypatch.setattr(config, "upload_retry_attempts", 1)

    assert batch_upload._upload_blob_bytes(
        UploadBucket(),
        "pages/page.png",
        b"processed-image",
        "image/png",
    )
    assert captured["if_generation_match"] == 0
