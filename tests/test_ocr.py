from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace

from PIL import Image

from patientjournals.batch import ocr_context as batch_ocr
from patientjournals.batch import submit_requests
from patientjournals.batch import upload as batch_upload
from patientjournals.config import config
from patientjournals.shared.ocr import (
    OcrAttempt,
    OcrDocument,
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
        self.payload = payload
        self.content_type = "image/png" if name.endswith(".png") else None

    def download_as_bytes(self) -> bytes:
        if self.payload is None:
            raise FileNotFoundError(self.name)
        return self.payload

    def upload_from_string(self, value: str | bytes, content_type: str) -> None:
        self.payload = value.encode("utf-8") if isinstance(value, str) else value


class FakeBucket:
    def __init__(self) -> None:
        self.objects: dict[str, FakeBlob] = {}

    def blob(self, name: str) -> FakeBlob:
        if name not in self.objects:
            self.objects[name] = FakeBlob(self, name)
        return self.objects[name]


def test_batch_ocr_sidecar_is_reused_only_for_matching_bytes(monkeypatch) -> None:
    monkeypatch.setattr(config, "ocr_enabled", True)
    monkeypatch.setattr(config, "ocr_required", True)
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

    first = batch_ocr.ocr_document_for_blob(image_blob)
    second = batch_ocr.ocr_document_for_blob(image_blob)

    assert first == second
    assert first is not None and first.image_sha256 == digest
    assert calls == 1
    assert bucket.blob("pages/page.png.ocr.json").payload is not None


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
