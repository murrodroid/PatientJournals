from __future__ import annotations

import json

from patientjournals.batch import submit_requests
from patientjournals.batch.ocr_context import CloudBlobIdentity
from patientjournals.config import config
from patientjournals.validation.input_manifest import (
    ExtractionImageBinding,
    InputImageManifestRecord,
    input_manifest_record_for_blob,
    read_extraction_image_bindings,
    verify_extraction_image_bindings,
    write_extraction_image_bindings,
)


def _identity(name: str, generation: str) -> CloudBlobIdentity:
    return CloudBlobIdentity(
        bucket="bucket",
        name=name,
        generation=generation,
        size=123,
        crc32c="crc",
        md5_hash="md5",
        etag=None,
    )


def _input_record() -> InputImageManifestRecord:
    return InputImageManifestRecord(
        key="pages/page.png",
        mime_type="image/png",
        image_source=_identity("pages/page.png", "7").to_dict(),
        ocr_sidecar_name="pages/page.png.ocr.json",
        ocr_sidecar_source=_identity("pages/page.png.ocr.json", "8").to_dict(),
        ocr_sidecar_sha256="sidecar",
        ocr_image_sha256="image",
        ocr_document_sha256="document",
        ocr_backend="test",
        ocr_line_count=1,
    )


class _Blob:
    def __init__(
        self,
        bucket: "_Bucket",
        name: str,
        generation: str,
    ) -> None:
        self.bucket = bucket
        self.name = name
        self.generation = generation
        self.size = 123
        self.crc32c = "crc"
        self.md5_hash = "md5"
        self.etag = None
        self.content_type = "image/png"
        self.signed_kwargs: dict[str, object] = {}

    def reload(self) -> None:
        return None

    def generate_signed_url(self, **kwargs) -> str:
        self.signed_kwargs = kwargs
        self.bucket.last_signed_blob = self
        return "https://signed.example/page"


class _Bucket:
    name = "bucket"

    def __init__(self) -> None:
        self.objects: dict[str, _Blob] = {
            "pages/page.png": _Blob(self, "pages/page.png", "7")
        }
        self.copy_kwargs: dict[str, object] = {}
        self.last_signed_blob: _Blob | None = None

    def blob(self, name: str, generation: int | None = None) -> _Blob:
        existing = self.objects.get(name)
        if existing is not None:
            if generation is not None:
                assert int(existing.generation) == int(generation)
            return existing
        return _Blob(self, name, str(generation or ""))

    def copy_blob(self, source, destination_bucket, **kwargs) -> _Blob:
        assert destination_bucket is self
        assert source.name == "pages/page.png"
        self.copy_kwargs = kwargs
        staged = _Blob(self, str(kwargs["new_name"]), "99")
        self.objects[staged.name] = staged
        return staged


def test_gemini_first_pass_uses_write_once_staged_generation(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(config, "batch_requests_gcs_prefix", "requests")
    bucket = _Bucket()
    path, bindings = write_extraction_image_bindings(
        bucket=bucket,
        records=(_input_record(),),
        provider="gemini",
        run_dir_name="run",
        path=tmp_path / "bindings.jsonl",
        workers=1,
    )

    binding = bindings[0]
    assert binding.reference_mode == "immutable_staged_uri"
    assert binding.request_uri.startswith("gs://bucket/requests/run/extraction_images/")
    assert bucket.copy_kwargs["if_source_generation_match"] == 7
    assert bucket.copy_kwargs["if_generation_match"] == 0
    verify_extraction_image_bindings(
        bucket=bucket,
        bindings=read_extraction_image_bindings(path),
        expected_keys=["pages/page.png"],
    )

    source_blob = bucket.objects["pages/page.png"]
    request = submit_requests._build_request_line(
        source_blob,
        "bucket",
        for_vertex=True,
        ocr_context=None,
        media_uri_override=binding.request_uri,
    )
    assert request["request"]["contents"][0]["parts"][0]["fileData"][
        "fileUri"
    ] == binding.request_uri


def test_image_manifest_can_pin_exact_bytes_without_ocr() -> None:
    blob = _Bucket().objects["pages/page.png"]

    record = input_manifest_record_for_blob(blob, include_ocr=False)

    assert record.key == "pages/page.png"
    assert record.source.generation == "7"
    assert record.ocr_enabled is False
    assert record.ocr_sidecar_name == ""
    assert record.ocr_sidecar_source == {}


def test_anthropic_first_pass_signed_url_is_generation_qualified(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(config, "model", "claude-sonnet-4-5")
    monkeypatch.setattr(config, "output_schema", {"type": "object"})
    monkeypatch.setattr(config, "batch_include_response_schema", True)
    monkeypatch.setattr(config, "subagents", False)
    monkeypatch.setattr(submit_requests, "ocr_context_for_blob", lambda _blob: None)
    bucket = _Bucket()
    source = _identity("pages/page.png", "7")
    binding = ExtractionImageBinding(
        key="pages/page.png",
        provider="anthropic",
        reference_mode="generation_qualified_signed_url",
        source_image=source.to_dict(),
        request_image=source.to_dict(),
        request_uri="gs://bucket/pages/page.png?generation=7",
    )
    manifest = tmp_path / "requests.jsonl"
    manifest.write_text(
        json.dumps(
            submit_requests._build_anthropic_manifest_line(
                bucket.objects["pages/page.png"], image_binding=binding
            )
        )
        + "\n",
        encoding="utf-8",
    )

    requests = submit_requests._build_anthropic_batch_requests(
        bucket=bucket,
        requests_path=manifest,
    )

    assert requests[0]["params"]["messages"][0]["content"][0]["source"][
        "url"
    ] == "https://signed.example/page"
    assert bucket.last_signed_blob is not None
    assert bucket.last_signed_blob.generation == "7"
    assert bucket.last_signed_blob.signed_kwargs["query_parameters"] == {
        "generation": "7"
    }
