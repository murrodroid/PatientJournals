from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, Sequence

from PIL import Image

from patientjournals.config.prompts import ocr_context_header


OCR_COORDINATE_SCALE = 1000
_BREAK_NAMES = {
    0: "UNKNOWN",
    1: "SPACE",
    2: "SURE_SPACE",
    3: "EOL_SURE_SPACE",
    4: "HYPHEN",
    5: "LINE_BREAK",
}


@dataclass(frozen=True)
class OcrLine:
    """One OCR line with a compact, normalized axis-aligned bounding box."""

    text: str
    box: tuple[int, int, int, int]

    def to_dict(self) -> dict[str, object]:
        return {"text": self.text, "box": list(self.box)}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "OcrLine":
        raw_box = payload.get("box")
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            raise ValueError("OCR line box must contain four coordinates.")
        return cls(
            text=str(payload.get("text") or ""),
            box=tuple(int(value) for value in raw_box),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class OcrDocument:
    """OCR derived from, and cryptographically bound to, one image payload."""

    image_sha256: str
    width: int
    height: int
    lines: tuple[OcrLine, ...]
    backend: str
    coordinate_scale: int = OCR_COORDINATE_SCALE

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "image_sha256": self.image_sha256,
            "width": self.width,
            "height": self.height,
            "coordinate_scale": self.coordinate_scale,
            "backend": self.backend,
            "lines": [line.to_dict() for line in self.lines],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "OcrDocument":
        raw_lines = payload.get("lines")
        if not isinstance(raw_lines, list):
            raise ValueError("OCR document lines must be a list.")
        return cls(
            image_sha256=str(payload.get("image_sha256") or ""),
            width=int(payload.get("width") or 0),
            height=int(payload.get("height") or 0),
            coordinate_scale=int(
                payload.get("coordinate_scale") or OCR_COORDINATE_SCALE
            ),
            backend=str(payload.get("backend") or "unknown"),
            lines=tuple(
                OcrLine.from_dict(line)
                for line in raw_lines
                if isinstance(line, dict)
            ),
        )

    @classmethod
    def from_json(cls, value: str | bytes) -> "OcrDocument":
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("OCR sidecar must contain a JSON object.")
        return cls.from_dict(payload)


@dataclass(frozen=True)
class OcrAttempt:
    document: OcrDocument | None
    error: str | None = None


@dataclass(frozen=True)
class OcrImageInput:
    image_bytes: bytes
    width: int
    height: int
    image_sha256: str


class OcrBackend(Protocol):
    name: str

    def detect(
        self,
        *,
        image_bytes: bytes,
        width: int,
        height: int,
        image_sha256: str,
        language_hints: Sequence[str],
    ) -> OcrDocument: ...


def image_identity(image_bytes: bytes) -> tuple[int, int, str]:
    """Read canonical dimensions and digest from the exact serialized bytes."""

    with Image.open(BytesIO(image_bytes)) as image:
        width, height = image.size
    return width, height, sha256(image_bytes).hexdigest()


def _break_name(symbol: object) -> str:
    symbol_property = getattr(symbol, "property", None)
    detected_break = getattr(symbol_property, "detected_break", None)
    value = getattr(detected_break, "type_", None)
    if value is None:
        value = getattr(detected_break, "type", None)
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    try:
        return _BREAK_NAMES.get(int(value), "UNKNOWN")
    except (TypeError, ValueError):
        return "UNKNOWN"


def _word_box(word: object) -> tuple[int, int, int, int] | None:
    bounding_box = getattr(word, "bounding_box", None)
    vertices = getattr(bounding_box, "vertices", None) or ()
    points: list[tuple[int, int]] = []
    for vertex in vertices:
        x = getattr(vertex, "x", 0) or 0
        y = getattr(vertex, "y", 0) or 0
        points.append((int(x), int(y)))
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _normalize_box(
    box: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    scale: int = OCR_COORDINATE_SCALE,
) -> tuple[int, int, int, int]:
    def normalized(value: int, dimension: int) -> int:
        if dimension <= 0:
            return 0
        return min(scale, max(0, round(value * scale / dimension)))

    left, top, right, bottom = box
    return (
        normalized(left, width),
        normalized(top, height),
        normalized(right, width),
        normalized(bottom, height),
    )


def extract_google_vision_lines(
    response: object,
    *,
    width: int,
    height: int,
) -> tuple[OcrLine, ...]:
    """Collapse Vision's symbol hierarchy into token-efficient visual lines."""

    annotation = getattr(response, "full_text_annotation", None)
    pages = getattr(annotation, "pages", None) or ()
    output: list[OcrLine] = []

    for page in pages:
        for block in getattr(page, "blocks", None) or ():
            for paragraph in getattr(block, "paragraphs", None) or ():
                pieces: list[str] = []
                boxes: list[tuple[int, int, int, int]] = []

                def flush() -> None:
                    text = "".join(pieces).strip()
                    if text:
                        if boxes:
                            left = min(box[0] for box in boxes)
                            top = min(box[1] for box in boxes)
                            right = max(box[2] for box in boxes)
                            bottom = max(box[3] for box in boxes)
                        else:
                            left = top = right = bottom = 0
                        output.append(
                            OcrLine(
                                text=text,
                                box=_normalize_box(
                                    (left, top, right, bottom),
                                    width=width,
                                    height=height,
                                ),
                            )
                        )
                    pieces.clear()
                    boxes.clear()

                for word in getattr(paragraph, "words", None) or ():
                    symbols = list(getattr(word, "symbols", None) or ())
                    word_text = "".join(
                        str(getattr(symbol, "text", "") or "")
                        for symbol in symbols
                    )
                    if word_text:
                        if pieces and not pieces[-1].endswith((" ", "-")):
                            pieces.append(" ")
                        pieces.append(word_text)
                    box = _word_box(word)
                    if box is not None:
                        boxes.append(box)

                    break_name = _break_name(symbols[-1]) if symbols else "UNKNOWN"
                    if break_name in {"EOL_SURE_SPACE", "LINE_BREAK"}:
                        flush()
                    elif break_name == "HYPHEN":
                        pieces.append("-")
                        flush()
                    elif break_name in {"SPACE", "SURE_SPACE"}:
                        pieces.append(" ")
                flush()

    return tuple(output)


class GoogleVisionOcrBackend:
    name = "google_vision"

    def __init__(self, client: object):
        self.client = client

    def detect(
        self,
        *,
        image_bytes: bytes,
        width: int,
        height: int,
        image_sha256: str,
        language_hints: Sequence[str],
    ) -> OcrDocument:
        from google.cloud import vision

        kwargs: dict[str, Any] = {"image": vision.Image(content=image_bytes)}
        if language_hints:
            kwargs["image_context"] = vision.ImageContext(
                language_hints=list(language_hints)
            )
        response = self.client.document_text_detection(**kwargs)
        error = getattr(response, "error", None)
        error_message = str(getattr(error, "message", "") or "").strip()
        if error_message:
            raise RuntimeError(f"Google Vision OCR failed: {error_message}")
        return OcrDocument(
            image_sha256=image_sha256,
            width=width,
            height=height,
            lines=extract_google_vision_lines(
                response,
                width=width,
                height=height,
            ),
            backend=self.name,
        )

    def detect_batch(
        self,
        *,
        images: Sequence[OcrImageInput],
        language_hints: Sequence[str],
    ) -> tuple[OcrAttempt, ...]:
        """Send up to 16 images through one Vision images:annotate RPC."""

        from google.cloud import vision

        if not images:
            return ()
        if len(images) > 16:
            raise ValueError("Google Vision accepts at most 16 images per batch RPC.")

        image_context = (
            vision.ImageContext(language_hints=list(language_hints))
            if language_hints
            else None
        )
        feature = vision.Feature(
            type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION
        )
        requests = [
            vision.AnnotateImageRequest(
                image=vision.Image(content=item.image_bytes),
                features=[feature],
                image_context=image_context,
            )
            for item in images
        ]
        batch_response = self.client.batch_annotate_images(requests=requests)
        responses = tuple(getattr(batch_response, "responses", ()) or ())
        if len(responses) != len(images):
            raise RuntimeError(
                "Google Vision returned "
                f"{len(responses)} response(s) for {len(images)} image(s)."
            )

        attempts: list[OcrAttempt] = []
        for item, response in zip(images, responses, strict=True):
            error = getattr(response, "error", None)
            error_message = str(getattr(error, "message", "") or "").strip()
            if error_message:
                attempts.append(
                    OcrAttempt(
                        document=None,
                        error=f"Google Vision OCR failed: {error_message}",
                    )
                )
                continue
            attempts.append(
                OcrAttempt(
                    document=OcrDocument(
                        image_sha256=item.image_sha256,
                        width=item.width,
                        height=item.height,
                        lines=extract_google_vision_lines(
                            response,
                            width=item.width,
                            height=item.height,
                        ),
                        backend=self.name,
                    )
                )
            )
        return tuple(attempts)


class _UnavailableOcrBackend:
    name = "unavailable"

    def __init__(self, message: str):
        self.message = message

    def detect(self, **_kwargs: object) -> OcrDocument:
        raise RuntimeError(self.message)


@lru_cache(maxsize=8)
def _configured_backend(
    backend_name: str,
    auth_mode: str,
    service_account_file: str,
) -> OcrBackend:
    try:
        if backend_name != "google_vision":
            raise ValueError(f"Unsupported OCR backend '{backend_name}'.")

        from google.cloud import vision

        if auth_mode == "service_account":
            path = Path(service_account_file).expanduser()
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            if not path.exists():
                raise FileNotFoundError(f"OCR service account file not found: {path}")
            from google.oauth2 import service_account

            credentials = service_account.Credentials.from_service_account_file(
                str(path)
            )
            client = vision.ImageAnnotatorClient(credentials=credentials)
        elif auth_mode == "adc":
            client = vision.ImageAnnotatorClient()
        else:
            raise ValueError(
                "Google Vision OCR requires service_account or adc authentication; "
                f"received '{auth_mode}'."
            )
        return GoogleVisionOcrBackend(client)
    except Exception as exc:  # noqa: BLE001 - converted to configured fail-open behavior
        return _UnavailableOcrBackend(str(exc))


_DOCUMENT_CACHE: dict[tuple[str, str, tuple[str, ...]], OcrDocument] = {}
_DOCUMENT_CACHE_LOCK = Lock()
_DOCUMENT_CACHE_LIMIT = 256


def detect_ocr(
    image_bytes: bytes,
    *,
    backend: OcrBackend,
    language_hints: Sequence[str] = (),
) -> OcrDocument:
    width, height, digest = image_identity(image_bytes)
    cache_key = (backend.name, digest, tuple(language_hints))
    with _DOCUMENT_CACHE_LOCK:
        cached = _DOCUMENT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    document = backend.detect(
        image_bytes=image_bytes,
        width=width,
        height=height,
        image_sha256=digest,
        language_hints=language_hints,
    )
    if document.image_sha256 != digest:
        raise ValueError("OCR backend returned a digest for different image bytes.")
    if (document.width, document.height) != (width, height):
        raise ValueError("OCR backend returned dimensions for different image bytes.")

    with _DOCUMENT_CACHE_LOCK:
        if len(_DOCUMENT_CACHE) >= _DOCUMENT_CACHE_LIMIT:
            _DOCUMENT_CACHE.pop(next(iter(_DOCUMENT_CACHE)))
        _DOCUMENT_CACHE[cache_key] = document
    return document


def detect_configured_ocr(
    image_bytes: bytes,
    *,
    backend: OcrBackend | None = None,
) -> OcrAttempt:
    """Run configured OCR, failing open unless ``ocr_required`` is set."""

    from patientjournals.config import config

    if not bool(config.ocr_enabled):
        return OcrAttempt(document=None)

    active_backend = backend or _configured_backend(
        str(config.ocr_backend or "google_vision").strip().lower(),
        str(config.gcp_auth_mode or "adc").strip().lower(),
        str(config.service_account_file or ""),
    )
    try:
        document = detect_ocr(
            image_bytes,
            backend=active_backend,
            language_hints=tuple(config.ocr_language_hints or ()),
        )
        return OcrAttempt(document=document)
    except Exception as exc:  # noqa: BLE001 - required/optional behavior is configuration
        if bool(config.ocr_required):
            raise RuntimeError(f"OCR is required but failed: {exc}") from exc
        return OcrAttempt(document=None, error=str(exc))


def detect_configured_ocr_batch(
    image_payloads: Sequence[bytes],
    *,
    backend: OcrBackend | None = None,
) -> tuple[OcrAttempt, ...]:
    """OCR multiple images with one backend RPC when the backend supports it."""

    from patientjournals.config import config

    if not image_payloads:
        return ()
    if not bool(config.ocr_enabled):
        return tuple(OcrAttempt(document=None) for _ in image_payloads)

    active_backend = backend or _configured_backend(
        str(config.ocr_backend or "google_vision").strip().lower(),
        str(config.gcp_auth_mode or "adc").strip().lower(),
        str(config.service_account_file or ""),
    )
    language_hints = tuple(config.ocr_language_hints or ())
    inputs = tuple(
        OcrImageInput(
            image_bytes=image_bytes,
            width=identity[0],
            height=identity[1],
            image_sha256=identity[2],
        )
        for image_bytes in image_payloads
        for identity in (image_identity(image_bytes),)
    )

    try:
        batch_detect = getattr(active_backend, "detect_batch", None)
        if callable(batch_detect):
            attempts = tuple(
                batch_detect(images=inputs, language_hints=language_hints)
            )
        else:
            attempts = tuple(
                OcrAttempt(
                    document=active_backend.detect(
                        image_bytes=item.image_bytes,
                        width=item.width,
                        height=item.height,
                        image_sha256=item.image_sha256,
                        language_hints=language_hints,
                    )
                )
                for item in inputs
            )
        if len(attempts) != len(inputs):
            raise RuntimeError(
                f"OCR backend returned {len(attempts)} result(s) for "
                f"{len(inputs)} image(s)."
            )

        validated: list[OcrAttempt] = []
        for item, attempt in zip(inputs, attempts, strict=True):
            document = attempt.document
            if document is not None and (
                document.image_sha256 != item.image_sha256
                or (document.width, document.height) != (item.width, item.height)
            ):
                validated.append(
                    OcrAttempt(
                        document=None,
                        error="OCR backend returned metadata for different image bytes.",
                    )
                )
            else:
                validated.append(attempt)
        return tuple(validated)
    except Exception as exc:  # noqa: BLE001 - record one RPC failure per image
        if bool(config.ocr_required):
            raise RuntimeError(f"OCR batch is required but failed: {exc}") from exc
        return tuple(
            OcrAttempt(document=None, error=str(exc)) for _ in image_payloads
        )


def render_ocr_context(document: OcrDocument | None) -> str:
    """Render every line with minimal syntax and no repeated field names."""

    if document is None or not document.lines:
        return ""
    rows = [ocr_context_header(document.coordinate_scale)]
    rows.extend(
        f"{left},{top},{right},{bottom}|{line.text}"
        for line in document.lines
        for left, top, right, bottom in (line.box,)
    )
    return "\n".join(rows)
