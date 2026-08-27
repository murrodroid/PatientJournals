from pathlib import Path
from io import BytesIO
import io
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageEnhance

from patientjournals.shared.ocr import (
    OcrBackend,
    OcrDocument,
    detect_configured_ocr,
    image_identity,
    render_ocr_context,
)


@dataclass(frozen=True)
class PreparedPage:
    """The exact image payload and OCR context supplied to a model request."""

    image_bytes: bytes
    mime_type: str
    width: int
    height: int
    image_sha256: str
    preprocessing: dict[str, Any]
    ocr: OcrDocument | None = None
    ocr_error: str | None = None

    @property
    def ocr_context(self) -> str:
        return render_ocr_context(self.ocr)


def load_image(path):
    path = Path(path)
    img = Image.open(path)
    return img.convert("RGB")


def resize_image(img, max_dim=3000):
    w, h = img.size
    longest = max(w, h)
    if longest <= max_dim:
        return img
    scale = max_dim / float(longest)
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.BICUBIC)


def crop_margins(img, left=0, top=0, right=0, bottom=0):
    w, h = img.size

    left = max(0, int(left))
    top = max(0, int(top))
    right = max(0, int(right))
    bottom = max(0, int(bottom))

    x1 = min(left, w - 1)
    y1 = min(top, h - 1)
    x2 = max(w - right, x1 + 1)
    y2 = max(h - bottom, y1 + 1)

    x2 = min(x2, w)
    y2 = min(y2, h)

    return img.crop((x1, y1, x2, y2))


def enhance_contrast(img, factor=1.0):
    if factor == 1.0:
        return img
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def image_to_bytes(img, format_hint="PNG"):
    buf = BytesIO()
    img.save(buf, format=format_hint)
    data = buf.getvalue()
    mime_type = {
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
        "TIFF": "image/tiff",
    }.get(format_hint.upper(), "application/octet-stream")
    return data, mime_type


def preprocess_pil_image_with_metadata(
    source: Image.Image,
    *,
    max_dim=3000,
    margins=(0, 0, 0, 0),
    contrast_factor=1.0,
    output_format="PNG",
    source_metadata: dict[str, Any] | None = None,
):
    original_format = source.format
    original_mode = source.mode
    original_size = source.size
    img = source.convert("RGB")

    img = resize_image(img, max_dim=max_dim)
    resized_size = img.size

    left, top, right, bottom = margins
    img = crop_margins(img, left=left, top=top, right=right, bottom=bottom)
    cropped_size = img.size

    img = enhance_contrast(img, factor=contrast_factor)
    image_bytes, mime_type = image_to_bytes(img, format_hint=output_format)
    output_width, output_height, image_sha256 = image_identity(image_bytes)
    metadata: dict[str, Any] = {
        **(source_metadata or {}),
        "original_width": original_size[0],
        "original_height": original_size[1],
        "original_mode": original_mode,
        "original_format": original_format,
        "max_dim": max_dim,
        "resized_width": resized_size[0],
        "resized_height": resized_size[1],
        "margins": {
            "left": int(left),
            "top": int(top),
            "right": int(right),
            "bottom": int(bottom),
        },
        "cropped_width": cropped_size[0],
        "cropped_height": cropped_size[1],
        "output_width": output_width,
        "output_height": output_height,
        "contrast_factor": contrast_factor,
        "output_format": output_format,
        "mime_type": mime_type,
        "output_bytes": len(image_bytes),
        "image_sha256": image_sha256,
    }
    return image_bytes, mime_type, metadata


def preprocess_image_with_metadata(
    path,
    max_dim=3000,
    margins=(0, 0, 0, 0),
    contrast_factor=1.0,
    output_format="PNG",
):
    path = Path(path)
    with Image.open(path) as source:
        return preprocess_pil_image_with_metadata(
            source,
            max_dim=max_dim,
            margins=margins,
            contrast_factor=contrast_factor,
            output_format=output_format,
            source_metadata={
                "source_path": str(path),
                "source_bytes": path.stat().st_size if path.exists() else None,
            },
        )


def prepare_page(
    path,
    *,
    max_dim=3000,
    margins=(0, 0, 0, 0),
    contrast_factor=1.0,
    output_format="PNG",
    ocr_backend: OcrBackend | None = None,
) -> PreparedPage:
    """Preprocess then OCR the final serialized payload without re-encoding it."""

    image_bytes, mime_type, metadata = preprocess_image_with_metadata(
        path,
        max_dim=max_dim,
        margins=margins,
        contrast_factor=contrast_factor,
        output_format=output_format,
    )
    width = int(metadata["output_width"])
    height = int(metadata["output_height"])
    image_sha256 = str(metadata["image_sha256"])
    attempt = detect_configured_ocr(image_bytes, backend=ocr_backend)
    context = render_ocr_context(attempt.document)
    metadata.update(
        {
            "ocr_enabled": attempt.document is not None or attempt.error is not None,
            "ocr_backend": attempt.document.backend if attempt.document else None,
            "ocr_line_count": len(attempt.document.lines) if attempt.document else 0,
            "ocr_context_characters": len(context),
            "ocr_estimated_input_tokens": (len(context) + 3) // 4,
            "ocr_error": attempt.error,
        }
    )
    return PreparedPage(
        image_bytes=image_bytes,
        mime_type=mime_type,
        width=width,
        height=height,
        image_sha256=image_sha256,
        preprocessing=metadata,
        ocr=attempt.document,
        ocr_error=attempt.error,
    )


def preprocess_image(
    path,
    max_dim=3000,
    margins=(0, 0, 0, 0),
    contrast_factor=1.0,
    output_format="PNG",
):
    image_bytes, mime_type, _metadata = preprocess_image_with_metadata(
        path,
        max_dim=max_dim,
        margins=margins,
        contrast_factor=contrast_factor,
        output_format=output_format,
    )
    return image_bytes, mime_type

if __name__ == "__main__":
    def preview_preprocessed_image(
        image_path: str | Path,
        max_dim: int = 3000,
        margins: tuple[int, int, int, int] = (0, 0, 0, 0),
        contrast_factor: float = 1.0,
        output_format: str = "PNG",
    ) -> tuple[bytes, str]:
        image_bytes, mime_type = preprocess_image(
            path=str(image_path),
            max_dim=max_dim,
            margins=margins,
            contrast_factor=contrast_factor,
            output_format=output_format,
        )

        img = Image.open(io.BytesIO(image_bytes))
        img.show()
        return image_bytes, mime_type
    
    preview_preprocessed_image(
    "data/test_image.png",
    max_dim=3000,
    margins=(400, 0, 0, 0),
    contrast_factor=1.1,
    output_format="PNG",
    )
