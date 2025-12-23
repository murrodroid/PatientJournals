from pathlib import Path
from io import BytesIO
import io
from mimetypes import guess_type

from PIL import Image, ImageEnhance
from google import genai
from google.genai import types


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


def preprocess_image(
    path,
    max_dim=3000,
    margins=(0, 0, 0, 0),
    contrast_factor=1.0,
    output_format="PNG",
):
    img = load_image(path)
    img = resize_image(img, max_dim=max_dim)

    left, top, right, bottom = margins
    img = crop_margins(img, left=left, top=top, right=right, bottom=bottom)

    img = enhance_contrast(img, factor=contrast_factor)
    image_bytes, mime_type = image_to_bytes(img, format_hint=output_format)
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