from pathlib import Path
from io import BytesIO
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


def crop_margins(img, margin_px=0):
    if margin_px <= 0:
        return img
    w, h = img.size
    left = margin_px
    top = margin_px
    right = max(w - margin_px, left + 1)
    bottom = max(h - margin_px, top + 1)
    return img.crop((left, top, right, bottom))


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
    margin_px=0,
    contrast_factor=1.0,
    output_format="PNG",
):
    img = load_image(path)
    img = resize_image(img, max_dim=max_dim)
    img = crop_margins(img, margin_px=margin_px)
    img = enhance_contrast(img, factor=contrast_factor)
    image_bytes, mime_type = image_to_bytes(img, format_hint=output_format)
    return image_bytes, mime_type
