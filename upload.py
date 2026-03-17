from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re

from google.cloud import storage
import pypdfium2 as pdfium
from tqdm import tqdm

from config import config
from preprocess import (
    resize_image,
    crop_margins,
    enhance_contrast,
    image_to_bytes,
)


_PAGE_NAME_PATTERN = re.compile(r"^page_(\d+)\.([A-Za-z0-9]+)$")


def _apply_image_settings(img):
    settings = config.image_settings or {}
    max_dim = settings.get("max_dim", 3000)
    margins = tuple(settings.get("margins", (0, 0, 0, 0)))
    contrast_factor = settings.get("contrast_factor", 1.0)
    output_format = settings.get("output_format", "PNG")

    img = resize_image(img, max_dim=max_dim)
    left, top, right, bottom = margins
    img = crop_margins(img, left=left, top=top, right=right, bottom=bottom)
    img = enhance_contrast(img, factor=contrast_factor)
    image_bytes, mime_type = image_to_bytes(img, format_hint=output_format)

    return image_bytes, mime_type, output_format


def _extension_for_format(output_format: str) -> str:
    fmt = output_format.strip().lower()
    if fmt in {"jpeg", "jpg"}:
        return "jpg"
    if fmt == "png":
        return "png"
    if fmt == "webp":
        return "webp"
    if fmt in {"tif", "tiff"}:
        return "tiff"
    return fmt or "png"


def _normalize_prefix(prefix: str) -> str:
    value = prefix.strip()
    if not value:
        return ""
    return f"{value.strip('/')}/"


def _allowed_page_extensions() -> set[str]:
    configured = {
        str(ext).lower().lstrip(".")
        for ext in (config.batch_input_extensions or ())
    }
    settings_output_format = (config.image_settings or {}).get("output_format", "PNG")
    configured.add(_extension_for_format(str(settings_output_format)).lower())
    return {ext for ext in configured if ext}


def _extract_page_number_from_blob_name(blob_name: str) -> tuple[int, str] | None:
    base_name = Path(blob_name).name
    match = _PAGE_NAME_PATTERN.match(base_name)
    if not match:
        return None
    try:
        page_number = int(match.group(1))
    except ValueError:
        return None
    extension = match.group(2).lower()
    return page_number, extension


def _list_uploaded_page_numbers(
    bucket: storage.Bucket,
    folder_prefix: str,
) -> set[int]:
    allowed_extensions = _allowed_page_extensions()
    uploaded_pages: set[int] = set()
    for blob in bucket.list_blobs(prefix=folder_prefix):
        if blob.name.endswith("/"):
            continue
        parsed = _extract_page_number_from_blob_name(blob.name)
        if parsed is None:
            continue
        page_number, extension = parsed
        if extension in allowed_extensions:
            uploaded_pages.add(page_number)
    return uploaded_pages


def _pdf_page_count(pdf_path: Path) -> int:
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        total_pages = len(doc)
    finally:
        doc.close()
    if total_pages <= 0:
        raise ValueError(f"Unable to determine page count for {pdf_path}")
    return total_pages


def _page_number_digits(total_pages: int) -> int:
    configured_digits = int(config.page_number_digits or 4)
    return max(1, configured_digits, len(str(total_pages)))


def _upload_blob_bytes(
    bucket: storage.Bucket,
    blob_path: str,
    image_bytes: bytes,
    mime_type: str,
) -> None:
    blob = bucket.blob(blob_path)
    blob.upload_from_string(image_bytes, content_type=mime_type)


def _list_target_pdfs(target_folder: str | None) -> list[Path]:
    if not target_folder:
        raise KeyError("target_folder is missing from config")
    folder = Path(target_folder).expanduser()
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"target_folder not found or not a directory: {folder}")
    pdfs = sorted(p for p in folder.glob("*.pdf") if p.is_file())
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {folder}")
    return pdfs


def _resolve_service_account_path(service_account_file: str) -> Path:
    candidate = Path(service_account_file).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            f"service_account_file not found: {candidate}"
        )
    return candidate


def _build_bucket() -> storage.Bucket:
    bucket_name = (config.gcs_bucket_name or "").strip()
    if not bucket_name:
        raise ValueError(
            "config.gcs_bucket_name is empty. "
            "Set the target GCS bucket before uploading PDF pages."
        )
    service_account_path = _resolve_service_account_path(
        config.service_account_file
    )
    client = storage.Client.from_service_account_json(
        str(service_account_path)
    )
    return client.bucket(bucket_name)


def _upload_single_pdf(
    pdf_path: Path,
    bucket: storage.Bucket,
    render_dpi: int,
    batch_size: int,
    existing_page_numbers: set[int] | None = None,
) -> None:
    uploaded_pages = existing_page_numbers or set()
    upload_workers = max(1, int(config.upload_workers or 1))
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        total_pages = len(doc)
        if total_pages <= 0:
            raise ValueError(f"Unable to determine page count for {pdf_path}")
        digits = _page_number_digits(total_pages)
        scale = render_dpi / 72.0

        pages_prefix = _normalize_prefix(config.gcs_pages_prefix or "")
        folder_name = pdf_path.name
        output_format = (config.image_settings or {}).get("output_format", "PNG")
        extension = _extension_for_format(str(output_format))
        with tqdm(
            total=total_pages,
            desc=f"Uploading {pdf_path.name}",
            unit="page",
        ) as progress:
            for start in range(0, total_pages, batch_size):
                end = min(start + batch_size, total_pages)
                progress.set_postfix_str(f"batch {start // batch_size + 1}")
                pending_uploads: list[tuple[str, bytes, str, int]] = []
                for index in range(start, end):
                    page_number = index + 1
                    if page_number in uploaded_pages:
                        progress.update(1)
                        continue

                    page = doc.get_page(index)
                    img = None
                    try:
                        bitmap = page.render(scale=scale)
                        try:
                            img = bitmap.to_pil()
                        finally:
                            bitmap.close()

                        image_bytes, mime_type, _ = _apply_image_settings(img)
                        blob_path = (
                            f"{pages_prefix}{folder_name}/"
                            f"page_{page_number:0{digits}d}.{extension}"
                        )
                        pending_uploads.append(
                            (blob_path, image_bytes, mime_type, page_number)
                        )
                    finally:
                        if img is not None:
                            img.close()
                        page.close()

                if not pending_uploads:
                    continue

                if upload_workers == 1 or len(pending_uploads) == 1:
                    for blob_path, image_bytes, mime_type, page_number in pending_uploads:
                        _upload_blob_bytes(
                            bucket=bucket,
                            blob_path=blob_path,
                            image_bytes=image_bytes,
                            mime_type=mime_type,
                        )
                        uploaded_pages.add(page_number)
                        progress.update(1)
                    continue

                with ThreadPoolExecutor(
                    max_workers=min(upload_workers, len(pending_uploads))
                ) as executor:
                    futures = {
                        executor.submit(
                            _upload_blob_bytes,
                            bucket,
                            blob_path,
                            image_bytes,
                            mime_type,
                        ): page_number
                        for blob_path, image_bytes, mime_type, page_number in pending_uploads
                    }
                    for future in as_completed(futures):
                        future.result()
                        uploaded_pages.add(futures[future])
                        progress.update(1)
    finally:
        doc.close()


def _upload_pdf_paths(
    pdf_paths: list[Path],
    bucket: storage.Bucket,
) -> list[str]:
    render_dpi = int(config.pdf_render_dpi or 300)
    batch_size = int(config.batch_upload_limit or 20)
    if batch_size <= 0:
        batch_size = 1

    uploaded: list[str] = []
    for pdf_path in pdf_paths:
        folder_prefix = f"{_normalize_prefix(config.gcs_pages_prefix or '')}{pdf_path.name}/"
        existing_pages = _list_uploaded_page_numbers(bucket, folder_prefix)
        _upload_single_pdf(
            pdf_path=pdf_path,
            bucket=bucket,
            render_dpi=render_dpi,
            batch_size=batch_size,
            existing_page_numbers=existing_pages,
        )
        uploaded.append(pdf_path.name)
    return uploaded


def upload_missing_pdfs(
    pdf_paths: list[Path] | None = None,
    bucket: storage.Bucket | None = None,
) -> list[str]:
    candidate_paths = pdf_paths or _list_target_pdfs(config.target_folder)
    active_bucket = bucket or _build_bucket()

    incomplete_paths: list[Path] = []
    pages_prefix = _normalize_prefix(config.gcs_pages_prefix or "")
    for pdf_path in candidate_paths:
        total_pages = _pdf_page_count(pdf_path)
        folder_prefix = f"{pages_prefix}{pdf_path.name}/"
        uploaded_pages = _list_uploaded_page_numbers(active_bucket, folder_prefix)
        if len(uploaded_pages) < total_pages:
            incomplete_paths.append(pdf_path)

    if not incomplete_paths:
        return []
    return _upload_pdf_paths(incomplete_paths, active_bucket)


def upload_all_pdfs() -> None:
    pdf_paths = _list_target_pdfs(config.target_folder)
    bucket = _build_bucket()
    _upload_pdf_paths(pdf_paths, bucket)


def upload_pdf_pages() -> None:
    upload_all_pdfs()


if __name__ == "__main__":
    upload_all_pdfs()
