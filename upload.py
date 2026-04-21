from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import time

from google.cloud import storage
from PIL import Image, UnidentifiedImageError
import pypdfium2 as pdfium
from tqdm import tqdm

from config import config
from preprocess import (
    resize_image,
    crop_margins,
    enhance_contrast,
    image_to_bytes,
)
from tools import list_input_files
from upload_tuning import UploadAutoTuner, build_upload_tuner


_PAGE_NAME_PATTERN = re.compile(r"^page_(\d+)\.([A-Za-z0-9]+)$")
_ALLOWED_FP_MODES = {"all", "only_fp", "exclude_fp"}
_ALLOWED_UPLOAD_SOURCES = {"pdf", "images", "auto"}


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
) -> bool:
    timeout_seconds = max(1.0, float(config.upload_timeout_seconds or 120.0))
    max_attempts = max(1, int(config.upload_retry_attempts or 1))
    initial_delay = max(0.0, float(config.upload_retry_initial_delay_seconds or 0.0))
    max_delay = max(initial_delay, float(config.upload_retry_max_delay_seconds or initial_delay))
    blob = bucket.blob(blob_path)
    for attempt in range(1, max_attempts + 1):
        try:
            blob.upload_from_string(
                image_bytes,
                content_type=mime_type,
                timeout=timeout_seconds,
            )
            return True
        except Exception as exc:
            if attempt >= max_attempts:
                tqdm.write(
                    f"Upload failed after {max_attempts} attempts: {blob_path} ({exc})"
                )
                return False
            delay = min(max_delay, initial_delay * (2 ** (attempt - 1)))
            if delay > 0:
                time.sleep(delay)
    return False


def _is_fp_pdf_path(path: Path, root: Path, fp_suffix: str) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    parent_parts = rel.parts[:-1]
    return any(part.endswith(fp_suffix) for part in parent_parts) or path.stem.endswith(fp_suffix)


def _apply_fp_mode_filter(
    paths: list[Path],
    *,
    root: Path,
    fp_mode: str,
    fp_suffix: str,
) -> list[Path]:
    mode = fp_mode.lower()
    if mode not in _ALLOWED_FP_MODES:
        raise ValueError(
            f"Unsupported fp_mode: {fp_mode}. "
            f"Expected one of: {sorted(_ALLOWED_FP_MODES)}"
        )

    if mode == "only_fp":
        return [p for p in paths if _is_fp_pdf_path(p, root, fp_suffix)]
    if mode == "exclude_fp":
        return [p for p in paths if not _is_fp_pdf_path(p, root, fp_suffix)]
    return paths


def _ensure_unique_pdf_names(paths: list[Path]) -> None:
    by_name: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in paths:
        existing = by_name.get(path.name)
        if existing is None:
            by_name[path.name] = path
            continue
        duplicates.setdefault(path.name, [existing]).append(path)

    if not duplicates:
        return

    examples = []
    for name, duplicate_paths in sorted(duplicates.items()):
        shown = ", ".join(str(p) for p in duplicate_paths[:3])
        suffix = "..." if len(duplicate_paths) > 3 else ""
        examples.append(f"{name}: {shown}{suffix}")
    raise ValueError(
        "Duplicate PDF file names detected. "
        "Batch upload/submit maps pages by PDF file name, so names must be unique. "
        f"Conflicts: {'; '.join(examples)}"
    )


def _list_target_pdfs(target_folder: str | None) -> list[Path]:
    if not target_folder:
        raise KeyError("target_folder is missing from config")
    folder = Path(target_folder).expanduser()
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"target_folder not found or not a directory: {folder}")
    recursive = bool(config.recursive)
    fp_mode = str(config.fp_mode or "all")
    fp_suffix = str(config.fp_suffix or "_fp")

    candidates = folder.rglob("*") if recursive else folder.glob("*")
    pdfs = sorted(
        p for p in candidates
        if p.is_file() and p.suffix.lower() == ".pdf"
    )
    pdfs = _apply_fp_mode_filter(
        pdfs,
        root=folder,
        fp_mode=fp_mode,
        fp_suffix=fp_suffix,
    )
    _ensure_unique_pdf_names(pdfs)
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {folder} "
            f"(recursive={recursive}, fp_mode={fp_mode})"
        )
    return pdfs


def _resolve_image_upload_root(image_folder: str | Path | None = None) -> Path:
    configured_folder = str(config.upload_images_folder or "").strip()
    folder_value: str | Path | None = image_folder
    if folder_value is None:
        folder_value = configured_folder or config.target_folder

    if not folder_value:
        raise KeyError(
            "No image upload folder configured. "
            "Set config.upload_images_folder or config.target_folder."
        )

    folder = Path(folder_value).expanduser()
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(
            f"image upload folder not found or not a directory: {folder}"
        )
    return folder


def _resolve_image_upload_recursive(recursive: bool | None = None) -> bool:
    if recursive is not None:
        return bool(recursive)
    return bool(config.upload_images_recursive)


def _resolve_image_upload_glob() -> str:
    pattern = str(config.upload_images_glob or "").strip()
    if not pattern:
        return "*.png"
    return pattern


def _should_skip_local_image(path: Path) -> bool:
    # macOS AppleDouble sidecar files (e.g. ._foo.png) are not real images.
    return path.name.startswith("._")


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
            "Set the target GCS bucket before uploading pages/images."
        )
    service_account_path = _resolve_service_account_path(
        config.service_account_file
    )
    client = storage.Client.from_service_account_json(
        str(service_account_path)
    )
    return client.bucket(bucket_name)


def _make_upload_tuner() -> UploadAutoTuner | None:
    if not bool(config.upload_auto_tune):
        return None
    return build_upload_tuner(
        profile=str(config.upload_profile or "normal"),
        initial_workers=max(1, int(config.upload_workers or 1)),
        initial_batch_limit=max(1, int(config.batch_upload_limit or 1)),
        max_workers_override=max(0, int(config.upload_max_workers or 0)),
    )


def _effective_workers(tuner: UploadAutoTuner | None) -> int:
    if tuner is None:
        return max(1, int(config.upload_workers or 1))
    return max(1, int(tuner.current_workers))


def _effective_batch_limit(tuner: UploadAutoTuner | None) -> int:
    if tuner is None:
        return max(1, int(config.batch_upload_limit or 1))
    return max(1, int(tuner.current_batch_limit))


def _upload_single_pdf(
    pdf_path: Path,
    bucket: storage.Bucket,
    render_dpi: int,
    batch_size: int,
    existing_page_numbers: set[int] | None = None,
    tuner: UploadAutoTuner | None = None,
) -> None:
    uploaded_pages = existing_page_numbers or set()
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
            start = 0
            batch_index = 0
            dynamic_batch_size = max(1, int(batch_size or 1))
            while start < total_pages:
                batch_index += 1
                dynamic_batch_size = _effective_batch_limit(tuner)
                end = min(start + dynamic_batch_size, total_pages)
                progress.set_postfix_str(
                    f"batch {batch_index} w={_effective_workers(tuner)}"
                )
                batch_started_at = time.perf_counter()
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

                start = end
                if not pending_uploads:
                    continue

                upload_workers = _effective_workers(tuner)
                failed_uploads = 0
                if upload_workers == 1 or len(pending_uploads) == 1:
                    for blob_path, image_bytes, mime_type, page_number in pending_uploads:
                        ok = _upload_blob_bytes(
                            bucket=bucket,
                            blob_path=blob_path,
                            image_bytes=image_bytes,
                            mime_type=mime_type,
                        )
                        if ok:
                            uploaded_pages.add(page_number)
                        else:
                            failed_uploads += 1
                        progress.update(1)
                else:
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
                            ): (page_number, blob_path)
                            for blob_path, image_bytes, mime_type, page_number in pending_uploads
                        }
                        for future in as_completed(futures):
                            ok = future.result()
                            page_number, _ = futures[future]
                            if ok:
                                uploaded_pages.add(page_number)
                            else:
                                failed_uploads += 1
                            progress.update(1)

                if tuner is not None:
                    elapsed = time.perf_counter() - batch_started_at
                    tuner.record_batch(
                        items=len(pending_uploads),
                        seconds=elapsed,
                        had_errors=failed_uploads > 0,
                    )
                    dynamic_batch_size = _effective_batch_limit(tuner)
    finally:
        doc.close()


def _upload_pdf_paths(
    pdf_paths: list[Path],
    bucket: storage.Bucket,
) -> list[str]:
    render_dpi = max(1, int(config.pdf_render_dpi or 300))
    batch_size = max(1, int(config.batch_upload_limit or 20))
    tuner = _make_upload_tuner()

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
            tuner=tuner,
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


def upload_missing_images(
    image_paths: list[str | Path] | None = None,
    bucket: storage.Bucket | None = None,
    image_folder: str | Path | None = None,
    recursive: bool | None = None,
) -> list[str]:
    if image_paths is None:
        root = _resolve_image_upload_root(image_folder)
        selection_cfg = {
            "target_folder": str(root),
            "input_glob": _resolve_image_upload_glob(),
            "recursive": _resolve_image_upload_recursive(recursive),
            "fp_mode": config.fp_mode,
            "fp_suffix": config.fp_suffix,
        }
        candidate_paths = [Path(path) for path in list_input_files(selection_cfg)]
    else:
        candidate_paths = [Path(path).expanduser() for path in image_paths]
        try:
            root = _resolve_image_upload_root(image_folder)
        except (FileNotFoundError, KeyError):
            root = Path.cwd()

    active_bucket = bucket or _build_bucket()
    pages_prefix = _normalize_prefix(config.gcs_pages_prefix or "")

    existing = {
        blob.name
        for blob in active_bucket.list_blobs(prefix=pages_prefix or None)
        if not blob.name.endswith("/")
    }

    tuner = _make_upload_tuner()
    fallback_batch_limit = max(1, int(config.batch_upload_limit or 1))
    uploaded: list[str] = []
    total_candidates = len(candidate_paths)
    cursor = 0
    batch_index = 0
    with tqdm(total=total_candidates, desc="Uploading images", unit="img") as progress:
        while cursor < total_candidates:
            batch_index += 1
            batch_limit = _effective_batch_limit(tuner)
            if batch_limit <= 0:
                batch_limit = fallback_batch_limit
            current_batch = candidate_paths[cursor: cursor + batch_limit]
            cursor += len(current_batch)
            pending_uploads: list[tuple[str, bytes, str]] = []
            batch_started_at = time.perf_counter()

            for local_path in current_batch:
                if not local_path.exists() or not local_path.is_file():
                    progress.update(1)
                    continue
                if _should_skip_local_image(local_path):
                    progress.update(1)
                    continue
                try:
                    rel = local_path.relative_to(root)
                except ValueError:
                    rel = Path(local_path.name)
                blob_path = f"{pages_prefix}{rel.as_posix()}"
                if blob_path in existing:
                    progress.update(1)
                    continue
                try:
                    with Image.open(local_path) as img:
                        image_bytes, mime_type, _ = _apply_image_settings(img)
                except (UnidentifiedImageError, OSError) as exc:
                    tqdm.write(f"Skipping unreadable image: {local_path} ({exc})")
                    progress.update(1)
                    continue
                pending_uploads.append((blob_path, image_bytes, mime_type))

            workers = _effective_workers(tuner)
            progress.set_postfix_str(f"batch {batch_index} w={workers}")
            failed_uploads = 0
            if workers == 1 or len(pending_uploads) <= 1:
                for blob_path, image_bytes, mime_type in pending_uploads:
                    ok = _upload_blob_bytes(
                        bucket=active_bucket,
                        blob_path=blob_path,
                        image_bytes=image_bytes,
                        mime_type=mime_type,
                    )
                    if ok:
                        existing.add(blob_path)
                        uploaded.append(blob_path)
                    else:
                        failed_uploads += 1
                    progress.update(1)
            else:
                with ThreadPoolExecutor(
                    max_workers=min(workers, len(pending_uploads))
                ) as executor:
                    futures = {
                        executor.submit(
                            _upload_blob_bytes,
                            active_bucket,
                            blob_path,
                            image_bytes,
                            mime_type,
                        ): blob_path
                        for blob_path, image_bytes, mime_type in pending_uploads
                    }
                    for future in as_completed(futures):
                        ok = future.result()
                        blob_path = futures[future]
                        if ok:
                            existing.add(blob_path)
                            uploaded.append(blob_path)
                        else:
                            failed_uploads += 1
                        progress.update(1)

            if tuner is not None and pending_uploads:
                elapsed = time.perf_counter() - batch_started_at
                tuner.record_batch(
                    items=len(pending_uploads),
                    seconds=elapsed,
                    had_errors=failed_uploads > 0,
                )
    return uploaded


def upload_all_pdfs() -> None:
    pdf_paths = _list_target_pdfs(config.target_folder)
    bucket = _build_bucket()
    _upload_pdf_paths(pdf_paths, bucket)


def upload_pdf_pages() -> None:
    upload_all_pdfs()


def upload_all_images(
    image_folder: str | Path | None = None,
    recursive: bool | None = None,
) -> list[str]:
    bucket = _build_bucket()
    return upload_missing_images(
        bucket=bucket,
        image_folder=image_folder,
        recursive=recursive,
    )


def _resolve_upload_source() -> str:
    source = str(config.upload_source or "pdf").strip().lower()
    if source not in _ALLOWED_UPLOAD_SOURCES:
        raise ValueError(
            f"Unsupported upload_source: {config.upload_source}. "
            f"Expected one of: {sorted(_ALLOWED_UPLOAD_SOURCES)}"
        )
    return source


def upload_all_sources() -> None:
    source = _resolve_upload_source()
    if source == "pdf":
        upload_all_pdfs()
        return

    if source == "images":
        upload_all_images(
            image_folder=config.upload_images_folder or None,
            recursive=config.upload_images_recursive,
        )
        return

    try:
        upload_all_pdfs()
        return
    except (FileNotFoundError, KeyError):
        pass

    upload_all_images(
        image_folder=config.upload_images_folder or None,
        recursive=config.upload_images_recursive,
    )


if __name__ == "__main__":
    upload_all_sources()
