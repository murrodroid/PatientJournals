from __future__ import annotations

import mimetypes
import random
import secrets
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from patientjournals.app.models import AppSettings
from patientjournals.data.bucket import (
    build_storage_bucket,
    list_bucket_blobs,
    normalize_prefix,
    select_bucket_image_blobs,
)
from patientjournals.data.inspection import configured_image_extensions
from patientjournals.shared.identity import image_name_from_reference


class ImageAccessService:
    """Short-lived image links for dataset inspection and submission previews."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._cloud_objects: dict[str, str] = {}
        self._local_images: dict[str, Path] = {}

    def update_settings(self, settings: AppSettings) -> None:
        self.settings = settings
        self._cloud_objects.clear()
        self._local_images.clear()

    def _register_local(self, path: Path) -> str:
        token = secrets.token_urlsafe(18)
        self._local_images[token] = path
        return f"/api/images/local?token={quote(token)}"

    def local_image_bytes(self, token: str) -> tuple[bytes, str]:
        path = self._local_images.get(str(token or ""))
        if path is None or not path.is_file():
            raise FileNotFoundError("The local image preview has expired.")
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return path.read_bytes(), content_type

    @staticmethod
    def _signed_url(blob: Any, *, minutes: int = 20) -> str:
        return str(
            blob.generate_signed_url(
                version="v4",
                method="GET",
                expiration=timedelta(minutes=max(1, minutes)),
            )
        )

    def _cloud_object_for_image(self, image_name: str, object_hint: str = "") -> str:
        clean_name = image_name_from_reference(image_name) or Path(image_name).name
        hint = str(object_hint or "").strip()
        is_gcs_uri = hint.startswith("gs://")
        if is_gcs_uri:
            _bucket, _separator, hint = hint[5:].partition("/")
        pages_prefix = normalize_prefix(self.settings.gcs_pages_prefix)
        is_bucket_object = bool(pages_prefix and hint.startswith(pages_prefix))
        if (
            (is_gcs_uri or is_bucket_object)
            and image_name_from_reference(hint) == clean_name
        ):
            return hint.lstrip("/")
        cached = self._cloud_objects.get(clean_name)
        if cached:
            return cached
        bucket = build_storage_bucket(self.settings.gcs_bucket_name or None)
        prefix = normalize_prefix(self.settings.gcs_pages_prefix)
        for blob in bucket.list_blobs(prefix=prefix or None):
            object_name = str(getattr(blob, "name", "") or "")
            if image_name_from_reference(object_name) == clean_name:
                self._cloud_objects[clean_name] = object_name
                return object_name
        raise FileNotFoundError(f"Image not found in the configured bucket: {clean_name}")

    def dataset_image_link(
        self,
        *,
        image_name: str,
        object_hint: str = "",
    ) -> dict[str, str]:
        clean_name = image_name_from_reference(image_name) or Path(image_name).name
        cloud_error = ""
        if self.settings.gcs_bucket_name:
            try:
                bucket = build_storage_bucket(self.settings.gcs_bucket_name)
                object_name = self._cloud_object_for_image(clean_name, object_hint)
                return {
                    "image_name": clean_name,
                    "source": "cloud",
                    "uri": f"gs://{getattr(bucket, 'name', self.settings.gcs_bucket_name)}/{object_name}",
                    "url": self._signed_url(bucket.blob(object_name)),
                }
            except Exception as exc:  # noqa: BLE001
                cloud_error = str(exc)

        root_value = self.settings.validation_images_root
        root = Path(root_value).expanduser() if root_value else None
        if root is not None and root.is_dir():
            for path in root.rglob(clean_name):
                if path.is_file() and path.name == clean_name:
                    return {
                        "image_name": clean_name,
                        "source": "local",
                        "uri": str(path),
                        "url": self._register_local(path),
                    }
        detail = f" Cloud lookup: {cloud_error}" if cloud_error else ""
        raise FileNotFoundError(f"Image not found: {clean_name}.{detail}")

    @staticmethod
    def _local_candidates(root: Path) -> list[Path]:
        extensions = {f".{item}" for item in configured_image_extensions()}
        return [
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in extensions
            and not path.name.startswith("._")
        ]

    def submission_preview(
        self,
        *,
        source: str,
        local_path: str = "",
        cloud_prefixes: Iterable[str] = (),
        sample_size: int = 6,
    ) -> dict[str, Any]:
        count = min(12, max(1, int(sample_size)))
        rng = random.SystemRandom()
        if source == "local":
            root = Path(local_path).expanduser()
            if not root.is_dir():
                raise FileNotFoundError(f"Local input folder not found: {root}")
            candidates = self._local_candidates(root)
            if not candidates:
                raise FileNotFoundError(f"No images found in {root}")
            chosen = rng.sample(candidates, min(count, len(candidates)))
            return {
                "source": "local",
                "selection_count": len(candidates),
                "samples": [
                    {
                        "image_name": path.name,
                        "location": str(path),
                        "url": self._register_local(path),
                    }
                    for path in chosen
                ],
            }

        prefixes = tuple(dict.fromkeys(str(item).strip() for item in cloud_prefixes if str(item).strip()))
        if not prefixes:
            raise ValueError("Select one or more cloud folders before previewing.")
        bucket = build_storage_bucket(self.settings.gcs_bucket_name or None)
        candidates_by_name: dict[str, Any] = {}
        for prefix in prefixes:
            blobs = select_bucket_image_blobs(
                list_bucket_blobs(bucket, prefix=normalize_prefix(prefix))
            )
            for blob in blobs:
                object_name = str(getattr(blob, "name", "") or "")
                image_name = image_name_from_reference(object_name)
                if image_name:
                    candidates_by_name.setdefault(image_name, blob)
        candidates = list(candidates_by_name.items())
        if not candidates:
            raise FileNotFoundError("No images found in the selected cloud folders.")
        chosen = rng.sample(candidates, min(count, len(candidates)))
        return {
            "source": "cloud",
            "selection_count": len(candidates),
            "samples": [
                {
                    "image_name": image_name,
                    "location": f"gs://{getattr(bucket, 'name', self.settings.gcs_bucket_name)}/{getattr(blob, 'name', '')}",
                    "url": self._signed_url(blob),
                }
                for image_name, blob in chosen
            ],
        }
