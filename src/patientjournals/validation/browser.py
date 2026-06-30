from __future__ import annotations

import csv
import mimetypes
import random
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from patientjournals.config import config
from patientjournals.data.bucket import (
    build_storage_bucket,
    list_bucket_blobs,
    normalize_prefix,
    select_bucket_image_blobs,
)
from patientjournals.shared.identity import image_name_from_reference
from patientjournals.validation.cli import (
    SamplingMode,
    _parse_corrected_value,
    _score_for_label,
    _stringify_value,
    build_image_index,
    build_validation_datapoints,
    choose_balanced_ucb_datapoint,
    choose_random_datapoint,
    display_image_name,
    load_dataset,
)
from patientjournals.validation.sync import upload_validation_run, write_validation_metadata


ImageSource = Literal["local", "cloud"]


@dataclass(frozen=True)
class ValidationImageRef:
    image_name: str
    source: ImageSource
    local_path: str = ""
    bucket_name: str = ""
    object_name: str = ""

    @property
    def uri(self) -> str:
        if self.source == "cloud":
            return f"gs://{self.bucket_name}/{self.object_name}"
        return self.local_path


def _safe_user(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return text.strip("._-") or "researcher"


def _create_validation_run_dir(username: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path.cwd() / "validations"
    stem = f"{_safe_user(username)}_{stamp}"
    candidate = base / stem
    index = 2
    while candidate.exists():
        candidate = base / f"{stem}_{index}"
        index += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def build_cloud_image_index(
    *,
    bucket: Any,
    bucket_name: str,
    prefixes: tuple[str, ...],
) -> dict[str, ValidationImageRef]:
    index: dict[str, ValidationImageRef] = {}
    for prefix_value in prefixes:
        prefix = normalize_prefix(prefix_value)
        blobs = select_bucket_image_blobs(list_bucket_blobs(bucket, prefix=prefix))
        for blob in blobs:
            object_name = str(getattr(blob, "name", "") or "")
            image_name = image_name_from_reference(object_name)
            if not image_name or image_name in index:
                continue
            index[image_name] = ValidationImageRef(
                image_name=image_name,
                source="cloud",
                bucket_name=bucket_name,
                object_name=object_name,
            )
    return index


def _local_image_index(root: str | Path) -> dict[str, ValidationImageRef]:
    paths = build_image_index(Path(root).expanduser())
    return {
        name: ValidationImageRef(
            image_name=name,
            source="local",
            local_path=str(path),
        )
        for name, path in paths.items()
    }


class BrowserValidationSession:
    """Server-side validation state for the browser validator."""

    def __init__(
        self,
        *,
        session_id: str,
        dataset_path: str | Path,
        dataset_label: str = "",
        username: str,
        allow_corrections: bool,
        sampling_mode: SamplingMode,
        image_source: ImageSource,
        image_index: dict[str, ValidationImageRef],
        bucket: Any | None = None,
        signed_url_ttl_minutes: int = 20,
    ) -> None:
        self.session_id = session_id
        self.dataset_path = Path(dataset_path).expanduser()
        self.dataset_label = dataset_label or self.dataset_path.name
        self.username = _safe_user(username)
        self.allow_corrections = allow_corrections
        self.sampling_mode = sampling_mode
        self.image_source = image_source
        self.image_index = image_index
        self.bucket = bucket
        self.signed_url_ttl_minutes = max(1, int(signed_url_ttl_minutes))
        self.seed = secrets.randbits(64)
        self.rng = random.Random(self.seed)
        self.rows = load_dataset(self.dataset_path)
        self.datapoints = build_validation_datapoints(self.rows, self.image_index)
        self.results: list[dict[str, Any]] = []
        self.validated_pairs: set[tuple[str, str]] = set()
        self.selection_counts: dict[str, int] = {}
        self.scored_counts: dict[str, int] = {}
        self.score_sums: dict[str, float] = {}
        self.total_pairs = len(self.datapoints)
        self.current_datapoint = None
        self.finished = False
        self.run_dir = _create_validation_run_dir(self.username)
        self.log_path = self.run_dir / "validation.log"
        self.csv_path = self.run_dir / f"{self.run_dir.name}_validations.csv"
        self.metadata_path: Path | None = None
        if not self.rows:
            raise ValueError("Dataset is empty.")
        if not self.datapoints:
            image_names = sorted(
                {
                    display_image_name(row)
                    for row in self.rows
                    if display_image_name(row)
                }
            )
            matched = len(set(image_names).intersection(self.image_index))
            raise ValueError(
                "No validation datapoints could be built. "
                f"Matched {matched} of {len(image_names)} dataset image(s)."
            )
        self.log(
            "Started browser validation. "
            f"Seed={self.seed} SamplingMode={self.sampling_mode} "
            f"ImageSource={self.image_source}"
        )

    def log(self, message: str) -> None:
        stamp = datetime.now().isoformat(timespec="seconds")
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(f"[{stamp}] {message}\n")

    def current_payload(self) -> dict[str, Any]:
        if self.finished:
            return self._complete_payload()
        if self.current_datapoint is None:
            self.current_datapoint = self._choose_next_datapoint()
            if self.current_datapoint is not None:
                field_name = self.current_datapoint.field_name
                self.selection_counts[field_name] = (
                    self.selection_counts.get(field_name, 0) + 1
                )
        if self.current_datapoint is None:
            self.finished = True
            self.save_results()
            return self._complete_payload()

        datapoint = self.current_datapoint
        image_ref = datapoint.image_path
        if not isinstance(image_ref, ValidationImageRef):
            raise ValueError("Invalid validation image reference.")
        field_value = _stringify_value(datapoint.field_value)
        return {
            "session_id": self.session_id,
            "status": "active",
            "run_id": self.run_dir.name,
            "dataset_file": self.dataset_label,
            "image_name": datapoint.image_name,
            "image_source": image_ref.source,
            "image_uri": image_ref.uri,
            "image_url": self._image_url(image_ref),
            "field_name": datapoint.field_name,
            "field_value": field_value,
            "correction_value": field_value,
            "allow_corrections": self.allow_corrections,
            "sampling_mode": self.sampling_mode,
            "decisions": len(self.results),
            "total_pairs": self.total_pairs,
            "remaining_pairs": max(0, self.total_pairs - len(self.validated_pairs)),
        }

    def mark(self, *, label: str, corrected_text: str = "") -> dict[str, Any]:
        if self.finished:
            return self._complete_payload()
        if self.current_datapoint is None:
            self.current_payload()
        if self.current_datapoint is None:
            self.finished = True
            return self._complete_payload()

        normalized_label = str(label or "").strip().lower()
        if normalized_label not in {
            "accept",
            "somewhat_accept",
            "reject",
            "unsure",
            "corrected",
        }:
            raise ValueError(f"Unknown validation label: {label}")
        if normalized_label == "corrected" and not self.allow_corrections:
            raise ValueError("Corrections are disabled for this validation session.")

        datapoint = self.current_datapoint
        image_ref = datapoint.image_path
        if not isinstance(image_ref, ValidationImageRef):
            raise ValueError("Invalid validation image reference.")
        corrected_field = None
        if normalized_label == "corrected":
            corrected_field = _parse_corrected_value(
                datapoint.field_name,
                corrected_text,
            )
        decided_at = datetime.now().isoformat(timespec="seconds")
        self.results.append(
            {
                "label": normalized_label,
                "column_name": datapoint.field_name,
                "image_name": datapoint.image_name,
                "file_name": datapoint.image_name,
                "dataset_file": self.dataset_label,
                "validator_id": self.username,
                "decided_at": decided_at,
                "corrected_field": corrected_field,
                "sampling_mode": self.sampling_mode,
                "image_source": image_ref.source,
                "image_uri": image_ref.uri,
                "session_id": self.session_id,
            }
        )
        self.validated_pairs.add((datapoint.image_name, datapoint.field_name))
        score = _score_for_label(normalized_label)
        if score is not None:
            field_name = datapoint.field_name
            self.scored_counts[field_name] = self.scored_counts.get(field_name, 0) + 1
            self.score_sums[field_name] = self.score_sums.get(field_name, 0.0) + score
        self.log(
            f"Marked {datapoint.image_name} {datapoint.field_name} "
            f"label={normalized_label}"
        )
        self.current_datapoint = None
        self.save_results()
        return self.current_payload()

    def finish(self) -> dict[str, Any]:
        self.finished = True
        self.save_results()
        uploaded: dict[str, str] = {}
        if self.csv_path.exists() and self.metadata_path and self.metadata_path.exists():
            uploaded = upload_validation_run(
                run_dir=self.run_dir,
                csv_path=self.csv_path,
                metadata_path=self.metadata_path,
            )
            if uploaded:
                self.log(
                    "Uploaded browser validation results: "
                    f"{uploaded.get('validation_csv_uri', '')}"
                )
        return {**self._complete_payload(), "uploaded": uploaded}

    def local_image_bytes(self) -> tuple[bytes, str]:
        if self.current_datapoint is None:
            self.current_payload()
        if self.current_datapoint is None:
            raise ValueError("No active validation image.")
        image_ref = self.current_datapoint.image_path
        if not isinstance(image_ref, ValidationImageRef) or image_ref.source != "local":
            raise ValueError("Current validation image is not local.")
        path = Path(image_ref.local_path).expanduser()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return path.read_bytes(), content_type

    def _choose_next_datapoint(self):
        if self.sampling_mode == "balanced_ucb":
            return choose_balanced_ucb_datapoint(
                self.datapoints,
                self.validated_pairs,
                self.selection_counts,
                self.scored_counts,
                self.score_sums,
                self.rng,
            )
        return choose_random_datapoint(self.datapoints, self.validated_pairs, self.rng)

    def _image_url(self, image_ref: ValidationImageRef) -> str:
        if image_ref.source == "local":
            return f"/api/validation/session/image?session_id={self.session_id}"
        if self.bucket is None:
            raise ValueError("Cloud image bucket is not available for this session.")
        try:
            return self.bucket.blob(image_ref.object_name).generate_signed_url(
                version="v4",
                method="GET",
                expiration=timedelta(minutes=self.signed_url_ttl_minutes),
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Could not create a signed GCS image URL. Signed URLs require "
                "credentials that can sign URLs, bucket object read access, and "
                "KMS decrypt access when the bucket is CMEK-encrypted. Use a "
                "service-account JSON key or grant the active ADC identity "
                "iam.serviceAccounts.signBlob on the signing service account."
            ) from exc

    def _complete_payload(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": "complete",
            "run_id": self.run_dir.name,
            "dataset_file": self.dataset_label,
            "decisions": len(self.results),
            "total_pairs": self.total_pairs,
            "remaining_pairs": max(0, self.total_pairs - len(self.validated_pairs)),
            "csv_path": str(self.csv_path) if self.csv_path.exists() else "",
            "metadata_path": str(self.metadata_path or ""),
        }

    def save_results(self) -> None:
        if not self.results:
            return
        fieldnames = [
            "label",
            "column_name",
            "image_name",
            "file_name",
            "dataset_file",
            "validator_id",
            "decided_at",
            "corrected_field",
            "sampling_mode",
            "image_source",
            "image_uri",
            "session_id",
        ]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        self.metadata_path = write_validation_metadata(
            run_dir=self.run_dir,
            csv_path=self.csv_path,
            dataset_path=self.dataset_path,
            validator_id=self.username,
            decision_count=len(self.results),
            sampling_mode=self.sampling_mode,
        )


class BrowserValidationManager:
    def __init__(self) -> None:
        self.sessions: dict[str, BrowserValidationSession] = {}

    def start_session(
        self,
        *,
        dataset_path: str | Path,
        dataset_label: str = "",
        username: str,
        allow_corrections: bool = True,
        sampling_mode: str = "balanced_ucb",
        image_source: str = "cloud",
        image_root: str = "",
        cloud_prefixes: tuple[str, ...] = (),
        bucket_name: str = "",
    ) -> dict[str, Any]:
        source = "cloud" if image_source == "cloud" else "local"
        resolved_sampling = sampling_mode
        if resolved_sampling not in {"random", "balanced_ucb"}:
            resolved_sampling = "balanced_ucb"
        bucket = None
        if source == "cloud":
            prefixes = tuple(prefix for prefix in cloud_prefixes if prefix) or (
                config.gcs_pages_prefix,
            )
            bucket = build_storage_bucket(bucket_name or None)
            resolved_bucket_name = str(
                getattr(bucket, "name", "") or bucket_name or config.gcs_bucket_name
            )
            image_index = build_cloud_image_index(
                bucket=bucket,
                bucket_name=resolved_bucket_name,
                prefixes=prefixes,
            )
        else:
            if not image_root:
                raise ValueError("Select a local image folder.")
            root = Path(image_root).expanduser()
            if not root.is_dir():
                raise FileNotFoundError(f"Image folder not found: {root}")
            image_index = _local_image_index(root)
        if not image_index:
            raise ValueError("No validation images found for the selected source.")
        session_id = f"val_{secrets.token_hex(8)}"
        session = BrowserValidationSession(
            session_id=session_id,
            dataset_path=dataset_path,
            dataset_label=dataset_label,
            username=username,
            allow_corrections=allow_corrections,
            sampling_mode=resolved_sampling,  # type: ignore[arg-type]
            image_source=source,
            image_index=image_index,
            bucket=bucket,
        )
        self.sessions[session_id] = session
        return session.current_payload()

    def get(self, session_id: str) -> BrowserValidationSession:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise ValueError(f"Validation session not found: {session_id}") from exc

    def current(self, session_id: str) -> dict[str, Any]:
        return self.get(session_id).current_payload()

    def mark(
        self,
        session_id: str,
        *,
        label: str,
        corrected_text: str = "",
    ) -> dict[str, Any]:
        return self.get(session_id).mark(label=label, corrected_text=corrected_text)

    def finish(self, session_id: str) -> dict[str, Any]:
        return self.get(session_id).finish()

    def local_image_bytes(self, session_id: str) -> tuple[bytes, str]:
        return self.get(session_id).local_image_bytes()
