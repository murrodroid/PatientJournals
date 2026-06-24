from __future__ import annotations

import getpass
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from patientjournals.app.jobs import _apply_runtime_overrides, _restore_runtime_overrides
from patientjournals.app.models import AppSettings
from patientjournals.app.settings_store import command_override_payload
from patientjournals.data.bucket import build_storage_bucket, normalize_prefix


CheckStatus = str
CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class AccessCheckResult:
    name: str
    status: CheckStatus
    detail: str
    fix: str = ""


@dataclass(frozen=True)
class AccessCheckReport:
    results: tuple[AccessCheckResult, ...]

    @property
    def failed(self) -> int:
        return sum(1 for result in self.results if result.status == "fail")

    @property
    def warnings(self) -> int:
        return sum(1 for result in self.results if result.status == "warn")

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.status == "pass")

    @property
    def ready(self) -> bool:
        return self.failed == 0


def _default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _run_text(runner: CommandRunner, command: Sequence[str]) -> tuple[int, str, str]:
    try:
        completed = runner(command)
    except FileNotFoundError as exc:
        return 127, "", str(exc)
    except subprocess.TimeoutExpired as exc:
        return 124, exc.stdout or "", exc.stderr or "Command timed out."
    except Exception as exc:  # noqa: BLE001
        return 1, "", str(exc)
    return (
        int(getattr(completed, "returncode", 1)),
        str(getattr(completed, "stdout", "") or "").strip(),
        str(getattr(completed, "stderr", "") or "").strip(),
    )


def _result(name: str, ok: bool, detail: str, fix: str = "") -> AccessCheckResult:
    return AccessCheckResult(
        name=name,
        status="pass" if ok else "fail",
        detail=detail,
        fix="" if ok else fix,
    )


def _warn(name: str, detail: str, fix: str = "") -> AccessCheckResult:
    return AccessCheckResult(name=name, status="warn", detail=detail, fix=fix)


def _sanitize_probe_part(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
    return clean.strip("-") or "user"


def _configured_prefixes(settings: AppSettings) -> tuple[tuple[str, str], ...]:
    return (
        ("Pages prefix", settings.gcs_pages_prefix),
        ("Batch requests prefix", settings.batch_requests_gcs_prefix),
        ("Batch outputs prefix", settings.batch_outputs_gcs_prefix),
        ("Datasets prefix", settings.datasets_gcs_prefix),
        ("Validations prefix", settings.validations_gcs_prefix),
    )


def _project_fix(member_hint: str) -> str:
    return (
        "Ask an admin to grant Vertex access, for example:\n"
        "gcloud projects add-iam-policy-binding PROJECT_ID "
        f"--member=\"{member_hint}\" --role=\"roles/aiplatform.user\""
    )


def _bucket_fix(member_hint: str) -> str:
    return (
        "Ask an admin to grant bucket object access, for example:\n"
        "gcloud storage buckets add-iam-policy-binding gs://BUCKET_NAME "
        f"--member=\"{member_hint}\" --role=\"roles/storage.objectAdmin\""
    )


def run_access_checks(
    settings: AppSettings,
    *,
    runner: CommandRunner | None = None,
    bucket_factory: Callable[[str | None], object] | None = None,
) -> AccessCheckReport:
    command_runner = runner or _default_runner
    results: list[AccessCheckResult] = []

    code, stdout, stderr = _run_text(command_runner, ("gcloud", "--version"))
    results.append(
        _result(
            "gcloud installed",
            code == 0,
            stdout.splitlines()[0] if stdout else stderr or "gcloud not found",
            "Install the Google Cloud CLI, then run gcloud init.",
        )
    )

    code, account, stderr = _run_text(
        command_runner,
        ("gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"),
    )
    active_account = account.splitlines()[0].strip() if account else ""
    results.append(
        _result(
            "gcloud account",
            code == 0 and bool(active_account),
            active_account or stderr or "No active gcloud account.",
            "Run gcloud auth login.",
        )
    )

    code, project, stderr = _run_text(
        command_runner,
        ("gcloud", "config", "get-value", "project"),
    )
    gcloud_project = project.splitlines()[0].strip() if project else ""
    configured_project = (settings.gcp_project_id or "").strip()
    if code == 0 and (configured_project or gcloud_project):
        detail = configured_project or gcloud_project
        if configured_project and gcloud_project and configured_project != gcloud_project:
            results.append(
                _warn(
                    "GCP project",
                    f"App uses {configured_project}; gcloud active project is {gcloud_project}.",
                    "This is allowed, but set them equal if users are confused.",
                )
            )
        else:
            results.append(AccessCheckResult("GCP project", "pass", detail))
    else:
        results.append(
            AccessCheckResult(
                "GCP project",
                "fail",
                stderr or "No project configured.",
                "Run gcloud config set project PROJECT_ID or fill GCP project in Settings.",
            )
        )

    auth_mode = (settings.auth_mode or "").strip().lower()
    if auth_mode == "adc":
        code, _token, stderr = _run_text(
            command_runner,
            ("gcloud", "auth", "application-default", "print-access-token"),
        )
        results.append(
            _result(
                "Application Default Credentials",
                code == 0,
                "ADC token available." if code == 0 else stderr or "ADC unavailable.",
                "Run gcloud auth application-default login.",
            )
        )
    elif auth_mode == "service_account":
        service_file = Path(settings.service_account_file or "").expanduser()
        if settings.service_account_file and not service_file.is_absolute():
            service_file = Path.cwd() / service_file
        results.append(
            _result(
                "Service account file",
                bool(settings.service_account_file) and service_file.exists(),
                str(service_file) if settings.service_account_file else "No service account path set.",
                "Select a readable service account JSON file in Settings, or use auth mode adc.",
            )
        )
    else:
        results.append(
            _warn(
                "Cloud auth mode",
                f"Auth mode is {auth_mode or 'empty'}; Vertex batch jobs use service_account or adc.",
                "Use service_account for shared service-account JSON or adc for gcloud user auth.",
            )
        )

    member_hint = (
        f"user:{active_account}"
        if active_account
        else "serviceAccount:NAME@PROJECT_ID.iam.gserviceaccount.com"
    )
    if configured_project or gcloud_project:
        results.append(
            _warn(
                "Vertex role",
                "IAM role cannot always be verified without admin IAM read access.",
                _project_fix(member_hint),
            )
        )

    bucket_name = (settings.gcs_bucket_name or "").strip()
    if not bucket_name:
        results.append(
            AccessCheckResult(
                "GCS bucket",
                "fail",
                "No bucket configured.",
                "Fill GCS bucket in Settings.",
            )
        )
        return AccessCheckReport(tuple(results))

    previous = _apply_runtime_overrides(command_override_payload(settings))
    try:
        try:
            bucket = (bucket_factory or build_storage_bucket)(bucket_name)
            exists = getattr(bucket, "exists", None)
            if callable(exists) and not exists():
                results.append(
                    AccessCheckResult(
                        "GCS bucket",
                        "fail",
                        f"Bucket {bucket_name} was not found or is not visible.",
                        _bucket_fix(member_hint),
                    )
                )
                return AccessCheckReport(tuple(results))
            results.append(AccessCheckResult("GCS bucket", "pass", f"Connected to gs://{bucket_name}."))
        except Exception as exc:  # noqa: BLE001
            results.append(
                AccessCheckResult(
                    "GCS bucket",
                    "fail",
                    str(exc),
                    _bucket_fix(member_hint),
                )
            )
            return AccessCheckReport(tuple(results))

        for label, prefix in _configured_prefixes(settings):
            normalized = normalize_prefix(prefix)
            try:
                iterator = bucket.list_blobs(prefix=normalized or None, max_results=1)
                list(iterator)
                results.append(
                    AccessCheckResult(
                        label,
                        "pass",
                        f"List allowed for gs://{bucket_name}/{normalized}".rstrip("/"),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                results.append(
                    AccessCheckResult(
                        label,
                        "fail",
                        str(exc),
                        _bucket_fix(member_hint),
                    )
                )

        user_part = _sanitize_probe_part(active_account or getpass.getuser())
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        object_name = f"diagnostics/access-checks/{user_part}/{stamp}.txt"
        try:
            blob = bucket.blob(object_name)
            blob.upload_from_string(
                "patientjournals access check\n",
                content_type="text/plain",
            )
            try:
                blob.download_as_text()
            finally:
                blob.delete()
            results.append(
                AccessCheckResult(
                    "GCS write/read/delete",
                    "pass",
                    f"Probe succeeded at gs://{bucket_name}/{object_name}.",
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                AccessCheckResult(
                    "GCS write/read/delete",
                    "fail",
                    str(exc),
                    _bucket_fix(member_hint),
                )
            )
    finally:
        _restore_runtime_overrides(previous)

    return AccessCheckReport(tuple(results))
