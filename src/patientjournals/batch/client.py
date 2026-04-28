from __future__ import annotations

import json
from pathlib import Path

from google import genai
from google.genai import types
from google.oauth2 import service_account

from patientjournals.config import config

_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def resolve_service_account_path(service_account_file: str) -> Path:
    candidate = Path(service_account_file).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            f"service_account_file not found: {candidate}"
        )
    return candidate


def infer_project_id_from_service_account(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    project_id = payload.get("project_id")
    if isinstance(project_id, str) and project_id.strip():
        return project_id.strip()
    return None


def get_batch_client(*, location: str | None = None) -> genai.Client:
    backend = (config.batch_backend or "").strip().lower()
    if backend != "vertex":
        api_key = (config.api_key or "").strip()
        if not api_key:
            raise ValueError(
                "batch_backend is not 'vertex' but config.api_key is empty."
            )
        return genai.Client(api_key=api_key)

    if not (config.service_account_file or "").strip():
        raise ValueError(
            "config.service_account_file is empty. "
            "Set it to your GCP service account JSON path for Vertex batch jobs."
        )

    service_account_path = resolve_service_account_path(
        config.service_account_file
    )
    project_id = config.gcp_project_id or infer_project_id_from_service_account(
        service_account_path
    )
    if not project_id:
        raise ValueError(
            "Unable to resolve GCP project id. Set config.gcp_project_id "
            "or ensure project_id exists in service_account_file."
        )

    vertex_location = (
        (location or "").strip()
        or (config.vertex_model_location or "").strip()
        or (config.gcp_location or "").strip()
    )
    if not vertex_location:
        raise ValueError(
            "Unable to resolve Vertex location. Set config.vertex_model_location "
            "or config.gcp_location."
        )

    credentials = service_account.Credentials.from_service_account_file(
        str(service_account_path),
        scopes=[_CLOUD_PLATFORM_SCOPE],
    )

    return genai.Client(
        vertexai=True,
        credentials=credentials,
        project=project_id,
        location=vertex_location,
        http_options=types.HttpOptions(api_version="v1"),
    )
