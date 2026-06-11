from types import SimpleNamespace

from patientjournals.batch import client as batch_client
from patientjournals.config import config


def test_get_batch_client_supports_adc(monkeypatch) -> None:
    created = {}

    monkeypatch.setattr(config, "batch_backend", "vertex")
    monkeypatch.setattr(config, "gcp_auth_mode", "adc")
    monkeypatch.setattr(config, "gcp_project_id", "")
    monkeypatch.setattr(config, "vertex_model_location", "global")
    monkeypatch.setattr(config, "gcp_location", "europe-north1")

    fake_credentials = object()
    monkeypatch.setattr(
        batch_client.google.auth,
        "default",
        lambda scopes: (fake_credentials, "adc-project"),
    )

    def fake_client(**kwargs):
        created.update(kwargs)
        return SimpleNamespace(vertexai=kwargs.get("vertexai"))

    monkeypatch.setattr(batch_client.genai, "Client", fake_client)

    result = batch_client.get_batch_client()

    assert result.vertexai is True
    assert created["credentials"] is fake_credentials
    assert created["project"] == "adc-project"
    assert created["location"] == "global"
