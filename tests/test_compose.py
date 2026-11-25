"""Rules the compose file must keep, checked without Docker: hardening of our own services and local-only ports."""

from pathlib import Path
from typing import Any

import pytest
import yaml

COMPOSE = Path(__file__).resolve().parents[1] / "docker-compose.yaml"
APP_IMAGE = "campaign-response-app:local"


@pytest.fixture(scope="module")
def services() -> dict[str, dict[str, Any]]:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]


def app_services(services: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {name: service for name, service in services.items() if service.get("image") == APP_IMAGE}


def test_compose_read_only_services_have_tmp(services: dict[str, dict[str, Any]]) -> None:
    """A read-only container without /tmp crashes as soon as matplotlib or tempfile needs a scratch folder."""
    missing = [name for name, service in services.items()
               if service.get("read_only") and "/tmp" not in service.get("tmpfs", [])]

    assert missing == []


def test_compose_app_services_hardened(services: dict[str, dict[str, Any]]) -> None:
    ours = app_services(services)

    assert set(ours) >= {"api", "model-bootstrap"}
    for name, service in ours.items():
        assert service.get("read_only") is True, name
        assert service.get("cap_drop") == ["ALL"], name
        assert "no-new-privileges:true" in service.get("security_opt", []), name


def test_compose_ports_bound_to_localhost(services: dict[str, dict[str, Any]]) -> None:
    ports = [port for service in services.values() for port in service.get("ports", [])]

    assert ports
    assert all(port.startswith("127.0.0.1:") for port in ports), ports
