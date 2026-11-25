"""Tests for the typed application settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from campaign_response.config import MIN_API_KEY_LENGTH, Settings

KEY = "k" * MIN_API_KEY_LENGTH


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.log_level == "INFO"
    assert settings.log_format == "text"
    assert settings.data_dir == Path("data")
    assert settings.mlflow_tracking_uri == "http://localhost:5500"
    assert settings.model_name == "campaign-response-classifier"
    assert settings.api_key is None


def test_settings_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRP_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("CRP_DATA_DIR", "other/data")

    settings = Settings(_env_file=None)

    assert settings.log_level == "DEBUG"
    assert settings.data_dir == Path("other/data")


def test_settings_reads_dotenv(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("CRP_LOG_FORMAT=json\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)

    assert settings.log_format == "json"


def test_settings_env_var_beats_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("CRP_LOG_FORMAT=json\n", encoding="utf-8")
    monkeypatch.setenv("CRP_LOG_FORMAT", "text")

    settings = Settings(_env_file=env_file)

    assert settings.log_format == "text"


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("CRP_LOG_LEVEL", "VERBOSE"),
        ("CRP_LOG_FORMAT", "xml"),
        ("CRP_MLFLOW_TRACKING_URI", "localhost:5500"),
        ("CRP_MLFLOW_TRACKING_URI", "ftp://mlflow:5000"),
        ("CRP_MODEL_NAME", "model name with spaces"),
        ("CRP_MODEL_NAME", "../other"),
    ],
)
def test_settings_invalid_value_ValidationError(variable: str, value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_dotenv_typo_ValidationError(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("CRP_LOG_LEVLE=DEBUG\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs are not permitted"):
        Settings(_env_file=env_file)


def test_settings_unknown_argument_ValidationError() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Settings(_env_file=None, log_levle="DEBUG")


def test_settings_api_key_from_secret_file(tmp_path: Path) -> None:
    (tmp_path / "crp_api_key").write_text(KEY + "\n", encoding="utf-8")

    settings = Settings(_env_file=None, _secrets_dir=tmp_path)

    assert settings.api_key is not None and settings.api_key.get_secret_value() == KEY
    assert KEY not in repr(settings)


def test_settings_api_key_env_beats_secret_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "crp_api_key").write_text(KEY, encoding="utf-8")
    monkeypatch.setenv("CRP_API_KEY", KEY.upper())

    settings = Settings(_env_file=None, _secrets_dir=tmp_path)

    assert settings.api_key is not None and settings.api_key.get_secret_value() == KEY.upper()


def test_settings_short_api_key_hidden_in_error(monkeypatch: pytest.MonkeyPatch) -> None:
    short = "x" * (MIN_API_KEY_LENGTH - 1)
    monkeypatch.setenv("CRP_API_KEY", short)

    with pytest.raises(ValidationError, match="at least 32") as error:
        Settings(_env_file=None)

    assert short not in str(error.value)
