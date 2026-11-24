"""Typed, validated runtime configuration (twelve-factor: whatever differs between deployments).

Values are read, from highest to lowest precedence, from: constructor arguments, environment variables
with the ``CRP_`` prefix, the ``.env`` file, and finally the defaults below. Model hyperparameters do not
belong here; they live in ``params.yaml`` (tracked by DVC).
"""

from pathlib import Path
from typing import Annotated, Final, Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
LogFormat = Literal["text", "json"]

MIN_API_KEY_LENGTH: Final = 32
# Docker mounts secrets here. pydantic-settings warns about a missing folder (an error in our tests), so the folder
# is used only where it exists, i.e. in the containers.
DOCKER_SECRETS_DIR: Final = Path("/run/secrets")


class Settings(BaseSettings):
    """Application settings, validated when the object is created."""

    model_config = SettingsConfigDict(
        env_prefix="CRP_",
        env_file=".env",
        env_file_encoding="utf-8",
        # Rejects unknown keys in .env and kwargs, but NOT misspelled environment variables (pydantic-settings).
        extra="forbid",
        # A file named like the variable, e.g. /run/secrets/crp_api_key, sets that field.
        secrets_dir=DOCKER_SECRETS_DIR if DOCKER_SECRETS_DIR.is_dir() else None,
        # Validation errors would otherwise quote the input, e.g. a slightly too short API key, into the log.
        hide_input_in_errors=True,
    )

    log_level: LogLevel = "INFO"
    log_format: LogFormat = "text"
    data_dir: Path = Path("data")
    mlflow_tracking_uri: str = Field(default="http://localhost:5500", pattern=r"^https?://\S+$")
    model_name: str = Field(default="campaign-response-classifier", pattern=r"^[A-Za-z0-9_.-]+$")
    # Optional here, as the CLI needs no key; the API refuses to start without one.
    api_key: Annotated[SecretStr, Field(min_length=MIN_API_KEY_LENGTH)] | None = None
