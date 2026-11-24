"""Typed, validated runtime configuration (twelve-factor: whatever differs between deployments).

Values are read, from highest to lowest precedence, from: constructor arguments, environment variables
with the ``CRP_`` prefix, the ``.env`` file, and finally the defaults below. Model hyperparameters do not
belong here; they live in ``params.yaml`` (tracked by DVC).
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
LogFormat = Literal["text", "json"]


class Settings(BaseSettings):
    """Application settings, validated when the object is created."""

    model_config = SettingsConfigDict(
        env_prefix="CRP_",
        env_file=".env",
        env_file_encoding="utf-8",
        # Rejects unknown keys in .env and kwargs, but NOT misspelled environment variables (pydantic-settings).
        extra="forbid",
    )

    log_level: LogLevel = "INFO"
    log_format: LogFormat = "text"
    data_dir: Path = Path("data")
    mlflow_tracking_uri: str = Field(default="http://localhost:5500", pattern=r"^https?://\S+$")
    model_name: str = Field(default="campaign-response-classifier", pattern=r"^[A-Za-z0-9_.-]+$")
