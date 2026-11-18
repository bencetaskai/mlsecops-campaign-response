"""Pipeline parameters from params.yaml, the file DVC tracks: changing a value re-runs exactly the stages using it."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from campaign_response.data.cleaning import CleaningParams
from campaign_response.data.splitting import SplitParams


class _CleaningSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_date: date
    min_age: int = Field(ge=0)
    max_age: int
    max_income: float = Field(gt=0)

    @model_validator(mode="after")
    def _ages_ordered(self) -> Self:
        if self.min_age >= self.max_age:
            raise ValueError(f"min_age ({self.min_age}) must be below max_age ({self.max_age})")
        return self


class _SplitSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    holdout_size: float = Field(gt=0, lt=1)
    seed: int


class _ParamsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cleaning: _CleaningSection
    split: _SplitSection


@dataclass(frozen=True)
class PipelineParams:
    cleaning: CleaningParams
    split: SplitParams


def load_params(path: Path) -> PipelineParams:
    """Read and validate params.yaml; unknown keys, missing keys and invalid values raise ValidationError."""
    # safe_load builds only plain data; yaml.load could construct arbitrary Python objects from the file.
    content = _ParamsFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    return PipelineParams(
        cleaning=CleaningParams(**content.cleaning.model_dump()),
        split=SplitParams(**content.split.model_dump()),
    )
