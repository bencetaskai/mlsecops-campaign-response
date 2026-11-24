"""Request and response models of the REST API: the trust boundary, where untrusted JSON becomes typed data.

Field names follow the raw file, so a record maps to a raw row without renaming and the prediction log has the same
columns as the training reference. Limits reject broken input only: values that are unusual but possible pass and
are masked by the same cleaning rules as in training, or show up as drift.
"""

import re
from datetime import date
from typing import Annotated, Final, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from campaign_response.data.cleaning import CleaningParams
from campaign_response.data.schema import VALUE_RANGES

MAX_RECORDS: Final = 1000
MAX_CHILDREN: Final = 10
MAX_RECENCY_DAYS: Final = 3650
MAX_AMOUNT: Final = 100_000
MAX_COUNT: Final = 1000
# Cleaning masks incomes above 200,000 anyway; this bound only keeps absurd numbers away from the pipeline.
MAX_INCOME: Final = 10_000_000
MAX_CUSTOMER_ID: Final = 2**31 - 1
EARLIEST_ENROLLMENT: Final = date(2000, 1, 1)
# A later date gives a negative tenure, which the model never saw; a test keeps this equal to params.yaml.
LATEST_ENROLLMENT: Final = CleaningParams().reference_date
_BIRTH_YEAR_LOW, _BIRTH_YEAR_HIGH = (int(bound) for bound in VALUE_RANGES["Year_Birth"])
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

EducationLevel = Literal["Basic", "Graduation", "2n Cycle", "Master", "PhD"]
Children = Annotated[int, Field(ge=0, le=MAX_CHILDREN)]
Amount = Annotated[int, Field(ge=0, le=MAX_AMOUNT)]
Count = Annotated[int, Field(ge=0, le=MAX_COUNT)]
# A strict int, not Literal[0, 1]: the literal would accept JSON true, because True == 1 in Python.
Flag = Annotated[int, Field(ge=0, le=1)]


def _iso_date_only(value: object) -> object:
    """Accept only YYYY-MM-DD text (or a date from Python callers), never a timestamp or another format."""
    if isinstance(value, date):
        return value
    if isinstance(value, str) and _ISO_DATE.fullmatch(value):
        return value
    raise ValueError("expected an ISO date YYYY-MM-DD")


# FastAPI validates the decoded body in Python mode, where a strict date rejects every string; so this one field is
# lax, and the validator above does the strict part.
EnrollmentDate = Annotated[
    date, BeforeValidator(_iso_date_only), Field(strict=False, ge=EARLIEST_ENROLLMENT, le=LATEST_ENROLLMENT),
]


class CustomerRecord(BaseModel):
    """One customer as sent to /predict: the 25 raw feature columns and an optional customer ID."""

    # strict: "5" is not 5 and true is not 1; at a trust boundary a wrong type is an error, not something to guess.
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    ID: int | None = Field(default=None, ge=1, le=MAX_CUSTOMER_ID)
    Year_Birth: int = Field(ge=_BIRTH_YEAR_LOW, le=_BIRTH_YEAR_HIGH)
    Education: EducationLevel
    # Free text on purpose: a new status is category drift for Evidently to report, not a client error.
    Marital_Status: str = Field(min_length=1, max_length=20, pattern=r"^[A-Za-z]+$")
    Income: float | None = Field(ge=0, le=MAX_INCOME, allow_inf_nan=False)
    Kidhome: Children
    Teenhome: Children
    Dt_Customer: EnrollmentDate
    Recency: int = Field(ge=0, le=MAX_RECENCY_DAYS)
    MntWines: Amount
    MntFruits: Amount
    MntMeatProducts: Amount
    MntFishProducts: Amount
    MntSweetProducts: Amount
    MntGoldProds: Amount
    NumDealsPurchases: Count
    NumWebPurchases: Count
    NumCatalogPurchases: Count
    NumStorePurchases: Count
    NumWebVisitsMonth: Count
    AcceptedCmp3: Flag
    AcceptedCmp4: Flag
    AcceptedCmp5: Flag
    AcceptedCmp1: Flag
    AcceptedCmp2: Flag
    Complain: Flag


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    records: list[CustomerRecord] = Field(min_length=1, max_length=MAX_RECORDS)


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ID: int | None
    probability: float = Field(ge=0, le=1)
    label: Literal[0, 1]


class HealthStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ok", "unavailable"]
    detail: str | None = None


class VersionInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    app_version: str
    git_sha: str
    model_name: str
    champion_version: int | None


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    model_name: str
    model_version: int = Field(ge=1)
    model_alias: str
    threshold: float = Field(ge=0, le=1)
    predictions: list[Prediction]
