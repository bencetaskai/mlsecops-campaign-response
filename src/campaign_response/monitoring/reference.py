"""The reference side of the drift comparison: the training data of the model that is serving now."""

import io
from dataclasses import dataclass
from typing import Final

import pandas as pd

from campaign_response.data.io import load_processed
from campaign_response.pipeline.train_pipeline import REFERENCE_ARTIFACT_DIR, REFERENCE_FILE
from campaign_response.registry.ports import CHAMPION, ModelRegistry, ModelVersion

REFERENCE_ARTIFACT: Final = f"{REFERENCE_ARTIFACT_DIR}/{REFERENCE_FILE}"


class ReferenceUnavailableError(RuntimeError):
    """No champion, or a champion whose run logged no reference sample (e.g. registered outside the pipeline)."""


@dataclass(frozen=True)
class Reference:
    version: ModelVersion
    data: pd.DataFrame


def load_reference(registry: ModelRegistry) -> Reference:
    """The champion's reference sample: its cleaned training split, as logged by its training run.

    Comparing against the champion's own training data, not a fixed file, keeps the report meaningful after each
    promotion: drift means "different from what the serving model learned from".
    """
    champion = registry.get_by_alias(CHAMPION)
    if champion is None:
        raise ReferenceUnavailableError("no champion model yet; call POST /train")
    try:
        content = registry.version_artifact(champion.version, REFERENCE_ARTIFACT)
    except LookupError as error:
        raise ReferenceUnavailableError(
            f"champion version {champion.version} has no reference sample ({REFERENCE_ARTIFACT})"
        ) from error
    return Reference(champion, load_processed(io.BytesIO(content)))
