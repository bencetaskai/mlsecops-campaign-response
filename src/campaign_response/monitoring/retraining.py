"""The retraining rule: a pure function from the drift counts to "retrain or not", with the reason.

The rule counts drifted columns instead of using a drifted share such as the lecture's 0.4: in the calibration
(02 §6) a realistic five-column shift reached a share of only 0.2, while a single drifted column must not trip the
alarm on its own. Small windows are never enough evidence, however many columns they flag.

The module does not import Evidently: params.py loads MonitoringParams in every pipeline stage, and Evidently's
import alone takes about 3 s.
"""

from dataclasses import dataclass

from campaign_response.data.schema import FEATURE_COLUMNS


@dataclass(frozen=True)
class MonitoringParams:
    """Calibrated on 1,440 reports: 0/60 false alarms at 200 and 400 rows, 60/60 five-column shifts detected."""

    drift_threshold: float = 0.2
    min_rows: int = 200
    min_drifted_columns: int = 3


@dataclass(frozen=True)
class RetrainingAdvice:
    dataset_drift: bool
    retrain_recommended: bool
    reason: str


def recommend_retraining(n_current: int, n_drifted: int, params: MonitoringParams) -> RetrainingAdvice:
    dataset_drift = n_drifted >= params.min_drifted_columns
    columns = f"{n_drifted} of {len(FEATURE_COLUMNS)} columns drifted (alarm at {params.min_drifted_columns})"
    if n_current < params.min_rows:
        reason = f"only {n_current} predictions in the window, at least {params.min_rows} needed; {columns}"
        return RetrainingAdvice(dataset_drift=dataset_drift, retrain_recommended=False, reason=reason)
    return RetrainingAdvice(dataset_drift=dataset_drift, retrain_recommended=dataset_drift, reason=columns)
