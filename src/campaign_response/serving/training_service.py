"""Training runs started through the API: one at a time, and the served champion refreshed right after each."""

import logging
import threading

from campaign_response.pipeline import train_pipeline
from campaign_response.pipeline.train_pipeline import TrainingContext, TrainingOutcome, TrainingRequest
from campaign_response.serving.model_cache import ChampionCache

logger = logging.getLogger(__name__)


class TrainingInProgressError(RuntimeError):
    """Another training run is still going; the API answers 409."""


class TrainingService:
    """Runs the training pipeline, refusing a second run while one is in progress.

    Two runs at once would race on the aliases: each gate compares against the champion it read at its start, and
    the later promotion would silently replace the earlier one. The lock is per process, so the API runs with a
    single worker; Airflow trains through the API, so its runs queue behind the same lock.
    """

    def __init__(self, context: TrainingContext, cache: ChampionCache) -> None:
        self._context = context
        self._cache = cache
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._lock.locked()

    def train(self, request: TrainingRequest) -> TrainingOutcome:
        # Non-blocking: a caller waiting minutes behind another run would time out anyway; 409 tells it at once.
        if not self._lock.acquire(blocking=False):
            raise TrainingInProgressError("a training run is already in progress; try again when it has finished")
        try:
            outcome = train_pipeline.run_training(request, self._context)
        finally:
            # Also after a failure: a run can move the champion alias before something later in it fails.
            self._cache.invalidate()
            self._lock.release()
        logger.info("training finished", extra={
            "model_version": outcome.version.version, "promoted": outcome.promoted, "trained_by": request.trained_by,
        })
        return outcome
