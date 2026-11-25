"""One-shot job: make sure a champion exists, training the first one if the registry has none.

Run once by compose after the API is up (``python -m campaign_response.bootstrap``). Without it, /predict answers
503 on a fresh stack until someone calls POST /train. Exit code 0: a champion exists; 1: it could not get one.
"""

import logging
import sys
import time
from collections.abc import Callable
from http import HTTPStatus
from typing import Final

import httpx2

from campaign_response.api_client import ApiClient
from campaign_response.config import Settings
from campaign_response.logging_setup import configure_logging, new_correlation_id
from campaign_response.registry.ports import CHAMPION
from campaign_response.serving.schemas import ModelOverview, TrainRequest

logger = logging.getLogger(__name__)

WAIT_ATTEMPTS: Final = 30
WAIT_SECONDS: Final = 5.0


class ApiNotReadyError(RuntimeError):
    """The API or the registry behind it did not answer within the waiting time."""


def wait_for_registry(
    client: ApiClient, attempts: int = WAIT_ATTEMPTS, delay: float = WAIT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> ModelOverview:
    """GET /model until it answers; connection failures and 503s (MLflow still starting) are retried."""
    for attempt in range(1, attempts + 1):
        try:
            return client.model_overview()
        except httpx2.TransportError as error:
            reason = type(error).__name__
        except httpx2.HTTPStatusError as error:
            if error.response.status_code != HTTPStatus.SERVICE_UNAVAILABLE:
                raise
            reason = "registry unavailable"
        logger.info("waiting for the API", extra={"attempt": attempt, "attempts": attempts, "reason": reason})
        if attempt < attempts:
            sleep(delay)
    raise ApiNotReadyError(f"the API did not answer within {attempts} attempts")


def ensure_champion(client: ApiClient, wait: Callable[[ApiClient], ModelOverview] = wait_for_registry) -> bool:
    """True if a champion exists afterwards: already there, or trained and promoted now."""
    overview = wait(client)
    champion = next((entry for entry in overview.versions if entry.alias == CHAMPION), None)
    if champion is not None:
        logger.info("champion exists; nothing to do", extra={"model_version": champion.version})
        return True
    logger.info("no champion; training the first one")
    outcome = client.train(TrainRequest(trained_by="bootstrap"))
    logger.info("bootstrap training finished", extra={
        "model_version": outcome.model_version, "promoted": outcome.promoted, "gate_reason": outcome.gate_reason,
        "duration_s": outcome.duration_s,
    })
    if not outcome.promoted:
        logger.error("the first model did not pass the gate; /predict stays unavailable",
                     extra={"gate_reason": outcome.gate_reason})
    return outcome.promoted


def main() -> int:
    settings = Settings()
    configure_logging(settings.log_level, settings.log_format)
    new_correlation_id()
    client = ApiClient(settings.api_url, settings.api_key)
    try:
        return 0 if ensure_champion(client) else 1
    # A failed job must say why in its log and exit non-zero, whatever the cause.
    except (ApiNotReadyError, httpx2.HTTPError) as error:
        logger.error("bootstrap failed", extra={"error_type": type(error).__name__, "detail": str(error)})
        return 1
    finally:
        client.close()


if __name__ == "__main__":  # pragma: no cover  # module run as a script; main() itself is tested
    sys.exit(main())
