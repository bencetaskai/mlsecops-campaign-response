"""The champion model held in memory, reloaded only when the champion alias moves to another version.

Loading a model from MLflow takes 4.5-7.5 s and an alias lookup about 140 ms (measured against the compose stack),
so the alias is checked at most once per TTL, and a model is loaded once per version.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from campaign_response.registry.ports import CHAMPION, Classifier, ModelRegistry, ModelVersion

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS: Final = 30.0


class ModelUnavailableError(RuntimeError):
    """No model can be served: none is cached and the registry cannot provide one."""


class NoChampionError(ModelUnavailableError):
    """The registry has no champion yet; POST /train creates one."""


@dataclass(frozen=True)
class LoadedModel:
    version: ModelVersion
    model: Classifier


class ChampionCache:
    """Serves the cached champion; refreshes it after the TTL, or at once after invalidate().

    Failures never replace a cached model: while MLflow is unreachable or a new version cannot be loaded, the last
    champion keeps serving and the next check retries. Only without a cached model do they become errors.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive, got {ttl_seconds}")
        self._registry = registry
        self._ttl = ttl_seconds
        self._clock = clock
        self._current: LoadedModel | None = None
        self._checked_at: float | None = None
        self._refresh_lock = threading.Lock()

    def get(self) -> LoadedModel:
        current = self._current
        if current is not None and not self._check_due():
            return current
        # Only one thread refreshes. It marks the check as done before the 5-7 s load, so with a cached model the
        # other threads return above and keep serving the old champion, which is still valid; without one they
        # have nothing to serve and wait here.
        with self._refresh_lock:
            return self._refresh()

    def invalidate(self) -> None:
        """Check the alias on the next get(), e.g. right after a training run in this process promoted a model."""
        self._checked_at = None

    def _check_due(self) -> bool:
        return self._checked_at is None or self._clock() - self._checked_at >= self._ttl

    def _refresh(self) -> LoadedModel:
        current = self._current
        # Another thread may have refreshed while this one waited for the lock.
        if current is not None and not self._check_due():
            return current
        self._checked_at = self._clock()
        champion = self._lookup_champion(current)
        if champion is None or (current is not None and current.version.version == champion.version):
            assert current is not None
            return current
        return self._load(champion, current)

    def _lookup_champion(self, current: LoadedModel | None) -> ModelVersion | None:
        """The champion to serve; None means keep the cached model."""
        try:
            champion = self._registry.get_by_alias(CHAMPION)
        # Broad on purpose, as in the health check: MLflow fails with several error types when unreachable.
        except Exception as error:
            if current is None:
                raise ModelUnavailableError("model registry unreachable") from error
            logger.warning("model registry unreachable; serving the cached champion",
                           extra={"error_type": type(error).__name__, "model_version": current.version.version})
            return None
        if champion is None and current is None:
            raise NoChampionError("no champion model yet; call POST /train")
        if champion is None:
            logger.warning("champion alias removed; serving the cached champion",
                           extra={"model_version": self._version_of(current)})
        return champion

    def _load(self, champion: ModelVersion, current: LoadedModel | None) -> LoadedModel:
        started = self._clock()
        try:
            model = self._registry.load_model(champion.version)
        # Broad on purpose: a refused file (untrusted types, pickle) or a failed download must not end serving.
        except Exception as error:
            logger.error("cannot load the champion", extra={
                "model_version": champion.version, "error_type": type(error).__name__,
                "serving_version": self._version_of(current),
            })
            if current is None:
                raise ModelUnavailableError(f"cannot load champion version {champion.version}") from error
            return current
        loaded = LoadedModel(champion, model)
        self._current = loaded
        logger.info("loaded champion", extra={
            "model_version": champion.version, "previous_version": self._version_of(current),
            "load_seconds": round(self._clock() - started, 2),
        })
        return loaded

    @staticmethod
    def _version_of(loaded: LoadedModel | None) -> int | None:
        return loaded.version.version if loaded is not None else None
