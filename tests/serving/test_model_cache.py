"""Tests for the champion cache with the in-memory registry and a clock the tests move by hand."""

import logging
import threading

import pandas as pd
import pytest
from fakes import FakeRegistry, FakeTracker

from campaign_response.registry.ports import CHAMPION, Classifier, ModelVersion
from campaign_response.serving.model_cache import ChampionCache, ModelUnavailableError, NoChampionError

TTL = 30.0


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class CountingRegistry(FakeRegistry):
    def __init__(self) -> None:
        super().__init__(FakeTracker())
        self.loads: list[int] = []
        self.lookups = 0

    def get_by_alias(self, alias: str) -> ModelVersion | None:
        self.lookups += 1
        return super().get_by_alias(alias)

    def load_model(self, version: int) -> Classifier:
        self.loads.append(version)
        return super().load_model(version)

    def add_version(self, champion: bool = True) -> int:
        with self.tracker.run("training", "train") as run_id:
            uri = self.tracker.log_model(run_id, object(), pd.DataFrame({"x": [1]}))
        version = self.register(uri, run_id).version
        if champion:
            self.set_alias(CHAMPION, version)
        return version


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def registry() -> CountingRegistry:
    return CountingRegistry()


@pytest.fixture
def cache(registry: CountingRegistry, clock: FakeClock) -> ChampionCache:
    return ChampionCache(registry, ttl_seconds=TTL, clock=clock)


def fail(*args: object) -> None:
    raise ConnectionError("cannot reach http://user:secret@mlflow:5000")


def test_cache_loads_champion_once(cache: ChampionCache, registry: CountingRegistry, clock: FakeClock) -> None:
    registry.add_version()

    first = cache.get()
    clock.now += TTL - 1
    second = cache.get()

    assert first is second
    assert first.version.version == 1
    assert (registry.loads, registry.lookups) == ([1], 1)


def test_cache_same_version_after_ttl_not_reloaded(
    cache: ChampionCache, registry: CountingRegistry, clock: FakeClock
) -> None:
    registry.add_version()
    cache.get()

    clock.now += TTL
    cache.get()

    assert (registry.loads, registry.lookups) == ([1], 2)


def test_cache_new_champion_seen_after_ttl(cache: ChampionCache, registry: CountingRegistry, clock: FakeClock) -> None:
    registry.add_version()
    cache.get()
    registry.add_version()

    before_ttl = cache.get()
    clock.now += TTL
    after_ttl = cache.get()

    assert (before_ttl.version.version, after_ttl.version.version) == (1, 2)
    assert registry.loads == [1, 2]


def test_cache_invalidate_checks_at_once(cache: ChampionCache, registry: CountingRegistry) -> None:
    registry.add_version()
    cache.get()
    registry.add_version()

    cache.invalidate()

    assert cache.get().version.version == 2


def test_cache_no_champion_NoChampionError(cache: ChampionCache, registry: CountingRegistry) -> None:
    registry.add_version(champion=False)

    with pytest.raises(NoChampionError, match="POST /train"):
        cache.get()


def test_cache_no_champion_retried_each_call(cache: ChampionCache, registry: CountingRegistry) -> None:
    with pytest.raises(NoChampionError):
        cache.get()
    registry.add_version()

    assert cache.get().version.version == 1


def test_cache_unreachable_without_model_ModelUnavailableError(
    cache: ChampionCache, registry: CountingRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registry, "get_by_alias", fail)

    with pytest.raises(ModelUnavailableError, match="unreachable") as error:
        cache.get()

    assert "secret" not in str(error.value)


def test_cache_unreachable_keeps_serving(
    cache: ChampionCache, registry: CountingRegistry, clock: FakeClock, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry.add_version()
    first = cache.get()
    monkeypatch.setattr(registry, "get_by_alias", fail)
    clock.now += TTL

    with caplog.at_level(logging.WARNING):
        served = cache.get()

    assert served is first
    assert caplog.records[0].__dict__["error_type"] == "ConnectionError"
    assert "secret" not in caplog.text


def test_cache_unreachable_retried_after_ttl_only(
    cache: ChampionCache, registry: CountingRegistry, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry.add_version()
    cache.get()
    calls: list[str] = []
    monkeypatch.setattr(registry, "get_by_alias", lambda alias: calls.append(alias) or fail())
    clock.now += TTL

    cache.get()
    cache.get()
    clock.now += TTL
    cache.get()

    assert len(calls) == 2


def test_cache_unloadable_version_keeps_old(
    cache: ChampionCache, registry: CountingRegistry, clock: FakeClock, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry.add_version()
    cache.get()
    registry.add_version()
    monkeypatch.setattr(registry, "load_model", fail)
    clock.now += TTL

    with caplog.at_level(logging.ERROR):
        served = cache.get()

    assert served.version.version == 1
    [record] = caplog.records
    assert (record.__dict__["model_version"], record.__dict__["serving_version"]) == (2, 1)


def test_cache_unloadable_first_model_ModelUnavailableError(
    cache: ChampionCache, registry: CountingRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry.add_version()
    monkeypatch.setattr(registry, "load_model", fail)

    with pytest.raises(ModelUnavailableError, match="cannot load champion version 1"):
        cache.get()


def test_cache_alias_removed_keeps_serving(
    cache: ChampionCache, registry: CountingRegistry, clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    registry.add_version()
    first = cache.get()
    del registry.aliases[CHAMPION]
    clock.now += TTL

    with caplog.at_level(logging.WARNING):
        served = cache.get()

    assert served is first
    assert caplog.records[0].message == "champion alias removed; serving the cached champion"


def test_cache_others_served_during_reload(
    cache: ChampionCache, registry: CountingRegistry, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry.add_version()
    cache.get()
    registry.add_version()
    loading, release = threading.Event(), threading.Event()
    original_load = registry.load_model

    def slow_load(version: int) -> Classifier:
        loading.set()
        assert release.wait(timeout=5)
        return original_load(version)

    monkeypatch.setattr(registry, "load_model", slow_load)
    clock.now += TTL
    results: list[int] = []
    loader = threading.Thread(target=lambda: results.append(cache.get().version.version))
    loader.start()
    assert loading.wait(timeout=5)

    during = cache.get()
    release.set()
    loader.join(timeout=5)

    assert during.version.version == 1
    assert results == [2]
    assert cache.get().version.version == 2


def test_cache_first_load_waiters_share_result(registry: CountingRegistry, clock: FakeClock) -> None:
    registry.add_version()
    cache = ChampionCache(registry, ttl_seconds=TTL, clock=clock)
    results: list[int] = []
    threads = [threading.Thread(target=lambda: results.append(cache.get().version.version)) for _ in range(5)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert results == [1] * 5
    assert registry.loads == [1]


def test_cache_refresh_after_wait_no_second_lookup(cache: ChampionCache, registry: CountingRegistry) -> None:
    """A thread that waited for the lock finds the work done; called directly, as the race cannot be timed."""
    registry.add_version()
    loaded = cache.get()

    assert cache._refresh() is loaded
    assert registry.lookups == 1


@pytest.mark.parametrize("ttl", [0.0, -1.0])
def test_cache_non_positive_ttl_ValueError(registry: CountingRegistry, ttl: float) -> None:
    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        ChampionCache(registry, ttl_seconds=ttl)
