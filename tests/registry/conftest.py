"""A real local MLflow for the adapter tests: a SQLite store and artifacts in a temporary folder."""

from collections.abc import Iterator

import mlflow
import pytest


@pytest.fixture(scope="module")
def mlflow_tracking_uri(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """One store per test module, as creating its tables takes about a second.

    The working directory moves to the temporary folder because a SQLite store without a server puts artifacts into
    ./mlruns; the process-wide tracking URI, which MlflowTracker sets, is restored afterwards.
    """
    folder = tmp_path_factory.mktemp("mlflow")
    previous = mlflow.get_tracking_uri()
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(folder)
        yield f"sqlite:///{(folder / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(previous)
