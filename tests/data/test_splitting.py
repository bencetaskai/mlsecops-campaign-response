"""Tests for the stratified train / holdout split."""

import logging

import pandas as pd
import pytest

from campaign_response.data.splitting import SplitParams, split_train_holdout


@pytest.fixture
def customers() -> pd.DataFrame:
    """200 customers, 30 of them positive (15%), like the real data."""
    return pd.DataFrame({
        "ID": pd.array(range(1000, 1200), dtype="Int64"),
        "Income": pd.array([float(i) for i in range(200)], dtype="Float64"),
        "Response": pd.array([1] * 30 + [0] * 170, dtype="Int64"),
    })


def test_split_sizes_quarter_holdout(customers: pd.DataFrame) -> None:
    train, holdout = split_train_holdout(customers)

    assert (len(train), len(holdout)) == (150, 50)


def test_split_positive_rate_preserved(customers: pd.DataFrame) -> None:
    train, holdout = split_train_holdout(customers)

    assert train["Response"].sum() == 22 or train["Response"].sum() == 23
    assert holdout["Response"].mean() == pytest.approx(0.15, abs=0.02)


def test_split_no_id_overlap_no_loss(customers: pd.DataFrame) -> None:
    train, holdout = split_train_holdout(customers)

    assert set(train["ID"]).isdisjoint(holdout["ID"])
    assert set(train["ID"]) | set(holdout["ID"]) == set(customers["ID"])


def test_split_same_seed_same_split(customers: pd.DataFrame) -> None:
    first, _ = split_train_holdout(customers)
    second, _ = split_train_holdout(customers)

    pd.testing.assert_frame_equal(first, second)


def test_split_other_seed_other_split(customers: pd.DataFrame) -> None:
    default, _ = split_train_holdout(customers)
    other, _ = split_train_holdout(customers, SplitParams(seed=7))

    assert set(default["ID"]) != set(other["ID"])


def test_split_custom_size(customers: pd.DataFrame) -> None:
    _, holdout = split_train_holdout(customers, SplitParams(holdout_size=0.2))

    assert len(holdout) == 40


def test_split_duplicate_ids_ValueError(customers: pd.DataFrame) -> None:
    customers.loc[1, "ID"] = 1000

    with pytest.raises(ValueError, match="1 duplicate ID values"):
        split_train_holdout(customers)


def test_split_input_unchanged(customers: pd.DataFrame) -> None:
    before = customers.copy()

    split_train_holdout(customers)

    pd.testing.assert_frame_equal(customers, before)


def test_split_logs_rates(customers: pd.DataFrame, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="campaign_response.data.splitting"):
        split_train_holdout(customers)

    assert caplog.records[-1].__dict__["holdout_rows"] == 50
