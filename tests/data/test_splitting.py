"""Tests for the ID-hash train / holdout split: stable, complete, disjoint."""

import logging

import pandas as pd
import pytest

from campaign_response.data.splitting import SplitParams, holdout_bucket, split_train_holdout


def customers(first_id: int, count: int) -> pd.DataFrame:
    """Customers with consecutive IDs, every 7th one positive (about 14%, like the real data)."""
    ids = range(first_id, first_id + count)
    return pd.DataFrame({
        "ID": pd.array(list(ids), dtype="Int64"),
        "Response": pd.array([int(i % 7 == 0) for i in ids], dtype="Int64"),
    })


def ids(frame: pd.DataFrame) -> set[int]:
    return {int(i) for i in frame["ID"]}


def test_bucket_deterministic_in_unit_interval() -> None:
    values = [holdout_bucket(customer_id, 42) for customer_id in range(1000)]

    assert values == [holdout_bucket(customer_id, 42) for customer_id in range(1000)]
    assert all(0 <= value < 1 for value in values)


def test_split_size_close_to_quarter() -> None:
    _, holdout = split_train_holdout(customers(1000, 2000))

    assert 0.22 <= len(holdout) / 2000 <= 0.28


def test_split_no_id_overlap_no_loss() -> None:
    frame = customers(1000, 2000)

    train, holdout = split_train_holdout(frame)

    assert ids(train).isdisjoint(ids(holdout))
    assert ids(train) | ids(holdout) == ids(frame)


def test_split_new_customers_never_move_old() -> None:
    old = customers(1000, 1500)
    grown = pd.concat([old, customers(5000, 500)], ignore_index=True)

    old_train, old_holdout = split_train_holdout(old)
    new_train, new_holdout = split_train_holdout(grown)

    assert ids(old_holdout) <= ids(new_holdout)
    assert ids(old_train) <= ids(new_train)


def test_split_row_order_irrelevant() -> None:
    frame = customers(1000, 2000)

    _, holdout = split_train_holdout(frame)
    _, shuffled_holdout = split_train_holdout(frame.sample(frac=1, random_state=0))

    assert ids(holdout) == ids(shuffled_holdout)


def test_split_other_seed_other_split() -> None:
    frame = customers(1000, 2000)

    _, default = split_train_holdout(frame)
    _, other = split_train_holdout(frame, SplitParams(seed=7))

    assert ids(default) != ids(other)


def test_split_custom_size() -> None:
    _, holdout = split_train_holdout(customers(1000, 2000), SplitParams(holdout_size=0.5))

    assert 0.45 <= len(holdout) / 2000 <= 0.55


def test_split_duplicate_ids_ValueError() -> None:
    frame = customers(1000, 10)
    frame.loc[1, "ID"] = 1000

    with pytest.raises(ValueError, match="1 duplicate ID values"):
        split_train_holdout(frame)


def test_split_missing_id_ValueError() -> None:
    frame = customers(1000, 10)
    frame.loc[1, "ID"] = pd.NA

    with pytest.raises(ValueError, match="missing ID values"):
        split_train_holdout(frame)


def test_split_input_unchanged() -> None:
    frame = customers(1000, 200)
    before = frame.copy()

    split_train_holdout(frame)

    pd.testing.assert_frame_equal(frame, before)


def test_split_logs_sizes(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="campaign_response.data.splitting"):
        _, holdout = split_train_holdout(customers(1000, 200))

    assert caplog.records[-1].__dict__["holdout_rows"] == len(holdout)
