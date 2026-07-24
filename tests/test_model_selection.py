from __future__ import annotations

import numpy as np

from chemometrics_mcp.core.model_selection import evaluate_nested_supervised
from chemometrics_mcp.core.splits import materialize_splits


def test_nested_regression_retains_outer_predictions_and_group_metrics() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(12, 5))
    y = X[:, 0] * 2 + rng.normal(scale=.05, size=12)
    ids = [f"s{i}" for i in range(12)]
    groups = [f"p{i // 2}" for i in range(12)]
    splits = materialize_splits(ids, groups, n_splits=3)
    result = evaluate_nested_supervised(X, y, ids, groups, splits)
    assert result["status"] == "ok" and len(result["predictions"]) == 12
    assert result["metrics"]["primary_level"] == "group"
    assert result["split_id"] == splits.split_id
    assert all(
        selected["selection_level"] == "group"
        and selected["inner_candidate_scores"]
        for selected in result["selected_configs"]
    )
    assert result == evaluate_nested_supervised(X, y, ids, groups, splits)


def test_classification_and_missing_groups_are_safe() -> None:
    X = np.array([[0, 0], [0, 1], [2, 2], [2, 3], [0, .2], [2, 2.2], [0, .5], [2, 2.5]])
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    ids = [f"s{i}" for i in range(len(y))]
    groups = [f"p{i // 2}" for i in range(len(y))]
    splits = materialize_splits(ids, groups, n_splits=4)
    result = evaluate_nested_supervised(X, y, ids, groups, splits, "classification")
    assert result["status"] == "ok" and "balanced_accuracy" in result["metrics"]["group"]
    assert evaluate_nested_supervised(X, y, ids, None, splits)["status"] == "blocked"


def test_too_few_inner_groups_blocks() -> None:
    ids, groups = ["a", "b", "c", "d"], ["g1", "g1", "g2", "g2"]
    splits = materialize_splits(ids, groups, n_splits=2)
    result = evaluate_nested_supervised([[1, 2], [2, 3], [3, 4], [4, 5]], [1, 2, 3, 4], ids, groups, splits)
    assert result["status"] == "blocked"


def test_invalid_outer_assignments_and_nonfinite_data_block() -> None:
    ids = ["a", "b", "c", "d", "e", "f"]
    groups = ["g1", "g1", "g2", "g2", "g3", "g3"]
    splits = materialize_splits(ids, groups, n_splits=3)
    X = np.arange(18.0).reshape(6, 3)
    y = np.arange(6.0)
    assert evaluate_nested_supervised(
        np.where(X == 0, np.nan, X), y, ids, groups, splits
    )["status"] == "blocked"
