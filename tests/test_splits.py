from __future__ import annotations

import numpy as np
import pytest

from chemometrics_mcp.core.splits import aggregate_predictions_by_group, compute_scan_and_group_metrics, confidence_interval, materialize_splits


def test_group_splits_are_disjoint_and_reproducible() -> None:
    ids, groups = ["a", "b", "c", "d"], ["p1", "p1", "p2", "p2"]
    first = materialize_splits(ids, groups, n_splits=2)
    second = materialize_splits(ids, groups, n_splits=2)
    assert first == second
    for fold in first.folds:
        assert {groups[ids.index(x)] for x in fold.train_ids}.isdisjoint({groups[ids.index(x)] for x in fold.test_ids})


def test_missing_groups_and_group_equals_class_are_blockers() -> None:
    assert materialize_splits(["a", "b"], None).issues[0]["code"] == "missing_groups"
    split = materialize_splits(["a", "b", "c", "d"], ["A", "A", "B", "B"], n_splits=2, y=["A", "A", "B", "B"], task_name="classification")
    assert any(issue["code"] == "group_equals_class" for issue in split.issues)


def test_group_metrics_ignore_duplicate_technical_scans() -> None:
    basic = compute_scan_and_group_metrics([1, 2], [1, 4], ["p1", "p2"])
    duplicated = compute_scan_and_group_metrics([1, 1, 2], [1, 1, 4], ["p1", "p1", "p2"])
    assert basic["group"] == duplicated["group"]


def test_regression_classification_aggregation_and_bootstrap() -> None:
    grouped = aggregate_predictions_by_group(["a", "a", "b"], ["a", "b", "b"], ["p1", "p1", "p2"])
    assert grouped["p1"]["y_true"] == "a"
    metrics = compute_scan_and_group_metrics(["a", "a", "b"], ["a", "b", "b"], ["p1", "p1", "p2"], "classification")
    assert metrics["primary_level"] == "group" and "accuracy" in metrics["group"]
    assert confidence_interval([1, 2, 3], seed=3) == confidence_interval([1, 2, 3], seed=3)
    assert confidence_interval(np.array([1.0, 2.0]), seed=3)["estimate"] == 1.5
    with pytest.raises(ValueError): confidence_interval([])
    single = compute_scan_and_group_metrics([1.0], [1.2], ["one"])
    assert single["group"]["r2"] is None
    constant = compute_scan_and_group_metrics(
        [1.0, 1.0], [1.0, 1.1], ["one", "two"]
    )
    assert constant["group"]["r2"] is None
