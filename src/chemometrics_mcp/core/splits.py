"""Leakage-safe split materialization and preparation-level metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from chemometrics_mcp.core.project_store import data_hash


@dataclass(frozen=True, slots=True)
class FoldIndices:
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    fold: int


@dataclass(frozen=True, slots=True)
class MaterializedSplits:
    split_id: str
    strategy: str
    group_key: str | None
    seed: int
    folds: tuple[FoldIndices, ...]
    issues: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def _blocker(code: str, message: str) -> dict[str, Any]:
    return {"code": code, "message": message, "level": "blocker", "stage": "splits"}


def _normalize_groups(sample_ids: Sequence[str], groups: Sequence[Any] | Mapping[str, Any] | None) -> list[Any] | None:
    if groups is None:
        return None
    if isinstance(groups, Mapping):
        if set(groups) != set(sample_ids):
            return None
        return [groups[item] for item in sample_ids]
    return list(groups) if len(groups) == len(sample_ids) else None


def materialize_splits(sample_ids: Sequence[str], groups: Sequence[Any] | Mapping[str, Any] | None, strategy: str = "group_kfold", n_splits: int = 5, seed: int = 42, y: Sequence[Any] | None = None, task_name: str | None = None) -> MaterializedSplits:
    ids = tuple(str(item) for item in sample_ids)
    normalized = _normalize_groups(ids, groups)
    grouped = strategy in {"group_kfold", "leave_one_group_out"}
    if not ids or len(set(ids)) != len(ids):
        return MaterializedSplits("invalid", strategy, "group", seed, (), (_blocker("invalid_sample_ids", "Sample ids must be non-empty and unique."),))
    if grouped and normalized is None:
        return MaterializedSplits("invalid", strategy, "group", seed, (), (_blocker("missing_groups", "Grouped strategies require one explicit group for each sample."),))
    if strategy == "explicit_holdout":
        if normalized is None or any(value not in {"train", "test"} for value in normalized):
            return MaterializedSplits("invalid", strategy, "explicit_holdout", seed, (), (_blocker("invalid_explicit_holdout", "Explicit holdout groups must be exactly 'train' or 'test'."),))
        train = tuple(item for item, group in zip(ids, normalized) if group == "train")
        test = tuple(item for item, group in zip(ids, normalized) if group == "test")
        folds = (FoldIndices(train, test, 0),) if train and test else ()
        issues = () if folds else (_blocker("invalid_explicit_holdout", "Explicit holdout must include train and test samples."),)
    elif strategy == "group_kfold":
        unique_groups = len(set(normalized or ()))
        if n_splits < 2 or n_splits > unique_groups:
            return MaterializedSplits("invalid", strategy, "group", seed, (), (_blocker("invalid_n_splits", "n_splits must be between 2 and the number of groups."),))
        splitter = GroupKFold(n_splits=n_splits)
        folds = tuple(FoldIndices(tuple(ids[i] for i in train), tuple(ids[i] for i in test), fold) for fold, (train, test) in enumerate(splitter.split(ids, groups=normalized)))
        issues = ()
    elif strategy == "leave_one_group_out":
        splitter = LeaveOneGroupOut()
        folds = tuple(FoldIndices(tuple(ids[i] for i in train), tuple(ids[i] for i in test), fold) for fold, (train, test) in enumerate(splitter.split(ids, groups=normalized)))
        issues = ()
    else:
        return MaterializedSplits("invalid", strategy, None, seed, (), (_blocker("unsupported_strategy", f"Unsupported strategy: {strategy}"),))
    issue_list = list(issues)
    group_by_id = dict(zip(ids, normalized or ()))
    for fold in folds:
        if set(group_by_id[item] for item in fold.train_ids) & set(group_by_id[item] for item in fold.test_ids):
            issue_list.append(_blocker("group_overlap", "Train and test groups overlap."))
        if y is not None and "class" in (task_name or "").lower():
            labels = dict(zip(ids, y))
            train_labels = {labels[item] for item in fold.train_ids}
            heldout = {labels[item] for item in fold.test_ids}
            if not heldout <= train_labels:
                issue_list.append(_blocker("group_equals_class", "A held-out class is absent from training; group may encode the class."))
    split_id = "split-" + data_hash({"sample_ids": ids, "groups": normalized, "strategy": strategy, "n_splits": n_splits, "seed": seed})[:16]
    return MaterializedSplits(split_id, strategy, "group" if grouped else "explicit_holdout", seed, folds, tuple(issue_list))


def _mean_or_mode(values: Sequence[Any], classification: bool) -> Any:
    if classification:
        counts = Counter(values)
        return sorted(counts, key=lambda value: (-counts[value], str(value)))[0]
    return float(np.mean(np.asarray(values, dtype=float)))


def aggregate_predictions_by_group(y_true: Sequence[Any], y_pred: Sequence[Any], groups: Sequence[Any], *, classification: bool | None = None) -> dict[str, dict[str, Any]]:
    if not (len(y_true) == len(y_pred) == len(groups)):
        raise ValueError("y_true, y_pred, and groups must have equal length")
    classification = any(isinstance(value, str) for value in y_true) if classification is None else classification
    buckets: dict[str, dict[str, list[Any]]] = defaultdict(lambda: {"true": [], "pred": []})
    for truth, prediction, group in zip(y_true, y_pred, groups):
        buckets[str(group)]["true"].append(truth)
        buckets[str(group)]["pred"].append(prediction)
    return {group: {"y_true": _mean_or_mode(values["true"], classification), "y_pred": _mean_or_mode(values["pred"], classification), "n_scans": len(values["true"])} for group, values in sorted(buckets.items())}


def _metrics(
    y_true: Sequence[Any], y_pred: Sequence[Any], classification: bool
) -> dict[str, float | None]:
    if classification:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            balanced = balanced_accuracy_score(y_true, y_pred)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced),
        }
    truth, predicted = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    residual = predicted - truth
    r2 = (
        float(r2_score(truth, predicted))
        if len(truth) > 1 and float(np.var(truth)) > 0
        else None
    )
    if r2 is not None and not np.isfinite(r2):
        r2 = None
    return {
        "rmse": float(np.sqrt(mean_squared_error(truth, predicted))),
        "mae": float(mean_absolute_error(truth, predicted)),
        "r2": r2,
        "bias": float(np.mean(residual)),
        "residual_std": float(np.std(residual, ddof=0)),
    }


def compute_scan_and_group_metrics(y_true: Sequence[Any], y_pred: Sequence[Any], groups: Sequence[Any], task_name: str = "regression") -> dict[str, Any]:
    classification = "class" in task_name.lower()
    grouped = aggregate_predictions_by_group(y_true, y_pred, groups, classification=classification)
    group_true = [item["y_true"] for item in grouped.values()]
    group_pred = [item["y_pred"] for item in grouped.values()]
    return {"scan": _metrics(y_true, y_pred, classification), "group": _metrics(group_true, group_pred, classification), "primary_level": "group", "n_groups": len(grouped), "aggregated_predictions": grouped}


def confidence_interval(values: Sequence[float], confidence: float = 0.95, n_bootstrap: int = 1000, seed: int = 42) -> dict[str, float]:
    if len(values) == 0 or not 0 < confidence < 1 or n_bootstrap < 1:
        raise ValueError("values, confidence, and n_bootstrap must be valid")
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("values must be a finite one-dimensional sequence")
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(array, size=(n_bootstrap, len(array)), replace=True), axis=1)
    alpha = (1 - confidence) / 2
    return {"estimate": float(np.mean(array)), "lower": float(np.quantile(means, alpha)), "upper": float(np.quantile(means, 1 - alpha)), "confidence": confidence}
