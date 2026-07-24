"""Small deterministic nested evaluator with group-safe model selection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC, SVR

from chemometrics_mcp.core.pipelines import build_pipeline
from chemometrics_mcp.core.splits import MaterializedSplits, compute_scan_and_group_metrics


@dataclass(frozen=True, slots=True)
class Candidate:
    identity: str
    preprocessor: str
    family: str
    parameters: dict[str, Any]


def default_candidates(task_name: str) -> tuple[Candidate, ...]:
    if "class" in task_name.lower():
        return (
            Candidate("raw:logistic:c1", "raw", "logistic", {"C": 1.0}),
            Candidate("snv:svm:c1", "snv", "svm", {"C": 1.0}),
        )
    return (
        Candidate("raw:pls:n1", "raw", "pls", {"n_components": 1}),
        Candidate("raw:pls:n2", "raw", "pls", {"n_components": 2}),
        Candidate("snv:ridge:a1", "snv", "ridge", {"alpha": 1.0}),
        Candidate("snv:svr:c1", "snv", "svr", {"C": 1.0, "epsilon": 0.1}),
    )


def _estimator(candidate: Candidate, seed: int = 42):
    if candidate.family == "logistic": return LogisticRegression(max_iter=300, random_state=seed, **candidate.parameters)
    if candidate.family == "svm": return SVC(random_state=seed, **candidate.parameters)
    if candidate.family == "pls": return PLSRegression(**candidate.parameters)
    if candidate.family == "ridge": return Ridge(**candidate.parameters)
    if candidate.family == "svr": return SVR(**candidate.parameters)
    raise ValueError(f"Unknown model family: {candidate.family}")


def _score(y: np.ndarray, prediction: np.ndarray, classification: bool) -> float:
    if classification:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return float(balanced_accuracy_score(y, prediction))
    return -float(np.sqrt(mean_squared_error(y, prediction)))


def _predict(model, X: np.ndarray, classification: bool) -> np.ndarray:
    predicted = np.asarray(model.predict(X)).reshape(-1)
    if classification:
        return predicted
    return predicted.astype(float)


def _inner_select(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    candidates: Sequence[Candidate],
    classification: bool,
    seed: int,
) -> tuple[Candidate | None, list[str], list[dict[str, Any]]]:
    unique = len(set(groups.tolist()))
    if unique < 3:
        return None, ["inner_too_few_groups_for_nested_selection"], []
    splitter = GroupKFold(n_splits=min(3, unique))
    scored: list[tuple[float, str, Candidate]] = []
    score_records: list[dict[str, Any]] = []
    warnings: list[str] = []
    for candidate in candidates:
        # PLS components cannot exceed either feature count or each inner training size.
        if candidate.family == "pls" and candidate.parameters.get("n_components", 1) > min(X.shape[1], len(X) - max(1, len(X) // unique)):
            continue
        inner_truth: list[Any] = []
        inner_predictions: list[Any] = []
        inner_groups: list[Any] = []
        try:
            for train, test in splitter.split(X, y, groups):
                if classification and not set(np.unique(y[test])).issubset(
                    set(np.unique(y[train]))
                ):
                    raise ValueError(
                        "held-out class absent from inner training groups"
                    )
                model = build_pipeline(
                    candidate.preprocessor, _estimator(candidate, seed)
                )
                model.fit(X[train], y[train])
                prediction = _predict(model, X[test], classification)
                inner_truth.extend(y[test].tolist())
                inner_predictions.extend(prediction.tolist())
                inner_groups.extend(groups[test].tolist())
            # Candidate selection is preparation-level, so adding technical
            # replicate scans cannot improve the primary selection score.
            metrics = compute_scan_and_group_metrics(
                inner_truth,
                inner_predictions,
                inner_groups,
                "classification" if classification else "regression",
            )
            primary = metrics["group"]
            score = (
                float(primary["balanced_accuracy"])
                if classification
                else -float(primary["rmse"])
            )
            scored.append((score, candidate.identity, candidate))
            score_records.append(
                {
                    "candidate": asdict(candidate),
                    "selection_level": "group",
                    "selection_score": score,
                    "metrics": metrics,
                }
            )
        except (ValueError, TypeError) as exc:
            warnings.append(f"candidate_failed:{candidate.identity}:{exc}")
    if not scored:
        return None, warnings or ["no_valid_candidates"], score_records
    return (
        max(scored, key=lambda row: (row[0], row[1]))[2],
        warnings,
        score_records,
    )


def evaluate_nested_supervised(X: Sequence[Sequence[float]], y: Sequence[Any], sample_ids: Sequence[str], groups: Sequence[Any] | None, splits: MaterializedSplits, task_name: str = "regression", candidates: Sequence[Candidate] | None = None, seed: int = 42) -> dict[str, Any]:
    """Evaluate bounded candidates with inner selection inside each outer fold.

    ``groups`` must be supplied explicitly even when ``splits`` is already
    materialized; labels are never used as a grouping fallback.
    """
    matrix, target = np.asarray(X, dtype=float), np.asarray(y)
    classification = "class" in task_name.lower()
    warnings: list[str] = []
    if (
        matrix.ndim != 2
        or not matrix.size
        or not np.isfinite(matrix).all()
        or target.ndim != 1
    ):
        return {"status": "blocked", "warnings": ["invalid_numeric_matrix_or_target"], "predictions": [], "metrics": {}}
    if groups is None or len(groups) != len(sample_ids) or len(target) != len(sample_ids) or matrix.shape[0] != len(sample_ids):
        return {"status": "blocked", "warnings": ["explicit_groups_required"], "predictions": [], "metrics": {}}
    if any(issue.get("level") == "blocker" for issue in splits.issues):
        return {"status": "blocked", "warnings": ["split_manifest_has_blockers"], "predictions": [], "metrics": {}}
    ids = [str(item) for item in sample_ids]
    if len(set(ids)) != len(ids) or not splits.folds:
        return {"status": "blocked", "warnings": ["invalid_sample_ids_or_empty_splits"], "predictions": [], "metrics": {}}
    index = {item: pos for pos, item in enumerate(ids)}
    group_values = np.asarray(groups, dtype=object)
    configs = tuple(candidates or default_candidates(task_name))
    if len(configs) > 8:
        return {"status": "blocked", "warnings": ["candidate_budget_exceeded"], "predictions": [], "metrics": {}}
    records: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    seen_test_ids: set[str] = set()
    for fold in splits.folds:
        if (
            set(fold.train_ids) & set(fold.test_ids)
            or not set(fold.train_ids).issubset(index)
            or not set(fold.test_ids).issubset(index)
            or seen_test_ids & set(fold.test_ids)
        ):
            return {"status": "blocked", "warnings": warnings + ["invalid_outer_fold_assignments"], "predictions": records, "metrics": {}}
        seen_test_ids.update(fold.test_ids)
        train = np.asarray([index[item] for item in fold.train_ids], dtype=int)
        test = np.asarray([index[item] for item in fold.test_ids], dtype=int)
        if classification and not set(np.unique(target[test])).issubset(
            set(np.unique(target[train]))
        ):
            return {"status": "blocked", "warnings": warnings + [f"fold_{fold.fold}:held_out_class_absent_from_training"], "predictions": records, "metrics": {}}
        chosen, fold_warnings, inner_scores = _inner_select(
            matrix[train],
            target[train],
            group_values[train],
            configs,
            classification,
            seed,
        )
        warnings.extend(f"fold_{fold.fold}:{item}" for item in fold_warnings)
        if chosen is None:
            return {"status": "blocked", "warnings": warnings + ["inner_group_selection_unavailable"], "predictions": records, "metrics": {}}
        try:
            model = build_pipeline(chosen.preprocessor, _estimator(chosen, seed))
            model.fit(matrix[train], target[train])
            predicted = _predict(model, matrix[test], classification)
        except (ValueError, TypeError) as exc:
            return {"status": "blocked", "warnings": warnings + [f"outer_fit_failed:{exc}"], "predictions": records, "metrics": {}}
        selected.append({"fold": fold.fold, "candidate": asdict(chosen), "inner_candidate_scores": inner_scores, "selection_level": "group", "train_ids": list(fold.train_ids), "test_ids": list(fold.test_ids)})
        records.extend({"sample_id": ids[idx], "group": str(group_values[idx]), "y_true": target[idx].item() if hasattr(target[idx], "item") else target[idx], "y_pred": value.item() if hasattr(value, "item") else value, "fold": fold.fold} for idx, value in zip(test, predicted))
    metrics = compute_scan_and_group_metrics([row["y_true"] for row in records], [row["y_pred"] for row in records], [row["group"] for row in records], task_name)
    return {"status": "ok", "seed": seed, "split_id": splits.split_id, "candidate_identities": [item.identity for item in configs], "selected_configs": selected, "fold_assignments": [{"fold": fold.fold, "train_ids": list(fold.train_ids), "test_ids": list(fold.test_ids)} for fold in splits.folds], "predictions": records, "metrics": metrics, "warnings": warnings}
