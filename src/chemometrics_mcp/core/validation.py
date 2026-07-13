"""Pure deterministic validation logic for chemometric analysis results.

No MCP imports. No side effects beyond returning values.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

from chemometrics_contracts import (
    AnalysisResult,
    DatasetInspection,
    SpectralDataset,
    ValidationSummary,
    ValidationWarning,
)


_REPLICATE_LEAKAGE_THRESHOLD = 0.80
_GROUPED_STRATEGIES = frozenset({"grouped_kfold_5", "grouped_kfold_3", "group_kfold"})


def check_suspicious_metrics(result: AnalysisResult) -> ValidationWarning | None:
    """Warn if accuracy or balanced_accuracy >= 0.99 for classification tasks."""
    if "classification" not in result.task_name.lower():
        return None
    for metric_name in ("accuracy", "balanced_accuracy"):
        value = result.metrics.get(metric_name, 0)
        if value >= 0.99:
            return ValidationWarning(
                code="suspicious_high_metric",
                severity="warning",
                category="reliability",
                message=(
                    f"Model {result.model_name!r} achieved {metric_name}={value:.3f} "
                    f"— verify this is not a data leakage artifact."
                ),
            )
    return None


def check_small_sample_per_class(
    result: AnalysisResult,
    dataset: SpectralDataset | None,
) -> ValidationWarning | None:
    """Warn if any class has fewer than 5 samples (if labels are available).

    Emits one warning for the first underrepresented class found.
    Callers that want all warnings should loop externally; this function returns
    the first offending class to keep the signature simple and consistent with
    the other checkers.
    """
    if dataset is None or dataset.labels is None:
        return None
    counts = Counter(dataset.labels)
    for class_name, n in counts.items():
        if n < 5:
            return ValidationWarning(
                code="small_sample_per_class",
                severity="warning",
                category="data_quality",
                message=(
                    f"Class {class_name!r} has only {n} sample(s). "
                    f"Results for this class may be unreliable."
                ),
            )
    return None


def _check_small_sample_per_class_all(
    result: AnalysisResult,
    dataset: SpectralDataset | None,
) -> list[ValidationWarning]:
    """Return one warning per underrepresented class (internal use by run_all_checks)."""
    if dataset is None or dataset.labels is None:
        return []
    counts = Counter(dataset.labels)
    warnings = []
    for class_name, n in counts.items():
        if n < 5:
            warnings.append(
                ValidationWarning(
                    code="small_sample_per_class",
                    severity="warning",
                    category="data_quality",
                    message=(
                        f"Class {class_name!r} has only {n} sample(s). "
                        f"Results for this class may be unreliable."
                    ),
                )
            )
    return warnings


def check_class_imbalance(
    result: AnalysisResult,
    dataset: SpectralDataset | None,
) -> ValidationWarning | None:
    """Warn if majority class / minority class ratio > 3.0."""
    if dataset is None or dataset.labels is None:
        return None
    if "classification" not in result.task_name.lower():
        return None
    counts = Counter(dataset.labels)
    if len(counts) < 2:
        return None
    max_count = max(counts.values())
    min_count = min(counts.values())
    if min_count == 0:
        return None
    ratio = max_count / min_count
    if ratio > 3.0:
        return ValidationWarning(
            code="class_imbalance",
            severity="warning",
            category="data_quality",
            message=f"Class imbalance ratio {ratio:.1f}x. Consider stratified validation.",
        )
    return None


def check_missing_metadata(result: AnalysisResult) -> ValidationWarning | None:
    """Warn if result has no run_metadata (cannot trace provenance)."""
    if result.run_metadata is None:
        return ValidationWarning(
            code="missing_run_metadata",
            severity="info",
            category="provenance",
            message=(
                f"Result for model {result.model_name!r} has no run metadata. "
                f"Traceability is limited."
            ),
        )
    return None


def check_regression_target_leakage(result: AnalysisResult) -> ValidationWarning | None:
    """Warn if regression result has R² > 0.99 (suspiciously perfect fit)."""
    if "regression" not in result.task_name.lower():
        return None
    r2 = result.metrics.get("r2", 0)
    if r2 > 0.99:
        return ValidationWarning(
            code="suspicious_regression_r2",
            severity="warning",
            category="reliability",
            message=(
                f"Regression model {result.model_name!r} R²={r2:.3f} "
                f"— investigate for target leakage."
            ),
        )
    return None


def check_replicate_leakage(
    result: AnalysisResult,
    dataset: SpectralDataset | None,
    inspection: DatasetInspection | None,
) -> list[ValidationWarning]:
    """Detect scan-level replicate leakage via prediction consistency within groups.

    For each candidate group column, samples sharing the same group value are
    treated as replicates. If >80% of prediction pairs within a group are
    identical, the model may be memorizing replicate structure rather than
    learning generalizable patterns.

    Returns one warning per group column that shows leakage, listing the
    most affected group values.
    """
    if dataset is None or inspection is None:
        return []
    if not result.predictions:
        return []
    if not list(inspection.candidate_group_columns):
        return []

    predictions = list(result.predictions)
    n_samples = len(predictions)

    if dataset.metadata and len(dataset.metadata) == n_samples:
        pass
    elif dataset.sample_ids and len(dataset.sample_ids) == n_samples:
        pass
    else:
        return []

    warnings: list[ValidationWarning] = []

    for group_col in inspection.candidate_group_columns:
        group_values: list[str] = []
        if dataset.metadata:
            for row in dataset.metadata:
                val = row.get(group_col)
                group_values.append(str(val) if val is not None else "__missing__")
        elif dataset.sample_ids:
            group_values = [str(s) for s in dataset.sample_ids]
        else:
            continue

        if len(group_values) != n_samples:
            continue

        groups: dict[str, list[int]] = defaultdict(list)
        for idx, gv in enumerate(group_values):
            groups[gv].append(idx)

        multi_member_groups = {gv: idxs for gv, idxs in groups.items() if len(idxs) >= 2}
        if not multi_member_groups:
            continue

        affected_groups: list[str] = []

        for gv, idxs in multi_member_groups.items():
            if len(idxs) > 20:
                import random
                sampled = random.Random(42).sample(list(combinations(idxs, 2)), 190)
            else:
                sampled = list(combinations(idxs, 2))

            if not sampled:
                continue

            matching = sum(1 for i, j in sampled if predictions[i] == predictions[j])
            consistency = matching / len(sampled)

            if consistency > _REPLICATE_LEAKAGE_THRESHOLD:
                affected_groups.append(f"{gv} ({consistency:.0%})")

        if affected_groups:
            warnings.append(
                ValidationWarning(
                    code="replicate_leakage",
                    severity="warning",
                    category="reliability",
                    message=(
                        f"Model {result.model_name!r} shows high prediction consistency "
                        f"within replicate groups (column {group_col!r}). "
                        f"Affected groups: {', '.join(affected_groups[:5])}. "
                        f"Consider group-aware cross-validation."
                    ),
                    details={"group_column": group_col, "affected_groups": affected_groups[:5]},
                )
            )

    return warnings


def check_group_leakage(
    results: Sequence[AnalysisResult],
    inspection: DatasetInspection | None,
) -> list[ValidationWarning]:
    """Warn when candidate group columns exist but grouped CV was not used.

    Fires when candidate group columns are detected in the dataset but the
    validation strategy is None or not a grouped strategy. This catches the
    common failure mode where GroupKFold falls back to regular KFold.
    """
    if inspection is None:
        return []
    if not list(inspection.candidate_group_columns):
        return []

    warnings: list[ValidationWarning] = []

    for result in results:
        strategy = result.validation_strategy
        is_grouped = strategy in _GROUPED_STRATEGIES if strategy else False

        if not is_grouped:
            strategy_desc = strategy if strategy else "not specified"
            warnings.append(
                ValidationWarning(
                    code="group_leakage_risk",
                    severity="warning",
                    category="reliability",
                    message=(
                        f"Dataset has candidate group columns "
                        f"({', '.join(inspection.candidate_group_columns)}) "
                        f"but model {result.model_name!r} used validation strategy "
                        f"{strategy_desc!r} (not group-aware). "
                        f"Replicates may have leaked across train/test splits."
                    ),
                    details={
                        "group_columns": list(inspection.candidate_group_columns),
                        "validation_strategy": strategy,
                        "model_name": result.model_name,
                    },
                )
            )

    return warnings


def check_split_instability(results: Sequence[AnalysisResult]) -> list[ValidationWarning]:
    """Detect models whose primary metric varies substantially across repeated runs.

    Groups results by ``model_name``. For each group with more than one result,
    computes the coefficient of variation (CV = std / mean) for the primary
    metric. Classification uses ``"accuracy"`` (falling back to
    ``"balanced_accuracy"``); regression uses ``"r2"`` (falling back to
    ``"rmse"``).

    Returns a warning for CV > 0.10 (severity ``"warning"`` for 0.10–0.25,
    ``"error"`` for > 0.25).
    """
    groups: dict[str, list[AnalysisResult]] = defaultdict(list)
    for r in results:
        groups[r.model_name].append(r)

    warnings: list[ValidationWarning] = []

    for model_name, group in groups.items():
        if len(group) <= 1:
            continue

        is_classification = any("classification" in r.task_name.lower() for r in group)

        if is_classification:
            primary_key = "accuracy"
            fallback_key = "balanced_accuracy"
        else:
            primary_key = "r2"
            fallback_key = "rmse"

        values: list[float] = []
        for r in group:
            if primary_key in r.metrics:
                values.append(float(r.metrics[primary_key]))
            elif fallback_key in r.metrics:
                values.append(float(r.metrics[fallback_key]))

        if len(values) < 2:
            continue

        arr = np.array(values, dtype=float)
        mean_val = float(np.mean(arr))
        if mean_val == 0.0:
            continue
        cv = float(np.std(arr, ddof=0) / abs(mean_val))

        if cv < 0.10:
            continue

        severity = "error" if cv > 0.25 else "warning"

        warnings.append(
            ValidationWarning(
                code="split_instability",
                severity=severity,
                category="reliability",
                message=(
                    f"Model {model_name!r} shows split instability: "
                    f"CV={cv:.3f} for primary metric across {len(values)} runs."
                ),
                details={"model_name": model_name, "cv": cv, "n_runs": len(values)},
            )
        )

    return warnings


_FTIR_SPECIFIC_PREPROCESSING = frozenset({"baseline_correction", "area_normalization"})


def check_modality_consistency(
    results: Sequence[AnalysisResult],
    dataset: SpectralDataset | None,
) -> list[ValidationWarning]:
    if dataset is None or dataset.modality is None:
        return []

    modality = dataset.modality.upper()
    warnings: list[ValidationWarning] = []

    for result in results:
        for method in result.preprocessing:
            if modality == "NIR" and method in _FTIR_SPECIFIC_PREPROCESSING:
                warnings.append(
                    ValidationWarning(
                        code="modality_preprocessing_mismatch",
                        severity="warning",
                        category="preprocessing",
                        message=(
                            f"Preprocessing method {method!r} is FTIR-specific "
                            f"but applied to NIR data in model {result.model_name!r}."
                        ),
                        details={
                            "modality": modality,
                            "method": method,
                            "model_name": result.model_name,
                        },
                    )
                )

    return warnings


def check_cross_modality_comparability(
    results_by_modality: Mapping[str, Sequence[AnalysisResult]],
) -> list[ValidationWarning]:
    if len(results_by_modality) < 2:
        return []

    warnings: list[ValidationWarning] = []
    modalities = sorted(results_by_modality.keys())

    preprocessing_sets: dict[str, set[str]] = {}
    feature_counts: dict[str, set[int]] = {}
    for mod, res_list in results_by_modality.items():
        prep_set: set[str] = set()
        n_features_set: set[int] = set()
        for r in res_list:
            prep_set.update(r.preprocessing)
            if r.selected_features:
                n_features_set.add(len(r.selected_features))
        preprocessing_sets[mod] = prep_set
        feature_counts[mod] = n_features_set

    all_prep = set()
    for s in preprocessing_sets.values():
        all_prep.update(s)
    has_different_prep = len(all_prep) > 0 and any(
        s != all_prep for s in preprocessing_sets.values()
    )

    all_feat_counts: set[int] = set()
    for s in feature_counts.values():
        all_feat_counts.update(s)
    has_different_features = len(all_feat_counts) > 1

    warnings.append(
        ValidationWarning(
            code="cross_modality_comparison",
            severity="info",
            category="comparability",
            message=(
                f"Comparing results across modalities "
                f"({', '.join(modalities)}). "
                f"Different preprocessing may have been applied, "
                f"and different feature counts make direct metric comparison misleading."
            ),
            details={
                "modalities": modalities,
                "preprocessing_by_modality": {
                    k: sorted(v) for k, v in preprocessing_sets.items()
                },
            },
        )
    )

    if has_different_prep:
        warnings.append(
            ValidationWarning(
                code="cross_modality_preprocessing_differs",
                severity="warning",
                category="comparability",
                message=(
                    f"Preprocessing methods differ across modalities "
                    f"({', '.join(modalities)}). "
                    f"Metric comparisons may not be meaningful."
                ),
                details={
                    "modalities": modalities,
                    "preprocessing_by_modality": {
                        k: sorted(v) for k, v in preprocessing_sets.items()
                    },
                },
            )
        )

    if has_different_features:
        warnings.append(
            ValidationWarning(
                code="cross_modality_feature_count_differs",
                severity="warning",
                category="comparability",
                message=(
                    f"Feature counts differ across modalities "
                    f"({', '.join(modalities)}). "
                    f"Direct metric comparison is misleading."
                ),
                details={
                    "modalities": modalities,
                    "feature_counts_by_modality": {
                        k: sorted(v) for k, v in feature_counts.items()
                    },
                },
            )
        )

    model_names_by_modality: dict[str, set[str]] = {}
    for mod, res_list in results_by_modality.items():
        model_names_by_modality[mod] = {r.model_name for r in res_list}
    all_models: set[str] = set()
    for s in model_names_by_modality.values():
        all_models.update(s)
    shared_models = set.intersection(*model_names_by_modality.values()) if model_names_by_modality else set()
    if shared_models:
        warnings.append(
            ValidationWarning(
                code="cross_modality_hyperparameters",
                severity="info",
                category="comparability",
                message=(
                    f"Models {sorted(shared_models)} appear in multiple modalities. "
                    f"Hyperparameters were likely not tuned per-modality."
                ),
                details={
                    "modalities": modalities,
                    "shared_models": sorted(shared_models),
                },
            )
        )

    return warnings


def run_all_checks(
    results: Sequence[AnalysisResult],
    dataset: SpectralDataset | None = None,
    inspection: DatasetInspection | None = None,
    split_instability: bool = True,
) -> ValidationSummary:
    """Run all checks against each result. Return a ValidationSummary."""
    if not results:
        return ValidationSummary(passed=None, checks={}, warnings=())

    all_warnings: list[ValidationWarning] = []

    # Track whether each check category has ever been triggered (failed)
    suspicious_metrics_triggered = False
    small_sample_triggered = False
    class_imbalance_triggered = False
    missing_metadata_triggered = False
    regression_leakage_triggered = False
    replicate_leakage_triggered = False
    group_leakage_triggered = False

    for result in results:
        # suspicious metrics
        w = check_suspicious_metrics(result)
        if w is not None:
            all_warnings.append(w)
            suspicious_metrics_triggered = True

        # small sample per class — emit one per underrepresented class
        small_sample_warnings = _check_small_sample_per_class_all(result, dataset)
        if small_sample_warnings:
            all_warnings.extend(small_sample_warnings)
            small_sample_triggered = True

        # class imbalance
        w = check_class_imbalance(result, dataset)
        if w is not None:
            all_warnings.append(w)
            class_imbalance_triggered = True

        # missing metadata
        w = check_missing_metadata(result)
        if w is not None:
            all_warnings.append(w)
            missing_metadata_triggered = True

        # regression leakage
        w = check_regression_target_leakage(result)
        if w is not None:
            all_warnings.append(w)
            regression_leakage_triggered = True

    # replicate leakage (per result, can produce multiple warnings)
    for result in results:
        repl_warnings = check_replicate_leakage(result, dataset, inspection)
        if repl_warnings:
            all_warnings.extend(repl_warnings)
            replicate_leakage_triggered = True

    # group leakage (across all results)
    group_warnings = check_group_leakage(results, inspection)
    if group_warnings:
        all_warnings.extend(group_warnings)
        group_leakage_triggered = True

    # split instability
    split_instability_triggered = False
    if split_instability:
        instability_warnings = check_split_instability(results)
        if instability_warnings:
            all_warnings.extend(instability_warnings)
            split_instability_triggered = True

    # modality consistency
    modality_consistency_triggered = False
    modality_warnings = check_modality_consistency(results, dataset)
    if modality_warnings:
        all_warnings.extend(modality_warnings)
        modality_consistency_triggered = True

    checks = {
        "suspicious_metrics": not suspicious_metrics_triggered,
        "small_sample_per_class": not small_sample_triggered,
        "class_imbalance": not class_imbalance_triggered,
        "metadata_present": not missing_metadata_triggered,
        "regression_leakage": not regression_leakage_triggered,
        "replicate_leakage": not replicate_leakage_triggered,
        "group_leakage": not group_leakage_triggered,
        "split_instability": not split_instability_triggered,
        "modality_consistency": not modality_consistency_triggered,
    }

    passed = all(checks.values())

    return ValidationSummary(
        passed=passed,
        checks=checks,
        warnings=tuple(all_warnings),
    )
