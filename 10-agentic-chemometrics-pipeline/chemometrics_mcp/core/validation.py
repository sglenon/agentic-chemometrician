"""Pure deterministic validation logic for chemometric analysis results.

No MCP imports. No side effects beyond returning values.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

from chemometrics_contracts import AnalysisResult, SpectralDataset, ValidationSummary, ValidationWarning


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


def run_all_checks(
    results: Sequence[AnalysisResult],
    dataset: SpectralDataset | None = None,
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

    checks = {
        "suspicious_metrics": not suspicious_metrics_triggered,
        "small_sample_per_class": not small_sample_triggered,
        "class_imbalance": not class_imbalance_triggered,
        "metadata_present": not missing_metadata_triggered,
        "regression_leakage": not regression_leakage_triggered,
    }

    passed = all(checks.values())

    return ValidationSummary(
        passed=passed,
        checks=checks,
        warnings=tuple(all_warnings),
    )
