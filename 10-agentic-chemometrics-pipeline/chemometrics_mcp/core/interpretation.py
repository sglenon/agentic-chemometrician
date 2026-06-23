"""Pure logic for model selection and result interpretation.

No MCP imports are allowed in this module — it is a pure computation layer
that may be tested in isolation from the MCP server.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

from chemometrics_contracts import (
    AnalysisResult,
    InterpretationSummary,
    ModelSelectionRecommendation,
    SpectralDataset,
    ValidationSummary,
    ValidationWarning,
)

# ---------------------------------------------------------------------------
# Complexity lookup
# ---------------------------------------------------------------------------

_COMPLEXITY_SCORES: dict[str, float] = {
    "pca": 0.9,
    "pca_lda": 0.9,
    "svm_rbf": 0.5,
    "svr": 0.5,
    "random_forest": 0.6,
    "plsr": 0.8,
    "kmeans": 0.9,
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_model(
    result: AnalysisResult,
    validation_summary: ValidationSummary | None = None,
) -> dict:
    """Score a single AnalysisResult on multiple criteria.

    Returns a dict with keys: performance, reliability, interpretability,
    complexity, composite.  All values are floats in [0, 1].
    """
    metrics = result.metrics or {}

    # 1. Performance
    task = (result.task_name or "").lower()
    if "classif" in task or "lda" in task:
        # Classification: prefer balanced_accuracy over accuracy
        if "balanced_accuracy" in metrics:
            performance = float(metrics["balanced_accuracy"])
        elif "accuracy" in metrics:
            performance = float(metrics["accuracy"])
        else:
            performance = 0.0
    elif "regress" in task or "plsr" in task:
        r2 = metrics.get("r2", 0.0)
        performance = float(r2) if float(r2) > 0 else 0.0
    elif "pca" in task or "cluster" in task or "unsupervised" in task:
        if "explained_variance_ratio_cumulative" in metrics:
            performance = float(metrics["explained_variance_ratio_cumulative"])
        else:
            performance = 0.5
    else:
        # Fall back on any common metric we can find
        if "balanced_accuracy" in metrics:
            performance = float(metrics["balanced_accuracy"])
        elif "accuracy" in metrics:
            performance = float(metrics["accuracy"])
        elif "r2" in metrics and float(metrics["r2"]) > 0:
            performance = float(metrics["r2"])
        elif "explained_variance_ratio_cumulative" in metrics:
            performance = float(metrics["explained_variance_ratio_cumulative"])
        else:
            performance = 0.0

    performance = max(0.0, min(1.0, performance))

    # 2. Reliability: penalise for warnings with severity "warning" or "error"
    reliability = 1.0
    for w in result.warnings:
        if w.severity in ("warning", "error"):
            reliability -= 0.3
    reliability = max(0.0, min(1.0, reliability))

    # 3. Interpretability
    interpretability = 1.0 if result.selected_features else 0.5

    # 4. Complexity (inverse of complexity — simpler = higher score)
    complexity = _COMPLEXITY_SCORES.get(result.model_name, 0.5)

    # Composite
    composite = (
        0.4 * performance
        + 0.3 * reliability
        + 0.2 * interpretability
        + 0.1 * complexity
    )
    composite = max(0.0, min(1.0, composite))

    return {
        "performance": performance,
        "reliability": reliability,
        "interpretability": interpretability,
        "complexity": complexity,
        "composite": composite,
    }


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def select_best_model(
    results: Sequence[AnalysisResult],
    validation_summary: ValidationSummary | None = None,
    task_name: str | None = None,
) -> ModelSelectionRecommendation:
    """Select the best defensible model from candidate results."""
    # 1. Filter by task_name if specified
    candidates = list(results)
    if task_name is not None:
        candidates = [r for r in candidates if r.task_name == task_name]

    # 2. Empty case
    if not candidates:
        return ModelSelectionRecommendation(
            selected_model=None,
            rationale="No results to compare.",
        )

    # 3. Score each result
    scored = [(r, score_model(r, validation_summary)) for r in candidates]

    # 4. Sort by composite descending
    scored.sort(key=lambda x: x[1]["composite"], reverse=True)

    best_result, best_scores = scored[0]
    best_name = best_result.model_name
    best_composite = best_scores["composite"]

    # 5. Build all candidate names
    all_names = tuple(r.model_name for r, _ in scored)

    # 6. Build rationale string
    if len(scored) > 1:
        runner_up_result, runner_up_scores = scored[1]
        runner_up_name = runner_up_result.model_name
        runner_up_composite = runner_up_scores["composite"]

        # Describe the key reason
        reasons: list[str] = []
        if best_scores["performance"] > runner_up_scores["performance"]:
            reasons.append("higher performance score")
        if best_scores["reliability"] > runner_up_scores["reliability"]:
            reasons.append("higher validation reliability")
        if best_scores["interpretability"] > runner_up_scores["interpretability"]:
            reasons.append("non-empty feature importance")
        if not reasons:
            reasons.append("better composite score")

        reason_str = " and ".join(reasons)
        rationale = (
            f"Selected {best_name!r} (composite score {best_composite:.2f}) "
            f"over {runner_up_name!r} ({runner_up_composite:.2f}) "
            f"due to {reason_str}. "
            f"Human review required before accepting."
        )
    else:
        rationale = (
            f"Selected {best_name!r} (composite score {best_composite:.2f}) "
            f"as the only candidate. "
            f"Human review required before accepting."
        )

    # 7. requires_human_approval
    suspicious_codes = {"suspicious_high_metric", "suspicious_metric"}
    has_suspicious = any(
        w.code in suspicious_codes
        for r, _ in scored
        for w in r.warnings
    )
    validation_failed = (
        validation_summary is not None and validation_summary.passed is False
    )
    requires_human_approval = has_suspicious or validation_failed

    # 8. Collect warnings from all results, deduplicate by code
    seen_codes: set[str] = set()
    deduped_warnings: list[ValidationWarning] = []
    for r, _ in scored:
        for w in r.warnings:
            if w.code not in seen_codes:
                seen_codes.add(w.code)
                deduped_warnings.append(w)

    return ModelSelectionRecommendation(
        selected_model=best_name,
        candidate_models=all_names,
        rationale=rationale,
        requires_human_approval=requires_human_approval,
        warnings=tuple(deduped_warnings),
    )


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------


def interpret_results(
    results: Sequence[AnalysisResult],
    dataset: SpectralDataset | None = None,
    validation_summary: ValidationSummary | None = None,
) -> InterpretationSummary:
    """Build an InterpretationSummary from tool-produced outputs only.

    CRITICAL: Uses ONLY data from result.selected_features, result.metrics,
    result.interpretation, and dataset.axis. Never invents feature names,
    chemical assignments, or wavelength interpretations.
    """
    warnings: list[ValidationWarning] = []

    # Collect important features per model
    models_with_features: list[tuple[str, list]] = []
    models_without_features: list[str] = []

    for r in results:
        if r.selected_features:
            models_with_features.append((r.model_name, list(r.selected_features)))
        else:
            models_without_features.append(r.model_name)
            warnings.append(
                ValidationWarning(
                    code="no_feature_importance",
                    message=(
                        f"Model {r.model_name!r} produced no feature importance data. "
                        f"selected_features is empty."
                    ),
                    category="interpretation",
                    severity="warning",
                    affected_stage="interpretation",
                )
            )

    # Add any validation_summary warnings
    if validation_summary is not None:
        for w in validation_summary.warnings:
            warnings.append(w)

    # important_features: union of all selected_features, deduplicated
    all_features: list = []
    seen_features: set = set()
    for _, feats in models_with_features:
        for f in feats:
            key = f if not isinstance(f, list) else tuple(f)
            if key not in seen_features:
                seen_features.add(key)
                all_features.append(f)

    # model_comparisons: one line per result
    model_comparisons: list[str] = []
    for r in results:
        metrics = r.metrics or {}
        metric_parts: list[str] = []
        for key in ("accuracy", "balanced_accuracy", "r2", "explained_variance_ratio_cumulative"):
            if key in metrics:
                metric_parts.append(f"{key}={metrics[key]:.4g}")
        metric_str = ", ".join(metric_parts) if metric_parts else "no metrics"

        if r.selected_features:
            top_features = list(r.selected_features)[:5]
            feat_str = str(top_features) if len(r.selected_features) <= 5 else str(top_features) + "..."
            line = f"{r.model_name}: {metric_str}, top features: {feat_str}"
        else:
            line = f"{r.model_name}: {metric_str} (no feature importance available)"
        model_comparisons.append(line)

    # Build summary text
    if not models_with_features:
        summary = (
            "No feature importance data available from the models run. "
            "Run random_forest or plsr to obtain wavelength importance."
        )
    else:
        n_models_with = len(models_with_features)

        # Find features appearing in >= 2 models
        if n_models_with >= 2:
            feature_occurrence: Counter = Counter()
            for _, feats in models_with_features:
                for f in feats:
                    key = f if not isinstance(f, list) else tuple(f)
                    feature_occurrence[key] += 1
            consistent = [f for f, cnt in feature_occurrence.items() if cnt >= 2]
        else:
            consistent = []

        lines: list[str] = [
            f"Feature importance is available from {n_models_with} model(s).",
        ]
        if consistent:
            lines.append(
                f"Most consistently important wavelengths (appearing in ≥2 models): {consistent}"
            )
        lines.append(
            "Note: Wavelength importance reflects model evidence only. "
            "Do not interpret as chemical causality without domain expert review."
        )
        summary = "\n".join(lines)

    return InterpretationSummary(
        summary=summary,
        important_features=tuple(all_features),
        model_comparisons=tuple(model_comparisons),
        warnings=tuple(warnings),
    )
