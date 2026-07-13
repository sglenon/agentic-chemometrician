"""Pure failure classification and fallback recommendation logic.

No MCP imports. No side effects beyond returning values.
"""
from __future__ import annotations

from typing import Sequence

from chemometrics_contracts import (
    NextModelRecommendation,
    RecommendNextModelRequest,
    ValidationWarning,
)

# ---------------------------------------------------------------------------
# Failure classification keyword tables (checked in priority order)
# ---------------------------------------------------------------------------

_CLASSIFICATION_KEYWORDS: list[tuple[str, list[str]]] = [
    ("data", ["missing", "nan", "empty", "shape", "dimension", "corrupt"]),
    ("preprocessing", ["preprocessing", "snv", "msc", "savitzky", "normali"]),
    ("convergence", ["converge", "iteration", "max_iter", "diverge"]),
    ("sample_size", ["sample", "n_sample", "too few", "insufficient"]),
    ("dependency", ["import", "module", "package", "not found", "install"]),
    ("unsupported_task", ["unsupported", "not implement", "deferred"]),
]

# ---------------------------------------------------------------------------
# Fallback tables
# ---------------------------------------------------------------------------

_CLASSIFICATION_FALLBACKS: dict[str, str | None] = {
    "data": None,                # data issues cannot be solved by changing model
    "preprocessing": "svm_rbf",  # simpler preprocessing path
    "convergence": "random_forest",  # tree-based, no convergence issues
    "sample_size": "pca",        # unsupervised; no sample-size constraints
    "dependency": None,          # cannot recommend if dependency missing
    "unsupported_task": None,    # cannot help without task clarification
    "unknown": "random_forest",  # safe default
}

_MODEL_SPECIFIC_FALLBACKS: dict[str, str] = {
    "svm_rbf": "random_forest",
    "svr": "plsr",
    "pca_lda": "svm_rbf",       # LDA fails on small/singular covariance
    "plsr": "svr",
    "random_forest": "svm_rbf",
    "kmeans": "pca",
    "xgboost": "random_forest",
    "xgboost_reg": "plsr",
}

# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def classify_failure(failure_reason: str) -> str:
    """Classify a failure reason string into one of the standard categories.

    Categories are checked in priority order. Returns ``"unknown"`` if no
    keyword matches.
    """
    reason_lower = failure_reason.lower()
    for category, keywords in _CLASSIFICATION_KEYWORDS:
        if any(kw in reason_lower for kw in keywords):
            return category
    return "unknown"


def recommend_fallback(
    failed_model: str,
    failure_classification: str,
    candidate_models: Sequence[str] = (),
) -> str | None:
    """Return the recommended fallback model name, or None if no recommendation is possible.

    Algorithm:
    1. If *candidate_models* is non-empty, prefer a candidate.  Try the
       model-specific fallback first, then the classification fallback.
    2. If no candidate matches (or candidates is empty): use
       ``_MODEL_SPECIFIC_FALLBACKS.get(failed_model)``.
    3. If still None: use ``_CLASSIFICATION_FALLBACKS.get(failure_classification)``.
    """
    model_specific = _MODEL_SPECIFIC_FALLBACKS.get(failed_model)
    classification_based = _CLASSIFICATION_FALLBACKS.get(failure_classification)

    if candidate_models:
        candidates_set = set(candidate_models)
        # Prefer model-specific fallback if it's in the candidate list
        if model_specific and model_specific in candidates_set:
            return model_specific
        # Otherwise prefer classification-based fallback if it's in the candidate list
        if classification_based and classification_based in candidates_set:
            return classification_based
        # No preferred fallback found in candidates — fall through to unconstrained logic

    # No candidate constraint (or no candidate matched)
    if model_specific is not None:
        return model_specific
    return classification_based


def _build_rationale(
    failed_model: str,
    failure_classification: str,
    fallback_model: str | None,
) -> str:
    """Build a human-readable rationale string."""
    _CLASSIFICATION_EXPLANATIONS: dict[str, str] = {
        "data": "data quality issues",
        "preprocessing": "preprocessing failures",
        "convergence": "convergence issues",
        "sample_size": "insufficient sample size",
        "dependency": "missing dependency",
        "unsupported_task": "unsupported task type",
        "unknown": "an unclassified error",
    }
    _FALLBACK_EXPLANATIONS: dict[str, str] = {
        "random_forest": "tree-based models do not have convergence constraints",
        "svm_rbf": "SVM with RBF kernel follows a simpler preprocessing path",
        "pca": "PCA is unsupervised and has no sample-size constraints",
        "plsr": "PLSR is robust for regression tasks with limited samples",
        "svr": "SVR is robust for regression tasks",
    }

    classification_label = _CLASSIFICATION_EXPLANATIONS.get(
        failure_classification, failure_classification
    )

    if fallback_model is None:
        return (
            f"No automatic fallback is possible for failure type "
            f"{failure_classification!r}. Human review and manual model "
            f"selection required."
        )

    fallback_explanation = _FALLBACK_EXPLANATIONS.get(
        fallback_model, f"it is a suitable alternative to {failed_model!r}"
    )
    return (
        f"Model {failed_model!r} failed due to {classification_label}. "
        f"Recommended fallback: {fallback_model!r} ({fallback_explanation}). "
        f"Comparability may be limited — human review required before accepting "
        f"fallback results."
    )


def build_recommendation(request: RecommendNextModelRequest) -> NextModelRecommendation:
    """Build a complete NextModelRecommendation from the request.

    Always sets ``requires_human_approval=True`` per project non-negotiable gates.
    """
    failure_classification = classify_failure(request.failure_reason)
    fallback_model = recommend_fallback(
        request.failed_model,
        failure_classification,
        request.candidate_models,
    )
    rationale = _build_rationale(request.failed_model, failure_classification, fallback_model)

    # Warnings: always include fallback_required
    warnings: list[ValidationWarning] = [
        ValidationWarning(
            code="fallback_required",
            message=f"Model {request.failed_model!r} failed: {request.failure_reason[:200]}...",
            severity="warning",
            category="reliability",
            affected_stage="modeling",
        )
    ]

    if fallback_model is None:
        warnings.append(
            ValidationWarning(
                code="no_automatic_fallback",
                message="No automatic fallback available. Manual intervention required.",
                severity="error",
                category="reliability",
            )
        )

    if failure_classification == "data":
        warnings.append(
            ValidationWarning(
                code="data_failure",
                message=(
                    "Data quality issue prevented model training. "
                    "Inspect the dataset before retrying."
                ),
                severity="error",
                category="data_quality",
            )
        )

    return NextModelRecommendation(
        failed_model=request.failed_model,
        failure_classification=failure_classification,
        fallback_model=fallback_model,
        rationale=rationale,
        requires_human_approval=True,
        warnings=tuple(warnings),
    )
