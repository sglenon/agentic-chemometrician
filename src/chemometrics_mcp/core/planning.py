"""Pure deterministic planning logic for the propose_analysis_plan tool.

No MCP imports allowed here. All functions operate purely on contract types
from chemometrics_contracts. This module can be tested in isolation.
"""
from __future__ import annotations

from chemometrics_contracts import (
    AnalysisPlan,
    DatasetInspection,
    ProposeAnalysisPlanRequest,
    ValidationWarning,
)

# ---------------------------------------------------------------------------
# Task inference
# ---------------------------------------------------------------------------

def infer_task_name(
    inspection: DatasetInspection,
    task_hint: str | None,
    allow_supervised_planning: bool = True,
) -> str | None:
    """Infer the primary task type from an inspection and optional hint.

    Returns
    -------
    str | None
        The task name, or ``None`` if the label columns are ambiguous.
    """
    if not allow_supervised_planning:
        return "unsupervised_exploration"

    label_cols = list(inspection.candidate_label_columns)

    if len(label_cols) == 0:
        return "unsupervised_exploration"

    if len(label_cols) > 1:
        # Ambiguous — caller must specify
        return None

    # Exactly 1 candidate label column — apply hint overrides
    if task_hint == "regression":
        return "regression"
    if task_hint == "binary_classification":
        return "binary_classification"
    if task_hint == "clustering":
        return "clustering"

    return "multi_class_classification"


# ---------------------------------------------------------------------------
# Preprocessing recommendations
# ---------------------------------------------------------------------------

_STANDARD_PREPROCESSING = ("snv", "msc", "sg_1st_deriv", "sg_2nd_deriv")
_FTIR_EXTRA = ("baseline_correction", "area_normalization")


def recommend_preprocessing(inspection: DatasetInspection) -> list[str]:
    """Return NIR/spectral preprocessing candidates in priority order."""
    candidates: list[str] = list(_STANDARD_PREPROCESSING)
    if inspection.modality and inspection.modality.upper() == "FTIR":
        candidates.extend(_FTIR_EXTRA)
    return candidates


# ---------------------------------------------------------------------------
# Validation strategy
# ---------------------------------------------------------------------------

def recommend_validation_strategy(
    inspection: DatasetInspection,
    task_name: str | None = None,
) -> str:
    """Return a cross-validation strategy based on task type, sample size and groups.

    Regression tasks use ``grouped_kfold_5`` (plain KFold) because
    ``StratifiedKFold`` requires discrete class labels.
    """
    if task_name == "regression":
        # Plain KFold for regression — stratification requires discrete labels
        return "grouped_kfold_5"

    if list(inspection.candidate_group_columns):
        return "grouped_kfold_5"

    sample_count = inspection.sample_count
    if sample_count is None or sample_count >= 50:
        return "stratified_kfold_5"
    if sample_count >= 20:
        return "stratified_kfold_3"
    return "loocv"


# ---------------------------------------------------------------------------
# Model family recommendations
# ---------------------------------------------------------------------------

def recommend_model_families(task_name: str | None, inspection: DatasetInspection) -> list[str]:  # noqa: ARG001
    """Return model families suitable for *task_name*."""
    if task_name in ("multi_class_classification", "binary_classification"):
        return ["svm_rbf", "random_forest", "logistic_regression", "pca_lda", "xgboost"]
    if task_name == "regression":
        return ["plsr", "svr", "ridge", "random_forest_reg", "xgboost_reg"]
    if task_name in ("unsupervised_exploration", "clustering"):
        return ["pca", "kmeans"]
    # None — ambiguous; exploratory fallback only
    return ["pca"]


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

_AMBIGUOUS_LABEL_WARNING = ValidationWarning(
    code="ambiguous_label_columns",
    message=(
        "Multiple candidate label columns found. "
        "User must specify 'label_column' in inspect_dataset "
        "or 'task_hint' in propose_analysis_plan."
    ),
    severity="warning",
    category="planning",
)


def _small_sample_warning(sample_count: int) -> ValidationWarning:
    return ValidationWarning(
        code="small_sample",
        message=f"Only {sample_count} samples. Results may be unreliable.",
        severity="warning",
        category="data_quality",
    )


# ---------------------------------------------------------------------------
# Human-readable plan builder
# ---------------------------------------------------------------------------

def _human_readable_plan(
    task_name: str | None,
    preprocessing_candidates: list[str],
    validation_strategy: str,
    model_families: list[str],
    inspection: DatasetInspection,
) -> str:
    modality = inspection.modality or "Spectral"
    lines: list[str] = [
        f"Analysis Plan for {modality} Data",
        "=====================================",
        f"Task: {task_name or 'ambiguous (not determined)'}",
        f"Preprocessing candidates: {', '.join(preprocessing_candidates)}",
        f"Validation strategy: {validation_strategy}",
        f"Model families: {', '.join(model_families)}",
        "",
    ]

    samples = inspection.sample_count
    features = inspection.feature_count
    if samples is not None and features is not None:
        lines.append(
            f"This plan was generated from a dataset with {samples} samples "
            f"and {features} spectral features."
        )
    elif samples is not None:
        lines.append(f"This plan was generated from a dataset with {samples} samples.")
    else:
        lines.append("This plan was generated from the provided dataset inspection.")

    label_cols = list(inspection.candidate_label_columns)
    if label_cols:
        label_col_str = label_cols[0] if len(label_cols) == 1 else ", ".join(label_cols)
        lines.append(f"Label column: {label_col_str}.")

    lines.append("Approval required before running.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main assembly function
# ---------------------------------------------------------------------------

def build_plan(request: ProposeAnalysisPlanRequest) -> AnalysisPlan:
    """Convert a :class:`ProposeAnalysisPlanRequest` into an :class:`AnalysisPlan`.

    This is the single entry point for the planning core. It is purely
    deterministic and has no side effects.
    """
    inspection = request.dataset_inspection

    task_name = infer_task_name(
        inspection,
        request.task_hint,
        request.allow_supervised_planning,
    )

    preprocessing_candidates = recommend_preprocessing(inspection)
    validation_strategy = recommend_validation_strategy(inspection, task_name)
    model_families = recommend_model_families(task_name, inspection)

    # Collect warnings — existing inspection warnings first
    warnings: list[ValidationWarning] = list(inspection.warnings)

    if task_name is None:
        warnings.append(_AMBIGUOUS_LABEL_WARNING)

    if inspection.sample_count is not None and inspection.sample_count < 30:
        warnings.append(_small_sample_warning(inspection.sample_count))

    human_plan = _human_readable_plan(
        task_name,
        preprocessing_candidates,
        validation_strategy,
        model_families,
        inspection,
    )

    return AnalysisPlan(
        task_name=task_name,
        preprocessing_candidates=tuple(preprocessing_candidates),
        validation_strategy=validation_strategy,
        model_families=tuple(model_families),
        human_readable_plan=human_plan,
        warnings=tuple(warnings),
    )
