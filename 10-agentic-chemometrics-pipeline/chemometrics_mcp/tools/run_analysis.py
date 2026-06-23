"""MCP tool: run_analysis

Executes an approved AnalysisPlan against a SpectralDataset.
Saves structured run artifacts (metrics, predictions, figures, preprocessing details).
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np

from chemometrics_contracts import (
    AnalysisResult,
    AnalysisRun,
    ArtifactReference,
    RunAnalysisRequest,
    RunMetadata,
    SpectralDataset,
    ToolResponse,
    ValidationWarning,
)

from chemometrics_mcp.artifacts import artifact_ref, ensure_run_dir, make_run_id
from chemometrics_mcp.core import modeling, preprocessing


def _spectral_dataset_from_dict(d: dict) -> SpectralDataset:
    """Reconstruct a SpectralDataset from a plain MCP JSON dict."""
    return SpectralDataset(
        x=tuple(tuple(float(v) for v in row) for row in d.get("x", [])),
        axis=tuple(float(v) for v in d.get("axis", [])),
        metadata=tuple(dict(m) for m in d.get("metadata", [])),
        labels=tuple(d["labels"]) if d.get("labels") is not None else None,
        modality=d.get("modality"),
        sample_ids=tuple(d["sample_ids"]) if d.get("sample_ids") is not None else None,
    )


def run(
    request: RunAnalysisRequest,
    *,
    runs_root: str | Path = "runs",
) -> ToolResponse[AnalysisRun]:
    """Execute the run_analysis tool.

    Parameters
    ----------
    request:
        Validated :class:`RunAnalysisRequest` instance.
    runs_root:
        Root directory under which run artifact directories are created.

    Returns
    -------
    :class:`ToolResponse` with an :class:`AnalysisRun` payload on success,
    or ``ok=False`` with an ``error`` message on failure.
    """
    dataset = request.dataset
    approved_plan = request.approved_plan

    # 1. Validate: dataset.x must be non-empty
    if not dataset.x:
        return ToolResponse(
            tool_name="run_analysis",
            ok=False,
            error="Dataset is empty: dataset.x has no rows.",
            message="Cannot run analysis on an empty dataset.",
        )

    # 2. Convert to numpy
    X = np.array(dataset.x, dtype=float)
    y: np.ndarray | None = np.array(dataset.labels) if dataset.labels is not None else None
    axis: np.ndarray | None = np.array(dataset.axis, dtype=float) if dataset.axis else None

    # 3. Choose preprocessing: first entry in preprocessing_candidates, fallback "raw"
    preprocessing_method = (
        approved_plan.preprocessing_candidates[0]
        if approved_plan.preprocessing_candidates
        else "raw"
    )

    # 4. Apply preprocessing
    try:
        X_proc, prep_details = preprocessing.apply(X, preprocessing_method)
    except ValueError as exc:
        return ToolResponse(
            tool_name="run_analysis",
            ok=False,
            error=f"Preprocessing failed ({preprocessing_method!r}): {exc}",
            message="Preprocessing step raised a ValueError.",
        )

    # 5. Build CV splitter
    cv = modeling.make_cv_splitter(
        approved_plan.validation_strategy or "stratified_kfold_5",
        y,
    )

    task_name = approved_plan.task_name or "unsupervised_exploration"

    # Set up run artifacts
    run_id = request.run_id or make_run_id(slug="run")
    artifact_dir = ensure_run_dir(run_id, runs_root)

    created_at = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    results: list[AnalysisResult] = []
    failed_models: list[str] = []
    warnings: list[ValidationWarning] = []
    result_artifacts: list[ArtifactReference] = []

    # 6. For each model in the plan
    for model_name in approved_plan.model_families:
        try:
            raw_result, fig_data = modeling.run_cv_model(
                X_proc,
                y,
                axis,
                model_name,
                cv,
                task_name,
                preprocessing_method,
            )
        except (ValueError, Exception) as exc:  # noqa: BLE001
            failed_models.append(model_name)
            warnings.append(
                ValidationWarning(
                    code="model_failed",
                    message=f"Model {model_name!r} failed: {exc}",
                    category="modeling",
                    severity="error",
                    affected_stage="analysis",
                )
            )
            continue

        # c. Save fig_data as JSON
        fig_filename = f"{model_name}_figure_data.json"
        fig_path = artifact_dir / fig_filename
        fig_path.write_text(
            json.dumps(fig_data, indent=2, default=str), encoding="utf-8"
        )

        # d. Build ArtifactReference for the figure data
        fig_artifact = artifact_ref(
            run_id,
            fig_filename,
            kind="figure_data",
            label=f"{model_name} figure data",
            mime_type="application/json",
            runs_root=runs_root,
        )
        result_artifacts.append(fig_artifact)

        # e. Construct final AnalysisResult with figures populated
        final_result = AnalysisResult(
            task_name=raw_result.task_name,
            model_name=raw_result.model_name,
            preprocessing=raw_result.preprocessing,
            metrics=raw_result.metrics,
            predictions=raw_result.predictions,
            selected_features=raw_result.selected_features,
            figures=(fig_artifact,),
            warnings=raw_result.warnings,
            interpretation=raw_result.interpretation,
            run_metadata=raw_result.run_metadata,
        )
        results.append(final_result)

    # 7. Save run_summary.json
    run_summary = {
        "run_id": run_id,
        "task_name": task_name,
        "preprocessing": prep_details,
        "n_samples": int(X_proc.shape[0]),
        "n_features": int(X_proc.shape[1]),
        "model_names": list(approved_plan.model_families),
        "failed_models": failed_models,
        "created_at": created_at,
    }
    summary_filename = "run_summary.json"
    summary_path = artifact_dir / summary_filename
    summary_path.write_text(
        json.dumps(run_summary, indent=2, default=str), encoding="utf-8"
    )

    summary_artifact = artifact_ref(
        run_id,
        summary_filename,
        kind="run_summary",
        label="Run summary",
        mime_type="application/json",
        runs_root=runs_root,
    )

    # 8. Build AnalysisRun
    run_meta = RunMetadata(
        run_id=run_id,
        tool_name="run_analysis",
        status="completed",
        created_at=created_at,
        parameters={
            "task_name": task_name,
            "preprocessing": preprocessing_method,
            "model_families": list(approved_plan.model_families),
            "validation_strategy": approved_plan.validation_strategy,
        },
    )

    analysis_run = AnalysisRun(
        run_metadata=run_meta,
        results=tuple(results),
        failed_models=tuple(failed_models),
        warnings=tuple(warnings),
        artifacts=(summary_artifact,),
    )

    # 9. Return ToolResponse
    if not results and failed_models:
        return ToolResponse(
            tool_name="run_analysis",
            ok=False,
            error=f"All models failed: {failed_models}",
            payload=analysis_run,
            warnings=tuple(warnings),
            message="All requested models failed to run.",
        )

    return ToolResponse(
        tool_name="run_analysis",
        ok=True,
        payload=analysis_run,
        warnings=tuple(warnings),
        artifacts=(summary_artifact,) + tuple(result_artifacts),
        metadata=run_meta,
        message=(
            f"Analysis complete: {len(results)} model(s) succeeded, "
            f"{len(failed_models)} failed. "
            f"Run ID: {run_id}."
        ),
    )
