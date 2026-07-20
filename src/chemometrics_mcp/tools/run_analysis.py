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

import logging

from chemometrics_mcp.artifacts import artifact_ref, ensure_run_dir, make_run_id
from chemometrics_mcp.core import modeling, preprocessing
from chemometrics_mcp.core.validation import check_easy_task, check_split_instability

try:
    from chemometrics_mcp.core.figures import render_figure as _render_figure
except ImportError:
    _render_figure = None

log = logging.getLogger(__name__)


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

    # 3. Build preprocessing candidates list
    preprocessing_candidates = (
        list(approved_plan.preprocessing_candidates)
        if approved_plan.preprocessing_candidates
        else ["raw"]
    )

    task_name = approved_plan.task_name or "unsupervised_exploration"

    # 4. Build CV splitter — regression must not use StratifiedKFold
    _is_regression = task_name == "regression"
    _default_cv_strategy = "grouped_kfold_5" if _is_regression else "stratified_kfold_5"
    cv = modeling.make_cv_splitter(
        approved_plan.validation_strategy or _default_cv_strategy,
        y,
    )

    run_id = request.run_id or make_run_id(slug="run")
    artifact_dir = ensure_run_dir(run_id, runs_root)

    created_at = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    results: list[AnalysisResult] = []
    failed_models: list[str] = []
    warnings: list[ValidationWarning] = []
    result_artifacts: list[ArtifactReference] = []
    all_prep_details: list[dict] = []

    # 4b. Caveat: detect easy tasks via PCA separation heuristic
    if y is not None and task_name:
        easy_task_warning = check_easy_task(X, y, task_name)
        if easy_task_warning is not None:
            warnings.append(easy_task_warning)

    # 5. For each preprocessing candidate
    for preprocessing_method in preprocessing_candidates:
        try:
            X_proc, prep_details = preprocessing.apply(X, preprocessing_method)
        except ValueError as exc:
            warnings.append(
                ValidationWarning(
                    code="preprocessing_failed",
                    message=f"Preprocessing failed ({preprocessing_method!r}): {exc}",
                    category="preprocessing",
                    severity="error",
                    affected_stage="analysis",
                )
            )
            continue

        all_prep_details.append(prep_details)

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
                    approved_plan.validation_strategy,
                )
            except (ValueError, Exception) as exc:  # noqa: BLE001
                failed_models.append(f"{preprocessing_method}/{model_name}")
                warnings.append(
                    ValidationWarning(
                        code="model_failed",
                        message=f"Model {model_name!r} failed with preprocessing {preprocessing_method!r}: {exc}",
                        category="modeling",
                        severity="error",
                        affected_stage="analysis",
                    )
                )
                continue

            fig_filename = f"{preprocessing_method}_{model_name}_figure_data.json"
            fig_path = artifact_dir / fig_filename
            fig_path.write_text(
                json.dumps(fig_data, indent=2, default=str), encoding="utf-8"
            )

            fig_artifact = artifact_ref(
                run_id,
                fig_filename,
                kind="figure_data",
                label=f"{model_name} figure data",
                mime_type="application/json",
                runs_root=runs_root,
            )
            result_artifacts.append(fig_artifact)

            if _render_figure is not None:
                try:
                    rendered_filename = f"{model_name}_figure.png"
                    rendered_path = artifact_dir / rendered_filename
                    _render_figure(fig_data, model_name, rendered_path, format="png")
                    rendered_artifact = artifact_ref(
                        run_id,
                        rendered_filename,
                        kind="figure",
                        label=f"{model_name} figure",
                        mime_type="image/png",
                        runs_root=runs_root,
                    )
                    result_artifacts.append(rendered_artifact)
                except Exception:  # noqa: BLE001
                    log.warning("Figure rendering failed for %s", model_name, exc_info=True)

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

    # 5b. Split-instability check for plans with >1 model family
    if len(approved_plan.model_families) > 1 and results:
        multi_seed_results: list[AnalysisResult] = []
        for preprocessing_method in preprocessing_candidates:
            try:
                X_proc, _ = preprocessing.apply(X, preprocessing_method)
            except ValueError:
                continue
            for model_name in approved_plan.model_families:
                if f"{preprocessing_method}/{model_name}" in failed_models:
                    continue
                try:
                    ms_results = modeling.run_cv_model_multi_seed(
                        X_proc,
                        y,
                        axis,
                        model_name,
                        task_name,
                        preprocessing_method,
                        approved_plan.validation_strategy,
                        n_seeds=3,
                    )
                    for ms_result, _ in ms_results:
                        multi_seed_results.append(ms_result)
                except Exception:  # noqa: BLE001
                    pass

        if multi_seed_results:
            instability_warnings = check_split_instability(multi_seed_results)
            warnings.extend(instability_warnings)

    # 5c. Preprocessing comparison artifact
    comparison_rows: list[dict] = []
    for result in results:
        prep_label = ", ".join(result.preprocessing) if result.preprocessing else "none"
        primary_metric = ""
        primary_value = None
        for k, v in result.metrics.items():
            if isinstance(v, float):
                primary_metric = k
                primary_value = v
                break
        comparison_rows.append({
            "preprocessing_method": prep_label,
            "model_name": result.model_name,
            "primary_metric": primary_metric,
            "primary_value": primary_value,
        })

    comparison_filename = "preprocessing_comparison.json"
    comparison_path = artifact_dir / comparison_filename
    comparison_path.write_text(
        json.dumps(comparison_rows, indent=2, default=str), encoding="utf-8"
    )

    comparison_artifact = artifact_ref(
        run_id,
        comparison_filename,
        kind="preprocessing_comparison",
        label="Preprocessing comparison",
        mime_type="application/json",
        runs_root=runs_root,
    )

    # 6. Save run_summary.json
    run_summary = {
        "run_id": run_id,
        "task_name": task_name,
        "preprocessing": all_prep_details,
        "preprocessing_candidates": preprocessing_candidates,
        "n_samples": int(X.shape[0]) if len(X) > 0 else 0,
        "n_features": int(X.shape[1]) if len(X) > 0 and len(X[0]) > 0 else 0,
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

    # 7. Build AnalysisRun
    run_meta = RunMetadata(
        run_id=run_id,
        tool_name="run_analysis",
        status="completed",
        created_at=created_at,
        parameters={
            "task_name": task_name,
            "preprocessing": preprocessing_candidates,
            "model_families": list(approved_plan.model_families),
            "validation_strategy": approved_plan.validation_strategy,
        },
    )

    analysis_run = AnalysisRun(
        run_metadata=run_meta,
        results=tuple(results),
        failed_models=tuple(failed_models),
        warnings=tuple(warnings),
        artifacts=(summary_artifact, comparison_artifact),
    )

    # 8. Return ToolResponse
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
        artifacts=(summary_artifact, comparison_artifact) + tuple(result_artifacts),
        metadata=run_meta,
        message=(
            f"Analysis complete: {len(results)} model(s) succeeded, "
            f"{len(failed_models)} failed. "
            f"Run ID: {run_id}."
        ),
    )
