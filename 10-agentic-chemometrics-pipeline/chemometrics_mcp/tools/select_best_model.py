"""MCP tool: select_best_model

Compares candidate models on performance, validation reliability,
interpretability, stability, complexity, and task suitability. Distinguishes
the best-measured model from the best-defensible model.
"""
from __future__ import annotations

import json
from pathlib import Path

from chemometrics_contracts import (
    ModelSelectionRecommendation,
    SelectBestModelRequest,
    ToolResponse,
)

from chemometrics_mcp.artifacts import artifact_ref, ensure_run_dir, make_run_id
from chemometrics_mcp.core import interpretation


def run(
    request: SelectBestModelRequest,
    *,
    runs_root: str | Path = "runs",
) -> ToolResponse[ModelSelectionRecommendation]:
    """Select the best defensible model from candidate results.

    Parameters
    ----------
    request:
        Validated :class:`SelectBestModelRequest` instance.
    runs_root:
        Root directory under which run artifact directories are created.

    Returns
    -------
    :class:`ToolResponse` with a :class:`ModelSelectionRecommendation` payload
    on success, or ``ok=False`` with an ``error`` message if no results provided.
    """
    # 1. Guard: no results
    if not request.results:
        return ToolResponse(
            tool_name="select_best_model",
            ok=False,
            error="No results to compare.",
            message="Provide at least one AnalysisResult in the results field.",
        )

    # 2. Core logic
    recommendation = interpretation.select_best_model(
        request.results,
        request.validation_summary,
        request.task_name,
    )

    # 3. Save artifact
    run_id = make_run_id(slug="select-best-model")
    artifact_dir = ensure_run_dir(run_id, runs_root)

    artifact_filename = "model_selection.json"
    artifact_path = artifact_dir / artifact_filename
    artifact_path.write_text(
        json.dumps(recommendation.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )

    ref = artifact_ref(
        run_id,
        artifact_filename,
        kind="model_selection",
        label="Model selection recommendation",
        mime_type="application/json",
        runs_root=runs_root,
    )

    return ToolResponse(
        tool_name="select_best_model",
        ok=True,
        payload=recommendation,
        warnings=recommendation.warnings,
        artifacts=(ref,),
        message=(
            f"Model selection complete. "
            f"Selected: {recommendation.selected_model!r}. "
            f"Candidates: {list(recommendation.candidate_models)}."
        ),
    )
