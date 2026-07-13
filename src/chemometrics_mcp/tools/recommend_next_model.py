"""MCP tool: recommend_next_model

Classifies model failures and recommends fallback models with rationale.
Requires human approval when the fallback changes the scientific plan.
Records failed model, failure reason, rationale, and comparability limitations.
"""
from __future__ import annotations

import json
from pathlib import Path

from chemometrics_contracts import (
    NextModelRecommendation,
    RecommendNextModelRequest,
    ToolResponse,
)

from chemometrics_mcp.artifacts import artifact_ref, ensure_run_dir, make_run_id
from chemometrics_mcp.core import fallback


def run(
    request: RecommendNextModelRequest,
    *,
    runs_root: str | Path = "runs",
) -> ToolResponse[NextModelRecommendation]:
    """Classify a model failure and recommend a fallback model.

    Parameters
    ----------
    request:
        Validated :class:`RecommendNextModelRequest` instance.
    runs_root:
        Root directory under which run artifact directories are created.

    Returns
    -------
    :class:`ToolResponse` with a :class:`NextModelRecommendation` payload.
    ``ok=True`` even when ``fallback_model`` is ``None`` — the tool succeeded
    at classifying the failure; the agent must handle the None case.
    """
    # Build recommendation via pure core logic
    recommendation = fallback.build_recommendation(request)

    # Save artifact
    run_id = make_run_id(slug="recommend-next-model")
    artifact_dir = ensure_run_dir(run_id, runs_root)

    artifact_filename = "fallback_recommendation.json"
    artifact_path = artifact_dir / artifact_filename
    artifact_path.write_text(
        json.dumps(recommendation.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )

    ref = artifact_ref(
        run_id,
        artifact_filename,
        kind="fallback_recommendation",
        label="Fallback model recommendation",
        mime_type="application/json",
        runs_root=runs_root,
    )

    return ToolResponse(
        tool_name="recommend_next_model",
        ok=True,
        payload=recommendation,
        warnings=recommendation.warnings,
        artifacts=(ref,),
        message=(
            f"Failure classified as {recommendation.failure_classification!r}. "
            f"Fallback recommendation: {recommendation.fallback_model!r}. "
            f"Human approval required before proceeding."
        ),
    )
