"""MCP tool: propose_analysis_plan

Converts a DatasetInspection into a bounded AnalysisPlan with recommended
tasks, preprocessing candidates, validation strategy, and model families.
The plan must be approved by a human before run_analysis executes it.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

from chemometrics_contracts import (
    AnalysisPlan,
    ProposeAnalysisPlanRequest,
    RunMetadata,
    ToolResponse,
)

from chemometrics_mcp.artifacts import artifact_ref, ensure_run_dir, make_run_id
from chemometrics_mcp.core.planning import build_plan


def run(
    request: ProposeAnalysisPlanRequest,
    *,
    runs_root: str | Path = "runs",
) -> ToolResponse[AnalysisPlan]:
    """Execute the propose_analysis_plan tool.

    Parameters
    ----------
    request:
        Validated :class:`ProposeAnalysisPlanRequest` instance.
    runs_root:
        Root directory under which run artifact directories are created.
        Tools may not write outside this directory.

    Returns
    -------
    :class:`ToolResponse` wrapping an :class:`AnalysisPlan` on success, or
    ``ok=False`` with an error message when label columns are ambiguous.
    """
    started_at = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    run_id = make_run_id(slug="plan")
    artifact_dir = ensure_run_dir(run_id, runs_root)

    plan = build_plan(request)

    # Ambiguous labels — cannot propose a plan
    if plan.task_name is None:
        return ToolResponse(
            tool_name="propose_analysis_plan",
            ok=False,
            payload=plan,
            warnings=plan.warnings,
            error=(
                "Cannot propose a plan: label column is ambiguous. "
                "Re-run inspect_dataset with an explicit label_column "
                "or pass task_hint."
            ),
            message=(
                "Cannot propose a plan: label column is ambiguous. "
                "Re-run inspect_dataset with an explicit label_column "
                "or pass task_hint."
            ),
        )

    # Save artifact
    plan_path = artifact_dir / "plan.json"
    plan_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")

    run_meta = RunMetadata(
        run_id=run_id,
        tool_name="propose_analysis_plan",
        status="completed",
        created_at=started_at,
        parameters={
            "task_hint": request.task_hint,
            "user_intent": request.user_intent,
            "allow_supervised_planning": request.allow_supervised_planning,
        },
    )

    plan_artifact = artifact_ref(
        run_id,
        plan_path.name,
        kind="plan_json",
        label="Analysis plan",
        mime_type="application/json",
        runs_root=runs_root,
    )

    return ToolResponse(
        tool_name="propose_analysis_plan",
        ok=True,
        payload=plan,
        warnings=plan.warnings,
        artifacts=(plan_artifact,),
        metadata=run_meta,
        message=(
            f"Proposed {plan.task_name!r} plan with "
            f"{len(plan.model_families)} model families and "
            f"{len(plan.preprocessing_candidates)} preprocessing candidates. "
            f"Artifact saved to {plan_artifact.uri}."
        ),
    )
