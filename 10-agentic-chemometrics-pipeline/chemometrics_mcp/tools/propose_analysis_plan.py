"""MCP tool: propose_analysis_plan

Status: DEFERRED — Phase 5 implementation target.

Converts a DatasetInspection into a bounded AnalysisPlan with recommended
tasks, preprocessing candidates, validation strategy, and model families.
"""
from __future__ import annotations

from chemometrics_contracts import (
    AnalysisPlan,
    ProposeAnalysisPlanRequest,
    ToolResponse,
    ValidationWarning,
)


_DEFERRED_WARNING = ValidationWarning(
    code="tool_deferred",
    message=(
        "propose_analysis_plan is not yet implemented. "
        "This tool will be available in Phase 5."
    ),
    category="system",
    severity="info",
    affected_stage="planning",
)


def run(request: ProposeAnalysisPlanRequest) -> ToolResponse[AnalysisPlan]:
    """Return a deferred response until Phase 5 is implemented."""
    return ToolResponse(
        tool_name="propose_analysis_plan",
        ok=False,
        error="Tool not yet implemented (Phase 5 target).",
        warnings=(_DEFERRED_WARNING,),
        message=(
            "propose_analysis_plan has not been implemented yet. "
            "See IMPLEMENTATION-PLAN.md Phase 5 for scope and acceptance criteria."
        ),
    )
