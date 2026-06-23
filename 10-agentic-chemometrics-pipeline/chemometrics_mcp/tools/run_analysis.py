"""MCP tool: run_analysis

Status: DEFERRED — Phase 6 implementation target.

Executes an approved AnalysisPlan against a SpectralDataset and saves
structured run artifacts (metrics, predictions, figures, preprocessing details).
"""
from __future__ import annotations

from chemometrics_contracts import (
    AnalysisRun,
    RunAnalysisRequest,
    ToolResponse,
    ValidationWarning,
)


_DEFERRED_WARNING = ValidationWarning(
    code="tool_deferred",
    message=(
        "run_analysis is not yet implemented. "
        "This tool will be available in Phase 6."
    ),
    category="system",
    severity="info",
    affected_stage="analysis",
)


def run(request: RunAnalysisRequest) -> ToolResponse[AnalysisRun]:
    """Return a deferred response until Phase 6 is implemented."""
    return ToolResponse(
        tool_name="run_analysis",
        ok=False,
        error="Tool not yet implemented (Phase 6 target).",
        warnings=(_DEFERRED_WARNING,),
        message=(
            "run_analysis has not been implemented yet. "
            "See IMPLEMENTATION-PLAN.md Phase 6 for scope and acceptance criteria."
        ),
    )
