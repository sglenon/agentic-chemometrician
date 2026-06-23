"""MCP tool: generate_report

Status: DEFERRED — Phase 10 implementation target.

Produces human-readable and agent-readable final reports from run artifacts.
Includes metrics, figures, validation warnings, caveats, interpretations,
next-step recommendations, and a human-review checklist.
"""
from __future__ import annotations

from chemometrics_contracts import (
    GenerateReportRequest,
    ReportSummary,
    ToolResponse,
    ValidationWarning,
)


_DEFERRED_WARNING = ValidationWarning(
    code="tool_deferred",
    message=(
        "generate_report is not yet implemented. "
        "This tool will be available in Phase 10."
    ),
    category="system",
    severity="info",
    affected_stage="reporting",
)


def run(request: GenerateReportRequest) -> ToolResponse[ReportSummary]:
    """Return a deferred response until Phase 10 is implemented."""
    return ToolResponse(
        tool_name="generate_report",
        ok=False,
        error="Tool not yet implemented (Phase 10 target).",
        warnings=(_DEFERRED_WARNING,),
        message=(
            "generate_report has not been implemented yet. "
            "See IMPLEMENTATION-PLAN.md Phase 10 for scope and acceptance criteria."
        ),
    )
