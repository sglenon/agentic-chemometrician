"""MCP tool: interpret_results

Status: DEFERRED — Phase 8 implementation target.

Summarizes feature or wavelength importance from tool-produced outputs,
compares evidence across models, and separates model evidence from chemical
conclusions.
"""
from __future__ import annotations

from chemometrics_contracts import (
    InterpretResultsRequest,
    InterpretationSummary,
    ToolResponse,
    ValidationWarning,
)


_DEFERRED_WARNING = ValidationWarning(
    code="tool_deferred",
    message=(
        "interpret_results is not yet implemented. "
        "This tool will be available in Phase 8."
    ),
    category="system",
    severity="info",
    affected_stage="interpretation",
)


def run(request: InterpretResultsRequest) -> ToolResponse[InterpretationSummary]:
    """Return a deferred response until Phase 8 is implemented."""
    return ToolResponse(
        tool_name="interpret_results",
        ok=False,
        error="Tool not yet implemented (Phase 8 target).",
        warnings=(_DEFERRED_WARNING,),
        message=(
            "interpret_results has not been implemented yet. "
            "See IMPLEMENTATION-PLAN.md Phase 8 for scope and acceptance criteria."
        ),
    )
