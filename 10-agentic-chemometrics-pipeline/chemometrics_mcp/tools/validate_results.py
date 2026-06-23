"""MCP tool: validate_results

Status: DEFERRED — Phase 7 implementation target.

Runs scientific reliability checks on analysis results: replicate leakage,
group leakage, class imbalance, small-sample warnings, split instability,
suspicious metric warnings, and target leakage.
"""
from __future__ import annotations

from chemometrics_contracts import (
    ToolResponse,
    ValidateResultsRequest,
    ValidationSummary,
    ValidationWarning,
)


_DEFERRED_WARNING = ValidationWarning(
    code="tool_deferred",
    message=(
        "validate_results is not yet implemented. "
        "This tool will be available in Phase 7."
    ),
    category="system",
    severity="info",
    affected_stage="validation",
)


def run(request: ValidateResultsRequest) -> ToolResponse[ValidationSummary]:
    """Return a deferred response until Phase 7 is implemented."""
    return ToolResponse(
        tool_name="validate_results",
        ok=False,
        error="Tool not yet implemented (Phase 7 target).",
        warnings=(_DEFERRED_WARNING,),
        message=(
            "validate_results has not been implemented yet. "
            "See IMPLEMENTATION-PLAN.md Phase 7 for scope and acceptance criteria."
        ),
    )
