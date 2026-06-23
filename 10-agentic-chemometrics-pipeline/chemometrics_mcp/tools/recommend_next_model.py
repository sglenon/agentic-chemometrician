"""MCP tool: recommend_next_model

Status: DEFERRED — Phase 9 implementation target.

Classifies model failures and recommends fallback models with rationale.
Requires human approval when the fallback changes the scientific plan.
"""
from __future__ import annotations

from chemometrics_contracts import (
    NextModelRecommendation,
    RecommendNextModelRequest,
    ToolResponse,
    ValidationWarning,
)


_DEFERRED_WARNING = ValidationWarning(
    code="tool_deferred",
    message=(
        "recommend_next_model is not yet implemented. "
        "This tool will be available in Phase 9."
    ),
    category="system",
    severity="info",
    affected_stage="fallback",
)


def run(request: RecommendNextModelRequest) -> ToolResponse[NextModelRecommendation]:
    """Return a deferred response until Phase 9 is implemented."""
    return ToolResponse(
        tool_name="recommend_next_model",
        ok=False,
        error="Tool not yet implemented (Phase 9 target).",
        warnings=(_DEFERRED_WARNING,),
        message=(
            "recommend_next_model has not been implemented yet. "
            "See IMPLEMENTATION-PLAN.md Phase 9 for scope and acceptance criteria."
        ),
    )
