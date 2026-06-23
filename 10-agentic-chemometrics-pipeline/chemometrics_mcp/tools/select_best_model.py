"""MCP tool: select_best_model

Status: DEFERRED — Phase 8 implementation target.

Compares candidate models on performance, validation reliability,
interpretability, stability, complexity, and task suitability. Distinguishes
the best-measured model from the best-defensible model.
"""
from __future__ import annotations

from chemometrics_contracts import (
    ModelSelectionRecommendation,
    SelectBestModelRequest,
    ToolResponse,
    ValidationWarning,
)


_DEFERRED_WARNING = ValidationWarning(
    code="tool_deferred",
    message=(
        "select_best_model is not yet implemented. "
        "This tool will be available in Phase 8."
    ),
    category="system",
    severity="info",
    affected_stage="model_selection",
)


def run(request: SelectBestModelRequest) -> ToolResponse[ModelSelectionRecommendation]:
    """Return a deferred response until Phase 8 is implemented."""
    return ToolResponse(
        tool_name="select_best_model",
        ok=False,
        error="Tool not yet implemented (Phase 8 target).",
        warnings=(_DEFERRED_WARNING,),
        message=(
            "select_best_model has not been implemented yet. "
            "See IMPLEMENTATION-PLAN.md Phase 8 for scope and acceptance criteria."
        ),
    )
