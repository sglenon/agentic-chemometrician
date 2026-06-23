"""MCP tool: interpret_results

Summarizes feature or wavelength importance from tool-produced outputs,
compares evidence across models, and separates model evidence from chemical
conclusions.
"""
from __future__ import annotations

import json
from pathlib import Path

from chemometrics_contracts import (
    InterpretResultsRequest,
    InterpretationSummary,
    ToolResponse,
)

from chemometrics_mcp.artifacts import artifact_ref, ensure_run_dir, make_run_id
from chemometrics_mcp.core import interpretation


def run(
    request: InterpretResultsRequest,
    *,
    runs_root: str | Path = "runs",
) -> ToolResponse[InterpretationSummary]:
    """Summarize feature/wavelength importance from tool-produced outputs.

    Parameters
    ----------
    request:
        Validated :class:`InterpretResultsRequest` instance.
    runs_root:
        Root directory under which run artifact directories are created.

    Returns
    -------
    :class:`ToolResponse` with an :class:`InterpretationSummary` payload
    on success, or ``ok=False`` with an ``error`` message if no results provided.
    """
    # 1. Guard: no results
    if not request.results:
        return ToolResponse(
            tool_name="interpret_results",
            ok=False,
            error="No results to interpret.",
            message="Provide at least one AnalysisResult in the results field.",
        )

    # 2. Core logic
    summary = interpretation.interpret_results(
        request.results,
        request.dataset,
        request.validation_summary,
    )

    # 3. Save artifact
    run_id = make_run_id(slug="interpret-results")
    artifact_dir = ensure_run_dir(run_id, runs_root)

    artifact_filename = "interpretation_summary.json"
    artifact_path = artifact_dir / artifact_filename
    artifact_path.write_text(
        json.dumps(summary.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )

    ref = artifact_ref(
        run_id,
        artifact_filename,
        kind="interpretation_summary",
        label="Interpretation summary",
        mime_type="application/json",
        runs_root=runs_root,
    )

    return ToolResponse(
        tool_name="interpret_results",
        ok=True,
        payload=summary,
        warnings=summary.warnings,
        artifacts=(ref,),
        message=(
            f"Interpretation complete. "
            f"{len(summary.important_features)} unique important feature(s) identified "
            f"across {len(request.results)} model(s)."
        ),
    )
