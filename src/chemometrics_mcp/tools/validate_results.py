"""MCP tool: validate_results

Runs scientific reliability checks on analysis results: replicate leakage,
group leakage, class imbalance, small-sample warnings, split instability,
suspicious metric warnings, and target leakage.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

from chemometrics_contracts import (
    RunMetadata,
    ToolResponse,
    ValidateResultsRequest,
    ValidationSummary,
)

from chemometrics_mcp.artifacts import artifact_ref, ensure_run_dir, make_run_id
from chemometrics_mcp.core import validation


def run(
    request: ValidateResultsRequest,
    *,
    runs_root: str | Path = "runs",
) -> ToolResponse[ValidationSummary]:
    """Execute the validate_results tool.

    Parameters
    ----------
    request:
        Validated :class:`ValidateResultsRequest` instance.
    runs_root:
        Root directory under which run artifact directories are created.

    Returns
    -------
    :class:`ToolResponse` with a :class:`ValidationSummary` payload on success,
    or ``ok=False`` with a message when there are no results to validate.
    """
    # 1. Collect results
    results = request.results
    if not results and request.analysis_run is not None:
        results = request.analysis_run.results

    if not results:
        return ToolResponse(
            tool_name="validate_results",
            ok=False,
            error="No results to validate.",
            message="No results to validate.",
        )

    # 2. Run all checks
    summary = validation.run_all_checks(results, request.dataset, request.dataset_inspection)

    # 3. Save validation_summary.json artifact
    run_id = make_run_id(slug="validate")
    artifact_dir = ensure_run_dir(run_id, runs_root)
    created_at = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    summary_filename = "validation_summary.json"
    summary_path = artifact_dir / summary_filename
    summary_path.write_text(
        json.dumps(summary.to_dict(), indent=2, default=str), encoding="utf-8"
    )

    summary_artifact = artifact_ref(
        run_id,
        summary_filename,
        kind="validation_summary",
        label="Validation summary",
        mime_type="application/json",
        runs_root=runs_root,
    )

    run_meta = RunMetadata(
        run_id=run_id,
        tool_name="validate_results",
        status="completed",
        created_at=created_at,
        parameters={"n_results": len(list(results))},
    )

    # 4. Return ToolResponse
    return ToolResponse(
        tool_name="validate_results",
        ok=True,
        payload=summary,
        warnings=summary.warnings,
        artifacts=(summary_artifact,),
        metadata=run_meta,
        message=(
            f"Validation complete: {'passed' if summary.passed else 'failed' if summary.passed is False else 'no results'}. "
            f"{len(summary.warnings)} warning(s) found."
        ),
    )
