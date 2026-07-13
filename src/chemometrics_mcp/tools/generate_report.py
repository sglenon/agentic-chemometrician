"""MCP tool: generate_report

Produces a human-readable Markdown report and a machine-readable JSON summary
from a completed AnalysisRun. Saves both as artifacts under the run directory.
"""
from __future__ import annotations

import json
from pathlib import Path

from chemometrics_contracts import (
    ArtifactReference,
    GenerateReportRequest,
    ReportSummary,
    ToolResponse,
    ValidationWarning,
)

from chemometrics_mcp.artifacts import artifact_ref, ensure_run_dir, make_run_id
from chemometrics_mcp.core.reporting import build_agent_readable_summary, build_markdown_report


def run(
    request: GenerateReportRequest,
    *,
    runs_root: str | Path = "runs",
) -> ToolResponse[ReportSummary]:
    """Generate a Markdown report and JSON summary from an AnalysisRun.

    Parameters
    ----------
    request:
        Validated :class:`GenerateReportRequest` instance.
    runs_root:
        Root directory under which run artifact directories are created.

    Returns
    -------
    :class:`ToolResponse` with a :class:`ReportSummary` payload on success,
    or ``ok=False`` with an error message on failure.
    """
    analysis_run = request.analysis_run
    validation_summary = request.validation_summary
    interpretation = request.interpretation

    # Determine run_id: reuse from analysis_run or generate a new one
    if analysis_run.run_metadata is not None:
        run_id = analysis_run.run_metadata.run_id
    else:
        run_id = make_run_id(slug="report")

    try:
        artifact_dir = ensure_run_dir(run_id, runs_root)

        # Build report content
        markdown_text = build_markdown_report(
            analysis_run,
            validation_summary=validation_summary,
            interpretation=interpretation,
        )
        agent_summary_dict = build_agent_readable_summary(
            analysis_run,
            validation_summary=validation_summary,
        )

        # Save report.md
        report_md_path = artifact_dir / "report.md"
        report_md_path.write_text(markdown_text, encoding="utf-8")

        # Save report_summary.json
        report_json_path = artifact_dir / "report_summary.json"
        report_json_path.write_text(
            json.dumps(agent_summary_dict, indent=2, default=str),
            encoding="utf-8",
        )

    except Exception as exc:  # noqa: BLE001
        return ToolResponse(
            tool_name="generate_report",
            ok=False,
            error=str(exc),
            message="Failed to write report artifacts.",
        )

    # Build artifact references
    report_md_ref = artifact_ref(
        run_id,
        "report.md",
        kind="report_markdown",
        label="Chemometrics Analysis Report (Markdown)",
        mime_type="text/markdown",
        runs_root=runs_root,
    )
    report_json_ref = artifact_ref(
        run_id,
        "report_summary.json",
        kind="report_json",
        label="Agent-readable report summary (JSON)",
        mime_type="application/json",
        runs_root=runs_root,
    )

    # Combine warnings from analysis_run and validation_summary
    combined_warnings: list[ValidationWarning] = list(analysis_run.warnings)
    if validation_summary is not None:
        combined_warnings.extend(validation_summary.warnings)

    # Truncate markdown for human_readable_summary
    human_summary = markdown_text[:500]

    report_summary = ReportSummary(
        report_title="Chemometrics Analysis Report",
        human_readable_summary=human_summary,
        agent_readable_summary=json.dumps(agent_summary_dict, default=str),
        primary_report=report_md_ref,
        artifacts=(report_md_ref, report_json_ref),
        warnings=tuple(combined_warnings),
    )

    return ToolResponse(
        tool_name="generate_report",
        ok=True,
        payload=report_summary,
        artifacts=(report_md_ref, report_json_ref),
        message=(
            f"Report generated for run {run_id}. "
            f"Markdown saved to {report_md_ref.uri}."
        ),
    )
