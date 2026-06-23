"""MCP server entry point for the agentic chemometrics pipeline.

Run with:
    python -m chemometrics_mcp.server

Or via the MCP stdio transport (for use with Claude Desktop, Codex, etc.):
    python -m chemometrics_mcp.server --transport stdio

The server registers all eight chemometrics tools. Tools that are not yet
implemented return explicit DEFERRED responses — they never silently fake results.

Security: no tool exposes arbitrary Python or shell execution. Artifact writes
are bounded to the configured runs directory via ``chemometrics_mcp.artifacts``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server
from mcp.server.models import InitializationOptions

from chemometrics_contracts import (
    AnalysisPlan,
    AnalysisRun,
    DatasetInspection,
    GenerateReportRequest,
    InspectDatasetRequest,
    InterpretResultsRequest,
    ModelSelectionRecommendation,
    NextModelRecommendation,
    ProposeAnalysisPlanRequest,
    RecommendNextModelRequest,
    ReportSummary,
    RunAnalysisRequest,
    SelectBestModelRequest,
    ToolResponse,
    ValidateResultsRequest,
    ValidationSummary,
)
from chemometrics_mcp.tools import (
    generate_report,
    inspect_dataset,
    interpret_results,
    propose_analysis_plan,
    recommend_next_model,
    run_analysis,
    select_best_model,
    validate_results,
)

_RUNS_ROOT = Path("runs")

server = Server("chemometrics-mcp")


def _tool_result(response: ToolResponse[Any]) -> list[types.TextContent]:
    return [types.TextContent(type="text", text=json.dumps(response.to_dict(), indent=2, default=str))]


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="inspect_dataset",
            description=(
                "Load a spectral data file and inspect its shape, axis range, candidate label "
                "columns, candidate group/replicate columns, modality, and data quality. "
                "Returns a DatasetInspection payload and saves a JSON artifact. "
                "Required first step in the end-to-end workflow."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source_uri": {
                        "type": "string",
                        "description": "Path or URI to the spectral data file (.xlsx/.xls).",
                    },
                    "dataset_id": {
                        "type": "string",
                        "description": "Optional identifier for this dataset (used in artifact naming).",
                    },
                    "modality_override": {
                        "type": "string",
                        "description": "Override inferred modality (e.g. 'NIR', 'FTIR').",
                    },
                    "label_column": {
                        "type": "string",
                        "description": "Metadata column to use as the primary class label.",
                    },
                    "sample_id_column": {
                        "type": "string",
                        "description": "Metadata column to use as sample identifiers.",
                    },
                },
                "required": ["source_uri"],
            },
        ),
        types.Tool(
            name="propose_analysis_plan",
            description=(
                "Convert a DatasetInspection into a bounded analysis plan with recommended tasks, "
                "preprocessing candidates, validation strategy, and initial model families. "
                "Returns a human-readable plan that requires user approval before running. "
                "STATUS: deferred — Phase 5 implementation target."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_inspection": {
                        "type": "object",
                        "description": "DatasetInspection output from inspect_dataset.",
                    },
                    "user_intent": {
                        "type": "string",
                        "description": "Optional user description of the analysis goal.",
                    },
                    "task_hint": {
                        "type": "string",
                        "description": "Optional task type hint (e.g. 'classification', 'regression').",
                    },
                },
                "required": ["dataset_inspection"],
            },
        ),
        types.Tool(
            name="run_analysis",
            description=(
                "Execute an approved AnalysisPlan against a SpectralDataset. Saves structured "
                "metrics, predictions, figures, preprocessing details, and warnings as run artifacts. "
                "Only runs tasks that appear in the approved plan. "
                "STATUS: deferred — Phase 6 implementation target."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "object",
                        "description": "SpectralDataset to analyse.",
                    },
                    "approved_plan": {
                        "type": "object",
                        "description": "AnalysisPlan approved by the user.",
                    },
                    "run_id": {
                        "type": "string",
                        "description": "Optional run ID; generated automatically if omitted.",
                    },
                },
                "required": ["dataset", "approved_plan"],
            },
        ),
        types.Tool(
            name="validate_results",
            description=(
                "Run scientific reliability checks on analysis results: replicate leakage, "
                "group leakage, class imbalance, small-sample warnings, split instability, "
                "suspicious metrics, and regression target leakage. Returns severity-labelled "
                "warnings and a ValidationSummary. "
                "STATUS: deferred — Phase 7 implementation target."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "description": "List of AnalysisResult objects to validate.",
                    },
                    "dataset": {
                        "type": "object",
                        "description": "Optional SpectralDataset used during analysis.",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="select_best_model",
            description=(
                "Compare candidate models on performance, validation reliability, interpretability, "
                "stability, complexity, and task suitability. Distinguishes the best-measured model "
                "from the best-defensible model. Requires human approval when validation warnings exist. "
                "STATUS: deferred — Phase 8 implementation target."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "description": "List of AnalysisResult objects to compare.",
                    },
                    "task_name": {
                        "type": "string",
                        "description": "Optional task type to filter candidate models.",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="recommend_next_model",
            description=(
                "Classify a model failure and recommend a fallback model with rationale. "
                "Requires human approval when the fallback changes the scientific plan. "
                "Records failed model, failure reason, rationale, and comparability limitations. "
                "STATUS: deferred — Phase 9 implementation target."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "failed_model": {
                        "type": "string",
                        "description": "Name of the model that failed.",
                    },
                    "failure_reason": {
                        "type": "string",
                        "description": "Description of the failure.",
                    },
                    "candidate_models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of models to consider as fallbacks.",
                    },
                },
                "required": ["failed_model", "failure_reason"],
            },
        ),
        types.Tool(
            name="interpret_results",
            description=(
                "Summarise feature or wavelength importance from tool-produced outputs. "
                "Compares evidence across models and separates model evidence from chemical "
                "conclusions. Flags unstable or weak interpretations. "
                "STATUS: deferred — Phase 8 implementation target."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "description": "List of AnalysisResult objects to interpret.",
                    },
                    "dataset": {
                        "type": "object",
                        "description": "Optional SpectralDataset for axis/metadata context.",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="generate_report",
            description=(
                "Produce a human-readable and agent-readable final report from run artifacts. "
                "Includes metrics, figures, validation warnings, caveats, interpretations, "
                "next-step recommendations, and a human-review checklist. "
                "STATUS: deferred — Phase 10 implementation target."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "analysis_run": {
                        "type": "object",
                        "description": "AnalysisRun with results and artifact references.",
                    },
                    "validation_summary": {
                        "type": "object",
                        "description": "Optional ValidationSummary to include in report.",
                    },
                    "interpretation": {
                        "type": "object",
                        "description": "Optional InterpretationSummary to include in report.",
                    },
                },
                "required": ["analysis_run"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    if name == "inspect_dataset":
        request = InspectDatasetRequest(
            source_uri=arguments["source_uri"],
            dataset_id=arguments.get("dataset_id"),
            modality_override=arguments.get("modality_override"),
            label_column=arguments.get("label_column"),
            sample_id_column=arguments.get("sample_id_column"),
        )
        return _tool_result(inspect_dataset.run(request, runs_root=_RUNS_ROOT))

    if name == "propose_analysis_plan":
        from chemometrics_contracts import DatasetInspection as DI

        raw = arguments.get("dataset_inspection", {})
        di = DI(
            sample_count=raw.get("sample_count"),
            feature_count=raw.get("feature_count"),
            axis_min=raw.get("axis_min"),
            axis_max=raw.get("axis_max"),
            modality=raw.get("modality"),
            candidate_label_columns=tuple(raw.get("candidate_label_columns", [])),
            candidate_group_columns=tuple(raw.get("candidate_group_columns", [])),
        )
        request = ProposeAnalysisPlanRequest(
            dataset_inspection=di,
            user_intent=arguments.get("user_intent"),
            task_hint=arguments.get("task_hint"),
        )
        return _tool_result(propose_analysis_plan.run(request))

    if name == "run_analysis":
        return _tool_result(
            ToolResponse(
                tool_name="run_analysis",
                ok=False,
                error="Tool not yet implemented (Phase 6 target).",
                message="run_analysis is deferred. See IMPLEMENTATION-PLAN.md Phase 6.",
            )
        )

    if name == "validate_results":
        return _tool_result(
            ToolResponse(
                tool_name="validate_results",
                ok=False,
                error="Tool not yet implemented (Phase 7 target).",
                message="validate_results is deferred. See IMPLEMENTATION-PLAN.md Phase 7.",
            )
        )

    if name == "select_best_model":
        return _tool_result(
            ToolResponse(
                tool_name="select_best_model",
                ok=False,
                error="Tool not yet implemented (Phase 8 target).",
                message="select_best_model is deferred. See IMPLEMENTATION-PLAN.md Phase 8.",
            )
        )

    if name == "recommend_next_model":
        request = RecommendNextModelRequest(
            failed_model=arguments["failed_model"],
            failure_reason=arguments["failure_reason"],
            candidate_models=tuple(arguments.get("candidate_models", [])),
        )
        return _tool_result(recommend_next_model.run(request))

    if name == "interpret_results":
        return _tool_result(
            ToolResponse(
                tool_name="interpret_results",
                ok=False,
                error="Tool not yet implemented (Phase 8 target).",
                message="interpret_results is deferred. See IMPLEMENTATION-PLAN.md Phase 8.",
            )
        )

    if name == "generate_report":
        return _tool_result(
            ToolResponse(
                tool_name="generate_report",
                ok=False,
                error="Tool not yet implemented (Phase 10 target).",
                message="generate_report is deferred. See IMPLEMENTATION-PLAN.md Phase 10.",
            )
        )

    return [
        types.TextContent(
            type="text",
            text=json.dumps({"error": f"Unknown tool: {name!r}"}),
        )
    ]


async def main() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="chemometrics-mcp",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
