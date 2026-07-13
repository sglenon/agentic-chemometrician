"""MCP server entry point for the agentic chemometrics pipeline.

Run with:
    python -m chemometrics_mcp.server

Or via the MCP stdio transport (for use with Claude Desktop, Codex, etc.):
    python -m chemometrics_mcp.server --transport stdio

The server registers all eight chemometrics tools. All tools are fully
implemented with real logic — none return deferred or stub responses.

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
    DatasetProfile,
    GenerateReportRequest,
    InspectDatasetRequest,
    InterpretResultsRequest,
    MethodMemory,
    ModelSelectionRecommendation,
    NextModelRecommendation,
    ProposeAnalysisPlanRequest,
    RecommendFromMemoryRequest,
    RecommendNextModelRequest,
    ReportSummary,
    RunAnalysisRequest,
    SaveMethodMemoryRequest,
    SearchMethodMemoryRequest,
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
    recommend_from_memory,
    recommend_next_model,
    run_analysis,
    save_method_memory,
    search_method_memory,
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
                "Returns a human-readable plan that requires user approval before running."
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
                "Only runs tasks that appear in the approved plan."
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
                "warnings and a ValidationSummary."
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
                "from the best-defensible model. Requires human approval when validation warnings exist."
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
                "Records failed model, failure reason, rationale, and comparability limitations."
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
                "conclusions. Flags unstable or weak interpretations."
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
                "next-step recommendations, and a human-review checklist."
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
        types.Tool(
            name="save_method_memory",
            description=(
                "Save a human-reviewed method recipe to persistent method memory. "
                "Stores preprocessing, model, validation strategy, metrics, and caveats "
                "for future recommendation. Rebuilds the method memory index."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "memory": {
                        "type": "object",
                        "description": "MethodMemory object to save.",
                    },
                },
                "required": ["memory"],
            },
        ),
        types.Tool(
            name="search_method_memory",
            description=(
                "Search the method memory index for previously reviewed analysis recipes. "
                "Filter by modality, task, model, minimum metric, and approval status."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "modality": {
                        "type": "string",
                        "description": "Filter by modality (e.g. 'NIR', 'FTIR').",
                    },
                    "task_name": {
                        "type": "string",
                        "description": "Filter by task name.",
                    },
                    "model_name": {
                        "type": "string",
                        "description": "Filter by model name.",
                    },
                    "min_metric": {
                        "type": "number",
                        "description": "Minimum key metric value.",
                    },
                    "approval_status": {
                        "type": "string",
                        "description": "Filter by approval status (default: 'approved').",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="recommend_from_memory",
            description=(
                "Recommend analysis methods from method memory based on a dataset profile. "
                "Matches by modality and ranks by dataset similarity and metric performance."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_profile": {
                        "type": "object",
                        "description": "DatasetProfile describing the current dataset.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of recommendations to return (default: 3).",
                    },
                },
                "required": ["dataset_profile"],
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
        from chemometrics_mcp.tools.run_analysis import _spectral_dataset_from_dict
        from chemometrics_contracts import SpectralDataset, AnalysisPlan

        dataset = _spectral_dataset_from_dict(arguments["dataset"])
        raw_plan = arguments["approved_plan"]
        plan = AnalysisPlan(
            task_name=raw_plan.get("task_name"),
            preprocessing_candidates=tuple(raw_plan.get("preprocessing_candidates", [])),
            validation_strategy=raw_plan.get("validation_strategy"),
            model_families=tuple(raw_plan.get("model_families", [])),
            human_readable_plan=raw_plan.get("human_readable_plan"),
        )
        request = RunAnalysisRequest(
            dataset=dataset,
            approved_plan=plan,
            run_id=arguments.get("run_id"),
        )
        return _tool_result(run_analysis.run(request, runs_root=_RUNS_ROOT))

    if name == "validate_results":
        from chemometrics_contracts import AnalysisResult, ValidationWarning, SpectralDataset, DatasetInspection
        from chemometrics_mcp.tools.run_analysis import _spectral_dataset_from_dict

        raw_results = arguments.get("results", [])
        results = tuple(
            AnalysisResult(
                task_name=r.get("task_name", ""),
                model_name=r.get("model_name", ""),
                preprocessing=tuple(r.get("preprocessing", [])),
                metrics=dict(r.get("metrics", {})),
                predictions=tuple(r.get("predictions", [])),
                selected_features=tuple(r.get("selected_features", [])),
                warnings=tuple(
                    ValidationWarning(code=w["code"], message=w["message"],
                        category=w.get("category", "validation"),
                        severity=w.get("severity", "warning"))
                    for w in r.get("warnings", [])
                ),
                validation_strategy=r.get("validation_strategy"),
            )
            for r in raw_results
        )
        dataset = None
        if arguments.get("dataset"):
            dataset = _spectral_dataset_from_dict(arguments["dataset"])

        raw_inspection = arguments.get("dataset_inspection")
        dataset_inspection = None
        if raw_inspection:
            dataset_inspection = DatasetInspection(
                sample_count=raw_inspection.get("sample_count"),
                feature_count=raw_inspection.get("feature_count"),
                axis_min=raw_inspection.get("axis_min"),
                axis_max=raw_inspection.get("axis_max"),
                modality=raw_inspection.get("modality"),
                candidate_label_columns=tuple(raw_inspection.get("candidate_label_columns", [])),
                candidate_group_columns=tuple(raw_inspection.get("candidate_group_columns", [])),
            )

        request = ValidateResultsRequest(
            results=results,
            dataset=dataset,
            dataset_inspection=dataset_inspection,
        )
        return _tool_result(validate_results.run(request, runs_root=_RUNS_ROOT))

    if name == "select_best_model":
        from chemometrics_contracts import AnalysisResult, ValidationWarning, ValidationSummary

        raw_results = arguments.get("results", [])
        results = tuple(
            AnalysisResult(
                task_name=r.get("task_name", ""),
                model_name=r.get("model_name", ""),
                metrics=dict(r.get("metrics", {})),
                selected_features=tuple(r.get("selected_features", [])),
                warnings=tuple(
                    ValidationWarning(
                        code=w["code"],
                        message=w["message"],
                        category=w.get("category", "validation"),
                        severity=w.get("severity", "warning"),
                    )
                    for w in r.get("warnings", [])
                ),
            )
            for r in raw_results
        )
        raw_vs = arguments.get("validation_summary")
        validation_summary = None
        if raw_vs:
            validation_summary = ValidationSummary(
                passed=raw_vs.get("passed"),
                checks=dict(raw_vs.get("checks", {})),
                warnings=tuple(
                    ValidationWarning(
                        code=w["code"],
                        message=w["message"],
                        category=w.get("category", "validation"),
                        severity=w.get("severity", "warning"),
                    )
                    for w in raw_vs.get("warnings", [])
                ),
            )
        request = SelectBestModelRequest(
            results=results,
            validation_summary=validation_summary,
            task_name=arguments.get("task_name"),
        )
        return _tool_result(select_best_model.run(request, runs_root=_RUNS_ROOT))

    if name == "recommend_next_model":
        request = RecommendNextModelRequest(
            failed_model=arguments["failed_model"],
            failure_reason=arguments["failure_reason"],
            candidate_models=tuple(arguments.get("candidate_models", [])),
        )
        return _tool_result(recommend_next_model.run(request, runs_root=_RUNS_ROOT))

    if name == "interpret_results":
        from chemometrics_contracts import AnalysisResult, ValidationWarning, ValidationSummary
        from chemometrics_mcp.tools.run_analysis import _spectral_dataset_from_dict

        raw_results = arguments.get("results", [])
        results = tuple(
            AnalysisResult(
                task_name=r.get("task_name", ""),
                model_name=r.get("model_name", ""),
                metrics=dict(r.get("metrics", {})),
                selected_features=tuple(r.get("selected_features", [])),
                warnings=tuple(
                    ValidationWarning(
                        code=w["code"],
                        message=w["message"],
                        category=w.get("category", "validation"),
                        severity=w.get("severity", "warning"),
                    )
                    for w in r.get("warnings", [])
                ),
            )
            for r in raw_results
        )
        dataset = None
        if arguments.get("dataset"):
            dataset = _spectral_dataset_from_dict(arguments["dataset"])
        raw_vs = arguments.get("validation_summary")
        validation_summary = None
        if raw_vs:
            validation_summary = ValidationSummary(
                passed=raw_vs.get("passed"),
                checks=dict(raw_vs.get("checks", {})),
                warnings=tuple(
                    ValidationWarning(
                        code=w["code"],
                        message=w["message"],
                        category=w.get("category", "validation"),
                        severity=w.get("severity", "warning"),
                    )
                    for w in raw_vs.get("warnings", [])
                ),
            )
        request = InterpretResultsRequest(
            results=results,
            dataset=dataset,
            validation_summary=validation_summary,
        )
        return _tool_result(interpret_results.run(request, runs_root=_RUNS_ROOT))

    if name == "generate_report":
        from chemometrics_contracts import (
            AnalysisResult,
            RunMetadata,
            InterpretationSummary,
        )

        def _analysis_run_from_dict(d: dict) -> AnalysisRun:
            """Reconstruct a nested AnalysisRun from a JSON dict."""

            def _warning_from_dict(w: dict) -> ValidationWarning:
                from chemometrics_contracts import ValidationWarning as VW
                return VW(
                    code=w["code"],
                    message=w["message"],
                    category=w.get("category", "validation"),
                    severity=w.get("severity", "warning"),
                    affected_stage=w.get("affected_stage"),
                )

            def _artifact_ref_from_dict(a: dict) -> ArtifactReference:
                return ArtifactReference(
                    kind=a["kind"],
                    uri=a["uri"],
                    label=a.get("label"),
                    mime_type=a.get("mime_type"),
                    description=a.get("description"),
                )

            def _run_metadata_from_dict(m: dict | None) -> RunMetadata | None:
                if m is None:
                    return None
                return RunMetadata(
                    run_id=m["run_id"],
                    tool_name=m["tool_name"],
                    dataset_id=m.get("dataset_id"),
                    status=m.get("status"),
                    created_at=m.get("created_at"),
                    parameters=dict(m.get("parameters", {})),
                )

            def _analysis_result_from_dict(r: dict) -> AnalysisResult:
                return AnalysisResult(
                    task_name=r["task_name"],
                    model_name=r["model_name"],
                    preprocessing=tuple(r.get("preprocessing", [])),
                    metrics=dict(r.get("metrics", {})),
                    predictions=tuple(r.get("predictions", [])),
                    selected_features=tuple(r.get("selected_features", [])),
                    figures=tuple(
                        _artifact_ref_from_dict(f) for f in r.get("figures", [])
                    ),
                    warnings=tuple(
                        _warning_from_dict(w) for w in r.get("warnings", [])
                    ),
                    interpretation=r.get("interpretation"),
                    run_metadata=_run_metadata_from_dict(r.get("run_metadata")),
                )

            return AnalysisRun(
                run_metadata=_run_metadata_from_dict(d.get("run_metadata")),
                results=tuple(
                    _analysis_result_from_dict(r) for r in d.get("results", [])
                ),
                failed_models=tuple(d.get("failed_models", [])),
                warnings=tuple(
                    _warning_from_dict(w) for w in d.get("warnings", [])
                ),
                artifacts=tuple(
                    _artifact_ref_from_dict(a) for a in d.get("artifacts", [])
                ),
            )

        raw_run = arguments["analysis_run"]
        analysis_run = _analysis_run_from_dict(raw_run)

        raw_vs = arguments.get("validation_summary")
        validation_summary = None
        if raw_vs:
            from chemometrics_contracts import ValidationWarning as VW
            validation_summary = ValidationSummary(
                passed=raw_vs.get("passed"),
                checks=dict(raw_vs.get("checks", {})),
                warnings=tuple(
                    VW(
                        code=w["code"],
                        message=w["message"],
                        category=w.get("category", "validation"),
                        severity=w.get("severity", "warning"),
                        affected_stage=w.get("affected_stage"),
                    )
                    for w in raw_vs.get("warnings", [])
                ),
            )

        raw_interp = arguments.get("interpretation")
        interpretation = None
        if raw_interp:
            interpretation = InterpretationSummary(
                summary=raw_interp.get("summary"),
                important_features=tuple(raw_interp.get("important_features", [])),
                model_comparisons=tuple(raw_interp.get("model_comparisons", [])),
                warnings=tuple(
                    ValidationWarning(
                        code=w["code"],
                        message=w["message"],
                        category=w.get("category", "validation"),
                        severity=w.get("severity", "warning"),
                        affected_stage=w.get("affected_stage"),
                    )
                    for w in raw_interp.get("warnings", [])
                ),
            )

        request = GenerateReportRequest(
            analysis_run=analysis_run,
            validation_summary=validation_summary,
            interpretation=interpretation,
        )
        return _tool_result(generate_report.run(request, runs_root=_RUNS_ROOT))

    if name == "save_method_memory":
        raw_mem = arguments["memory"]
        raw_dp = raw_mem["dataset_profile"]
        dp = DatasetProfile(
            modality=raw_dp["modality"],
            n_samples=raw_dp["n_samples"],
            n_features=raw_dp["n_features"],
            n_classes=raw_dp.get("n_classes"),
            axis_min=raw_dp.get("axis_min"),
            axis_max=raw_dp.get("axis_max"),
            label_column=raw_dp.get("label_column"),
        )
        memory = MethodMemory(
            memory_id=raw_mem["memory_id"],
            created_at=raw_mem["created_at"],
            modality=raw_mem["modality"],
            task_name=raw_mem["task_name"],
            dataset_profile=dp,
            preprocessing=raw_mem["preprocessing"],
            model_name=raw_mem["model_name"],
            validation_strategy=raw_mem["validation_strategy"],
            key_metrics=dict(raw_mem.get("key_metrics", {})),
            caveats=tuple(raw_mem.get("caveats", [])),
            reviewer_notes=raw_mem.get("reviewer_notes"),
            source_run_id=raw_mem.get("source_run_id", ""),
            approval_status=raw_mem.get("approval_status", "approved"),
        )
        request = SaveMethodMemoryRequest(memory=memory)
        return _tool_result(save_method_memory.run(request))

    if name == "search_method_memory":
        request = SearchMethodMemoryRequest(
            modality=arguments.get("modality"),
            task_name=arguments.get("task_name"),
            model_name=arguments.get("model_name"),
            min_metric=arguments.get("min_metric"),
            approval_status=arguments.get("approval_status", "approved"),
        )
        return _tool_result(search_method_memory.run(request))

    if name == "recommend_from_memory":
        raw_dp = arguments["dataset_profile"]
        dp = DatasetProfile(
            modality=raw_dp["modality"],
            n_samples=raw_dp["n_samples"],
            n_features=raw_dp["n_features"],
            n_classes=raw_dp.get("n_classes"),
            axis_min=raw_dp.get("axis_min"),
            axis_max=raw_dp.get("axis_max"),
            label_column=raw_dp.get("label_column"),
        )
        request = RecommendFromMemoryRequest(
            dataset_profile=dp,
            top_k=arguments.get("top_k", 3),
        )
        return _tool_result(recommend_from_memory.run(request))

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
