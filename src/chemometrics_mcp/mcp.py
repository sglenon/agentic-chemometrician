"""Strict MCP-facing request validation for the project workflow."""
from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from chemometrics_mcp.tools import project_workflow
from chemometrics_mcp.tools import preprocess_spectra as _preprocess_spectra_module


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CreateProjectRequest(_Request):
    source_root: str = Field(min_length=1)
    output_root: str | None = None
    project_id: str | None = Field(default=None, min_length=1)


class GetProjectRequest(_Request):
    output_root: str = Field(min_length=1)


class UpdateProjectManifestRequest(_Request):
    output_root: str = Field(min_length=1)
    updates: dict[str, Any]


class PlanProjectAnalysisRequest(_Request):
    output_root: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    task_kind: str | None = None
    task_kinds: list[str] | None = Field(
        default=None,
        description=(
            "Explicit list of task kinds for a composite plan (e.g. "
            "['unsupervised_exploration', 'mixture_quantification']). "
            "When provided, auto-routing via task_kind/objective keywords is "
            "bypassed. At most one split-supervised kind (classification or "
            "regression) is allowed per composite plan."
        ),
    )
    requested_claim_level: str = "exploratory"
    intended_use: str | None = None
    target: str | None = Field(default=None, min_length=1)
    analysis_options: dict[str, Any] = Field(default_factory=dict)
    compute_budget: dict[str, Any] | int | None = None


class ApproveProjectPlanRequest(_Request):
    output_root: str = Field(min_length=1)
    plan: dict[str, Any] | str
    approved_by: str = Field(min_length=1)
    notes: str | None = None


class ListCapabilitiesRequest(_Request):
    pass


class RunProjectAnalysisRequest(_Request):
    output_root: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    approval_id: str | None = Field(default=None, min_length=1)
    run_id: str | None = Field(default=None, min_length=1)


class GetProjectRunRequest(_Request):
    output_root: str = Field(min_length=1)
    run_id: str = Field(min_length=1)


class GenerateProjectReportRequest(_Request):
    output_root: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    include_notebook: bool = False


class PreprocessSpectraRequest(_Request):
    model_config = ConfigDict(extra="forbid", strict=False)  # steps list contains dicts with mixed types
    source_path: str = Field(min_length=1)
    steps: list[dict[str, Any]] = Field(
        min_length=1,
        description=(
            "Ordered list of preprocessing steps.  Each item must have a 'name' key "
            "(e.g. 'snv', 'sg_2nd_deriv', 'region_select') plus optional per-step "
            "parameters.  For 'region_select' use 'min' and 'max' for axis-unit bounds."
        ),
    )


_TOOLS: dict[str, tuple[str, type[_Request], Any]] = {
    "create_project": ("Create a folder-first project and a conservative draft manifest.", CreateProjectRequest, project_workflow.create_project),
    "get_project": ("Get a compact project summary without raw measurement arrays.", GetProjectRequest, project_workflow.get_project),
    "update_project_manifest": ("Apply explicit manifest field updates and persist a new version.", UpdateProjectManifestRequest, project_workflow.update_project_manifest),
    "plan_project_analysis": ("Build and persist a bounded analysis plan for a project.", PlanProjectAnalysisRequest, project_workflow.plan_project_analysis),
    "approve_project_plan": ("Approve an immutable plan already stored in the project.", ApproveProjectPlanRequest, project_workflow.approve_project_plan),
    "run_project_analysis": ("Execute an approved persisted project plan and return compact run status.", RunProjectAnalysisRequest, project_workflow.run_project_analysis),
    "get_project_run": ("Get compact persisted project run status without measurement arrays.", GetProjectRunRequest, project_workflow.get_project_run),
    "generate_project_report": ("Validate one run and return its scientist report, offline dashboard, figures, tables, and optional reproducibility notebook.", GenerateProjectReportRequest, project_workflow.generate_project_report),
    "list_chemometrics_capabilities": ("List available conservative chemometrics task capabilities.", ListCapabilitiesRequest, project_workflow.list_capabilities),
    "preprocess_spectra": (
        "Load spectra from a file or directory and apply an ordered list of preprocessing steps "
        "(SNV, MSC, Savitzky-Golay derivatives, baseline correction, area normalization, region select). "
        "Returns before/after spectral arrays and axis for overlay plots.  "
        "Operates outside the project pipeline — no project setup required.",
        PreprocessSpectraRequest,
        _preprocess_spectra_module.preprocess_spectra,
    ),
}


def tool_definitions() -> tuple[dict[str, Any], ...]:
    """Return generated schemas, keeping the MCP surface in sync with models."""
    return tuple({"name": name, "description": description, "inputSchema": model.model_json_schema()} for name, (description, model, _) in _TOOLS.items())


def is_tool(name: str) -> bool:
    return name in _TOOLS


def dispatch(name: str, arguments: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and execute a tool in the standard JSON-safe envelope."""
    if name not in _TOOLS:
        return {"tool_name": name, "ok": False, "payload": None, "error": "Unknown tool"}
    _, model, handler = _TOOLS[name]
    try:
        request = model.model_validate(arguments or {})
        payload = handler(**request.model_dump())
        return {"tool_name": name, "ok": True, "payload": payload, "error": None}
    except (ValidationError, ValueError, FileNotFoundError, KeyError) as exc:
        return {"tool_name": name, "ok": False, "payload": None, "error": str(exc)}
