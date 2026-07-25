"""Compact wrappers around the scientist-facing project workflow services.

This module intentionally contains no MCP registration: adapters can translate
its ordinary ``ValueError``/``FileNotFoundError`` exceptions into MCP errors.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping

from chemometrics_mcp.core.project_store import ProjectStore, slugify_project_id


def _load_service() -> type[Any]:
    module = importlib.import_module("chemometrics_mcp.core.project_service")
    return getattr(module, "ProjectService")


def _load_planning() -> Any:
    return importlib.import_module("chemometrics_mcp.core.project_planning")


def _service(output_root: str) -> Any:
    service_type = _load_service()
    # The public opening API also validates the persisted manifest.  Keep the
    # wrapper at that boundary instead of constructing a service around a raw
    # path (which would bypass the project's store contract).
    return service_type.open(output_root)


def _json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    return value


def _compact(value: Any) -> Any:
    """Ensure tool replies never expose potentially large raw measurements."""
    if isinstance(value, Mapping):
        return {key: _compact(item) for key, item in value.items() if key not in {"axis", "signal", "x", "y", "signals"}}
    if isinstance(value, list):
        return [_compact(item) for item in value]
    return value


def _issues(value: Any) -> list[dict[str, Any]]:
    return _compact(_json(value)) if value else []


def create_project(source_root: str, output_root: str | None = None, project_id: str | None = None,
                   infer_roles_from_filenames: bool = False) -> dict[str, Any]:
    service_type = _load_service()
    created = service_type.create(source_root, output_root=output_root, project_id=project_id,
                                  infer_roles_from_filenames=infer_roles_from_filenames)
    # ProjectService.create returns the opened service; support a direct
    # manifest return too for small compatible implementations.
    service = created if hasattr(created, "get_manifest") else None
    manifest = service.get_manifest() if service is not None else created
    return _compact({
        "project_id": manifest.project_id, "output_root": str(getattr(getattr(service, "store", None), "output_root", output_root or Path(source_root) / "chemometrics-output")),
        "manifest_hash": manifest.manifest_hash, "assets": len(manifest.assets), "samples": len(manifest.samples),
        "measurements": len(manifest.measurements), "issues": _issues(manifest.unresolved_issues),
    })


def get_project(output_root: str) -> dict[str, Any]:
    service = _service(output_root)
    summary = service.get_summary()
    return _compact(_json(summary))


def update_project_manifest(output_root: str, updates: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _service(output_root).update_manifest(updates)
    return _compact({"project_id": manifest.project_id, "version": manifest.version, "manifest_hash": manifest.manifest_hash,
                     "assets": len(manifest.assets), "samples": len(manifest.samples), "measurements": len(manifest.measurements),
                     "issues": _issues(manifest.unresolved_issues)})


def _save_plan(output_root: str, plan: Any) -> None:
    payload = _json(plan)
    ProjectStore(output_root).save_plan(payload, name=payload["plan_id"], version=1)


def plan_project_analysis(output_root: str, objective: str, task_kind: str | None = None,
                          requested_claim_level: str = "exploratory", intended_use: str | None = None,
                          compute_budget: Mapping[str, Any] | None = None, target: str | None = None,
                          analysis_options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    manifest = _service(output_root).get_manifest()
    planner = _load_planning()
    from chemometrics_contracts.project import ClaimLevel, ScientificIntent
    try:
        claim_level = ClaimLevel(requested_claim_level)
    except ValueError as exc:
        raise ValueError(f"Unknown requested claim level: {requested_claim_level!r}") from exc
    intent = ScientificIntent(objective=objective, task_kind=task_kind, target=target, claim_level=claim_level, intended_use=intended_use, metadata=dict(analysis_options or {}))
    budget = compute_budget
    if isinstance(compute_budget, Mapping):
        budget = compute_budget.get("max_pipelines", compute_budget.get("pipeline_limit"))
    plan = planner.build_analysis_plan(manifest, intent, task_kind=task_kind, compute_budget=budget)
    _save_plan(output_root, plan)
    payload = _json(plan)
    return _compact({
        "plan_id": payload["plan_id"],
        "plan_hash": payload.get("plan_hash"),
        "project_id": payload.get("project_id"),
        "manifest_hash": payload.get("manifest_hash"),
        "approval_required": payload.get("approval_required", True),
        "issues": payload.get("issues", []),
        "tasks": payload.get("tasks", []),
        "pipelines": [
            {
                "pipeline_id": item.get("pipeline_id"),
                "model_family": item.get("model_family"),
                "description": item.get("description"),
            }
            for item in payload.get("pipelines", [])
        ],
        "split_manifest": payload.get("split_manifest"),
    })


def _stored_plan(store: ProjectStore, plan_id: str) -> dict[str, Any]:
    candidates = sorted((store.output_root / "plans").glob("*.json"))
    for path in candidates:
        payload = store.read_json(path.relative_to(store.output_root))
        if payload.get("plan_id") == plan_id:
            return payload
    raise FileNotFoundError(f"No stored plan with id {plan_id!r}")


def approve_project_plan(output_root: str, plan: Mapping[str, Any] | str, approved_by: str, notes: str | None = None) -> dict[str, Any]:
    if not approved_by:
        raise ValueError("approved_by is required")
    store = ProjectStore(output_root)
    if isinstance(plan, str):
        payload = _stored_plan(store, plan)
    else:
        supplied = dict(plan)
        if not supplied.get("plan_id"):
            raise ValueError("plan payload requires plan_id")
        payload = _stored_plan(store, supplied["plan_id"])
        if supplied.get("plan_hash") != payload.get("plan_hash"):
            raise ValueError("supplied plan does not match the stored plan hash")
    if not payload.get("plan_id"):
        raise ValueError("plan payload requires plan_id")
    planner = _load_planning()
    plan_model = payload if hasattr(payload, "plan_id") else planner.PlannedAnalysisPlan.model_validate(payload)
    approval = planner.approve_plan(plan_model, approved_by=approved_by, notes=notes)
    approval_payload = _json(approval)
    approval_name = slugify_project_id(
        approval_payload.get("approval_id") or payload["plan_id"]
    )
    store.write_json(f"approvals/{approval_name}.json", approval_payload)
    verified = planner.verify_plan_approval(plan_model, approval)
    return _compact({
        "plan_id": payload["plan_id"],
        "plan_hash": payload.get("plan_hash"),
        "approved": bool(approval_payload.get("approved", False)),
        "approval_id": approval_payload.get("approval_id"),
        "verified": bool(verified),
        "notes": notes,
    })


def list_capabilities() -> dict[str, Any]:
    return _compact(_json(_load_planning().capability_catalog()))


def _safe_run_id(run_id: str) -> str:
    if not run_id or slugify_project_id(run_id) != run_id:
        raise ValueError("run_id must be a safe lowercase slug")
    return run_id


def run_project_analysis(output_root: str, plan_id: str, approval_id: str | None = None,
                         run_id: str | None = None) -> dict[str, Any]:
    if run_id is not None:
        _safe_run_id(run_id)
    from chemometrics_mcp.core.run_service import run_project_analysis as execute
    return _compact(_json(execute(output_root, plan_id, approval_id=approval_id, run_id=run_id)))


def get_project_run(output_root: str, run_id: str) -> dict[str, Any]:
    run_id = _safe_run_id(run_id)
    payload = ProjectStore(output_root).read_json(f"runs/{run_id}.json")
    if payload.get("run_id") != run_id:
        raise ValueError("stored run id does not match requested run id")
    return _compact({key: value for key, value in payload.items() if key not in {"axis", "signal", "evidence", "data_hashes"}})


def generate_project_report(
    output_root: str, run_id: str, include_notebook: bool = False
) -> dict[str, Any]:
    run_id = _safe_run_id(run_id)
    store = ProjectStore(output_root)
    run = store.read_json(f"runs/{run_id}.json")
    try:
        module = importlib.import_module("chemometrics_mcp.core.project_reporting")
        generator = getattr(module, "generate_report_for_run", None)
        report = (
            generator(
                output_root,
                run_id,
                include_notebook=include_notebook,
            )
            if generator
            else module.generate_evidence_report(run, output_root)
        )
    except (ModuleNotFoundError, AttributeError):
        report = {"run_id": run_id, "status": run.get("status"), "issues": run.get("issues", []), "summary": "Evidence report unavailable."}
    path = f"reports/{run_id}.json"
    store.write_json(path, _json(report))
    compact = _compact(_json(report))
    return {
        "run_id": run_id,
        "status": run.get("status"),
        "report": path,
        "scientist_report": compact.get("scientist_report_path"),
        "dashboard": compact.get("dashboard_path"),
        "notebook": compact.get("notebook_path"),
        "figures": compact.get("figure_paths", []),
        "tables": compact.get("table_paths", []),
        "issues": compact.get("issues", []),
        "summary": compact.get(
            "summary", compact.get("markdown", "Report persisted.")
        ),
    }
