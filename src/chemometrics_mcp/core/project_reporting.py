"""Evidence-led reports for persisted runs, without scientific overclaiming."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from chemometrics_mcp.core.project_store import (
    ProjectStore,
    sha256_file,
    slugify_project_id,
)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _artifact_path(item: Mapping[str, Any], root: Path) -> Path:
    raw = item.get("path", item.get("uri"))
    if not raw:
        raise ValueError("evidence artifact requires path or uri")
    path = Path(str(raw))
    return path if path.is_absolute() else root / path


def validate_local_evidence(run: Mapping[str, Any], project_root: str | Path) -> tuple[dict[str, Any], ...]:
    """Validate identity, hashes, and run-local artifact boundaries before reporting."""
    required = ("project_id", "run_id", "manifest_hash", "plan_hash")
    missing = [key for key in required if not run.get(key)]
    if missing:
        raise ValueError(f"run is missing required identity fields: {', '.join(missing)}")
    for key in ("manifest_hash", "plan_hash"):
        value = run[key]
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"{key} must be a lowercase SHA-256 hash")
    root = Path(project_root).resolve()
    project_file = root / "project.json"
    if project_file.is_file():
        project = ProjectStore(root).read_json("project.json")
        if project.get("project_id") != run["project_id"]:
            raise ValueError("run project_id does not match the local project")
    run_dir = (root / "runs" / str(run["run_id"])).resolve()
    artifacts = run.get("artifacts", ())
    if not isinstance(artifacts, (list, tuple)):
        raise ValueError("artifacts must be a list")
    validated: list[dict[str, Any]] = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ValueError("artifact entries must be objects")
        if artifact.get("run_id") not in (None, run["run_id"]):
            raise ValueError("cross-run evidence reference rejected")
        for identity in ("project_id", "manifest_hash", "plan_hash"):
            if artifact.get(identity) not in (None, run[identity]):
                raise ValueError(f"cross-{identity} evidence reference rejected")
        path = _artifact_path(artifact, root)
        if not _inside(path, root) or not _inside(path, run_dir):
            raise ValueError("cross-directory evidence reference rejected")
        if not path.is_file():
            raise FileNotFoundError(f"evidence artifact not found: {path}")
        declared = artifact.get("sha256")
        if (
            not isinstance(declared, str)
            or len(declared) != 64
            or any(char not in "0123456789abcdef" for char in declared)
        ):
            raise ValueError("evidence artifact requires declared sha256")
        actual = _hash(path)
        if actual != declared:
            raise ValueError(f"evidence hash mismatch for {path.name}")
        if artifact.get("kind") == "analysis_evidence":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("analysis evidence is not valid JSON") from exc
            for identity in (
                "project_id",
                "run_id",
                "manifest_hash",
                "plan_hash",
            ):
                if payload.get(identity) != run[identity]:
                    raise ValueError(
                        f"analysis evidence {identity} does not match run"
                    )
        validated.append({"pointer": str(path.relative_to(root)), "sha256": actual, "kind": artifact.get("kind", "artifact")})
    # Inline numerical claims may only point at validated, same-run evidence.
    pointers = {item["pointer"] for item in validated}
    for result in run.get("results", ()):
        pointer = result.get("evidence_pointer")
        if pointer is not None and pointer not in pointers:
            raise ValueError("result points outside validated run evidence")
    for finding in run.get("findings", ()):
        if finding.get("run_id", run["run_id"]) != run["run_id"]:
            raise ValueError("cross-run numerical finding rejected")
        pointer = finding.get("evidence_pointer")
        if pointer is not None and pointer not in pointers:
            raise ValueError("finding points outside validated run evidence")
    return tuple(validated)


def _ledger(run: Mapping[str, Any], artifacts: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    pointer = artifacts[0]["pointer"] if artifacts else "run-record"
    rows = []
    for result in run.get("results", ()):
        metrics = result.get("metrics", {})
        for name, value in metrics.items():
            rows.append({"kind": "metric", "name": name, "value": value, "run_id": run["run_id"],
                         "manifest_hash": run["manifest_hash"], "plan_hash": run["plan_hash"],
                         "pipeline": result.get("pipeline_id"), "task": result.get("task_id"), "evidence_pointer": result.get("evidence_pointer", pointer)})
    for finding in run.get("findings", ()):
        rows.append({"kind": "finding", "name": finding.get("name", "finding"), "value": finding.get("value", finding.get("text")),
                     "run_id": run["run_id"], "manifest_hash": run["manifest_hash"], "plan_hash": run["plan_hash"],
                     "pipeline": finding.get("pipeline_id"), "task": finding.get("task_id"), "evidence_pointer": finding.get("evidence_pointer", pointer)})
    for name, value in run.get("counts", {}).items():
        rows.append(
            {
                "kind": "study_count",
                "name": name,
                "value": value,
                "run_id": run["run_id"],
                "manifest_hash": run["manifest_hash"],
                "plan_hash": run["plan_hash"],
                "pipeline": None,
                "task": run.get("task_kind"),
                "evidence_pointer": pointer,
            }
        )
    return rows


def _text(item: Any) -> str:
    if isinstance(item, Mapping):
        return str(item.get("message", item.get("text", dict(item))))
    return str(item)


def generate_evidence_report(run: Mapping[str, Any], project_root: str | Path) -> dict[str, Any]:
    """Generate a concise report after strict local-evidence validation."""
    artifacts = validate_local_evidence(run, project_root)
    ledger = _ledger(run, artifacts)
    eligibility = run.get("claim_eligibility", {"claim_level": "descriptive", "eligible": False})
    lod = run.get("detection_limit_eligibility", {})
    observed = list(run.get("observed_spectral_evidence", ()))
    model = list(run.get("model_evidence", ()))
    tentative = list(run.get("tentative_explanations", ()))
    unsupported = list(run.get("unsupported_claims", ()))
    if lod.get("estimable") is not True:
        unsupported.extend(["LOD/LOQ not reported: detection-limit design eligibility is not estimable."])
    limitations = list(run.get("blockers", ())) + list(run.get("limitations", ()))
    counts = run.get("counts", {})
    recommendations = list(run.get("next_experiments", ())) or ["Collect independent preparations with documented validation design."]
    lines = [f"# Scientific report: {run['run_id']}", "", "## Observed spectral evidence", *(f"- {_text(item)}" for item in observed or ["No observed spectral evidence supplied."]),
             "", "## Model evidence", *(f"- {_text(item)}" for item in model or ["No model evidence supplied."]), "", "## Tentative explanations", *(f"- {_text(item)}" for item in tentative or ["None supplied; no chemical identity or purity conclusion is made."]),
             "", "## Unsupported claims", *(f"- {_text(item)}" for item in unsupported or ["None listed."]), "", "## Blockers and limitations", *(f"- {_text(item)}" for item in limitations or ["None listed."]),
             "", "## Study counts", f"- scans: {counts.get('scan_count', 'not declared')}", f"- independent preparations: {counts.get('preparation_count', 'not declared')}",
             "", "## Claim eligibility", f"- level: {eligibility.get('claim_level', 'descriptive')}", f"- eligible: {eligibility.get('eligible', False)}", "", "## Next experiment", *(f"- {_text(item)}" for item in recommendations)]
    return {"project_id": run["project_id"], "run_id": run["run_id"], "manifest_hash": run["manifest_hash"], "plan_hash": run["plan_hash"],
            "markdown": "\n".join(lines), "evidence_ledger": ledger,
            "machine_summary": {"claim_eligibility": eligibility, "scan_count": counts.get("scan_count"), "preparation_count": counts.get("preparation_count"), "limitations": limitations, "recommendations": recommendations}}


def persist_evidence_report(store: ProjectStore, report: Mapping[str, Any], name: str = "evidence-report") -> str:
    """Atomically persist a report through the project's storage boundary."""
    return str(store.write_json(f"reports/{name}.json", dict(report)))


def generate_report_for_run(
    output_root: str | Path,
    run_id: str,
    *,
    include_notebook: bool = False,
) -> dict[str, Any]:
    """Validate one run and persist its report and scientist-facing artifacts."""
    if slugify_project_id(run_id) != run_id:
        raise ValueError("run_id must be a safe lowercase slug")
    store = ProjectStore(output_root)
    run = store.read_json(f"runs/{run_id}.json")
    if run.get("run_id") != run_id:
        raise ValueError("stored run id does not match requested run id")
    # Validate existing evidence before generating anything new. This prevents
    # a report call from laundering or overwriting tampered run artifacts.
    validate_local_evidence(run, store.output_root)
    if not any(
        item.get("kind") == "scientist_dashboard"
        for item in run.get("artifacts", ())
    ):
        from chemometrics_mcp.core.dashboard import render_run_dashboard

        try:
            rendered = render_run_dashboard(store, run)
        except FileNotFoundError:
            # Compatibility for imported legacy run records which predate the
            # folder-first project manifest. Their existing evidence can still
            # be reported, but there is no trusted sample inventory to render.
            rendered = ()
        if rendered:
            run = {
                **run,
                "artifacts": [*run.get("artifacts", ()), *rendered],
            }
            store.save_run(run, run_id)
    if include_notebook and not any(
        item.get("kind") == "reproducibility_notebook"
        for item in run.get("artifacts", ())
    ):
        from chemometrics_mcp.core.dashboard import (
            render_reproducibility_notebook,
        )

        notebook = render_reproducibility_notebook(store, run)
        run = {
            **run,
            "artifacts": [*run.get("artifacts", ()), notebook],
        }
        store.save_run(run, run_id)
    report = generate_evidence_report(run, store.output_root)
    markdown_artifact = next(
        (
            item
            for item in run.get("artifacts", ())
            if item.get("kind") == "scientist_report"
        ),
        None,
    )
    if markdown_artifact is None:
        markdown_path = store.write_bytes(
            f"runs/{run_id}/report.md",
            (report["markdown"].rstrip() + "\n").encode("utf-8"),
        )
        markdown_artifact = {
            "kind": "scientist_report",
            "path": markdown_path.relative_to(store.output_root).as_posix(),
            "sha256": sha256_file(markdown_path),
            "media_type": "text/markdown",
            "project_id": run["project_id"],
            "run_id": run_id,
            "manifest_hash": run["manifest_hash"],
            "plan_hash": run["plan_hash"],
        }
        run = {
            **run,
            "artifacts": [*run.get("artifacts", ()), markdown_artifact],
        }
        store.save_run(run, run_id)
        report = generate_evidence_report(run, store.output_root)
    relative = f"reports/{run_id}.json"
    store.write_json(relative, report)
    dashboard = next(
        (
            item["path"]
            for item in run.get("artifacts", ())
            if item.get("kind") == "scientist_dashboard"
        ),
        None,
    )
    notebook = next(
        (
            item["path"]
            for item in run.get("artifacts", ())
            if item.get("kind") == "reproducibility_notebook"
        ),
        None,
    )
    return {
        **report,
        "report_path": relative,
        "scientist_report_path": markdown_artifact["path"],
        "dashboard_path": dashboard,
        "notebook_path": notebook,
        "figure_paths": [
            item["path"]
            for item in run.get("artifacts", ())
            if item.get("kind") == "scientist_figure"
        ],
        "table_paths": [
            item["path"]
            for item in run.get("artifacts", ())
            if item.get("kind") == "scientist_table"
        ],
    }
