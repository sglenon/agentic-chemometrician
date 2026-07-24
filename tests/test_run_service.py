from pathlib import Path

from chemometrics_contracts.project import ProjectAnalysisPlan, AnalysisTaskSpec, PlanApproval
from chemometrics_mcp.core.project_service import ProjectService
from chemometrics_mcp.core.project_store import data_hash
from chemometrics_mcp.core.run_service import run_project_analysis
from chemometrics_mcp.core.project_reporting import generate_report_for_run
from chemometrics_mcp.tools import project_workflow


def test_run_requires_matching_approval_and_persists_terminal_record(tmp_path: Path) -> None:
    source = tmp_path / "source"; source.mkdir()
    (source / "x.csv").write_text("x,y\n1,2\n2,3\n", encoding="utf-8")
    project = ProjectService.create(source)
    manifest = project.get_manifest()
    draft = ProjectAnalysisPlan(plan_id="plan-1", manifest_hash=manifest.manifest_hash,
                           task=AnalysisTaskSpec(task_id="t", task_type="spectral_comparison"), approval_required=True)
    plan = draft.model_copy(update={"plan_hash": data_hash(draft.model_dump(mode="json", exclude={"plan_hash"}))})
    project.store.write_json("plans/plan.json", plan.model_dump(mode="json"))
    blocked = run_project_analysis(project.store.output_root, "plan-1", run_id="blocked")
    assert blocked["status"] == "blocked"
    missing = run_project_analysis(
        project.store.output_root,
        "plan-1",
        approval_id="missing",
        run_id="missing-approval",
    )
    assert missing["status"] == "blocked"
    assert missing["issues"][-1]["code"] == "approval_not_found"
    approval = PlanApproval(approval_id="ok", plan_id="plan-1", plan_hash=plan.plan_hash, approved=True)
    project.store.write_json("approvals/ok.json", approval.model_dump(mode="json"))
    result = run_project_analysis(project.store.output_root, "plan-1", "ok", "done")
    assert result["status"] in {"succeeded", "blocked", "failed"}
    assert project.store.read_json("runs/done.json")["status"] == result["status"]


def test_folder_project_runs_and_reports_with_local_hashed_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "spectra"
    source.mkdir()
    (source / "product.csv").write_text(
        "1000,0.1\n1001,0.3\n1002,0.2\n", encoding="utf-8"
    )
    (source / "precursor.csv").write_text(
        "1000,0.2\n1001,0.1\n1002,0.2\n", encoding="utf-8"
    )
    created = project_workflow.create_project(
        str(source), project_id="complete"
    )
    output = created["output_root"]
    service = ProjectService.open(output)
    manifest = service.get_manifest()
    project_workflow.update_project_manifest(
        output,
        {
            "measurements": {
                row.measurement_id: {
                    "modality": "ftir",
                    "axis_kind": "wavenumber",
                    "axis_unit": "cm^-1",
                    "signal_kind": "absorbance",
                    "signal_unit": "absorbance",
                    "role": (
                        "product"
                        if "product" in str(row.metadata["measurement_name"])
                        else "precursor"
                    ),
                }
                for row in manifest.measurements
            },
            "samples": {
                sample.sample_id: {
                    "preparation_id": f"prep-{index}",
                    "role": (
                        "product"
                        if "product"
                        in str(sample.metadata["measurement_name"])
                        else "precursor"
                    ),
                }
                for index, sample in enumerate(manifest.samples)
            },
        },
    )
    plan = project_workflow.plan_project_analysis(
        output,
        "Compare product and precursor FTIR spectra",
        task_kind="spectral_comparison",
    )
    approval = project_workflow.approve_project_plan(
        output, plan["plan_id"], approved_by="scientist"
    )
    run = run_project_analysis(
        output,
        plan["plan_id"],
        approval["approval_id"],
        "complete-run",
    )
    assert run["status"] == "succeeded"
    record = service.store.read_json("runs/complete-run.json")
    artifact = record["artifacts"][0]
    assert artifact["path"] == "runs/complete-run/evidence.json"
    assert record["results"][0]["metrics"]
    report = generate_report_for_run(output, "complete-run")
    assert report["evidence_ledger"]
    assert any(
        row["kind"] == "metric" for row in report["evidence_ledger"]
    )
    assert "Similarity or peak matching" in report["markdown"]


def test_supervised_run_uses_approved_materialized_group_folds(
    tmp_path: Path,
) -> None:
    source = tmp_path / "regression"
    source.mkdir()
    for group in range(6):
        for replicate in range(2):
            offset = group + replicate * 0.01
            (source / f"g{group}-r{replicate}.csv").write_text(
                f"1000,{offset + 1}\n"
                f"1001,{offset * 2 + 2}\n"
                f"1002,{offset * 3 + 4}\n",
                encoding="utf-8",
            )
    created = project_workflow.create_project(
        str(source), project_id="regression"
    )
    output = created["output_root"]
    service = ProjectService.open(output)
    manifest = service.get_manifest()
    measurement_updates = {
        row.measurement_id: {
            "modality": "nir",
            "axis_kind": "wavelength",
            "axis_unit": "nm",
            "signal_kind": "absorbance",
            "signal_unit": "absorbance",
        }
        for row in manifest.measurements
    }
    sample_updates = {}
    for sample in manifest.samples:
        name = str(sample.metadata["measurement_name"])
        group = int(name.split("-")[0][1:])
        sample_updates[sample.sample_id] = {
            "preparation_id": f"prep-{group}",
            "technical_replicate_id": name,
            "metadata": {
                **dict(sample.metadata),
                "concentration": float(group),
            },
        }
    project_workflow.update_project_manifest(
        output,
        {
            "measurements": measurement_updates,
            "samples": sample_updates,
        },
    )
    plan = project_workflow.plan_project_analysis(
        output,
        "Model concentration from NIR",
        task_kind="regression",
        target="concentration",
    )
    assert plan["split_manifest"]["metadata"]["row_indices_declared"]
    approval = project_workflow.approve_project_plan(
        output, plan["plan_id"], approved_by="scientist"
    )
    result = run_project_analysis(
        output,
        plan["plan_id"],
        approval["approval_id"],
        "regression-run",
    )
    assert result["status"] == "succeeded"
    record = service.store.read_json("runs/regression-run.json")
    assert record["results"][0]["pipeline_id"] == "nested-selection"
    evidence = service.store.read_json(
        "runs/regression-run/evidence.json"
    )
    assert evidence["task_result"]["evaluation"]["split_id"] == plan[
        "split_manifest"
    ]["split_id"]
