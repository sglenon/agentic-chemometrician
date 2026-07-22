"""Execute full MCP chemometrics analysis for fwdftirjune262026 dataset and copy findings to mcp-expts/antigravity."""

import json
import shutil
from pathlib import Path
from chemometrics_contracts import (
    InspectDatasetRequest,
    ProposeAnalysisPlanRequest,
    RunAnalysisRequest,
    ValidateResultsRequest,
    SelectBestModelRequest,
    InterpretResultsRequest,
    GenerateReportRequest,
)
from chemometrics_mcp.core.datasets import load_ftir_real
from chemometrics_mcp.tools import (
    inspect_dataset,
    propose_analysis_plan,
    run_analysis,
    validate_results,
    select_best_model,
    interpret_results,
    generate_report,
)

DATA_DIR = Path("/Users/Lars/orca/workspaces/agentic-chemometrician/main/ftir-purity-dataset/fwdftirjune262026")
TARGET_DIR = Path("/Users/Lars/orca/workspaces/agentic-chemometrician/main/mcp-expts/antigravity")
RUNS_ROOT = Path("/Users/Lars/orca/workspaces/agentic-chemometrician/main/runs")
RUN_ID = "antigravity-ftir-purity-run"

def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=== Step 1: Inspect Dataset ===")
    inspect_req = InspectDatasetRequest(
        source_uri=str(DATA_DIR),
        dataset_id="fwdftirjune262026",
        modality_override="FTIR"
    )
    inspect_res = inspect_dataset.run(inspect_req, runs_root=RUNS_ROOT)
    print(inspect_res.message)
    dataset, inspection = load_ftir_real(DATA_DIR)
    
    print("=== Step 2: Propose Analysis Plan ===")
    plan_req = ProposeAnalysisPlanRequest(
        dataset_inspection=inspection,
        user_intent="Perform automated spectral chemometric classification of FTIR purity samples with preprocessing and validation guardrails.",
        task_hint="classification"
    )
    plan_res = propose_analysis_plan.run(plan_req)
    print(plan_res.message)
    plan = plan_res.payload
    
    print("=== Step 3: Run Analysis ===")
    run_req = RunAnalysisRequest(
        dataset=dataset,
        approved_plan=plan,
        run_id=RUN_ID
    )
    run_res = run_analysis.run(run_req, runs_root=RUNS_ROOT)
    print(run_res.message)
    analysis_run = run_res.payload
    
    print("=== Step 4: Validate Results ===")
    val_req = ValidateResultsRequest(
        results=analysis_run.results,
        dataset=dataset,
        dataset_inspection=inspection
    )
    val_res = validate_results.run(val_req, runs_root=RUNS_ROOT)
    print(val_res.message)
    validation_summary = val_res.payload
    
    print("=== Step 5: Select Best Model ===")
    sel_req = SelectBestModelRequest(
        results=analysis_run.results,
        validation_summary=validation_summary,
        task_name=plan.task_name
    )
    sel_res = select_best_model.run(sel_req, runs_root=RUNS_ROOT)
    print(sel_res.message)
    selection_rec = sel_res.payload
    
    print("=== Step 6: Interpret Results ===")
    interp_req = InterpretResultsRequest(
        results=analysis_run.results,
        dataset=dataset,
        validation_summary=validation_summary
    )
    interp_res = interpret_results.run(interp_req, runs_root=RUNS_ROOT)
    print(interp_res.message)
    interpretation_summary = interp_res.payload
    
    print("=== Step 7: Generate Final Report ===")
    rep_req = GenerateReportRequest(
        analysis_run=analysis_run,
        validation_summary=validation_summary,
        interpretation=interpretation_summary
    )
    rep_res = generate_report.run(rep_req, runs_root=RUNS_ROOT)
    print(rep_res.message)
    report_summary = rep_res.payload

    # Write summary files and copy artifacts to TARGET_DIR
    run_dir = RUNS_ROOT / RUN_ID
    
    # Save structured json files into TARGET_DIR
    (TARGET_DIR / "inspection.json").write_text(json.dumps(inspection.to_dict(), indent=2, default=str))
    (TARGET_DIR / "analysis_plan.json").write_text(json.dumps(plan.to_dict(), indent=2, default=str))
    (TARGET_DIR / "analysis_run.json").write_text(json.dumps(analysis_run.to_dict(), indent=2, default=str))
    (TARGET_DIR / "validation_summary.json").write_text(json.dumps(validation_summary.to_dict(), indent=2, default=str))
    (TARGET_DIR / "model_selection.json").write_text(json.dumps(selection_rec.to_dict(), indent=2, default=str))
    (TARGET_DIR / "interpretation.json").write_text(json.dumps(interpretation_summary.to_dict(), indent=2, default=str))
    (TARGET_DIR / "report_summary.json").write_text(json.dumps(report_summary.to_dict(), indent=2, default=str))

    # Copy report.md and structured json files into TARGET_DIR
    artifacts_src = run_dir / "artifacts"
    report_md_path = artifacts_src / "report.md"
    if report_md_path.exists():
        shutil.copy(report_md_path, TARGET_DIR / "REPORT.md")
        
    figures_dir = TARGET_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    if artifacts_src.exists():
        for item in artifacts_src.glob("*"):
            if item.name != "report.md":
                shutil.copy(item, figures_dir / item.name)

    print(f"\nAll artifacts successfully saved to {TARGET_DIR}")

if __name__ == "__main__":
    main()
