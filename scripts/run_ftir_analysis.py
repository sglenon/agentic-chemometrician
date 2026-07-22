"""Run the chemometrics MCP pipeline on the fwdftirjune262026 FTIR dataset."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chemometrics_contracts import (
    AnalysisPlan,
    DatasetInspection,
    InspectDatasetRequest,
    ProposeAnalysisPlanRequest,
    RunAnalysisRequest,
    ValidateResultsRequest,
    SelectBestModelRequest,
    InterpretResultsRequest,
    GenerateReportRequest,
    AnalysisResult,
    AnalysisRun,
    ValidationWarning,
    ValidationSummary,
    InterpretationSummary,
    RunMetadata,
    ArtifactReference,
)
from chemometrics_mcp.tools import (
    inspect_dataset,
    propose_analysis_plan,
    run_analysis,
    validate_results,
    select_best_model,
    interpret_results,
    generate_report,
)

RUNS_ROOT = PROJECT_ROOT / "runs"
OUTPUT_DIR = PROJECT_ROOT / "mcp-expts" / "opencode"
DATA_DIR = PROJECT_ROOT / "ftir-purity-dataset" / "fwdftirjune262026"


def _save_json(data, filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / filename
    out.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return out


def main() -> None:
    print("=" * 70)
    print("STEP 1: Inspect Dataset")
    print("=" * 70)

    inspect_req = InspectDatasetRequest(
        source_uri=str(DATA_DIR),
        dataset_id="fwdftir_june2026",
        modality_override="FTIR",
    )
    inspect_resp = inspect_dataset.run(inspect_req, runs_root=RUNS_ROOT)

    print(f"  ok: {inspect_resp.ok}")
    print(f"  message: {inspect_resp.message}")
    if inspect_resp.payload:
        p = inspect_resp.payload
        print(f"  samples: {p.sample_count}, features: {p.feature_count}")
        print(f"  modality: {p.modality}")
        print(f"  axis: {p.axis_min}–{p.axis_max} cm⁻¹")
        print(f"  candidate labels: {p.candidate_label_columns}")
        print(f"  warnings: {len(p.warnings)}")
        for w in p.warnings:
            print(f"    [{w.severity}] {w.code}: {w.message}")

    _save_json(inspect_resp.to_dict(), "01_inspection.json")

    if not inspect_resp.ok or not inspect_resp.payload:
        print("FATAL: inspection failed")
        return

    inspection = inspect_resp.payload

    print()
    print("=" * 70)
    print("STEP 2: Propose Analysis Plan")
    print("=" * 70)

    plan_req = ProposeAnalysisPlanRequest(
        dataset_inspection=inspection,
        user_intent="Classify FTIR spectra by purity group for pharmaceutical raw material identification",
        task_hint="classification",
    )
    plan_resp = propose_analysis_plan.run(plan_req, runs_root=RUNS_ROOT)

    print(f"  ok: {plan_resp.ok}")
    print(f"  message: {plan_resp.message}")
    if plan_resp.payload:
        pl = plan_resp.payload
        print(f"  task: {pl.task_name}")
        print(f"  models: {pl.model_families}")
        print(f"  preprocessing: {pl.preprocessing_candidates}")
        print(f"  validation: {pl.validation_strategy}")
        print(f"  warnings: {len(pl.warnings)}")
        for w in pl.warnings:
            print(f"    [{w.severity}] {w.code}: {w.message}")

    _save_json(plan_resp.to_dict(), "02_plan.json")

    if not plan_resp.ok or not plan_resp.payload:
        print("FATAL: plan proposal failed")
        return

    plan = plan_resp.payload

    print()
    print("=" * 70)
    print("STEP 3: Run Analysis")
    print("=" * 70)

    from chemometrics_mcp.core.datasets import load_ftir_real

    dataset, _ = load_ftir_real(DATA_DIR, modality_override="FTIR")

    run_req = RunAnalysisRequest(
        dataset=dataset,
        approved_plan=plan,
        run_id=None,
    )
    run_resp = run_analysis.run(run_req, runs_root=RUNS_ROOT)

    print(f"  ok: {run_resp.ok}")
    print(f"  message: {run_resp.message}")
    if run_resp.payload:
        ar = run_resp.payload
        print(f"  results: {len(ar.results)}")
        print(f"  failed models: {ar.failed_models}")
        print(f"  warnings: {len(ar.warnings)}")
        for r in ar.results:
            print(f"    {r.model_name} ({r.preprocessing}): {r.metrics}")
        for w in ar.warnings:
            print(f"    [{w.severity}] {w.code}: {w.message}")

    _save_json(run_resp.to_dict(), "03_analysis_run.json")

    if not run_resp.payload:
        print("FATAL: analysis run failed")
        return

    analysis_run = run_resp.payload

    print()
    print("=" * 70)
    print("STEP 4: Validate Results")
    print("=" * 70)

    val_req = ValidateResultsRequest(
        results=analysis_run.results,
        dataset=dataset,
        dataset_inspection=inspection,
    )
    val_resp = validate_results.run(val_req, runs_root=RUNS_ROOT)

    print(f"  ok: {val_resp.ok}")
    print(f"  message: {val_resp.message}")
    if val_resp.payload:
        vs = val_resp.payload
        print(f"  passed: {vs.passed}")
        print(f"  checks: {vs.checks}")
        print(f"  warnings: {len(vs.warnings)}")
        for w in vs.warnings:
            print(f"    [{w.severity}] {w.code}: {w.message}")

    _save_json(val_resp.to_dict(), "04_validation.json")

    validation_summary = val_resp.payload

    print()
    print("=" * 70)
    print("STEP 5: Select Best Model")
    print("=" * 70)

    sel_req = SelectBestModelRequest(
        results=analysis_run.results,
        validation_summary=validation_summary,
        task_name=plan.task_name,
    )
    sel_resp = select_best_model.run(sel_req, runs_root=RUNS_ROOT)

    print(f"  ok: {sel_resp.ok}")
    print(f"  message: {sel_resp.message}")
    if sel_resp.payload:
        rec = sel_resp.payload
        print(f"  selected_model: {rec.selected_model}")
        print(f"  rationale: {rec.rationale}")
        print(f"  requires_human_approval: {rec.requires_human_approval}")
        print(f"  candidates: {rec.candidate_models[:5]}")
        print(f"  warnings: {len(rec.warnings)}")
        for w in rec.warnings:
            print(f"    [{w.severity}] {w.code}: {w.message}")

    _save_json(sel_resp.to_dict(), "05_model_selection.json")

    print()
    print("=" * 70)
    print("STEP 6: Interpret Results")
    print("=" * 70)

    interp_req = InterpretResultsRequest(
        results=analysis_run.results,
        dataset=dataset,
        validation_summary=validation_summary,
    )
    interp_resp = interpret_results.run(interp_req, runs_root=RUNS_ROOT)

    print(f"  ok: {interp_resp.ok}")
    print(f"  message: {interp_resp.message}")
    if interp_resp.payload:
        ip = interp_resp.payload
        print(f"  summary: {ip.summary}")
        print(f"  important features: {len(ip.important_features)}")
        for f in ip.important_features[:10]:
            print(f"    {f}")
        print(f"  model comparisons: {len(ip.model_comparisons)}")
        for mc in ip.model_comparisons[:5]:
            print(f"    {mc}")
        print(f"  warnings: {len(ip.warnings)}")
        for w in ip.warnings:
            print(f"    [{w.severity}] {w.code}: {w.message}")

    _save_json(interp_resp.to_dict(), "06_interpretation.json")

    interpretation = interp_resp.payload

    print()
    print("=" * 70)
    print("STEP 7: Generate Report")
    print("=" * 70)

    report_req = GenerateReportRequest(
        analysis_run=analysis_run,
        validation_summary=validation_summary,
        interpretation=interpretation,
    )
    report_resp = generate_report.run(report_req, runs_root=RUNS_ROOT)

    print(f"  ok: {report_resp.ok}")
    print(f"  message: {report_resp.message}")
    if report_resp.payload:
        rp = report_resp.payload
        print(f"  report title: {rp.report_title}")
        print(f"  report length: {len(rp.human_readable_summary)} chars")
        print()
        print("--- REPORT PREVIEW (first 3000 chars) ---")
        print(rp.human_readable_summary[:3000])
        print("--- END PREVIEW ---")

    _save_json(report_resp.to_dict(), "07_report.json")

    print()
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"All artifacts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
