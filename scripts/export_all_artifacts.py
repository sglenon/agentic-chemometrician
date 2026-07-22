"""Comprehensive script to run MCP chemometrics analysis, ground truth evaluation, figures generation, and save all outputs to mcp-expts-2/antigravity."""

import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_predict

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

ROOT_DIR = Path("/Users/Lars/orca/workspaces/agentic-chemometrician/main")
DATA_DIR = ROOT_DIR / "ftir-purity-dataset/fwdftirjune262026"
TARGET_DIR = ROOT_DIR / "mcp-expts-2/antigravity"
RUNS_ROOT = ROOT_DIR / "runs"
RUN_ID = "antigravity-ftir-purity-run-final"

GT_MG = {
    "C": (0.0, 0.0, 54.0),
    "D": (0.0, 6.5, 17.4),
    "E": (0.0, 16.5, 17.4),
    "F": (11.1, 6.8, 10.6),
    "N": (0.0, 30.9, 37.6),
    "G": (6.2, 0.0, 21.2),
    "H": (6.3, 0.0, 13.3),
    "I": (14.8, 0.0, 16.4),
    "J": (3.3, 12.5, 26.4),
    "M": (3.2, 2.6, 10.6),
    "L": (3.3, 6.2, 25.2),
}

def wt_fracs(label: str) -> tuple[float, float, float]:
    fc, ami, s31 = GT_MG[label]
    tot = fc + ami + s31
    return fc / tot, ami / tot, s31 / tot

def to_dict_safe(obj):
    if obj is None:
        return None
    if isinstance(obj, (int, float, bool, str)):
        return obj
    if hasattr(obj, "value") and not isinstance(obj, type):
        return getattr(obj, "value")
    import dataclasses
    if dataclasses.is_dataclass(obj):
        res = {}
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            res[f.name] = to_dict_safe(v)
        return res
    if isinstance(obj, (list, tuple, set)):
        return [to_dict_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_dict_safe(v) for k, v in obj.items()}
    if hasattr(obj, "to_dict"):
        try:
            res = obj.to_dict()
            if res is not None:
                return res
        except Exception:
            pass
    return str(obj)

def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = TARGET_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)

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
        user_intent="Perform end-to-end FTIR spectral analysis including mixture purity identification, preprocessing comparison, validation guardrails, and ground truth regression.",
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

    print("=== Step 8: Ground Truth & Leakage Analysis ===")
    X = np.array(dataset.x)
    axis = np.array(dataset.axis)
    sample_ids = list(dataset.sample_ids)
    labels = list(dataset.labels)

    gt_info = {}
    for lab in sorted(set(labels)):
        if lab in GT_MG:
            fc, ami, s31 = wt_fracs(lab)
            gt_info[lab] = {
                "mg_3fc": GT_MG[lab][0],
                "mg_8ami": GT_MG[lab][1],
                "mg_S31": GT_MG[lab][2],
                "wt_pct_3fc": round(100 * fc, 2),
                "wt_pct_8ami": round(100 * ami, 2),
                "wt_pct_S31": round(100 * s31, 2),
            }
        else:
            gt_info[lab] = {
                "note": "Pure S31 reference preparation (solvent/solid)",
                "wt_pct_3fc": 0.0,
                "wt_pct_8ami": 0.0,
                "wt_pct_S31": 100.0,
            }

    X_sg1 = savgol_filter(X, window_length=11, polyorder=2, deriv=1, axis=1)
    y_cls = np.array(labels)
    grp = y_cls.copy()

    rf_cls = RandomForestClassifier(n_estimators=100, random_state=42)
    pred_shuf = cross_val_predict(rf_cls, X_sg1, y_cls, cv=KFold(5, shuffle=True, random_state=42))
    acc_shuf = float(np.mean(pred_shuf == y_cls))

    pred_logo = cross_val_predict(rf_cls, X_sg1, y_cls, groups=grp, cv=LeaveOneGroupOut())
    acc_logo = float(np.mean(pred_logo == y_cls))

    mix_idx = [i for i, lab in enumerate(labels) if lab in GT_MG]
    X_mix = X_sg1[mix_idx]
    g_mix = np.array([labels[i] for i in mix_idx])
    y_ami = np.array([wt_fracs(labels[i])[1] * 100 for i in mix_idx])
    y_fc = np.array([wt_fracs(labels[i])[0] * 100 for i in mix_idx])
    y_s31 = np.array([wt_fracs(labels[i])[2] * 100 for i in mix_idx])

    reg_results = {}
    for name, yv in [("8ami_wt_pct", y_ami), ("3fc_wt_pct", y_fc), ("S31_wt_pct", y_s31)]:
        preds = np.zeros(len(yv), dtype=float)
        logo = LeaveOneGroupOut()
        for tr, te in logo.split(X_mix, yv, groups=g_mix):
            nc = max(1, min(2, len(tr) - 1))
            pls = PLSRegression(n_components=nc)
            pls.fit(X_mix[tr], yv[tr])
            pred_vals = pls.predict(X_mix[te]).reshape(-1)
            preds[te] = pred_vals
        rmse = float(np.sqrt(mean_squared_error(yv, preds)))
        ss_res = float(np.sum((preds - yv) ** 2))
        ss_tot = float(np.sum((yv - yv.mean()) ** 2))
        q2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        reg_results[name] = {
            "logo_cv_rmse_wtpct": round(rmse, 2),
            "logo_cv_q2": round(q2, 3),
            "true_values": [round(float(v), 2) for v in yv],
            "pred_values": [round(float(v), 2) for v in preds],
        }

    gt_analysis_payload = {
        "ground_truth_by_class": gt_info,
        "classification_leakage_assessment": {
            "shuffled_5fold_accuracy": round(acc_shuf, 3),
            "leave_one_group_out_accuracy": round(acc_logo, 3),
            "leakage_gap": round(acc_shuf - acc_logo, 3),
        },
        "quantitative_impurity_regression": reg_results,
    }

    # Save JSON files directly into TARGET_DIR
    files_to_save = {
        "ground_truth_analysis.json": gt_analysis_payload,
        "inspection.json": inspection,
        "analysis_plan.json": plan,
        "analysis_run.json": analysis_run,
        "validation_summary.json": validation_summary,
        "model_selection.json": selection_rec,
        "interpretation.json": interpretation_summary,
        "report_summary.json": report_summary,
    }

    for fname, data_obj in files_to_save.items():
        dict_data = to_dict_safe(data_obj)
        json_text = json.dumps(dict_data, indent=2, default=str)
        (TARGET_DIR / fname).write_text(json_text, encoding="utf-8")
        print(f"Exported {fname} ({len(json_text)} bytes)")

    # Copy primary report
    if report_summary.primary_report and report_summary.primary_report.uri:
        rp = Path(report_summary.primary_report.uri)
        if not rp.is_absolute():
            rp = ROOT_DIR / rp
        if rp.exists():
            shutil.copy(rp, TARGET_DIR / "REPORT.md")
            print("Copied primary REPORT.md")

    # Copy figures from artifacts
    for art in getattr(analysis_run, "artifacts", []):
        if hasattr(art, "uri") and art.uri:
            ap = Path(art.uri)
            if not ap.is_absolute():
                ap = ROOT_DIR / ap
            if ap.exists() and ap.suffix in (".png", ".svg", ".jpg"):
                shutil.copy(ap, figures_dir / ap.name)

    # Custom visualizations
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    plt.figure(figsize=(10, 5))
    unique_labels = sorted(set(labels))
    cmap = plt.cm.tab10
    for i, lab in enumerate(unique_labels):
        indices = [j for j, l in enumerate(labels) if l == lab]
        for idx in indices:
            plt.plot(axis, X[idx], color=cmap(i % 10), alpha=0.6, label=lab if idx == indices[0] else "")
    plt.gca().invert_xaxis()
    plt.title("FTIR Raw Transmittance Spectra (%T) across Mixture Groups", fontsize=12, fontweight='bold')
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Transmittance (%T)")
    plt.legend(loc="best", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(figures_dir / "raw_spectra_by_group.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    for i, lab in enumerate(unique_labels):
        indices = [j for j, l in enumerate(labels) if l == lab]
        for idx in indices:
            plt.plot(axis, X_sg1[idx], color=cmap(i % 10), alpha=0.6, label=lab if idx == indices[0] else "")
    plt.gca().invert_xaxis()
    plt.title("Savitzky-Golay 1st Derivative FTIR Spectra", fontsize=12, fontweight='bold')
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("d(%T)/d(cm⁻¹)")
    plt.legend(loc="best", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(figures_dir / "sg_1st_derivative.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    gt_classes = [c for c in unique_labels if c in GT_MG]
    fc_vals = [gt_info[c]["wt_pct_3fc"] for c in gt_classes]
    ami_vals = [gt_info[c]["wt_pct_8ami"] for c in gt_classes]
    s31_vals = [gt_info[c]["wt_pct_S31"] for c in gt_classes]

    x_pos = np.arange(len(gt_classes))
    width = 0.5
    plt.bar(x_pos, s31_vals, width, label="S31 (Schiff Base)", color="#2b5c8f")
    plt.bar(x_pos, ami_vals, width, bottom=s31_vals, label="8ami (Aminoquinoline)", color="#d95f02")
    plt.bar(x_pos, fc_vals, width, bottom=np.array(s31_vals)+np.array(ami_vals), label="3fc (Formylchromone)", color="#7570b3")
    plt.xlabel("Mixture Label")
    plt.ylabel("Ground Truth Weight Percentage (wt%)")
    plt.title("Ground Truth Ternary/Binary Compositions from Weighed Masses", fontsize=12, fontweight='bold')
    plt.xticks(x_pos, gt_classes)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(figures_dir / "ground_truth_compositions.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4.5))
    bars = plt.bar(["Shuffled 5-Fold\n(Replicate Leakage)", "Leave-One-Group-Out\n(Strict Out-of-Group)"], 
                   [acc_shuf * 100, acc_logo * 100], 
                   color=["#e74c3c", "#2ecc71"], width=0.45)
    plt.ylabel("CV Accuracy (%)")
    plt.title("Impact of Replicate Leakage on Model Validation Accuracy", fontsize=11, fontweight='bold')
    plt.ylim(0, 110)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / "replicate_leakage_impact.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    y_true_ami = reg_results["8ami_wt_pct"]["true_values"]
    y_pred_ami = reg_results["8ami_wt_pct"]["pred_values"]
    plt.scatter(y_true_ami, y_pred_ami, color='#d95f02', s=60, edgecolors='black', label=f"8ami (RMSE={reg_results['8ami_wt_pct']['logo_cv_rmse_wtpct']}%)")
    
    y_true_fc = reg_results["3fc_wt_pct"]["true_values"]
    y_pred_fc = reg_results["3fc_wt_pct"]["pred_values"]
    plt.scatter(y_true_fc, y_pred_fc, color='#7570b3', s=60, edgecolors='black', marker='^', label=f"3fc (RMSE={reg_results['3fc_wt_pct']['logo_cv_rmse_wtpct']}%)")

    max_val = max(max(y_true_ami), max(y_true_fc)) + 5
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.7, label="Ideal Parity")
    plt.xlabel("Ground Truth Weight % (Weighed Mass)")
    plt.ylabel("PLS LOGO-CV Predicted Weight %")
    plt.title("PLS Quantitative Regression on Measured Impurity wt%", fontsize=11, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "pls_impurity_regression_parity.png", dpi=300)
    plt.close()

    print(f"\nAll artifacts and figures successfully exported to {TARGET_DIR}")

if __name__ == "__main__":
    main()
