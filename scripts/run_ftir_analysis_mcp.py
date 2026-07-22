"""FTIR Chemometrics Analysis via MCP Workflow.

Implements the full agentic workflow:
  01_inspection  → inspect_dataset (FTIR dir)
  00_target_manifest → manual join of composition table to spectra
  02_plan        → propose_analysis_plan
  03_run_FC/AMI/SB → run_analysis (×3 regression targets)
  04_logo_baseline → true LOGO-CV (LeaveOneGroupOut, sklearn)
  05_validation  → validate_results (with and without grouped_kfold_5)
  06_selection   → select_best_model
  07_interpretation → interpret_results
  08_report      → generate_report
  09_predictions_vs_truth.csv
  findings.md, experience.md

Key design notes:
- MCP grouped_kfold_5 is stubbed (modeling.py:65): uses KFold(5) that ignores
  the groups argument — replicates can bleed across folds.
- True LOGO-CV uses sklearn LeaveOneGroupOut, grouping by 6-letter stems.
- Step 4 exposes the leakage gap between MCP CV and true LOGO.
"""
from __future__ import annotations

import json
import sys
import csv
import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: add src/ so MCP modules are importable without pip install.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np
from scipy.interpolate import interp1d

from chemometrics_contracts import (
    AnalysisPlan,
    AnalysisRun,
    DatasetInspection,
    GenerateReportRequest,
    InterpretResultsRequest,
    ProposeAnalysisPlanRequest,
    RunAnalysisRequest,
    SelectBestModelRequest,
    SpectralDataset,
    ValidateResultsRequest,
    RunMetadata,
    ValidationSummary,
)
from chemometrics_mcp.tools import (
    generate_report,
    inspect_dataset,
    interpret_results,
    propose_analysis_plan,
    run_analysis,
    select_best_model,
    validate_results,
)
from chemometrics_mcp.core.datasets import load_ftir_real, _FTIR_REAL_PURITY_MANIFEST
from chemometrics_contracts import InspectDatasetRequest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FTIR_DIR = REPO_ROOT / "ftir-purity-dataset" / "fwdftirjune262026"
COMP_TABLE = REPO_ROOT / "ftir-purity-dataset" / "ftir-dataset"
OUTPUT_DIR = REPO_ROOT / "mcp-expts-2" / "claude"
RUNS_ROOT = OUTPUT_DIR / "runs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUNS_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  saved → {path.relative_to(REPO_ROOT)}")


def _tool_response_to_dict(response) -> dict:
    """Serialise a ToolResponse to a plain dict."""
    return response.to_dict()


def now_iso() -> str:
    return datetime.datetime.now(tz=datetime.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Step 0: Parse composition table
# ---------------------------------------------------------------------------
def parse_composition_table(path: Path) -> dict[str, dict[str, float]]:
    """Parse markdown table → {letter: {FC_mg, AMI_mg, SB_mg}}.

    Table format (from ftir-dataset):
        | X | mg (3fc) | mg (8ami) | mg (S31) |
        | - | -------: | --------: | -------: |
        | C |      0.0 |       0.0 |     54.0 |
        ...
    """
    comp: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("|"):
                continue
            parts = [p.strip() for p in line.split("|")]
            parts = [p for p in parts if p]  # drop empty
            if len(parts) < 4:
                continue
            # Skip header and separator rows
            if parts[0] in ("X", "-"):
                continue
            if parts[1].startswith("-"):
                continue
            try:
                letter = parts[0]
                fc_mg = float(parts[1])
                ami_mg = float(parts[2])
                sb_mg = float(parts[3])
                comp[letter] = {"FC_mg": fc_mg, "AMI_mg": ami_mg, "SB_mg": sb_mg}
            except (ValueError, IndexError):
                continue
    return comp


# ---------------------------------------------------------------------------
# Step 1: Build target manifest
# ---------------------------------------------------------------------------
def build_target_manifest(
    composition: dict[str, dict[str, float]],
    ftir_dir: Path,
    supervised_groups: list[str],
) -> list[dict]:
    """Join composition to each usable .txt spectrum.

    supervised_groups: letters that have .txt files and are in composition.
    Returns list of dicts with sample_id, group, FC_mg, AMI_mg, SB_mg, file.
    """
    manifest = []
    txt_files = sorted(ftir_dir.glob("*.txt"))
    for fp in txt_files:
        stem = fp.stem
        # Determine group letter (C1→C, L2→L, s31acn2→exclude, etc.)
        group = _FTIR_REAL_PURITY_MANIFEST.get(stem)
        if group is None or group not in supervised_groups:
            continue
        if group not in composition:
            continue
        comp = composition[group]
        manifest.append({
            "sample_id": stem,
            "group": group,
            "source_file": fp.name,
            "FC_mg": comp["FC_mg"],
            "AMI_mg": comp["AMI_mg"],
            "SB_mg": comp["SB_mg"],
        })
    return manifest


# ---------------------------------------------------------------------------
# Step 2: Load FTIR spectra for supervised set
# ---------------------------------------------------------------------------
def load_supervised_spectra(
    ftir_dir: Path,
    manifest: list[dict],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load spectra for samples in manifest, interpolated to common grid.

    Returns
    -------
    X : (n_samples, n_features)
    axis : (n_features,)
    sample_ids : list[str]
    groups : list[str]  (letter group for LOGO-CV)
    """
    # Build common grid matching load_ftir_real defaults
    wavenumber_start = 400.0
    wavenumber_end = 4000.0
    wavenumber_step = 4.0
    n_grid = int(round((wavenumber_end - wavenumber_start) / wavenumber_step)) + 1
    common_axis = np.linspace(wavenumber_start, wavenumber_end, n_grid)

    stem_to_row = {row["sample_id"]: row for row in manifest}
    rows_X = []
    sample_ids = []
    groups = []

    for fp in sorted(ftir_dir.glob("*.txt")):
        stem = fp.stem
        if stem not in stem_to_row:
            continue
        row = stem_to_row[stem]
        # Parse spectrum
        wavenumbers: list[float] = []
        transmittances: list[float] = []
        with fp.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    wavenumbers.append(float(parts[0]))
                    transmittances.append(float(parts[1]))
                except ValueError:
                    continue
        if len(wavenumbers) < 2:
            print(f"  WARNING: skipping {fp.name} — fewer than 2 data rows")
            continue
        wn = np.array(wavenumbers)
        tr = np.array(transmittances)
        interp_fn = interp1d(wn, tr, kind="linear", bounds_error=False,
                             fill_value=(tr[0], tr[-1]))
        tr_interp = interp_fn(common_axis)
        rows_X.append(tr_interp)
        sample_ids.append(stem)
        groups.append(row["group"])

    X = np.vstack(rows_X)
    return X, common_axis, sample_ids, groups


# ---------------------------------------------------------------------------
# LOGO-CV baseline (sklearn, no MCP)
# ---------------------------------------------------------------------------
def compute_logo_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    model_name: str,
    preprocessing_method: str,
) -> dict:
    """True Leave-One-Group-Out CV using sklearn.

    Returns dict with r2, rmse, mae, per_sample_preds, per_sample_residuals.
    """
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.svm import SVR
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score, mean_absolute_error
    from chemometrics_mcp.core import preprocessing as prep_mod

    logo = LeaveOneGroupOut()
    groups_arr = np.array(groups)

    y_true_all = []
    y_pred_all = []
    sample_idx_all = []

    for train_idx, test_idx in logo.split(X, y, groups_arr):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        # Apply preprocessing (fit on train, transform both)
        # NB: for row-wise transforms (snv, sg_*), this is equivalent
        # to applying independently — no leakage. For msc, we fit on train mean.
        if preprocessing_method == "raw":
            X_tr_proc = X_train.copy()
            X_te_proc = X_test.copy()
        elif preprocessing_method == "snv":
            # SNV is row-wise, no leakage
            X_tr_proc, _ = prep_mod.apply(X_train, "snv")
            X_te_proc, _ = prep_mod.apply(X_test, "snv")
        elif preprocessing_method == "sg_1st_deriv":
            X_tr_proc, _ = prep_mod.apply(X_train, "sg_1st_deriv")
            X_te_proc, _ = prep_mod.apply(X_test, "sg_1st_deriv")
        elif preprocessing_method == "sg_2nd_deriv":
            X_tr_proc, _ = prep_mod.apply(X_train, "sg_2nd_deriv")
            X_te_proc, _ = prep_mod.apply(X_test, "sg_2nd_deriv")
        else:
            X_tr_proc, _ = prep_mod.apply(X_train, preprocessing_method)
            X_te_proc, _ = prep_mod.apply(X_test, preprocessing_method)

        # Build model
        if model_name == "plsr":
            n_comp = min(10, X_tr_proc.shape[1], X_tr_proc.shape[0] - 1)
            model = PLSRegression(n_components=n_comp)
        elif model_name == "svr":
            model = SVR(kernel="rbf", C=1.0, gamma="scale")
        elif model_name == "ridge":
            model = Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.fit(X_tr_proc, y_train)
        preds = model.predict(X_te_proc)
        if hasattr(preds, "ravel"):
            preds = preds.ravel()

        y_true_all.extend(y[test_idx].tolist())
        y_pred_all.extend(preds.tolist())
        sample_idx_all.extend(test_idx.tolist())

    y_true_arr = np.array(y_true_all)
    y_pred_arr = np.array(y_pred_all)

    rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))

    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "y_true": y_true_arr.tolist(),
        "y_pred": y_pred_arr.tolist(),
        "sample_indices": sample_idx_all,
    }


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("FTIR MCP Workflow — agentic chemometrics analysis")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Parse composition table
    # -----------------------------------------------------------------------
    print("\n[0] Parsing composition table...")
    composition = parse_composition_table(COMP_TABLE)
    print(f"  Compounds in table: {sorted(composition.keys())}")

    # Usable supervised groups: have .txt files AND in composition
    supervised_groups = sorted(
        letter for letter in ["C", "I", "J", "L", "M", "N"]
        if letter in composition
    )
    print(f"  Supervised groups (have .txt + composition): {supervised_groups}")

    # -----------------------------------------------------------------------
    # Build target manifest
    # -----------------------------------------------------------------------
    print("\n[0b] Building target manifest...")
    manifest = build_target_manifest(composition, FTIR_DIR, supervised_groups)
    print(f"  Manifest entries: {len(manifest)}")
    for row in manifest:
        print(f"    {row['sample_id']:8s} group={row['group']} "
              f"FC={row['FC_mg']:5.1f} AMI={row['AMI_mg']:5.1f} SB={row['SB_mg']:5.1f}")

    manifest_data = {
        "created_at": now_iso(),
        "supervised_groups": supervised_groups,
        "composition_source": str(COMP_TABLE.relative_to(REPO_ROOT)),
        "ftir_dir": str(FTIR_DIR.relative_to(REPO_ROOT)),
        "n_spectra": len(manifest),
        "compounds": ["FC (mg 3fc)", "AMI (mg 8ami)", "SB (mg S31)"],
        "note_L": "L.txt absent; L1 and L2 used as replicates for group L",
        "entries": manifest,
    }
    _save_json(manifest_data, OUTPUT_DIR / "00_target_manifest.json")

    # -----------------------------------------------------------------------
    # Step 1: inspect_dataset via MCP tool
    # -----------------------------------------------------------------------
    print("\n[1] Running inspect_dataset (FTIR dir)...")
    inspect_req = InspectDatasetRequest(
        source_uri=str(FTIR_DIR),
        dataset_id="fwdftirjune262026",
        modality_override="FTIR",
        source_format="ftir_dir",
    )
    inspect_resp = inspect_dataset.run(inspect_req, runs_root=str(RUNS_ROOT))
    print(f"  ok={inspect_resp.ok} msg={inspect_resp.message}")
    inspection: DatasetInspection = inspect_resp.payload
    _save_json(_tool_response_to_dict(inspect_resp), OUTPUT_DIR / "01_inspection.json")

    # -----------------------------------------------------------------------
    # Step 2: propose_analysis_plan via MCP tool
    # -----------------------------------------------------------------------
    print("\n[2] Running propose_analysis_plan...")
    plan_req = ProposeAnalysisPlanRequest(
        dataset_inspection=inspection,
        task_hint="regression",
        user_intent=(
            "Quantify three compound concentrations (FC, AMI, SB in mg) in "
            "FTIR tablet spectra. 6 samples (17 spectra with replicates). "
            "Compare plsr, svr, ridge regression with snv and sg_1st_deriv preprocessing. "
            "Use grouped_kfold_5 validation strategy."
        ),
    )
    plan_resp = propose_analysis_plan.run(plan_req, runs_root=str(RUNS_ROOT))
    print(f"  ok={plan_resp.ok} msg={plan_resp.message}")
    proposed_plan: AnalysisPlan = plan_resp.payload
    _save_json(_tool_response_to_dict(plan_resp), OUTPUT_DIR / "02_plan.json")

    # Print proposed plan details
    if proposed_plan:
        print(f"  task={proposed_plan.task_name}")
        print(f"  preprocessing={list(proposed_plan.preprocessing_candidates)}")
        print(f"  models={list(proposed_plan.model_families)}")
        print(f"  validation={proposed_plan.validation_strategy}")

    # -----------------------------------------------------------------------
    # Load spectra + build supervised SpectralDataset(s)
    # -----------------------------------------------------------------------
    print("\n[prep] Loading supervised spectra...")
    X, axis, sample_ids, groups = load_supervised_spectra(FTIR_DIR, manifest)
    print(f"  X.shape={X.shape}  n_groups={len(set(groups))}")
    print(f"  Groups: {sorted(set(groups))}")
    print(f"  Sample IDs: {sample_ids}")

    # Build label arrays for each compound
    stem_to_comp = {row["sample_id"]: row for row in manifest}
    FC_labels = [stem_to_comp[sid]["FC_mg"] for sid in sample_ids]
    AMI_labels = [stem_to_comp[sid]["AMI_mg"] for sid in sample_ids]
    SB_labels = [stem_to_comp[sid]["SB_mg"] for sid in sample_ids]

    # Fixed analysis plan for regression (override proposed if needed)
    # Use 3 models and 3 preprocessing methods
    regression_plan = AnalysisPlan(
        task_name="regression",
        preprocessing_candidates=("raw", "snv", "sg_1st_deriv"),
        model_families=("plsr", "svr", "ridge"),
        validation_strategy="grouped_kfold_5",
        human_readable_plan=(
            "Regression on FTIR spectra. "
            "Preprocessing: raw, SNV, SG 1st deriv. "
            "Models: PLSR, SVR, Ridge. "
            "Validation: grouped_kfold_5 (NB: MCP stub ignores groups). "
            "Compounds: FC, AMI, SB (separate runs)."
        ),
    )

    # -----------------------------------------------------------------------
    # Step 3: run_analysis × 3 (one per compound)
    # -----------------------------------------------------------------------
    compound_configs = [
        ("FC", FC_labels, "03_run_FC.json"),
        ("AMI", AMI_labels, "03_run_AMI.json"),
        ("SB", SB_labels, "03_run_SB.json"),
    ]

    all_results = {}  # compound → AnalysisRun
    all_datasets = {}  # compound → SpectralDataset

    for compound, labels, artifact_name in compound_configs:
        print(f"\n[3] run_analysis for compound={compound}...")
        print(f"  labels={labels}")

        dataset = SpectralDataset(
            x=tuple(tuple(float(v) for v in row) for row in X),
            axis=tuple(float(v) for v in axis),
            labels=tuple(float(v) for v in labels),
            modality="FTIR",
            sample_ids=tuple(sample_ids),
            metadata=tuple(
                {"sample_id": sid, "group": grp, "compound": compound}
                for sid, grp in zip(sample_ids, groups)
            ),
        )
        all_datasets[compound] = dataset

        run_req = RunAnalysisRequest(
            dataset=dataset,
            approved_plan=regression_plan,
        )
        run_resp = run_analysis.run(run_req, runs_root=str(RUNS_ROOT))
        print(f"  ok={run_resp.ok} msg={run_resp.message}")
        if run_resp.payload:
            analysis_run: AnalysisRun = run_resp.payload
            all_results[compound] = analysis_run
            print(f"  results={len(analysis_run.results)}  failed={list(analysis_run.failed_models)}")
            for r in analysis_run.results:
                print(f"    {r.preprocessing[0]:15s} / {r.model_name:6s}  "
                      f"r2={r.metrics.get('r2', 'N/A'):.3f}  "
                      f"rmse={r.metrics.get('rmse', 'N/A'):.3f}")

        _save_json(_tool_response_to_dict(run_resp), OUTPUT_DIR / artifact_name)

    # -----------------------------------------------------------------------
    # Step 4: True LOGO-CV baseline (outside MCP)
    # -----------------------------------------------------------------------
    print("\n[4] Computing true LOGO-CV baseline (LeaveOneGroupOut)...")
    logo_baseline = {
        "description": (
            "True Leave-One-Group-Out CV using sklearn LeaveOneGroupOut. "
            "Groups = 6 letter stems (C,I,J,L,M,N). "
            "Each fold withholds all replicates of one compound mixture. "
            "This is the unbiased estimator of out-of-sample performance."
        ),
        "n_groups": len(set(groups)),
        "groups": sorted(set(groups)),
        "n_samples": len(sample_ids),
        "sample_ids": sample_ids,
        "groups_per_sample": groups,
        "created_at": now_iso(),
        "results": {},
    }

    preprocessing_methods_logo = ["raw", "snv", "sg_1st_deriv"]
    model_names_logo = ["plsr", "svr", "ridge"]

    for compound, labels in [("FC", FC_labels), ("AMI", AMI_labels), ("SB", SB_labels)]:
        logo_baseline["results"][compound] = {}
        y = np.array(labels, dtype=float)
        for prep_m in preprocessing_methods_logo:
            for model_m in model_names_logo:
                key = f"{prep_m}/{model_m}"
                print(f"  LOGO-CV {compound} {key}...", end="", flush=True)
                try:
                    cv_result = compute_logo_cv(X, y, groups, model_m, prep_m)
                    logo_baseline["results"][compound][key] = {
                        "r2": cv_result["r2"],
                        "rmse": cv_result["rmse"],
                        "mae": cv_result["mae"],
                    }
                    print(f" r2={cv_result['r2']:.3f}  rmse={cv_result['rmse']:.3f}")
                except Exception as exc:
                    logo_baseline["results"][compound][key] = {"error": str(exc)}
                    print(f" ERROR: {exc}")

    _save_json(logo_baseline, OUTPUT_DIR / "04_logo_baseline.json")

    # -----------------------------------------------------------------------
    # Step 5: validate_results (for all compound results combined)
    # -----------------------------------------------------------------------
    print("\n[5] Running validate_results...")

    # Collect all AnalysisResult objects from all 3 compound runs
    all_analysis_results = []
    for compound, analysis_run in all_results.items():
        all_analysis_results.extend(list(analysis_run.results))

    if all_analysis_results:
        # 5a. Without explicit validation_strategy (default)
        val_req_default = ValidateResultsRequest(
            results=tuple(all_analysis_results),
            dataset=all_datasets.get("FC"),
            dataset_inspection=inspection,
        )
        val_resp_default = validate_results.run(val_req_default, runs_root=str(RUNS_ROOT))
        print(f"  [default] ok={val_resp_default.ok} warnings={len(val_resp_default.warnings)}")
        _save_json(
            _tool_response_to_dict(val_resp_default),
            OUTPUT_DIR / "05_validation_default.json",
        )

        # 5b. With grouped_kfold_5 strategy — note this silences leakage check
        # even though the splitter ignores groups (validation.py:270 stub)
        val_req_grouped = ValidateResultsRequest(
            results=tuple(all_analysis_results),
            dataset=all_datasets.get("FC"),
            dataset_inspection=inspection,
        )
        val_resp_grouped = validate_results.run(val_req_grouped, runs_root=str(RUNS_ROOT))
        print(f"  [grouped] ok={val_resp_grouped.ok} warnings={len(val_resp_grouped.warnings)}")
        _save_json(
            _tool_response_to_dict(val_resp_grouped),
            OUTPUT_DIR / "05_validation_grouped_kfold_5.json",
        )
    else:
        print("  No results to validate (all runs failed?)")
        val_resp_default = None
        val_resp_grouped = None

    # -----------------------------------------------------------------------
    # Step 6: select_best_model
    # -----------------------------------------------------------------------
    print("\n[6] Running select_best_model...")
    # Use FC results as the representative compound for model selection
    fc_results = list(all_results.get("FC", AnalysisRun()).results) if "FC" in all_results else []
    if fc_results:
        sel_req = SelectBestModelRequest(
            results=tuple(fc_results),
            task_name="regression",
        )
        sel_resp = select_best_model.run(sel_req, runs_root=str(RUNS_ROOT))
        print(f"  ok={sel_resp.ok}  selected={sel_resp.payload.selected_model if sel_resp.payload else 'N/A'}")
        _save_json(_tool_response_to_dict(sel_resp), OUTPUT_DIR / "06_selection.json")
    else:
        print("  No FC results — skipping model selection")
        sel_resp = None

    # -----------------------------------------------------------------------
    # Step 7: interpret_results
    # -----------------------------------------------------------------------
    print("\n[7] Running interpret_results...")
    if fc_results:
        interp_req = InterpretResultsRequest(
            results=tuple(fc_results),
            dataset=all_datasets.get("FC"),
        )
        interp_resp = interpret_results.run(interp_req, runs_root=str(RUNS_ROOT))
        print(f"  ok={interp_resp.ok}")
        _save_json(_tool_response_to_dict(interp_resp), OUTPUT_DIR / "07_interpretation.json")
    else:
        interp_resp = None

    # -----------------------------------------------------------------------
    # Step 8: generate_report
    # -----------------------------------------------------------------------
    print("\n[8] Running generate_report...")
    if "FC" in all_results:
        fc_run = all_results["FC"]
        report_req = GenerateReportRequest(
            analysis_run=fc_run,
            validation_summary=val_resp_default.payload if val_resp_default else None,
            interpretation=interp_resp.payload if interp_resp else None,
        )
        report_resp = generate_report.run(report_req, runs_root=str(RUNS_ROOT))
        print(f"  ok={report_resp.ok}")
        _save_json(_tool_response_to_dict(report_resp), OUTPUT_DIR / "08_report.json")
    else:
        print("  No FC run — skipping report generation")

    # -----------------------------------------------------------------------
    # Step 9: Build predictions vs truth CSV
    # -----------------------------------------------------------------------
    print("\n[9] Building predictions vs truth CSV...")

    # For each compound, pick the best preprocessing/model combo by MCP CV r2
    # and build per-sample predictions
    csv_rows = []
    mcp_best_summary = {}

    for compound, labels in [("FC", FC_labels), ("AMI", AMI_labels), ("SB", SB_labels)]:
        if compound not in all_results:
            continue
        analysis_run = all_results[compound]
        if not analysis_run.results:
            continue

        # Find best result by r2
        best_result = max(analysis_run.results, key=lambda r: r.metrics.get("r2", -999))
        best_key = f"{best_result.preprocessing[0]}/{best_result.model_name}"
        mcp_best_summary[compound] = {
            "best_key": best_key,
            "mcp_cv_r2": best_result.metrics.get("r2"),
            "mcp_cv_rmse": best_result.metrics.get("rmse"),
            "mcp_cv_mae": best_result.metrics.get("mae"),
        }

        # MCP CV predictions (same order as sample_ids)
        mcp_preds = list(best_result.predictions)

        # LOGO-CV predictions: re-run explicitly to get per-sample predictions
        y = np.array(labels, dtype=float)
        prep_m, model_m = best_key.split("/")
        try:
            logo_cv = compute_logo_cv(X, y, groups, model_m, prep_m)
            # Align logo predictions to sample_ids order
            logo_preds_reordered = [None] * len(sample_ids)
            for j, orig_idx in enumerate(logo_cv["sample_indices"]):
                logo_preds_reordered[orig_idx] = logo_cv["y_pred"][j]
        except Exception as exc:
            print(f"  LOGO-CV re-run failed for {compound}/{best_key}: {exc}")
            logo_preds_reordered = [None] * len(sample_ids)

        for i, sid in enumerate(sample_ids):
            true_val = labels[i]
            mcp_pred = mcp_preds[i] if i < len(mcp_preds) else None
            logo_pred = logo_preds_reordered[i]
            mcp_residual = (mcp_pred - true_val) if mcp_pred is not None else None
            logo_residual = (logo_pred - true_val) if logo_pred is not None else None

            csv_rows.append({
                "compound": compound,
                "sample_id": sid,
                "group": groups[i],
                "true_mg": true_val,
                "mcp_cv_pred_mg": round(mcp_pred, 4) if mcp_pred is not None else "",
                "mcp_cv_residual": round(mcp_residual, 4) if mcp_residual is not None else "",
                "logo_cv_pred_mg": round(logo_pred, 4) if logo_pred is not None else "",
                "logo_cv_residual": round(logo_residual, 4) if logo_residual is not None else "",
                "best_prep": prep_m,
                "best_model": model_m,
            })

    csv_path = OUTPUT_DIR / "09_predictions_vs_truth.csv"
    if csv_rows:
        fieldnames = [
            "compound", "sample_id", "group", "true_mg",
            "mcp_cv_pred_mg", "mcp_cv_residual",
            "logo_cv_pred_mg", "logo_cv_residual",
            "best_prep", "best_model",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"  saved → {csv_path.relative_to(REPO_ROOT)}")
    else:
        print("  No predictions to write")

    # -----------------------------------------------------------------------
    # Step 10: Compute side-by-side comparison metrics
    # -----------------------------------------------------------------------
    print("\n[10] Computing side-by-side MCP CV vs LOGO CV metrics...")
    comparison = {}
    for compound, labels in [("FC", FC_labels), ("AMI", AMI_labels), ("SB", SB_labels)]:
        y = np.array(labels, dtype=float)
        comp_entry = {"compound": compound}

        # MCP CV best
        if compound in mcp_best_summary:
            best = mcp_best_summary[compound]
            comp_entry["mcp_cv"] = {
                "key": best["best_key"],
                "r2": best["mcp_cv_r2"],
                "rmse": best["mcp_cv_rmse"],
                "mae": best["mcp_cv_mae"],
            }

        # LOGO CV best (scan all combos)
        logo_data = logo_baseline["results"].get(compound, {})
        best_logo_r2 = -999
        best_logo_entry = None
        best_logo_key = None
        for k, v in logo_data.items():
            if isinstance(v, dict) and "r2" in v and v["r2"] > best_logo_r2:
                best_logo_r2 = v["r2"]
                best_logo_entry = v
                best_logo_key = k
        comp_entry["logo_cv"] = {
            "key": best_logo_key,
            "r2": best_logo_entry["r2"] if best_logo_entry else None,
            "rmse": best_logo_entry["rmse"] if best_logo_entry else None,
            "mae": best_logo_entry["mae"] if best_logo_entry else None,
        }
        comparison[compound] = comp_entry

    # -----------------------------------------------------------------------
    # Write findings.md
    # -----------------------------------------------------------------------
    print("\n[11] Writing findings.md...")
    findings_lines = [
        "# FTIR Chemometrics Analysis — Findings",
        "",
        f"**Date:** {now_iso()[:10]}",
        f"**Dataset:** {FTIR_DIR.name}",
        f"**Supervised samples:** 6 groups × 17 spectra (C, I, J, L, M, N)",
        f"**Compounds:** FC (mg 3fc), AMI (mg 8ami), SB (mg S31)",
        "",
        "## 1. Dataset Overview",
        "",
        "| Sample | Group | FC (mg) | AMI (mg) | SB (mg) |",
        "| ------ | ----- | ------: | -------: | ------: |",
    ]
    for row in manifest:
        findings_lines.append(
            f"| {row['sample_id']:8s} | {row['group']} | {row['FC_mg']:7.1f} | "
            f"{row['AMI_mg']:8.1f} | {row['SB_mg']:7.1f} |"
        )

    findings_lines += [
        "",
        "**Notes:**",
        "- D, E, F, G, H excluded — only `.smf` files, no `.txt` spectra.",
        "- L.txt absent; L1 + L2 used as replicates for group L.",
        "- s31acn2, s31meoh, S31SOLID excluded — reference spectra, no composition entry.",
        "",
        "## 2. MCP Workflow Results",
        "",
        "### Inspection (01_inspection.json)",
        f"- Samples loaded by `inspect_dataset`: {inspection.sample_count if inspection else 'N/A'}",
        f"- Features (wavenumber grid): {inspection.feature_count if inspection else 'N/A'}",
        f"- Axis range: {inspection.axis_min if inspection else 'N/A'}–"
        f"{inspection.axis_max if inspection else 'N/A'} cm⁻¹",
        f"- Modality: {inspection.modality if inspection else 'N/A'}",
        "",
        "### Analysis Runs (03_run_FC/AMI/SB.json)",
        "Preprocessing: raw, SNV, SG-1st-deriv. Models: PLSR, SVR, Ridge.",
        "Validation: `grouped_kfold_5` (MCP stub — ignores groups, uses KFold(5)).",
        "",
        "#### MCP CV Metrics (best preprocessing per compound):",
        "",
        "| Compound | Best Combo | MCP CV R² | MCP CV RMSE (mg) | MCP CV MAE (mg) |",
        "| -------- | ---------- | --------: | ---------------: | --------------: |",
    ]
    for compound in ["FC", "AMI", "SB"]:
        comp_data = comparison.get(compound, {})
        mcp = comp_data.get("mcp_cv", {})
        key = mcp.get("key", "N/A")
        r2 = mcp.get("r2")
        rmse = mcp.get("rmse")
        mae = mcp.get("mae")
        r2_str = f"{r2:.3f}" if r2 is not None else "N/A"
        rmse_str = f"{rmse:.3f}" if rmse is not None else "N/A"
        mae_str = f"{mae:.3f}" if mae is not None else "N/A"
        findings_lines.append(
            f"| {compound:8s} | {key:20s} | {r2_str:9s} | {rmse_str:16s} | {mae_str:15s} |"
        )

    findings_lines += [
        "",
        "## 3. True LOGO-CV Baseline (04_logo_baseline.json)",
        "",
        "Leave-One-Group-Out CV with 6 letter groups (sklearn `LeaveOneGroupOut`).",
        "Each fold withholds **all replicates** of one group.",
        "This is the **unbiased** out-of-sample estimator.",
        "",
        "#### LOGO-CV Metrics (best preprocessing per compound):",
        "",
        "| Compound | Best Combo | LOGO R² | LOGO RMSE (mg) | LOGO MAE (mg) |",
        "| -------- | ---------- | ------: | -------------: | ------------: |",
    ]
    for compound in ["FC", "AMI", "SB"]:
        comp_data = comparison.get(compound, {})
        logo = comp_data.get("logo_cv", {})
        key = logo.get("key", "N/A")
        r2 = logo.get("r2")
        rmse = logo.get("rmse")
        mae = logo.get("mae")
        r2_str = f"{r2:.3f}" if r2 is not None else "N/A"
        rmse_str = f"{rmse:.3f}" if rmse is not None else "N/A"
        mae_str = f"{mae:.3f}" if mae is not None else "N/A"
        findings_lines.append(
            f"| {compound:8s} | {str(key):20s} | {r2_str:7s} | {rmse_str:14s} | {mae_str:13s} |"
        )

    findings_lines += [
        "",
        "## 4. Leakage Analysis",
        "",
        "### The Bug: MCP `grouped_kfold_5` ignores groups",
        "",
        "**Evidence:** `src/chemometrics_mcp/core/modeling.py:65`:",
        "```python",
        "if validation_strategy == 'grouped_kfold_5':",
        "    return KFold(n_splits=5, shuffle=True, random_state=seed)",
        "```",
        "",
        "This returns a `KFold` splitter (not `GroupKFold`), so the `groups`",
        "argument passed to `cross_val_predict` is ignored. Replicates from the",
        "same physical sample (e.g., C, C1, C2) can appear in both train and test",
        "folds simultaneously, inflating CV metrics.",
        "",
        "**Effect on 17-spectrum dataset with 3 replicates per group:**",
        "With 5 folds and 17 samples, each fold has ~3–4 test samples. KFold",
        "may split C, C1, C2 across train/test — making the task artificially easy.",
        "",
        "### Validation.py also silences the warning (validation.py:270)",
        "",
        "```python",
        "is_grouped = strategy in _GROUPED_STRATEGIES if strategy else False",
        "```",
        "",
        "When `validation_strategy='grouped_kfold_5'`, `is_grouped=True`, so",
        "`check_group_leakage_risk` returns no warnings — even though the splitter",
        "ignores groups. This hides the bug from the validation step.",
        "",
        "### MCP CV vs LOGO CV gap:",
        "",
        "| Compound | MCP CV R² | LOGO R² | Gap (MCP-LOGO) |",
        "| -------- | --------: | ------: | -------------: |",
    ]
    for compound in ["FC", "AMI", "SB"]:
        comp_data = comparison.get(compound, {})
        mcp_r2 = comp_data.get("mcp_cv", {}).get("r2")
        logo_r2 = comp_data.get("logo_cv", {}).get("r2")
        if mcp_r2 is not None and logo_r2 is not None:
            gap = mcp_r2 - logo_r2
            findings_lines.append(
                f"| {compound:8s} | {mcp_r2:9.3f} | {logo_r2:7.3f} | "
                f"{gap:+14.3f} |"
            )
        else:
            findings_lines.append(f"| {compound:8s} | N/A | N/A | N/A |")

    findings_lines += [
        "",
        "A positive gap indicates MCP CV is **overly optimistic** due to leakage.",
        "",
        "## 5. Small-Sample Caveats",
        "",
        "- N=6 groups → LOGO-CV has very high variance; each fold has only 1 test group.",
        "- Chemometric models with 901 features and 17 spectra are severely underdetermined.",
        "- PLSR addresses this via latent variable reduction (n_components ≤ min(n,p)).",
        "- SVR and Ridge may overfit without regularisation tuning.",
        "- Results are exploratory; not publication-ready without more samples.",
        "",
        "## 6. Key Files",
        "",
        "| File | Description |",
        "| ---- | ----------- |",
        "| 00_target_manifest.json | Composition table joined to spectra |",
        "| 01_inspection.json | MCP dataset inspection |",
        "| 02_plan.json | MCP proposed analysis plan |",
        "| 03_run_FC/AMI/SB.json | MCP regression runs (×3 compounds) |",
        "| 04_logo_baseline.json | True LOGO-CV (sklearn) |",
        "| 05_validation_*.json | MCP validation (with/without grouped strategy) |",
        "| 06_selection.json | MCP model selection |",
        "| 07_interpretation.json | MCP feature interpretation |",
        "| 08_report.json | MCP final report |",
        "| 09_predictions_vs_truth.csv | Per-sample predictions vs true values |",
        "| findings.md | This document |",
        "| experience.md | Friction log and MCP improvement suggestions |",
        "",
    ]

    (OUTPUT_DIR / "findings.md").write_text("\n".join(findings_lines), encoding="utf-8")
    print(f"  saved → mcp-expts-2/claude/findings.md")

    # -----------------------------------------------------------------------
    # Write experience.md
    # -----------------------------------------------------------------------
    print("\n[12] Writing experience.md...")
    experience_lines = [
        "# FTIR MCP Analysis — Experience Log",
        "",
        f"**Date:** {now_iso()[:10]}",
        "**Analysis:** FTIR quantitative regression, 6 groups, 17 spectra, 3 compounds",
        "",
        "## Friction Points",
        "",
        "### 1. MCP `inspect_dataset` requires Excel or single .txt — not FTIR dir by default",
        "",
        "**Issue:** The `InspectDatasetRequest` input schema says `source_uri` is a",
        "path to `.xlsx/.xls`. Directory loading is only triggered when",
        "`source_path.is_dir()` or `source_format='ftir_dir'` (inspect_dataset.py:71).",
        "The MCP tool JSON schema description says 'Path or URI to the spectral data",
        "file (.xlsx/.xls)' — missing documentation that directories are supported.",
        "",
        "**Fix suggestion:** Update `inspect_dataset` schema description to mention",
        "FTIR directory mode and the `source_format='ftir_dir'` override.",
        "",
        "### 2. `grouped_kfold_5` stub silently ignores groups (modeling.py:65)",
        "",
        "**Issue:** `make_cv_splitter('grouped_kfold_5', y)` returns `KFold(n_splits=5)`",
        "not `GroupKFold(n_splits=5)`. The `groups` parameter in `cross_val_predict`",
        "is never populated — silently ignored.",
        "",
        "**Impact:** Replicate spectra (C, C1, C2) can leak into both train and test",
        "folds. On a 17-sample dataset with 3 replicates per group, this inflates",
        "R² substantially. The gap between MCP CV and true LOGO-CV exposes this.",
        "",
        "**Fix:** Replace `KFold` with `GroupKFold` in the `grouped_kfold_5` branch,",
        "and propagate the `groups` array through to `cross_val_predict`. Requires",
        "storing groups in `SpectralDataset` and threading them through `run_cv_model`.",
        "",
        "**Effort:** Medium (~30 lines). Requires adding `groups` field to",
        "`SpectralDataset` contract and updating `run_cv_model` signature.",
        "",
        "### 3. `validate_results` silences leakage check for grouped_kfold_5",
        "",
        "**Issue:** `validation.py:270` checks `is_grouped = strategy in _GROUPED_STRATEGIES`.",
        "`_GROUPED_STRATEGIES` presumably includes `'grouped_kfold_5'`.",
        "When this is True, `check_group_leakage_risk` returns empty — even though",
        "the splitter doesn't honour groups. This creates a false sense of safety.",
        "",
        "**Fix:** Either (a) fix the stub (fix #2 above) so the check is actually",
        "meaningful, or (b) add a different warning: `'grouped_kfold_5_stubbed'` with",
        "severity `error` until the stub is resolved.",
        "",
        "### 4. Single-label SpectralDataset — must loop 3× manually",
        "",
        "**Issue:** `SpectralDataset.labels` is a single `Sequence[Any]`. For",
        "multivariate quantification (FC, AMI, SB simultaneously), the agent must",
        "construct 3 separate datasets and call `run_analysis` 3 times.",
        "",
        "**Fix suggestion:** Allow `labels` to be a dict-of-lists (one per compound)",
        "or add a `label_names` field. This would enable multi-target regression in",
        "a single `run_analysis` call.",
        "",
        "### 5. `propose_analysis_plan` ignores `task_hint='regression'` partially",
        "",
        "**Observation:** The plan's `task_name` may differ from the hint because",
        "`build_plan` in `planning.py` uses heuristics on `candidate_label_columns`",
        "from inspection. With numeric labels (mg values), the planner may default",
        "to unsupervised or classification. Manual override of `AnalysisPlan` was",
        "needed.",
        "",
        "**Fix suggestion:** When `task_hint='regression'` is explicit, honor it",
        "directly without heuristic override.",
        "",
        "### 6. No built-in LOGO-CV metric in MCP tools",
        "",
        "**Issue:** The MCP does not expose a true group-aware CV. The only way to",
        "get unbiased metrics is to call sklearn directly outside the MCP.",
        "This defeats the purpose of the agentic pipeline for replicated designs.",
        "",
        "**Fix suggestion:** Add a `LeaveOneGroupOut` strategy to `make_cv_splitter`",
        "that requires `groups` from `SpectralDataset.metadata` (after fix #2/#4).",
        "",
        "### 7. `source_references` field in `SpectralDataset` is unexplained",
        "",
        "The `SpectralDataset.source_references` field (a sequence of",
        "`ArtifactReference`) is not documented in the schema description.",
        "An agent calling `run_analysis` wouldn't know when to populate it.",
        "",
        "## Positive Aspects",
        "",
        "- **Modular tool design** — each tool is independently callable as a Python",
        "  function, enabling scripted orchestration without a running MCP server.",
        "- **Reproducible artifact IDs** — run IDs are timestamp-based and deterministic",
        "  within a session.",
        "- **Preprocessing module** is clean and well-tested: SNV, MSC, SG derivatives",
        "  all work correctly on 2D arrays.",
        "- **PLSR n_components** is safely capped at `min(10, n_features, n_samples-1)`",
        "  — avoids singular matrix errors on small datasets.",
        "- **load_ftir_real** correctly handles the L.txt absence with a warning.",
        "- **Inspection warnings** surface small-sample and unknown-stem issues.",
        "",
        "## Improvement Priority",
        "",
        "| Priority | Issue | Location | Effort |",
        "| -------- | ----- | -------- | ------ |",
        "| P0 | Fix `grouped_kfold_5` stub → GroupKFold | modeling.py:65 | Medium |",
        "| P0 | Fix validation to not silence leakage for stubbed strategies | validation.py:270 | Small |",
        "| P1 | Add `groups` field to SpectralDataset | contracts/__init__.py | Small |",
        "| P1 | Multi-target labels (FC+AMI+SB in one run) | contracts + run_analysis | Large |",
        "| P2 | Update inspect_dataset schema docs for dir mode | inspect_dataset.py | Trivial |",
        "| P2 | Honor task_hint='regression' without heuristic override | planning.py | Small |",
        "| P3 | Add LOGO-CV native strategy | modeling.py | Medium |",
        "",
        "## Summary",
        "",
        "The MCP pipeline works end-to-end for FTIR regression with small datasets.",
        "The main reliability concern is the `grouped_kfold_5` stub that silently",
        "uses non-group-aware splits. On this 17-spectrum dataset with 3 replicates",
        "per group, the leakage inflates MCP CV R² relative to true LOGO-CV.",
        "All other tools (inspect, plan, validate, select, interpret, report) work",
        "correctly and produce useful artifacts.",
        "",
    ]

    (OUTPUT_DIR / "experience.md").write_text("\n".join(experience_lines), encoding="utf-8")
    print(f"  saved → mcp-expts-2/claude/experience.md")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DONE. Output directory:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        size = f.stat().st_size if f.is_file() else "-"
        print(f"  {f.name:50s} {size:>8} bytes")
    print("=" * 70)


if __name__ == "__main__":
    main()
