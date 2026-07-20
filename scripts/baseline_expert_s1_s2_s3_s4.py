"""Expert baseline pipelines for evaluation scenarios S1–S4.

Each function runs a minimal expert pipeline seeded at seed=42, returns metrics
+ wall_clock_s. These baselines are the reference for agent performance.

Scenarios
---------
S1 — wear-layer regression (vinyl NIR, ~90 samples, 3 classes of WL thickness)
S2 — species classification (lumber NIR, ~44 samples, 6 species)
S3 — binary classification (vinyl vs. wood NIR, from nir_first_pass.py)
S4 — FTIR purity (real FTIR, ~20 samples, group-based labels)

Usage
-----
    python scripts/baseline_expert_s1_s2_s3_s4.py \
        --workbook PATH_TO_XLSX \
        --ftir-dir PATH_TO_FTIR_DIR \
        [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Shared SNV transformer (copied from nir_first_pass for zero imports from it)
# ---------------------------------------------------------------------------
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class SNV(BaseEstimator, TransformerMixin):
    """Standard Normal Variate preprocessing."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, ddof=1, keepdims=True)
        return (X - mean) / np.where(std == 0, 1, std)


# ---------------------------------------------------------------------------
# Helper: load flooring NIR workbook
# ---------------------------------------------------------------------------

def _load_flooring_nir(workbook_path: Path) -> tuple[Any, np.ndarray, np.ndarray]:
    """Load the flooring NIR Excel and return (metadata_df, X, wavelengths)."""
    import pandas as pd

    xl = pd.ExcelFile(workbook_path)
    meta = xl.parse("Spectra Metadata")
    spectra_raw = xl.parse([s for s in xl.sheet_names if "spectra" in s.lower() and "metadata" not in s.lower()][0])
    wavelengths = spectra_raw.iloc[:, 0].to_numpy(dtype=float)
    X_df = spectra_raw.iloc[:, 1:].T
    X_df.index = X_df.index.astype(int)
    meta = meta.set_index("Spectrum ID").loc[X_df.index].reset_index()
    X = X_df.to_numpy(dtype=float)
    return meta, X, wavelengths


# ---------------------------------------------------------------------------
# S1 — Wear-layer regression (vinyl only)
# ---------------------------------------------------------------------------

def run_s1_baseline(workbook_path: str | Path, seed: int = 42) -> dict[str, Any]:
    """Expert pipeline: SNV + Ridge, LeaveOneGroupOut on sample description.

    Returns metrics: r2, rmse, mae, best_cv, wall_clock_s, tool_calls_estimate.
    """
    import re
    t0 = time.perf_counter()

    meta, X, _ = _load_flooring_nir(Path(workbook_path))

    # Filter to vinyl samples with a parsed wear-layer thickness
    def _parse_wl(desc: str):
        m = re.search(r"\bwl\s+(\d+)", str(desc).lower())
        return int(m.group(1)) if m else None

    desc_col = "Measurement Description"
    wl_vals = np.array([_parse_wl(d) for d in meta[desc_col]])
    vinyl_mask = np.array([str(d).lower().startswith("vinyl") for d in meta[desc_col]])
    keep = vinyl_mask & (wl_vals != None)  # noqa: E711
    keep = np.array([bool(k) for k in keep])

    X_s1 = X[keep]
    y_s1 = wl_vals[keep].astype(float)
    groups = meta.loc[keep, desc_col].to_numpy()

    # Expert model: SNV → Ridge
    pipe = Pipeline([
        ("snv", SNV()),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])

    # LeaveOneGroupOut (expert strategy)
    logo = LeaveOneGroupOut()
    n_groups = len(set(groups))

    if n_groups >= 2:
        y_pred = cross_val_predict(pipe, X_s1, y_s1, cv=logo, groups=groups)
        cv_name = "LeaveOneGroupOut"
    else:
        # Fallback to 5-fold if not enough groups
        y_pred = cross_val_predict(pipe, X_s1, y_s1, cv=KFold(n_splits=5, shuffle=True, random_state=seed))
        cv_name = "KFold(5)"

    r2 = float(r2_score(y_s1, y_pred))
    rmse = float(np.sqrt(np.mean((y_s1 - y_pred) ** 2)))
    mae = float(mean_absolute_error(y_s1, y_pred))

    wall_clock_s = time.perf_counter() - t0

    return {
        "scenario": "S1_wear_layer",
        "task": "regression",
        "model": "Ridge(alpha=1.0)",
        "preprocessing": "SNV + StandardScaler",
        "cv_strategy": cv_name,
        "n_samples": int(X_s1.shape[0]),
        "n_features": int(X_s1.shape[1]),
        "metrics": {
            "r2": round(r2, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
        },
        "wall_clock_s": round(wall_clock_s, 3),
        "tool_calls_estimate": 1,
        "caveats": [
            "LeaveOneGroupOut on description strings — same description = same group (may not be physical sample groups)",
            "Ridge alpha=1.0 not tuned — expert Occam's razor for small n",
        ],
    }


# ---------------------------------------------------------------------------
# S2 — Species classification (lumber only)
# ---------------------------------------------------------------------------

def run_s2_baseline(workbook_path: str | Path, seed: int = 42) -> dict[str, Any]:
    """Expert pipeline: SNV + RandomForest, StratifiedKFold(5).

    Returns metrics: accuracy, balanced_accuracy, f1_weighted, wall_clock_s.
    """
    import re
    t0 = time.perf_counter()

    meta, X, _ = _load_flooring_nir(Path(workbook_path))

    _SPECIES_MAP = [
        ("particle board", "particle_board"),
        ("particle_board", "particle_board"),
        ("mahogany", "mahogany"),
        ("poplar", "poplar"),
        ("pine", "pine"),
        ("fir", "fir"),
        ("oak", "oak"),
    ]

    def _parse_species(desc: str):
        lower = str(desc).lower()
        if not lower.startswith("lumber"):
            return None
        parts = [p.strip() for p in lower.split(",")]
        if len(parts) < 2:
            return None
        raw = parts[1]
        for keyword, canonical in _SPECIES_MAP:
            if keyword in raw:
                return canonical
        return None

    desc_col = "Measurement Description"
    species = np.array([_parse_species(d) for d in meta[desc_col]])
    keep = np.array([s is not None for s in species])

    X_s2 = X[keep]
    y_s2 = species[keep].astype(str)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y_s2)

    # Expert: SNV → RandomForest with stratified 5-fold
    pipe = Pipeline([
        ("snv", SNV()),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    y_pred_enc = cross_val_predict(pipe, X_s2, y_enc, cv=cv)
    y_pred = le.inverse_transform(y_pred_enc)

    acc = float(accuracy_score(y_s2, y_pred))
    bal_acc = float(balanced_accuracy_score(y_s2, y_pred))
    f1_w = float(f1_score(y_s2, y_pred, average="weighted", zero_division=0))

    wall_clock_s = time.perf_counter() - t0

    from collections import Counter
    class_counts = Counter(y_s2.tolist())
    small_classes = [cls for cls, n in class_counts.items() if n < 5]

    return {
        "scenario": "S2_species",
        "task": "multi_class_classification",
        "model": "RandomForestClassifier(n_estimators=100)",
        "preprocessing": "SNV",
        "cv_strategy": "StratifiedKFold(5)",
        "n_samples": int(X_s2.shape[0]),
        "n_features": int(X_s2.shape[1]),
        "n_classes": int(len(set(y_s2))),
        "metrics": {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "f1_weighted": round(f1_w, 4),
        },
        "class_counts": {k: int(v) for k, v in class_counts.items()},
        "small_classes_n_lt_5": small_classes,
        "wall_clock_s": round(wall_clock_s, 3),
        "tool_calls_estimate": 1,
        "caveats": [
            "Class imbalance likely — some species have < 5 samples",
            "Stratified 5-fold ignores any physical replicate grouping",
        ],
    }


# ---------------------------------------------------------------------------
# S3 — Binary classification (vinyl vs. wood, from nir_first_pass)
# ---------------------------------------------------------------------------

def run_s3_baseline(workbook_path: str | Path, seed: int = 42) -> dict[str, Any]:
    """Adapt nir_first_pass.py; capture best balanced_accuracy under LeaveOneDescriptionOut."""
    t0 = time.perf_counter()

    # Add the 09- utils directory to path if needed
    # workbook lives in project root, so parent = project root
    utils_dir = Path(workbook_path).parent / "09-can-llms-be-used-for-chemometrics" / "utils"
    if utils_dir.exists():
        sys.path.insert(0, str(utils_dir))

    from nir_first_pass import load_workbook, evaluate_models

    meta, spectra, wavelengths = load_workbook(Path(workbook_path))

    valid = meta["material_family"].isin(["vinyl", "wood_lumber"])
    X_valid = spectra.loc[valid.to_numpy()].to_numpy(dtype=float)
    y_valid = (meta.loc[valid, "material_family"] == "vinyl").astype(int).to_numpy()
    groups = meta.loc[valid, "Measurement Description"].to_numpy()

    results = evaluate_models(X_valid, y_valid, groups)

    # Find best balanced_accuracy under LeaveOneDescriptionOut
    lodo_results = results.get("leave_one_description_out_cv", {})
    best_model = None
    best_bal_acc = -1.0
    for model_name, metrics in lodo_results.items():
        ba = metrics.get("balanced_accuracy")
        if ba is not None and ba > best_bal_acc:
            best_bal_acc = ba
            best_model = model_name

    wall_clock_s = time.perf_counter() - t0

    return {
        "scenario": "S3_binary",
        "task": "binary_classification",
        "model": best_model,
        "cv_strategy": "LeaveOneDescriptionOut",
        "n_samples": int(X_valid.shape[0]),
        "n_features": int(X_valid.shape[1]),
        "best_metrics": {
            "balanced_accuracy": round(best_bal_acc, 4) if best_bal_acc >= 0 else None,
        },
        "all_model_results_lodo": {
            name: m.get("balanced_accuracy") for name, m in lodo_results.items()
        },
        "wall_clock_s": round(wall_clock_s, 3),
        "tool_calls_estimate": 1,
        "caveats": [
            "Groups = full description strings — same vinyl brand + WL = same group",
            "Baseline predates current agentic workflow; comparison is indicative only",
        ],
    }


# ---------------------------------------------------------------------------
# S4 — FTIR purity (real FTIR, GroupKFold on purity_group)
# ---------------------------------------------------------------------------

def run_s4_baseline(ftir_dir: str | Path, seed: int = 42) -> dict[str, Any]:
    """Expert pipeline: SNV + LogisticRegression, GroupKFold(n_splits=min(n_groups, 5)).

    Returns metrics: accuracy, balanced_accuracy, wall_clock_s.
    """
    t0 = time.perf_counter()

    from scipy.interpolate import interp1d

    ftir_path = Path(ftir_dir)
    txt_files = sorted(ftir_path.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {ftir_path}")

    _MANIFEST = {
        "C": "C", "C1": "C", "C2": "C",
        "I": "I", "I1": "I", "I2": "I",
        "J": "J", "J1": "J", "J2": "J",
        "L1": "L", "L2": "L",
        "M": "M", "M1": "M", "M2": "M",
        "N": "N", "N1": "N", "N2": "N",
        "s31acn2": "s31", "s31meoh": "s31", "S31SOLID": "s31",
    }

    n_grid = 901
    common_axis = np.linspace(400.0, 4000.0, n_grid)
    rows, labels_list, group_list = [], [], []

    for f in txt_files:
        stem = f.stem
        purity = _MANIFEST.get(stem, "unknown")
        wns, trs = [], []
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    wns.append(float(parts[0]))
                    trs.append(float(parts[1]))
                except ValueError:
                    continue
        if len(wns) < 2:
            continue
        wns_arr, trs_arr = np.array(wns), np.array(trs)
        interp_fn = interp1d(wns_arr, trs_arr, kind="linear", bounds_error=False,
                             fill_value=(trs_arr[0], trs_arr[-1]))
        rows.append(interp_fn(common_axis))
        labels_list.append(purity)
        group_list.append(purity)  # group = purity label (same group = same compound)

    X_s4 = np.vstack(rows)
    y_s4 = np.array(labels_list)
    groups_arr = np.array(group_list)

    n_groups = len(set(group_list))
    n_splits = min(n_groups, 5)

    pipe = Pipeline([
        ("snv", SNV()),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=seed, solver="lbfgs")),
    ])

    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        y_pred = cross_val_predict(pipe, X_s4, y_s4, cv=gkf, groups=groups_arr)
        cv_name = f"GroupKFold(n_splits={n_splits})"
    else:
        y_pred = cross_val_predict(pipe, X_s4, y_s4, cv=KFold(n_splits=2, shuffle=True, random_state=seed))
        cv_name = "KFold(2) — fallback: too few groups"

    acc = float(accuracy_score(y_s4, y_pred))
    bal_acc = float(balanced_accuracy_score(y_s4, y_pred))

    wall_clock_s = time.perf_counter() - t0

    from collections import Counter
    class_counts = Counter(y_s4.tolist())

    return {
        "scenario": "S4_ftir_purity",
        "task": "multi_class_classification",
        "model": "LogisticRegression(C=1.0)",
        "preprocessing": "SNV + StandardScaler",
        "cv_strategy": cv_name,
        "n_samples": int(X_s4.shape[0]),
        "n_features": int(X_s4.shape[1]),
        "n_groups": int(n_groups),
        "metrics": {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
        },
        "class_counts": {k: int(v) for k, v in class_counts.items()},
        "wall_clock_s": round(wall_clock_s, 3),
        "tool_calls_estimate": 1,
        "caveats": [
            "n=~20 very small — all metrics unreliable; treat as proof-of-concept only",
            "GroupKFold groups = purity labels; n_splits = min(n_groups, 5)",
            "Compound identity (letter codes → AMI/FC/SB) not confirmed — labels are purity group codes",
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run expert baselines for S1–S4 evaluation scenarios.")
    parser.add_argument("--workbook", type=Path, default=None,
                        help="Path to flooring NIR Excel workbook (required for S1/S2/S3).")
    parser.add_argument("--ftir-dir", type=Path, default=None,
                        help="Path to FTIR .txt directory (required for S4).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenarios", nargs="+", default=["s1", "s2", "s3", "s4"],
                        choices=["s1", "s2", "s3", "s4"],
                        help="Scenarios to run.")
    args = parser.parse_args()

    results = {}

    if "s1" in args.scenarios:
        if args.workbook is None:
            print("SKIP S1: --workbook not provided", file=sys.stderr)
        else:
            print("Running S1 baseline (wear-layer regression)...", file=sys.stderr)
            results["s1"] = run_s1_baseline(args.workbook, seed=args.seed)

    if "s2" in args.scenarios:
        if args.workbook is None:
            print("SKIP S2: --workbook not provided", file=sys.stderr)
        else:
            print("Running S2 baseline (species classification)...", file=sys.stderr)
            results["s2"] = run_s2_baseline(args.workbook, seed=args.seed)

    if "s3" in args.scenarios:
        if args.workbook is None:
            print("SKIP S3: --workbook not provided", file=sys.stderr)
        else:
            print("Running S3 baseline (binary classification)...", file=sys.stderr)
            results["s3"] = run_s3_baseline(args.workbook, seed=args.seed)

    if "s4" in args.scenarios:
        if args.ftir_dir is None:
            print("SKIP S4: --ftir-dir not provided", file=sys.stderr)
        else:
            print("Running S4 baseline (FTIR purity)...", file=sys.stderr)
            results["s4"] = run_s4_baseline(args.ftir_dir, seed=args.seed)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
