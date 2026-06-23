from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class SNV(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, ddof=1, keepdims=True)
        return (X - mean) / np.where(std == 0, 1, std)


class SavGolDerivative(BaseEstimator, TransformerMixin):
    def __init__(self, window_length=15, polyorder=2, deriv=1):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return savgol_filter(
            np.asarray(X, dtype=float),
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            axis=1,
            mode="interp",
        )


class PLSDA(BaseEstimator):
    def __init__(self, n_components=5):
        self.n_components = n_components

    def fit(self, X, y):
        self.classes_ = np.asarray([0, 1])
        n_comp = min(self.n_components, X.shape[0] - 1, X.shape[1])
        self.model_ = PLSRegression(n_components=max(1, n_comp), scale=False)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        score = np.asarray(self.model_.predict(X)).ravel()
        score = np.clip(score, 0, 1)
        return np.column_stack([1 - score, score])


def classify_description(description: str) -> str:
    text = str(description).strip().lower()
    if text.startswith("vinyl"):
        return "vinyl"
    if text.startswith("lumber"):
        return "wood_lumber"
    return "control_other"


def parse_description(description: str) -> dict:
    text = str(description).strip()
    low = text.lower()
    out = {"material_family": classify_description(text)}
    if low.startswith("vinyl"):
        parts = [p.strip() for p in text.split(",")]
        out["vinyl_brand"] = parts[1] if len(parts) > 1 else None
        wl = re.search(r"\bwl\s*(\d+)", low)
        out["wear_layer_mil"] = int(wl.group(1)) if wl else None
        price = re.search(r"(\d+(?:\.\d+)?)\s*$", text)
        out["price_usd_sqft"] = float(price.group(1)) if price else None
    elif low.startswith("lumber"):
        parts = [p.strip() for p in text.split(",")]
        out["wood_type"] = parts[1] if len(parts) > 1 else None
    return out


def load_workbook(path: Path):
    meta = pd.read_excel(path, sheet_name="Spectra Metadata")
    spectra_raw = pd.read_excel(path, sheet_name="Spectra P100001492")
    wavelengths = spectra_raw.iloc[:, 0].to_numpy(dtype=float)
    spectra = spectra_raw.iloc[:, 1:].T
    spectra.index = spectra.index.astype(int)
    spectra.columns = wavelengths.astype(int)
    meta = meta.set_index("Spectrum ID").loc[spectra.index].reset_index()
    if "index" in meta.columns and "Spectrum ID" not in meta.columns:
        meta = meta.rename(columns={"index": "Spectrum ID"})
    parsed = pd.DataFrame([parse_description(x) for x in meta["Measurement Description"]])
    meta = pd.concat([meta.reset_index(drop=True), parsed], axis=1)
    return meta, spectra, wavelengths


def cv_metrics(y_true, y_pred, y_score) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics["roc_auc"] = None
    return {k: (None if v is None else round(float(v), 4)) for k, v in metrics.items()}


def evaluate_models(X, y, groups):
    models = {
        "raw_logistic": Pipeline(
            [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"))]
        ),
        "snv_logistic": Pipeline(
            [
                ("snv", SNV()),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
            ]
        ),
        "first_derivative_logistic": Pipeline(
            [
                ("deriv", SavGolDerivative()),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
            ]
        ),
        "snv_first_derivative_logistic": Pipeline(
            [
                ("snv", SNV()),
                ("deriv", SavGolDerivative()),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
            ]
        ),
        "snv_plsda_5lv": Pipeline([("snv", SNV()), ("scale", StandardScaler()), ("model", PLSDA(n_components=5))]),
    }
    strategies = {
        "stratified_5fold_replicate_cv": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        "leave_one_description_out_cv": LeaveOneGroupOut(),
    }
    results = {}
    for cv_name, cv in strategies.items():
        results[cv_name] = {}
        for name, model in models.items():
            if cv_name.startswith("stratified"):
                preds = cross_val_predict(model, X, y, cv=cv, method="predict")
                scores = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
            else:
                splits = cv.split(X, y, groups=groups)
                preds = cross_val_predict(model, X, y, cv=splits, method="predict", groups=groups)
                splits = cv.split(X, y, groups=groups)
                scores = cross_val_predict(model, X, y, cv=splits, method="predict_proba", groups=groups)[:, 1]
            results[cv_name][name] = cv_metrics(y, preds, scores)
    return results


def plot_outputs(meta, spectra, wavelengths, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    X = spectra.to_numpy(dtype=float)
    labels = meta["material_family"].to_numpy()
    colors = {"vinyl": "#2563EB", "wood_lumber": "#16A34A", "control_other": "#DC2626"}

    fig, ax = plt.subplots(figsize=(10, 5))
    for label in ["vinyl", "wood_lumber", "control_other"]:
        mask = labels == label
        mean = X[mask].mean(axis=0)
        std = X[mask].std(axis=0)
        ax.plot(wavelengths, mean, label=f"{label} mean (n={mask.sum()})", color=colors[label])
        ax.fill_between(wavelengths, mean - std, mean + std, color=colors[label], alpha=0.12, linewidth=0)
    ax.set_title("Mean NIR spectra by derived material family")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance / reported y-axis")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "mean_spectra_by_family.png", dpi=180)
    plt.close(fig)

    valid = labels != "control_other"
    X_snv_deriv = SavGolDerivative().fit_transform(SNV().fit_transform(X[valid]))
    scores = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(X_snv_deriv))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for label in ["vinyl", "wood_lumber"]:
        mask = labels[valid] == label
        ax.scatter(scores[mask, 0], scores[mask, 1], s=46, alpha=0.82, label=label, color=colors[label])
    ax.set_title("PCA scores after SNV + first derivative")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "pca_snv_first_derivative.png", dpi=180)
    plt.close(fig)

    diff = X[labels == "vinyl"].mean(axis=0) - X[labels == "wood_lumber"].mean(axis=0)
    top_idx = np.argsort(np.abs(diff))[-15:][::-1]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(wavelengths, diff, color="#111827")
    ax.scatter(wavelengths[top_idx], diff[top_idx], color="#EF4444", s=26, zorder=3)
    ax.axhline(0, color="#9CA3AF", linewidth=1)
    ax.set_title("Vinyl minus wood/lumber mean spectrum")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Mean absorbance difference")
    fig.tight_layout()
    fig.savefig(out_dir / "mean_difference_vinyl_minus_wood.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("can-llms-be-used-for-chemometrics/agent-memory/first_pass"))
    args = parser.parse_args()

    meta, spectra, wavelengths = load_workbook(args.workbook)
    meta["spectrum_mean"] = spectra.mean(axis=1).to_numpy()
    meta["spectrum_std"] = spectra.std(axis=1).to_numpy()
    meta["spectrum_min"] = spectra.min(axis=1).to_numpy()
    meta["spectrum_max"] = spectra.max(axis=1).to_numpy()

    valid = meta["material_family"].isin(["vinyl", "wood_lumber"])
    X_valid = spectra.loc[valid.to_numpy()].to_numpy(dtype=float)
    y_valid = (meta.loc[valid, "material_family"] == "vinyl").astype(int).to_numpy()
    groups = meta.loc[valid, "Measurement Description"].to_numpy()
    results = evaluate_models(X_valid, y_valid, groups)

    pca_raw = PCA(n_components=5, random_state=42).fit(StandardScaler().fit_transform(spectra.to_numpy(dtype=float)))
    pca_snv_deriv = PCA(n_components=5, random_state=42).fit(
        StandardScaler().fit_transform(SavGolDerivative().fit_transform(SNV().fit_transform(X_valid)))
    )
    class_intensity = (
        meta.groupby("material_family")[["spectrum_mean", "spectrum_std", "spectrum_min", "spectrum_max"]]
        .agg(["mean", "std", "min", "max"])
        .round(5)
    )
    by_description = (
        meta.groupby(["material_family", "Measurement Description"])
        .size()
        .rename("n_spectra")
        .reset_index()
        .sort_values(["material_family", "Measurement Description"])
    )
    diff = spectra.loc[meta["material_family"].eq("vinyl").to_numpy()].mean(axis=0).to_numpy() - spectra.loc[
        meta["material_family"].eq("wood_lumber").to_numpy()
    ].mean(axis=0).to_numpy()
    top_diff = [
        {"wavelength_nm": int(wavelengths[i]), "vinyl_minus_wood_mean": round(float(diff[i]), 6)}
        for i in np.argsort(np.abs(diff))[-15:][::-1]
    ]

    labels_valid = meta.loc[valid, "material_family"].to_numpy()
    X_snv_deriv = StandardScaler().fit_transform(SavGolDerivative().fit_transform(SNV().fit_transform(X_valid)))
    silhouette = silhouette_score(X_snv_deriv, labels_valid)

    summary = {
        "workbook": str(args.workbook),
        "sheet_shapes": {"Spectra Metadata": [int(meta.shape[0]), 16], "Spectra P100001492": [249, 147]},
        "wavelength": {
            "count": int(len(wavelengths)),
            "min_nm": int(wavelengths.min()),
            "max_nm": int(wavelengths.max()),
            "step_nm": int(np.unique(np.diff(wavelengths))[0]),
        },
        "derived_class_counts": meta["material_family"].value_counts().to_dict(),
        "description_counts": by_description.to_dict(orient="records"),
        "date_range": [str(meta["Date and Time"].min()), str(meta["Date and Time"].max())],
        "data_quality": {
            "missing_spectral_cells": int(spectra.isna().sum().sum()),
            "duplicated_spectrum_ids": int(meta["Spectrum ID"].duplicated().sum()),
            "metadata_spectra_joined": int(len(meta)),
        },
        "pca_explained_variance_raw_all_classes_first5": [round(float(x), 4) for x in pca_raw.explained_variance_ratio_],
        "pca_explained_variance_snv_derivative_wood_vinyl_first5": [
            round(float(x), 4) for x in pca_snv_deriv.explained_variance_ratio_
        ],
        "silhouette_snv_derivative_wood_vs_vinyl": round(float(silhouette), 4),
        "top_abs_mean_difference_wavelengths": top_diff,
        "class_intensity_summary": json.loads(class_intensity.to_json()),
        "model_results": results,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    by_description.to_csv(args.out_dir / "description_counts.csv", index=False)
    plot_outputs(meta, spectra, wavelengths, args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
