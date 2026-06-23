# NIR Hardwood vs. Vinyl

Chemometric analysis of portable near-infrared (NIR) spectra for **hardwood vs. vinyl**
flooring classification. The dataset comes from a portable NIR instrument (trinamiX,
wavelength range 1454–2446 nm) collected as part of a flooring-materials method-development
study. The primary objective is hardwood-vs-vinyl separation, with subclasses (hardwood
species, vinyl brands / wear-layer grades) explored along the way.

## The data

- **Source workbook:** `2026-05-21_22-59_Method Dev_wavelength_range_1450-2450.xlsx`
  (repo root), two sheets — `Spectra Metadata` and `Spectra P100001492`.
- **Spectra:** 249 wavelength channels, 1454–2446 nm, 4 nm step.
- **Samples:** 146 spectra joined to metadata, collected 2025-06-24 → 2025-09-09.
- **Material families** derived from the `Measurement Description` text (hardwood species,
  vinyl brands / wear layers).

## Repository layout

Each numbered folder is a self-contained analysis notebook plus its exported figures.

| Folder | Contents |
| --- | --- |
| `00_analysis_summary` | Cross-model consensus and overall summary |
| `01_svm_classification` | SVM (vs. LDA) classification |
| `02_xgboost_classification` | XGBoost classification + feature importance |
| `03_knn_classification` | k-NN classification, k and distance-metric tuning |
| `04_clustering_analysis` | KMeans / hierarchical clustering, PCA views |
| `05_feature_selection_rfe` | Recursive feature elimination, RFE vs. VIP |
| `06_svr_wear_layer` | SVR regression on vinyl wear-layer thickness |
| `07_anova_wavelength_analysis` | ANOVA / Tukey HSD per-wavelength analysis |
| `08_model_interpretability` | SHAP, LIME, PDP, permutation importance |
| `09-can-llms-be-used-for-chemometrics` | Exploratory LLM-as-chemometrician sub-project |

Top-level `chris-floor_nir_analysis.ipynb` is the original end-to-end exploratory notebook.

## Setup

Requires Python 3.10.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** `xgboost` is pinned to `2.1.4` for SHAP compatibility. On macOS you may also
> need `libomp` installed (`brew install libomp`).

Launch the notebooks with `jupyter lab` (or open them in VS Code) and run from the repo
root so the relative path to the data workbook resolves.
