# Agentic Chemometrics Pipeline

This folder summarizes the current NIR flooring analyses and reframes them as the
prototype for a reusable **agentic chemometrics pipeline**: a workflow that reduces the
amount of expert labor needed to move from raw spectral data to a defensible,
human-reviewable analytical method.

The goal is not to replace chemometric expertise. The goal is to automate the repetitive
and first-pass judgment work: data loading, metadata cleanup, preprocessing comparison,
task selection, model benchmarking, validation checks, feature interpretation, and report
generation.

## Pipeline concept

```text
Raw spectral data
  -> ingestion and metadata harmonization
  -> preprocessing candidate generation
  -> task and model planning
  -> analysis execution
  -> validation and leakage checks
  -> interpretation and reporting
  -> human review
  -> method-memory library
```

## Why this matters

Spectral method development often requires many small, expert-guided decisions:

- Which metadata fields are trustworthy labels?
- Which preprocessing methods should be tried?
- Is the task classification, regression, clustering, or anomaly detection?
- Which models are strong baselines, and which are overkill?
- Is validation vulnerable to replicate leakage or group leakage?
- Are the selected wavelengths, peaks, or features chemically plausible?
- What should be reported as a result versus a caveat?

This project turns those decisions into a modular workflow that an agent can execute,
document, and hand back for expert review.

## Current proof of concept

The current case study uses portable near-infrared spectra of flooring materials:

- Material discrimination: hardwood vs. vinyl
- Multi-class classification: product/species/wear-layer groups
- Wood-species classification
- Vinyl wear-layer thickness regression
- Unsupervised clustering and PCA
- Wavelength selection and statistical testing
- Model interpretability and consensus feature importance

The key scientific lesson is that binary hardwood-vs-vinyl separation is easy, but the
pipeline is useful because it recognizes that, escalates to harder tasks, compares model
families, checks validation assumptions, and produces interpretable evidence.

## Existing modules in this repository

The notebooks already act like pipeline modules. The next step is to convert their shared
logic into callable Python components with a standard input/output contract.

| Module | Current source | Role in pipeline |
| --- | --- | --- |
| Summary/reporting | `00_analysis_summary.ipynb` | Consolidates metrics, figures, recommendations, and scientific takeaways |
| SVM/LDA classification | `../01_svm_classification/01_svm_classification.ipynb` | Supervised classification baseline and tuned SVM models |
| XGBoost classification | `../02_xgboost_classification/02_xgboost_classification.ipynb` | Nonlinear classifier plus model-native feature importance |
| KNN classification | `../03_knn_classification/03_knn_classification.ipynb` | Distance-based baseline and metric sensitivity |
| Clustering/PCA | `../04_clustering_analysis/04_clustering_analysis.ipynb` | Unsupervised structure discovery and label sanity checks |
| Feature selection | `../05_feature_selection_rfe/05_feature_selection_rfe.ipynb` | Recursive feature elimination and reduced-wavelength models |
| Regression | `../06_svr_wear_layer/06_svr_wear_layer.ipynb` | Quantitative prediction of vinyl wear-layer thickness |
| Statistical testing | `../07_anova_wavelength_analysis/07_anova_wavelength_analysis.ipynb` | Per-wavelength ANOVA, multiple-testing correction, Tukey HSD |
| Interpretability | `../08_model_interpretability/08_model_interpretability.ipynb` | SHAP, LIME, permutation importance, PDP, consensus importance |
| Agent first pass | `../09-can-llms-be-used-for-chemometrics/` | Early self-contained agentic analysis prototype |

## Proposed standard data contract

Each module should accept a common dataset object:

```python
SpectralDataset(
    X,              # spectra as samples x features
    axis,           # wavelengths, wavenumbers, m/z values, 2-theta values, etc.
    metadata,       # parsed sample metadata
    labels,         # optional task labels
    modality,       # NIR, FTIR, Raman, pXRD, XRF, LIBS, MS, ...
)
```

Each module should return a common result object:

```python
AnalysisResult(
    task_name,
    model_name,
    preprocessing,
    metrics,
    predictions,
    selected_features,
    figures,
    warnings,
    interpretation,
)
```

This makes the workflow composable: the planner can choose modules, the validator can
compare outputs, and the reporter can summarize results without knowing the internals of
each model.

## Modality adapters

The pipeline should have a shared core and modality-specific adapters.

| Modality | Example adapter responsibilities |
| --- | --- |
| NIR / FTIR / Raman | Baseline correction, SNV/MSC, smoothing, derivatives, band assignment |
| pXRD | Background subtraction, peak finding, phase matching, crystallinity metrics |
| XRF / LIBS | Line normalization, elemental feature extraction, matrix-effect warnings |
| MS / GC-MS / LC-MS | Peak picking, alignment, normalization, library matching |

The planner, validation engine, interpretation layer, reporting layer, and method-memory
library can remain mostly modality-agnostic.

## Method-memory library

The self-learning component should store human-reviewed method-development experience,
not every raw automated conclusion.

Potential records:

- Dataset fingerprint and modality
- Task type and label source
- Preprocessing recipes attempted
- Models attempted and their metrics
- Validation design and leakage warnings
- Important wavelengths, peaks, or features
- Figures and generated reports
- Human-approved conclusions
- Known caveats and failure modes

On a new dataset, the agent can query this library to recommend likely preprocessing
recipes, validation strategies, model baselines, and interpretation checks.

## Paper framing

The paper should focus on **expert-effort reduction in spectral method development**.

Possible thesis:

> We convert a collection of chemometric analyses into an agentic, modular pipeline for
> reducing expert labor in spectral method development. The system automates routine
> preprocessing, modeling, validation, interpretation, and report generation while
> preserving human review as the gatekeeper for scientific conclusions.

Suggested title directions:

- Agentic chemometric determination of material identity from spectral data
- An agent-assisted chemometric pipeline for spectral method development
- Reducing expert effort in spectroscopic chemometrics with a modular agentic workflow

## Next implementation steps

1. Extract shared data loading, metadata parsing, SNV, and Savitzky-Golay preprocessing
   into a reusable package.
2. Define the `SpectralDataset` and `AnalysisResult` contracts.
3. Wrap each notebook analysis as a callable module.
4. Add a planner that selects tasks based on available labels and metadata.
5. Add a validation engine with stratified, grouped, and leave-one-group-out options.
6. Add a report generator that writes figures, tables, warnings, and conclusions.
7. Add a method-memory store for reviewed results and reusable recipes.

