# Workflow: FTIR Follow-On Analysis

## Purpose

Run the standard analysis suite on an FTIR dataset, building on prior NIR results.
Produce paper-ready artifacts, agent trace, and a cross-modality comparison report.

## Dataset

- **Modality**: FTIR (typically 400–4000 cm⁻¹ wavenumber range)
- **Primary label column**: to be determined at inspection
- **Device**: to be confirmed at inspection

## What transfers from NIR

- Classification and regression task structure.
- Model families (SVM, random forest, XGBoost, LDA).
- Validation strategy patterns (grouped CV for replicates).
- Interpretation methods (feature importance framing).

## What does NOT transfer

- Preprocessing methods: FTIR requires baseline correction and area normalization
  instead of or in addition to SNV/MSC.
- Feature counts: FTIR and NIR have different spectral resolutions. Direct metric
  comparison is misleading.
- Model hyperparameters: must be re-tuned per modality.
- Wavelength/wavenumber importance: different physical origins.

## FTIR-specific preprocessing

- **baseline_correction**: Removes additive baseline offset common in FTIR.
- **area_normalization**: Corrects for path-length and concentration variations.
- **snv** and **msc**: Still applicable for scatter correction.
- **sg_1st_deriv** / **sg_2nd_deriv**: Applicable but window size must be appropriate
  for FTIR feature spacing.

## Steps

### Step 1 — Inspect FTIR dataset

Call `inspect_dataset` with:
```json
{
  "source_uri": "<path to FTIR data file>",
  "dataset_id": "ftir-<identifier>-v1",
  "modality_override": "FTIR",
  "label_column": "<label column>"
}
```

Confirm: sample count, feature count, FTIR modality, candidate labels found.

### Step 2 — Propose analysis plan

Call `propose_analysis_plan` with the inspection output and:
```json
{
  "user_intent": "Classify/quantify from FTIR spectra, comparing with prior NIR results.",
  "task_hint": "classification"
}
```

Ensure plan includes FTIR-specific preprocessing (baseline_correction, area_normalization).

Present the plan. **[HUMAN APPROVAL GATE]**

### Step 3 — Run analyses

Call `run_analysis` with the approved plan.
- Save all artifacts under a canonical run ID.
- Surface any model failures for fallback handling.

### Step 4 — Validate results

Call `validate_results`. Check for modality-preprocessing consistency.
Carry all warnings forward.

### Step 5 — Compare with NIR results

If NIR results are available:
- Note differences in preprocessing, feature counts, and model configurations.
- Do not directly compare accuracy/R² across modalities without caveats.
- Identify whether the same classes/tasks show consistent patterns.

**[HUMAN APPROVAL GATE before drawing cross-modality conclusions]**

### Step 6 — Select best model

Call `select_best_model`. **[HUMAN APPROVAL GATE if validation warnings exist]**

### Step 7 — Interpret results

Call `interpret_results`. Frame wavenumber importance as model evidence only.
Do not claim chemical causality from feature importance.

### Step 8 — Generate report

Call `generate_report`. If cross-modality results are present, the report will
include a Cross-Modality Comparison section with caveats.

Present human-review checklist.
**[HUMAN APPROVAL GATE for final conclusions]**

### Step 9 — Save artifacts

Confirm these artifacts are saved:
- [ ] Agent plan.
- [ ] Run metadata JSON.
- [ ] Results JSON with metrics tables.
- [ ] Validation report.
- [ ] Interpretation report.
- [ ] Figures (confusion matrices, ROC curves, importance plots, PCA).
- [ ] Final report (Markdown) with cross-modality caveats.
- [ ] Agent trace (decisions, tool calls, failures, fallbacks, approvals).
- [ ] Human-review checklist.

## Prohibited claims

- FTIR results prove NIR results were wrong or right.
- Direct metric comparison across modalities without caveats.
- Wavenumber importance proves chemical identity.
- The system replaces chemometric experts.
