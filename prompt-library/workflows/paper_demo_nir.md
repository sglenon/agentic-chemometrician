# Workflow: NIR Flooring Paper Demo

## Purpose

Run the standard analysis suite on the NIR flooring dataset to produce
paper-ready artifacts, agent trace, and final report.

## Dataset

- **File**: `2026-05-21_22-59_Method Dev_wavelength_range_1450-2450.xlsx`
- **Modality**: NIR (1454–2446 nm)
- **Samples**: 146 spectra
- **Features**: 249 wavelength points
- **Primary label column**: `Measurement Description`
  (21 classes: Lumber variants, Vinyl variants, Water bottle, test_lab_door)
- **Device**: SYS-IR-R-P.1 (P100001492)

## Analysis suite

The paper demo should cover:
1. **Binary classification** — flooring vs. non-flooring (or Lumber vs. Vinyl).
2. **Multi-class classification** — all 21 or grouped material classes.
3. **Preprocessing comparison** — raw vs. SNV vs. MSC vs. detrend.
4. **Feature/wavelength importance** — identify discriminating spectral regions.
5. **Model comparison** — SVM, random forest, XGBoost, LDA (where supported).
6. **Wear-layer or price regression** — if numeric label available in metadata.
7. **PCA / clustering sanity check** — confirm class separability.

## Steps

### Step 1 — Inspect dataset

Call `inspect_dataset` with:
```json
{
  "source_uri": "<path to Excel file>",
  "dataset_id": "nir-flooring-v1",
  "modality_override": "NIR",
  "label_column": "Measurement Description"
}
```

Confirm: 146 samples, 249 features, NIR modality, candidate labels found.

### Step 2 — Propose analysis plan

Call `propose_analysis_plan` with the inspection output and:
```json
{
  "user_intent": "Classify floor material types from NIR spectra for a paper demo.",
  "task_hint": "classification"
}
```

Present the plan. **[HUMAN APPROVAL GATE]**

### Step 3 — Run analyses

Call `run_analysis` with the approved plan.
- Save all artifacts under a canonical run ID.
- Surface any model failures for fallback handling.

### Step 4 — Validate results

Call `validate_results`. Classify results. Carry all warnings forward.

### Step 5 — Select best model

Call `select_best_model`. **[HUMAN APPROVAL GATE if validation warnings exist]**

### Step 6 — Interpret results

Call `interpret_results`. Frame wavelength importance as model evidence only.

### Step 7 — Generate report

Call `generate_report`. Present human-review checklist.
**[HUMAN APPROVAL GATE for final conclusions]**

### Step 8 — Save paper-demo artifacts

Confirm these artifacts are saved:
- [ ] Agent plan.
- [ ] Run metadata JSON.
- [ ] Results JSON with metrics tables.
- [ ] Validation report.
- [ ] Interpretation report.
- [ ] Figures (confusion matrices, ROC curves, importance plots, PCA).
- [ ] Final report (Markdown).
- [ ] Agent trace (decisions, tool calls, failures, fallbacks, approvals).
- [ ] Human-review checklist.

## Paper narrative support

- Binary task (Lumber vs. Vinyl) should show easy separation — note this.
- Multi-class task shows harder challenge — escalate caveats appropriately.
- Feature importance supports discussion of NIR spectral regions, not causal claims.
- Expert-effort reduction is described as "reducing routine steps" not "replacing experts".

## Prohibited claims in the paper

- The system replaces chemometric experts.
- NIR feature importance proves chemical identity.
- The 146-sample result generalises to all flooring NIR datasets.
- Binary accuracy translates to multi-class reliability.
