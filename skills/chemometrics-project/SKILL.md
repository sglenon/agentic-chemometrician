---
name: chemometrics-project
description: Run scientist-supervised, folder-first spectral analysis through the chemometrics MCP. Use for local FTIR, NIR, UV-Vis, PXRD, or mass-spectrometry folders; spectral/reference comparisons; exploratory PCA; classification or regression; constrained mixture screening; UV-Vis Job's plots; and evidence-led reports.
---

# Chemometrics Project

Use the project tools as a protocol. Automate data handling and computation;
leave scientific judgment and final conclusions to the researcher.

## Workflow

1. Call `create_project` with the user's source folder. Keep derived files in the
   returned output folder.
2. Call `get_project`. Summarize files, measurements, and blockers without
   requesting raw arrays.
3. Resolve the manifest with `update_project_manifest`.
   - Require explicit modality, axis kind/unit, signal kind/unit, sample role,
     and preparation hierarchy.
   - Never infer units, preparation IDs, targets, or reference roles from values.
   - Treat technical scans and independent preparations differently.
   - Ask the scientist only for unresolved facts that block the intended task.
4. Call `plan_project_analysis` with the scientific objective and an explicit
   task kind when known. Supply `target` for supervised or Job's-method work.
   Supply task-specific `analysis_options`.
5. Present the compact plan, blockers, pipeline candidates, split manifest,
   claim ceiling, and expected outputs. Do not approve it yourself.
6. After the user explicitly approves, call `approve_project_plan` with their
   identity, then `run_project_analysis` using the returned approval ID.
7. Call `get_project_run`, then `generate_project_report`. Set
   `include_notebook=true` only when the scientist requests a reproducibility
   notebook.
8. Return the local `dashboard`, `scientist_report`, figure, and table paths.
   Explain that the dashboard is offline and evidence-only. Do not create a
   separate analysis script or rerun the analysis inside the notebook.
9. Lead with observed evidence, limitations, and abstentions. Keep possible
   explanations separate and leave the final scientific interpretation to the
   user.

## Route the task

- Use `spectral_comparison` for product/precursor/reference similarity.
- Use `unsupervised_exploration` for descriptive FTIR/NIR PCA.
- Use `classification` or `regression` only with explicit targets and
  preparation IDs.
- Use `mixture_quantification` only with named, role-tagged, physically
  compatible references. Treat results as constrained screening coefficients,
  not certified purity.
- Use `uvvis_job_plot` with a mole-fraction target. Emit a ratio only when
  `design_metadata` declares blank correction, component controls,
  constant-total concentration, and the response definition.
- Use `pxrd_reference_matching` with explicit reference roles, a declared
  two-theta tolerance, and calibration/acquisition metadata when available.
- Use `ms_peak_matching` with approved `mass_tolerance`,
  `mass_tolerance_unit`, and mass-calibration provenance. Never guess adducts.

## Scientific guardrails

- Never bypass unit, role, hierarchy, leakage, plan-hash, or evidence-hash
  blockers.
- Do not describe scan-level performance as independent generalization.
- Require nested group-aware evaluation for selected supervised pipelines.
- Do not turn similarity, PCA separation, peak matches, or feature importance
  into chemical identity, causality, phase purity, or compound purity.
- Report LOD/LOQ as not estimable unless blanks, low-level standards,
  independent replicates, and a declared detection-limit method are present.
- Do not silently repair, clip, exclude, align, or normalize data. Report every
  applied transformation and retain the raw source evidence.
- Treat `runs/<run_id>/dashboard.html`, `figures/`, and `tables/` as generated
  run artifacts. Never replace them with an ad hoc Python analysis script.
- When evidence is insufficient, recommend the minimum next experiment instead
  of trying more arbitrary models.

## Useful analysis options

Pass options inside `plan_project_analysis.analysis_options` so approval binds
them to the plan:

```json
{
  "mass_tolerance": 5,
  "mass_tolerance_unit": "ppm",
  "calibration_metadata": {"mass_error_basis": "external calibration"}
}
```

For Job's method:

```json
{
  "wavelength": 500,
  "design_metadata": {
    "blank_corrected": true,
    "component_controls": true,
    "constant_total_concentration": true,
    "response_definition": "blank-corrected delta absorbance",
    "validated_response_method": true
  }
}
```
