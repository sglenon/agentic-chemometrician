# Agentic Chemometrician repository instructions

## Scientist analysis requests

When a user asks to analyze spectral or chemometric data, use the connected
`chemometrics` MCP as the analysis implementation. Read and follow
`skills/chemometrics-project/SKILL.md`.

- Begin with `list_chemometrics_capabilities`, then use the folder-first
  `create_project` → manifest review → plan → human approval → run → report
  workflow.
- Do not create a new Python/R analysis script, fit an unapproved model, or
  substitute a generic coding workflow for the MCP.
- Do not approve an analysis plan on the scientist's behalf.
- Return the generated run-local dashboard, report, figures, and tables.
  Generate the optional notebook only when requested.
- A generated notebook verifies and displays persisted evidence; it must not
  become a second analysis pipeline.
- If the MCP tools are unavailable, stop and report the connection problem
  instead of rebuilding the analysis in code.

### Preprocessing without a full project

Use `preprocess_spectra(source_path, steps)` when the scientist wants to inspect
or compare preprocessing steps on raw spectra files **before** committing to a
project. This tool runs outside the create/plan/approve/run cycle — no manifest,
no hash, no approval required — and returns before/after signals plus inline SVG
overlays. Supported steps: `snv`, `msc`, `sg_1st_deriv`, `sg_2nd_deriv`,
`sg_3rd_deriv`, `baseline`, `area_normalization`, `region_select`.

Do not write a standalone Python preprocessing script for this purpose. The MCP
tool produces the same output with no extra code and preserves full traceability.

### Composite multi-task runs

Pass `task_kinds` to `plan_project_analysis` when the scientist wants more than
one type of analysis on the same data in a single run. Example:

```
task_kinds=["unsupervised_exploration", "mixture_quantification"]
```

This produces one merged plan, one run, and one combined dashboard (PCA figures
and mixture figures side by side). Constraint: at most one split-supervised task
(`regression` or `classification`) per composite plan — requesting two is
rejected at plan time.

Do not run separate sequential projects or write ad-hoc scripts to achieve what
`task_kinds` already provides.

### File format notes

Single-column Y-only `.asc` files that carry JCAMP-style `##KEY=VALUE` headers
(`FIRSTX`, `LASTX`, `DELTAX`, `NPOINTS`) are ingested automatically — the
wavenumber axis is reconstructed from the headers. Files that lack the required
geometry headers are rejected; `.smf` files remain unsupported.

Measurements from different file formats on the same physical instrument grid
(e.g. `.txt` explicit-axis files and `.asc` JCAMP-reconstructed-axis files) align
correctly because the alignment check uses a float-noise tolerance
(`atol=1e-6`) rather than strict equality. Genuinely different grids or
resolutions are still blocked.

## Product-development requests

The restrictions above apply to scientific data-analysis requests, not to
explicit requests to develop, test, document, or maintain this repository.
For product work, modify the implementation normally and run relevant tests.
