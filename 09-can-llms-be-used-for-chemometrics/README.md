> **Note:** This folder is the early agentic prototype that directly preceded the
> current system. The lessons learned here — agent-guided preprocessing, self-contained
> tool scripts, and the agent-memory pattern — shaped the architecture of the full
> MCP-based pipeline in [`../10-agentic-chemometrics-pipeline/`](../10-agentic-chemometrics-pipeline/).

# Can LLMs be used for chemometrics?

An exploratory sub-project that asks whether an LLM acting as an analytical-chemist
agent can analyze portable-NIR spectra and build/interpret models for **hardwood vs.
vinyl** flooring classification.

The data comes from a portable NIR instrument (wavelength range 1454–2446 nm) originally
collected for a flooring-materials study. The primary question is hardwood-vs-vinyl
separation, but the dataset also contains subclasses (hardwood species, vinyl brands /
wear-layer grades) worth exploring.

## Folder layout

```
09-can-llms-be-used-for-chemometrics/
├── AGENTS.md                  # agent persona, research questions, working rules
├── README.md                  # this file
├── utils/
│   └── nir_first_pass.py      # end-to-end first-pass analysis script
├── src/                       # (empty) reusable tools/helpers go here
└── agent-memory/              # agent's findings, conclusions, and artifacts
    └── first_pass/
        ├── summary.json                    # all computed metrics
        ├── description_counts.csv          # spectra count per measurement description
        ├── mean_spectra_by_family.png      # mean ± std spectra per family
        ├── pca_snv_first_derivative.png    # PCA scores (SNV + 1st derivative)
        └── mean_difference_vinyl_minus_wood.png  # discriminative wavelengths
```

Per `AGENTS.md`, the agent works only inside this folder: tools live in `src/`/`utils/`,
and all findings/conclusions are written to `agent-memory/`.

## The data

- **Source workbook:** `2026-05-21_22-59_Method Dev_wavelength_range_1450-2450.xlsx`
  (repo root), two sheets — `Spectra Metadata` and `Spectra P100001492`.
- **Spectra:** 249 wavelength channels, 1454–2446 nm, 4 nm step.
- **Samples:** 146 spectra joined to metadata, collected 2025-06-24 → 2025-09-09.
- **Derived material families** (from the `Measurement Description` text):
  - `vinyl` — 90 spectra (brands: Home Decorators, LifeProof, TrafficMaster; wear layers
    6/12/22 mil; price per sqft parsed from the description)
  - `wood_lumber` — 44 spectra (fir, mahogany, oak, particle board, pine, poplar)
  - `control_other` — 12 spectra (water bottle, lab door) — excluded from modeling
- **Data quality:** 0 missing spectral cells, 0 duplicate spectrum IDs.

## The first-pass analysis (`utils/nir_first_pass.py`)

A single self-contained script that loads the workbook, parses the free-text descriptions
into structured fields (family, brand, wear layer, price, wood type), runs preprocessing
and modeling, and writes all artifacts to `agent-memory/first_pass/`.

Run it from the repo root:

```bash
python 09-can-llms-be-used-for-chemometrics/utils/nir_first_pass.py \
  "2026-05-21_22-59_Method Dev_wavelength_range_1450-2450.xlsx" \
  --out-dir 09-can-llms-be-used-for-chemometrics/agent-memory/first_pass
```

What it does:

- **Preprocessing** (custom scikit-learn transformers): SNV (standard normal variate) and
  Savitzky–Golay first derivative, used alone and combined.
- **Models:** logistic regression on raw / SNV / 1st-derivative / SNV+derivative inputs,
  plus a PLS-DA (5 latent variables) classifier.
- **Validation:** stratified 5-fold CV *and* leave-one-description-out CV (groups by
  measurement description) to guard against replicate leakage.
- **Unsupervised:** PCA on raw and on SNV+derivative inputs; silhouette score for the
  wood-vs-vinyl split.
- **Interpretation:** per-class intensity summaries and the wavelengths with the largest
  vinyl-minus-wood mean difference.

## Key findings (so far)

- **Wood vs. vinyl is trivially separable.** Every model reaches perfect scores
  (accuracy / balanced accuracy / macro-F1 / ROC-AUC = 1.0) under *both* stratified and
  leave-one-description-out CV. The classes are linearly separable even on raw spectra.
- **Strong baseline intensity difference.** Vinyl spectra sit much higher
  (mean ≈ 0.94) than wood/lumber (mean ≈ 0.35) — a large baseline/scatter offset that
  raw-input models exploit directly. PC1 of the raw data explains 97.4% of variance.
- **Most discriminative region:** ~2290–2370 nm (C–H combination/overtone bands), where
  vinyl–wood mean differences peak around 2306 nm.
- **After SNV + 1st derivative** (which removes the baseline offset), wood and vinyl still
  separate cleanly — silhouette ≈ 0.69, PC1 ≈ 76% — confirming the separation is not
  purely a scatter artifact.

See `agent-memory/first_pass/summary.json` for the full numbers.

## Caveats / next steps

- Perfect classification on a small, well-separated dataset means the *hardwood-vs-vinyl*
  task is too easy to be a discriminating test — the harder, more interesting questions
  are the subclasses (wood species, vinyl wear-layer/brand, price tiers).
- `control_other` samples are excluded from modeling; they're useful as out-of-distribution
  checks.
- The broader research questions in `AGENTS.md` (LLM-driven feature/subclass discovery,
  novel preprocessing, interpretation, etc.) remain open beyond this first pass.
