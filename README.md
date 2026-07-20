# Agentic Chemometrician

**MCP-based agentic chemometrics server** for spectral analysis. An agent-capable
client orchestrates deterministic chemometric tools through a human-in-the-loop
workflow — data loading, preprocessing, modeling, validation, interpretation, and
report generation.

## Quick start

Requires Python 3.10.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# or with dev dependencies:
pip install -e ".[dev]"
```

## MCP tools

| Tool | Purpose |
| --- | --- |
| `inspect_dataset` | Load spectral data, detect metadata, candidate labels, modality, and data quality warnings |
| `propose_analysis_plan` | Convert a dataset summary into a bounded, human-readable analysis plan for approval |
| `run_analysis` | Execute approved preprocessing and modeling tasks; save structured run artifacts |
| `validate_results` | Check for replicate leakage, group leakage, class imbalance, split instability, and suspicious metrics |
| `select_best_model` | Recommend the most defensible model by balancing performance, validation reliability, interpretability, and task suitability |
| `recommend_next_model` | Classify a model failure and return a fallback recommendation with rationale |
| `interpret_results` | Summarize feature and wavelength importance across models; flag unstable or overclaimed interpretations |
| `generate_report` | Produce a human-reviewable report with metrics, figures, validation warnings, caveats, and next-step recommendations |

The server is backed by a **shared, agent-neutral prompt library** (`prompt-library/`)
that defines scientific guardrails, skills, and workflows independent of client. The
workflow includes **human-in-the-loop approval gates** before consequential scientific
decisions — analysis plan, fallback model selection, final model selection when
validation warnings exist, and scientific conclusions.

## Project structure

```
src/
  chemometrics_contracts/   # Data contracts (SpectralDataset, AnalysisResult, ...)
  chemometrics_mcp/         # MCP server + tools + core analysis engine
    core/                   # Deterministic analysis logic (no MCP imports)
    tools/                  # MCP tool implementations (thin wrappers over core)
    server.py               # MCP server entry point
    artifacts.py            # Run artifact path management
tests/                      # Test suite (pytest / unittest)
prompt-library/             # Agent-neutral prompts: guardrails, skills, workflows
agent-memory/               # Changelog and trace of agent work
runs/                       # Generated run artifacts
ftir-purity-dataset/        # Reference dataset documentation
```

## The dataset

- **Source workbook:** `2026-05-21_22-59_Method Dev_wavelength_range_1450-2450.xlsx`
  (repo root), two sheets — `Spectra Metadata` and `Spectra P100001492`.
- **Instrument:** portable NIR (trinamiX), wavelength range 1454–2446 nm, 4 nm step,
  249 channels.
- **Samples:** 146 spectra joined to metadata, collected 2025-06-24 → 2025-09-09.
- **Material families:** hardwood species (fir, mahogany, oak, particle board, pine,
  poplar) and vinyl flooring brands / wear-layer grades (Home Decorators, LifeProof,
  TrafficMaster; 6/12/22 mil wear layers).

## Legacy references

| Location | Contents |
| --- | --- |
| `09-can-llms-be-used-for-chemometrics/` | Early self-contained agentic prototype (predecessor to this pipeline) |
| `archive/` | Notebook-based exploratory studies 00–08 and the original analysis notebook |

## Setup details

> **Note:** `xgboost` is pinned to `2.1.4` for SHAP compatibility (optional dev dep).
> On macOS you may also need `libomp` (`brew install libomp`).
