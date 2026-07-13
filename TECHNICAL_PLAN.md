# Technical Plan: Agentic Chemometrics Framework

> Revised 2026-07-11 to reflect actual implementation state.

## 1. Architecture & Project Structure

The repository has been consolidated from exploratory numbered notebooks into two Python packages under `src/`, a shared prompt library, and supporting directories:

```text
agentic-chemometrician/
├── src/
│   ├── chemometrics_contracts/     # Standalone data-contract package
│   │   └── __init__.py             # All dataclass definitions (22 types)
│   └── chemometrics_mcp/           # MCP server package
│       ├── __init__.py
│       ├── server.py               # MCP server entry point (mcp SDK, stdio transport)
│       ├── artifacts.py            # Run-ID generation, artifact path management
│       ├── tools/                  # MCP tool implementations (one module per tool)
│       │   ├── inspect_dataset.py
│       │   ├── propose_analysis_plan.py
│       │   ├── run_analysis.py
│       │   ├── validate_results.py
│       │   ├── select_best_model.py
│       │   ├── recommend_next_model.py
│       │   ├── interpret_results.py
│       │   ├── generate_report.py
│       │   ├── save_method_memory.py
│       │   ├── search_method_memory.py
│       │   ├── recommend_from_memory.py
│       │   └── orchestrate.py      # End-to-end pipeline orchestration
│       └── core/                   # Deterministic chemometrics implementations
│           ├── datasets.py         # trinamiX .xlsx parsing, metadata extraction
│           ├── preprocessing.py    # SNV, Savitzky-Golay, MSC, baseline correction, area normalization
│           ├── modeling.py         # scikit-learn + XGBoost pipelines (PLS, SVM, RF, XGBoost)
│           ├── planning.py         # Analysis plan proposal logic
│           ├── validation.py       # Leakage, imbalance, split-instability checks
│           ├── fallback.py         # Failure classification, fallback recommendation, rerun plans
│           ├── interpretation.py   # Feature/wavelength importance extraction, SHAP/LIME
│           ├── reporting.py        # Markdown/JSON report generation
│           ├── figures.py          # PNG/PDF figure rendering
│           ├── deserialization.py  # Shared dict→contract reconstruction
│           └── method_memory.py    # Reviewed-method memory store (save, search, recommend)
├── prompt-library/                 # Agent-neutral prompt library
│   ├── guardrails.md               # Scientific guardrails
│   ├── skills/                     # Skill prompts (4 files)
│   ├── workflows/                  # Workflow prompts (6 files)
│   └── output-contracts/           # Output format contracts (2 files)
├── tests/                          # 393 passing tests (contracts, tools, server smoke)
├── scripts/                        # Demo and utility scripts (run_nir_demo.py)
├── runs/                           # Agent-generated run artifacts
├── archive/                        # Archived exploratory notebooks (00–08)
├── ftir-purity-dataset/            # Second-modality dataset for generalization
├── agent-memory/                   # Changelog and agent traces
├── pyproject.toml                  # Package config (Python ≥3.10, setuptools)
└── requirements.txt                # Runtime + dev dependencies
```

### Key architectural decisions

- **Two packages, one repo.** `chemometrics_contracts` is a standalone package so contracts can be imported by tools, tests, and future external consumers without pulling in MCP dependencies.
- **MCP SDK, not FastMCP.** The server uses the `mcp` PyPI package directly (`mcp.server.Server`) with stdio transport. No wrapper framework.
- **Plan proposal is a deterministic tool**, not a prompt-only step. The `propose_analysis_plan` tool produces a bounded plan from dataset inspection results using heuristic rules in `core/planning.py`. The agent presents it to the user for approval.
- **Artifact isolation.** All writes go to `runs/<run-id>/artifacts/`. Path-separator injection is rejected by `artifacts.py`.

## 2. Data Contracts

All data contracts are frozen dataclasses in `chemometrics_contracts/__init__.py`. They enforce strict boundaries between the LLM agent and deterministic tools.

### Core domain types

| Type | Purpose |
|------|---------|
| `SpectralDataset` | Spectra matrix (`x`), wavelength axis, metadata rows, labels, modality, sample IDs, source references |
| `AnalysisResult` | Per-model output: task name, model name, preprocessing chain, metrics dict, predictions, selected features, figures, warnings, interpretation text |
| `AnalysisPlan` | Bounded plan: task name, preprocessing candidates, validation strategy, model families, human-readable summary |
| `AnalysisRun` | Aggregated run: metadata, results list, failed models, run-level warnings, artifact references |

### Inspection & validation types

| Type | Purpose |
|------|---------|
| `DatasetInspection` | Shape, axis range, modality, candidate label/group columns, data-quality warnings |
| `ValidationWarning` | Code, message, category, severity, affected stage, details |
| `ValidationSummary` | Pass/fail flag, per-check results, aggregated warnings |

### Decision & interpretation types

| Type | Purpose |
|------|---------|
| `ModelSelectionRecommendation` | Selected model, candidates, rationale, approval flag |
| `NextModelRecommendation` | Failed model, failure classification, fallback, rationale, approval flag |
| `InterpretationSummary` | Feature importance summary, model comparisons, caveats |
| `InterpretationResult` | Per-model SHAP/LIME importance values and metadata |
| `ReportSummary` | Report title, human/agent summaries, primary report artifact, all artifacts |

### Method memory types

| Type | Purpose |
|------|---------|
| `MethodMemory` | Top-level memory store metadata |
| `MethodMemoryEntry` | Single reviewed method record with configuration, metrics, and caveats |
| `MethodMemoryIndex` | Searchable index of method memory entries |
| `DatasetProfile` | Dataset characteristics summary for method memory matching |

### Infrastructure types

| Type | Purpose |
|------|---------|
| `ToolResponse[T]` | Generic envelope: tool name, ok/error, typed payload, warnings, artifacts, metadata |
| `RunMetadata` | Run ID, tool name, dataset ID, status, timestamp, parameters |
| `ArtifactReference` | Kind, URI, label, MIME type, description |

### Request types (one per tool)

`InspectDatasetRequest`, `ProposeAnalysisPlanRequest`, `RunAnalysisRequest`, `ValidateResultsRequest`, `SelectBestModelRequest`, `RecommendNextModelRequest`, `InterpretResultsRequest`, `GenerateReportRequest`, `SaveMethodMemoryRequest`, `SearchMethodMemoryRequest`, `RecommendFromMemoryRequest`.

## 3. The Deterministic Chemometrics Core

All numerical analysis lives in `chemometrics_mcp/core/`. The LLM never generates or executes arbitrary Python — it calls pre-built tools.

| Module | Implemented functionality |
|--------|--------------------------|
| `datasets.py` | Loads trinamiX `.xlsx` files; splits `Measurement Description` into type/brand/wear-layer metadata; detects modality from wavelength range; runs data-quality checks (NaN density, constant features, duplicate spectra) |
| `preprocessing.py` | 7 methods: `raw` (passthrough), SNV, MSC, Savitzky-Golay 1st derivative, Savitzky-Golay 2nd derivative (adaptive window), baseline correction (ALS), area normalization. Preprocessing comparison loop iterates all candidates from the plan |
| `modeling.py` | 8 pipeline wrappers: `svm_rbf`, `random_forest`, `pca_lda`, `xgboost` (classification); `pca`, `kmeans` (unsupervised); `plsr`, `svr`, `xgboost_reg` (regression). Cross-validated training with per-fold metrics extraction. Figures saved as JSON data and rendered to PNG/PDF via `core/figures.py` |
| `planning.py` | Heuristic plan proposal from `DatasetInspection`: selects task type, preprocessing candidates, validation strategy, and initial model families |
| `validation.py` | 10 checks: class-imbalance detection, suspiciously-high-metric flagging, small-sample warnings, missing-metadata detection, regression target-leakage heuristic, replicate-leakage check, group-leakage check, split-instability analysis, modality consistency, cross-modality comparability. GroupKFold referenced; falls back to regular KFold when group columns are absent |
| `fallback.py` | Failure classification (convergence, data incompatibility, metric anomaly); rule-based fallback model recommendation with rationale; `build_rerun_plan` for automated rerun-after-fallback |
| `interpretation.py` | Feature importance extraction (model-native coefficients, permutation importance); SHAP importance (KernelExplainer); LIME importance; wavelength-region summarization; cross-model comparison |
| `reporting.py` | Markdown report generation with metrics tables, validation warnings, interpretation summaries, human-review checklist, and next-steps section |
| `figures.py` | Renders figure data to PNG/PDF for paper-ready output |
| `deserialization.py` | Shared dict→contract reconstruction layer used by all tool handlers |
| `method_memory.py` | File-based reviewed-method memory store: save, search, and recommend operations with dataset-profile matching |

## 4. MCP Tool Surface

The server registers **11 tools**, each backed by a dedicated module in `tools/` that delegates to `core/`:

| # | Tool | Input | Output | Purpose |
|---|------|-------|--------|---------|
| 1 | `inspect_dataset` | `source_uri`, optional overrides | `DatasetInspection` | Load data, report shape/metadata/quality |
| 2 | `propose_analysis_plan` | `DatasetInspection`, optional intent/hints | `AnalysisPlan` | Generate bounded plan for user approval |
| 3 | `run_analysis` | `SpectralDataset`, approved `AnalysisPlan` | `AnalysisRun` | Execute preprocessing + cross-validated modeling |
| 4 | `validate_results` | `AnalysisResult[]`, optional dataset | `ValidationSummary` | Check for imbalance, suspicious metrics, leakage risks |
| 5 | `select_best_model` | `AnalysisResult[]`, optional validation summary | `ModelSelectionRecommendation` | Rank candidates by performance + defensibility |
| 6 | `recommend_next_model` | Failed model name, failure reason | `NextModelRecommendation` | Classify failure, suggest fallback with rationale |
| 7 | `interpret_results` | `AnalysisResult[]`, optional dataset | `InterpretationSummary` | Summarize feature importance across models |
| 8 | `generate_report` | `AnalysisRun`, optional validation/interpretation | `ReportSummary` | Produce human-reviewable markdown report |
| 9 | `save_method_memory` | Reviewed method entry | `MethodMemoryEntry` | Store human-approved analysis configuration |
| 10 | `search_method_memory` | Query, optional dataset profile | `MethodMemoryIndex` | Search reviewed methods by keyword or profile |
| 11 | `recommend_from_memory` | Dataset profile or query | `MethodMemoryEntry` | Recommend prior reviewed method with caveats |

### Intended tool-call sequence

```text
inspect_dataset → propose_analysis_plan → [user approval] → run_analysis
  → validate_results → select_best_model → interpret_results → generate_report
```

With `recommend_next_model` available as a branch when a model fails during `run_analysis`, and `save_method_memory`, `search_method_memory`, `recommend_from_memory` available for reviewed-method memory operations.

## 5. Prompt Library

The prompt library is agent-neutral — it works with any MCP-capable client (Claude, Codex, etc.).

### Skills (4)
- `dataset_inspection.md` — How to call and interpret `inspect_dataset`
- `analysis_planning.md` — How to present and get approval for analysis plans
- `interpretation.md` — How to present feature importance without overclaiming
- `report_writing.md` — How to structure and present the final report

### Workflows (6)
- `end_to_end_spectral_analysis.md` — Full 8-tool sequence with approval gates
- `model_selection.md` — How to use `select_best_model` with defensibility criteria
- `model_failure_fallback.md` — How to use `recommend_next_model` and handle retraining
- `validation_review.md` — How to present validation warnings and request human judgment
- `paper_demo_nir.md` — NIR flooring-specific workflow for the paper demo
- `ftir_followon.md` — FTIR purity-specific workflow for the generalization demo

### Output contracts (2)
- `agent_summary.md` — Agent-readable summary format
- `human_review_checklist.md` — Human-review gate checklist format

### Guardrails
- `guardrails.md` — Scientific boundaries, anti-hallucination rules, approval gates

## 6. Implementation Status & Remaining Work

### ✅ Complete (Product Plan Phases 1–4)

| Item | Status |
|------|--------|
| Repository consolidation (notebooks → `archive/`) | Done |
| `chemometrics_contracts` package (22 types) | Done |
| `chemometrics_mcp` package with 11 working tools | Done |
| `core/` modules: datasets, preprocessing, modeling, planning, validation, fallback, interpretation, reporting, figures, deserialization, method_memory | Done |
| MCP server with stdio transport | Done |
| Prompt library (guardrails, 4 skills, 6 workflows, 2 output contracts) | Done |
| Test suite (393 tests passing) | Done |
| Run artifact infrastructure (`runs/`, `artifacts.py`) | Done |
| Agent-memory changelog | Done |
| Replicate-leakage and group-leakage checks in `core/validation.py` | Done |
| Split-instability analysis in `core/validation.py` | Done |
| XGBoost model wired into `core/modeling.py` (classification + regression) | Done |
| Preprocessing comparison loop (iterates all candidates) | Done |
| Rendered figures as PNG/PDF (`core/figures.py`) | Done |
| Shared deserialization module (`core/deserialization.py`) | Done |
| Rerun-after-fallback automation (`build_rerun_plan` in `core/fallback.py`) | Done |
| Report "Next Steps" section (`_build_next_steps` in `core/reporting.py`) | Done |
| End-to-end orchestration (`run_full_pipeline` in `tools/orchestrate.py`) | Done |
| NIR demo trace script (`scripts/run_nir_demo.py`) | Done |
| Deterministic replicate-leakage sampling | Done |
| Modality-agnostic preprocessing (baseline correction, area normalization, adaptive SG window) | Done |
| FTIR dataset ingestion with synthetic spectra generator | Done |
| SHAP/LIME interpretability integration (`compute_shap_importance`, `compute_lime_importance`) | Done |
| Method memory system (save, search, recommend) | Done |
| Multi-modality validation checks (modality consistency, cross-modality comparability) | Done |
| FTIR workflow prompt (`prompt-library/workflows/ftir_followon.md`) | Done |

### 🔧 In Progress

All items from the Phase 3 Paper-Ready MVP and Phase 4 Generalization backlogs are complete. Remaining partial items are tracked in `IMPLEMENTATION-PLAN.md` (Phases 8–10 partial markers for model comparison tables, dedicated wavelength-importance visualization, formal report finalization, and paper-narrative verification).

### 📋 Future

| Item | Notes |
|------|-------|
| Paper-demo package | Phase 12: select canonical NIR demo, run standard suite, save canonical artifacts and paper-ready outputs |
| FTIR demo artifacts | Phase 14: run FTIR end-to-end with minimal prompt changes, save FTIR demo artifacts and report |

## 7. Dependencies

**Runtime:** `mcp`, `numpy`, `pandas`, `openpyxl`, `scikit-learn`, `scipy`

**Dev/optional:** `pytest`, `xgboost`, `shap`, `lime`, `matplotlib`, `seaborn`, `statsmodels`, `nbformat`, `langchain`

**Python:** ≥ 3.10
