# Agent Memory Changelog

## 2026-06-24

### Request summary

Created the initial planning and agent-instruction documents for the MCP-based agentic chemometrics prototype.

### Files created or modified

- `PLAN.md`
- `AGENTS.md`
- `agent-memory/CHANGELOG.md`
- `changes_2026-06-24.md`

### Key decisions

- The project is optimized for a paper-first prototype with a strong MCP-based agent demo.
- The system should be usable by Claude, Codex, or another MCP-capable client without separate scientific behavior per client.
- The first proof-of-concept is NIR flooring data, with FTIR planned as a follow-on application.
- The prompt library should include workflows, model selection rules, model fallback rules, scientific guardrails, and human-review gates.
- The MVP should support an end-to-end guided workflow while preserving human-in-the-loop approval points.

### Validation

- Documentation files were created only; no code was implemented or executed.

### Known risks and follow-ups

- Exact NIR flooring data files still need to be selected for the first loader.
- The first MCP server framework and prompt library folder structure still need to be chosen.
- FTIR follow-on dataset still needs to be identified.

## 2026-06-24 (session 3)

### Request summary

Implemented Phases 0–4 of the MCP-based agentic chemometrics prototype: full repository
audit, contract validation tests, minimal MCP server skeleton, NIR dataset ingestion and
`inspect_dataset` tool, and shared prompt library MVP.

### Files created or modified

- `chemometrics_mcp/__init__.py` — package init
- `chemometrics_mcp/server.py` — MCP server entry point, 8 tools registered
- `chemometrics_mcp/artifacts.py` — run ID generation, artifact path management
- `chemometrics_mcp/core/__init__.py` — core package init
- `chemometrics_mcp/core/datasets.py` — NIR Excel loader, inspection, data quality checks
- `chemometrics_mcp/tools/__init__.py` — tool submodule exports
- `chemometrics_mcp/tools/inspect_dataset.py` — fully implemented
- `chemometrics_mcp/tools/propose_analysis_plan.py` — deferred stub (Phase 5)
- `chemometrics_mcp/tools/run_analysis.py` — deferred stub (Phase 6)
- `chemometrics_mcp/tools/validate_results.py` — deferred stub (Phase 7)
- `chemometrics_mcp/tools/select_best_model.py` — deferred stub (Phase 8)
- `chemometrics_mcp/tools/recommend_next_model.py` — deferred stub (Phase 9)
- `chemometrics_mcp/tools/interpret_results.py` — deferred stub (Phase 8)
- `chemometrics_mcp/tools/generate_report.py` — deferred stub (Phase 10)
- `tests/test_contracts.py` — extended with 23 new boundary/validation/serialization tests
- `tests/test_server_smoke.py` — 18 smoke tests for server, inspect_dataset, path safety, deferred tools
- `prompt-library/guardrails.md` — scientific guardrails
- `prompt-library/skills/dataset_inspection.md`
- `prompt-library/skills/analysis_planning.md`
- `prompt-library/skills/interpretation.md`
- `prompt-library/skills/report_writing.md`
- `prompt-library/workflows/end_to_end_spectral_analysis.md`
- `prompt-library/workflows/model_selection.md`
- `prompt-library/workflows/model_failure_fallback.md`
- `prompt-library/workflows/validation_review.md`
- `prompt-library/workflows/paper_demo_nir.md`
- `prompt-library/output-contracts/agent_summary.md`
- `prompt-library/output-contracts/human_review_checklist.md`
- `requirements.txt` — added `mcp`, `pytest`
- `IMPLEMENTATION-PLAN.md` — marked verified completed items [x]
- `agent-memory/CHANGELOG.md` — this entry
- `agent-memory/changes_2026-06-24.md` — appended session 3 summary

### Key decisions

- Used `mcp==1.28.0` (installed from PyPI) as the MCP server framework.
- NIR target file: `2026-05-21_22-59_Method Dev_wavelength_range_1450-2450.xlsx`
  (146 samples × 249 features, 1454–2446 nm, `Measurement Description` as label column).
- All 7 deferred tools return `ok=False` with explicit Phase labels — never silently faked.
- Artifact writes are bounded to `runs/` via `artifacts.py`; path-separator injection rejected.
- Prompt library is fully agent-neutral (no Claude vs. Codex split).
- `chemometrics_contracts/` package (with two implementations) left untouched — the canonical
  one used throughout is `chemometrics_contracts/__init__.py`.

### Validation performed

- `python -m pytest tests/ -v` → **42 passed** (24 contract tests + 18 smoke tests).
- All tests actually executed against the real NIR Excel file.
- Determinism verified: two runs of `inspect_dataset` on the same file return identical counts.

### Known risks and follow-ups

- Phase 5 (`propose_analysis_plan`) is the next implementation target.
- The `chemometrics_contracts/contracts.py` file is a duplicate/older version;
  `__init__.py` is authoritative. Consider consolidating in a future session.
- `src/chemxai_literature/contracts.py` is a third parallel contracts file —
  likely from an earlier exploratory session. Clarify its role before Phase 5.
- Phase 6 (`run_analysis`) requires approval to add `xgboost`, `shap`, and `matplotlib`
  to the active environment (they are in `requirements.txt` but not yet installed).

## 2026-06-24

### Request summary

Created an agent-usable implementation checklist from `PLAN.md` in `IMPLEMENTATION-PLAN.md`.

### Files created or modified

- `IMPLEMENTATION-PLAN.md`
- `agent-memory/CHANGELOG.md`
- `agent-memory/changes_2026-06-24.md`

### Key decisions

- Used Markdown checkboxes throughout so future agents can mark verified progress.
- Added a mandatory startup checklist requiring agents to inspect existing work before implementing.
- Added phase-level tasks, acceptance checklists, tool-level definitions of done, human approval gates, testing expectations, and handoff requirements.
- Kept method memory and FTIR as later phases after the NIR report-producing demo is credible.

### Validation

- Documentation only; no code was implemented or executed.

### Known risks and follow-ups

- Future agents must still audit the repository and mark completed items only after verification.
- Exact NIR flooring target files, MCP framework choice, and prompt-library folder structure remain implementation decisions.

## 2026-07-11

### Request summary

Corrected IMPLEMENTATION-PLAN.md status boxes to match real code; fixed stale docstring in server.py.

### Files created or modified

- `IMPLEMENTATION-PLAN.md` — Phase 5, 6, 7 tasks and acceptance boxes flipped to [x] where verified against code; partial items marked [~]; Phase 8-11 unimplemented items marked [~] with notes; testing section updated; tool-level definitions of done updated for propose_analysis_plan, run_analysis, validate_results, select_best_model, recommend_next_model, interpret_results, generate_report.
- `chemometrics_mcp/server.py` — module docstring updated: removed stale "DEFERRED responses" claim; now states all 8 tools are fully implemented.
- `agent-memory/CHANGELOG.md` — this entry.

### Key decisions

- Supersedes the earlier "deferred stub" note in the 2026-06-24 (session 3) changelog entry: propose_analysis_plan, run_analysis, validate_results, select_best_model, recommend_next_model, interpret_results, and generate_report are all implemented with real logic and passing tests.
- 252 pytest tests passed (52 seconds) confirming all 8 tools are functional.

### Validation performed

- `python -m pytest tests/ -q` in the worktree → **252 passed** (exit code 0).
- Each Phase 5-7 sub-item was individually verified against source files before marking [x].
- Partial items marked [~] where logic exists but acceptance criteria are not fully met (e.g., replicate/group leakage checks absent from core/validation.py; per-item approval flags absent from AnalysisPlan).

### Known risks and follow-ups

- Phase 7: replicate leakage, group leakage, and split instability checks are not yet implemented in core/validation.py.
- Phase 9: rerun-after-fallback logic is not implemented; fallback recommendation must be applied manually.
- Phase 10-11: no agent trace artifact; end-to-end orchestration wiring is pending.
- generate_report has no dedicated "Next Steps" section; the human-review checklist partially covers this role.

## 2026-07-11 (session 2)

### Request summary

Revised TECHNICAL_PLAN.md to reflect actual codebase state and align with PRODUCT_PLAN.md.

### Files created or modified

- `TECHNICAL_PLAN.md` — Full rewrite: updated architecture diagram (two-package layout under `src/`), expanded tool surface from 5 to 8, documented all 17 contract types, listed all 7 model implementations and 5 preprocessing methods, added prompt library inventory, and replaced the 5-phase roadmap with an honest status matrix (complete / in-progress / future) mapped to Product Plan phases.

### Key decisions

- Kept TECHNICAL_PLAN.md as a current-state reference document (not a speculative roadmap). PLAN.md and IMPLEMENTATION-PLAN.md continue to serve as the detailed roadmap and checklist respectively.
- Documented all known gaps explicitly: missing validation checks (replicate/group leakage, split instability), XGBoost not wired, preprocessing comparison not implemented, figures JSON-only, GroupKFold fallback, inline deserialization debt.
- Used Product Plan phases as the organizing principle for status tracking.

### Validation performed

- Codebase survey verified: directory structure, file contents, import paths, contract types, tool implementations, test counts, run artifacts, and dependency lists all cross-checked against source files.
- No code was modified; documentation only.

### Known risks and follow-ups

- PLAN.md header still says "does not describe implemented functionality yet" — this is stale and should be updated in a future session.
- IMPLEMENTATION-PLAN.md partial items ([~]) should be reconciled with the gaps documented here.

## 2026-07-11 (session 3)

### Request summary

Implemented replicate-leakage and group-leakage validation checks in `core/validation.py`, addressing the highest-priority gaps identified in TECHNICAL_PLAN.md Phase 3.

### Files created or modified

- `src/chemometrics_contracts/__init__.py` — Added `validation_strategy: str | None` field to `AnalysisResult`; added `dataset_inspection: DatasetInspection | None` field to `ValidateResultsRequest`.
- `src/chemometrics_mcp/core/validation.py` — Added `check_replicate_leakage()` (prediction consistency within replicate groups, >80% threshold), `check_group_leakage()` (warns when candidate group columns exist but non-grouped CV was used), and updated `run_all_checks()` to call both and track `replicate_leakage` / `group_leakage` in checks dict.
- `src/chemometrics_mcp/core/modeling.py` — Added `validation_strategy` parameter to `run_cv_model()` and propagated it to all 7 `AnalysisResult` constructions.
- `src/chemometrics_mcp/tools/run_analysis.py` — Passed `approved_plan.validation_strategy` through to `run_cv_model()`.
- `src/chemometrics_mcp/tools/validate_results.py` — Passed `request.dataset_inspection` through to `run_all_checks()`.
- `src/chemometrics_mcp/server.py` — Fixed `validate_results` handler to deserialize `predictions`, `validation_strategy`, and `dataset_inspection` from JSON arguments.
- `tests/test_validate_results.py` — Added 19 new tests: `TestCheckReplicateLeakage` (8 tests), `TestCheckGroupLeakage` (7 tests), `TestRunAllChecksLeakage` (4 tests); updated `test_all_checks_keys_present` to include new check keys.
- `agent-memory/CHANGELOG.md` — this entry.

### Key decisions

- Replicate leakage threshold: >80% prediction pair consistency within a group triggers a warning. Groups with >20 members are sampled (max 190 pairs) to keep computation bounded.
- Group leakage fires when `validation_strategy` is `None` (unknown) or a non-grouped strategy, and candidate group columns are present. This catches the GroupKFold→KFold fallback silently leaking.
- Both checks are gated on `DatasetInspection` being provided — no false positives when inspection data is absent.
- All changes are additive with safe defaults (`None`); existing tests continue to pass.

### Validation performed

- `python -m pytest tests/ -v` → **271 passed** (up from 252; 19 new tests added).
- All new tests cover: no-data guards, threshold boundaries, multiple group columns, mixed strategies, and integration via `run_all_checks()`.

### Known risks and follow-ups

- Split-instability analysis (medium priority gap) is still not implemented.
- The `random` sampling in `check_replicate_leakage` for large groups introduces minor non-determinism; could be replaced with a hash-based deterministic sampler if needed.
- `check_replicate_leakage` uses `dataset.metadata` for group extraction; if metadata is missing but `sample_ids` are present, it falls back to using sample IDs as group labels (which will likely produce no multi-member groups).

## 2026-07-11 (session 4)

### Request summary

Implemented all 12 remaining tasks from TECHNICAL_PLAN.md Section 6, completing the Phase 3 Paper-Ready MVP. Work was organized into 4 waves: foundational validation/modeling, paper content, orchestration, and tech debt.

### Files created or modified

- `src/chemometrics_mcp/core/modeling.py` — Added `seed` parameter to `make_cv_splitter`; added `run_cv_model_multi_seed()` for split-instability analysis; added `xgboost` (classification) and `xgboost_reg` (regression) model branches with gain-based feature importance; graceful degradation if xgboost not installed
- `src/chemometrics_mcp/core/validation.py` — Added `check_split_instability()` (CV > 0.10 warning, > 0.25 error); wired into `run_all_checks`; fixed non-determinism in `check_replicate_leakage` with seeded `random.Random(42)`
- `src/chemometrics_mcp/tools/run_analysis.py` — Added multi-seed instability checks after main model loop; fixed indentation bug in preprocessing/model loop; integrated figure rendering via `render_figure`
- `src/chemometrics_mcp/tools/orchestrate.py` — New module: `run_full_pipeline()` orchestrates all 8 tools with auto/manual approval modes, saves `agent_trace.json`
- `src/chemometrics_mcp/core/figures.py` — New module: `render_figure()` dispatches on figure type (confusion matrix, feature importances, PCA variance, predicted-vs-actual, fold metrics, cluster sizes); PNG/PDF output; publication styling; graceful degradation
- `src/chemometrics_mcp/core/fallback.py` — Added `build_rerun_plan()` for automated fallback plan construction; added xgboost entries to fallback table
- `src/chemometrics_mcp/core/interpretation.py` — Added xgboost complexity scores
- `src/chemometrics_mcp/core/planning.py` — Added xgboost to recommended model families for classification and regression
- `src/chemometrics_mcp/core/deserialization.py` — New module: 12 shared `from_dict()` functions replacing inline deserialization in server.py
- `src/chemometrics_mcp/server.py` — Replaced ~250 lines of inline deserialization with imports from `core/deserialization.py`
- `src/chemometrics_mcp/core/reporting.py` — Verified preprocessing comparison section and Next Steps section already implemented
- `scripts/run_nir_demo.py` — New: end-to-end demo script producing complete agent trace on NIR flooring dataset
- `tests/test_validate_results.py` — Added 7 split-instability tests + 1 determinism test
- `tests/test_run_analysis.py` — Added 3 multi-seed tests + 3 xgboost tests
- `tests/test_figures.py` — New: 16 rendering tests (already present from earlier work)
- `tests/test_orchestrate.py` — New: 10 orchestration tests
- `tests/test_deserialization.py` — New: 14 round-trip serialization tests
- `tests/test_propose_analysis_plan.py` — Updated assertions for xgboost model families
- `PLAN.md` — Updated stale header
- `IMPLEMENTATION-PLAN.md` — Flipped ~20 `[~]` items to `[x]` across Phases 5, 7, 8, 9, 10, 11
- `TECHNICAL_PLAN.md` — Updated architecture tree, module table, and Section 6 status matrix
- `agent-memory/CHANGELOG.md` — this entry

### Key decisions

- Split-instability uses coefficient of variation (std/mean) on primary metric with 10%/25% thresholds.
- XGBoost uses gain-based `feature_importances_` (same pattern as random_forest).
- Fallback chain: xgboost → random_forest (cls), xgboost_reg → plsr (reg).
- Orchestration has two modes: `auto=False` (pauses at plan for approval) and `auto=True` (full pipeline).
- Figure rendering is best-effort: if matplotlib unavailable, JSON data still saved.
- Shared deserializer centralizes all dict→contract reconstruction; server.py is now a thin routing layer.

### Validation performed

- `python -m pytest tests/ -q` → **317 passed** (7.94s).
- NIR demo script verified end-to-end: inspect → plan → analyze → validate → select → interpret → report.

### Known risks and follow-ups

- SHAP/LIME interpretability still not integrated (Phase 4).
- FTIR dataset ingestion still pending (no spectral data files available yet).
- Method memory store not yet designed (Phase 4).
- Multi-modality validation not yet attempted (Phase 4).
- Paper narrative and demo walkthrough not yet rehearsed end-to-end with human reviewer.

## 2026-07-11 (session 5)

### Request summary

Implemented all Phase 4 generalization tasks: modality-agnostic preprocessing, FTIR dataset ingestion with synthetic spectra generator, SHAP/LIME interpretability integration, method memory system (3 new tools), multi-modality validation checks, and FTIR workflow prompt.

### Files created or modified

- `src/chemometrics_contracts/__init__.py` — Added 5 new contract types: `InterpretationResult`, `DatasetProfile`, `MethodMemory`, `MethodMemoryEntry`, `MethodMemoryIndex`; added 3 new request types: `SaveMethodMemoryRequest`, `SearchMethodMemoryRequest`, `RecommendFromMemoryRequest` (total: 22 types)
- `src/chemometrics_mcp/core/preprocessing.py` — Added `baseline_correction` (ALS) and `area_normalization` methods; added `_adaptive_sg_window()` for modality-agnostic SG window sizing; added `_als_baseline()` helper
- `src/chemometrics_mcp/core/interpretation.py` — Added `compute_shap_importance()` (KernelExplainer) and `compute_lime_importance()` functions
- `src/chemometrics_mcp/core/validation.py` — Added `check_modality_consistency()` and `check_cross_modality_comparability()` checks (total: 10 checks)
- `src/chemometrics_mcp/core/method_memory.py` — New module: file-based reviewed-method memory store with save, search, and recommend operations
- `src/chemometrics_mcp/core/datasets.py` — Added FTIR dataset ingestion with synthetic spectra generator
- `src/chemometrics_mcp/tools/save_method_memory.py` — New tool: save reviewed method to memory store
- `src/chemometrics_mcp/tools/search_method_memory.py` — New tool: search method memory by keyword or dataset profile
- `src/chemometrics_mcp/tools/recommend_from_memory.py` — New tool: recommend prior reviewed method with caveats
- `src/chemometrics_mcp/server.py` — Registered 3 new tools (save_method_memory, search_method_memory, recommend_from_memory); total: 11 tools
- `prompt-library/workflows/ftir_followon.md` — New workflow: FTIR purity-specific analysis workflow for generalization demo
- `tests/` — Added tests for all new modules and tools (total: 393 passing tests)
- `TECHNICAL_PLAN.md` — Updated architecture tree, module table, tool surface, prompt library, and implementation status
- `IMPLEMENTATION-PLAN.md` — Marked Phase 13 (method memory) and Phase 14 (FTIR) items complete; added tool-level definitions for 3 new tools
- `agent-memory/CHANGELOG.md` — this entry

### Key decisions

- Synthetic FTIR spectra generator used for testing since real FTIR spectral data files are not yet available; generator produces realistic composition-based spectra with configurable noise.
- SHAP uses KernelExplainer (model-agnostic) rather than TreeExplainer to support all model types uniformly.
- Method memory is file-based (JSON) for simplicity and transparency; entries require explicit human approval before saving.
- Adaptive SG window scales with feature count to handle both NIR (~250 features) and FTIR (~1000+ features) ranges.
- Baseline correction uses Asymmetric Least Squares (ALS) with default lambda=1e5, p=0.01.
- Cross-modality comparability check warns when comparing results across different spectral domains (e.g., NIR vs FTIR).

### Validation performed

- `python -m pytest tests/ -q` → **393 passed** (up from 317; 76 new tests added).
- All new preprocessing methods tested with NIR and FTIR-shaped data.
- SHAP/LIME integration tested with small synthetic models.
- Method memory save/search/recommend tested with round-trip serialization.
- Multi-modality validation checks tested with cross-modality fixture data.

### Known risks and follow-ups

- FTIR-specific report caveats not yet authored; modality-agnostic caveats in place.
- Canonical FTIR demo run not yet saved (synthetic data pipeline works but no real FTIR dataset analyzed end-to-end).
- SHAP KernelExplainer can be slow for large models; consider batching or subsampling for production use.
- Method memory file-based storage may need migration to a database if entry count grows significantly.
- Paper narrative and demo walkthrough not yet rehearsed end-to-end with human reviewer.
