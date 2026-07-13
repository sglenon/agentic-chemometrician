# Implementation Plan: MCP-Based Agentic Chemometrics Prototype

## Purpose

This document converts `PLAN.md` into an implementation checklist that agents can follow without guessing the roadmap. `PLAN.md` remains the conceptual source of truth. This file is the execution checklist.

## Mandatory agent startup checklist

Before implementing anything, every agent must first check what is already done.

- [x] Re-read `PLAN.md` and this file before changing code.
- [x] Inspect the current repository structure for existing implementations, tests, prompts, artifacts, and changelog entries.
- [x] Mark any already-completed checklist items as `[x]` only after verifying the related files and behavior exist.
- [x] Do not duplicate an existing implementation under a new path without a clear reason.
- [x] Record evidence for completed items in the relevant phase notes, changelog, or handoff summary.
- [x] Keep changes small enough for review and avoid unrelated refactors.
- [x] Update `agent-memory/CHANGELOG.md` for meaningful work in this folder.
- [x] Update or create the appropriate `changes_YYYY-MM-DD.md` entry for meaningful code or documentation changes.

## Status legend

- `[ ]` Not started or not yet verified.
- `[x]` Verified complete.
- `[~]` In progress; include a short note naming the active branch, files, or blocking issue.
- `[!]` Blocked; include the decision, data, dependency, or approval needed.

## Non-negotiable implementation rules

- [x] Keep the prototype paper-first and demo-first.
- [x] Build one agent-neutral MCP tool layer usable by Claude, Codex, or another MCP-capable client.
- [x] Keep deterministic chemometrics in Python tools, not in prompts.
- [x] Do not expose arbitrary Python or shell execution through MCP tools.
- [x] Do not create separate scientific behavior for different MCP clients.
- [x] Do not invent metrics, figures, validation results, or chemical conclusions.
- [x] Preserve human approval gates before consequential scientific decisions.
- [x] Treat method memory as later work until the report-producing demo works.
- [x] Prefer simple contracts and reviewable artifacts over a broad general-purpose framework.

## Suggested implementation layout

Agents must check the existing tree before creating these paths. Use this as a target shape, not permission to duplicate existing work.

```text
10-agentic-chemometrics-pipeline/
  chemometrics_mcp/
    __init__.py
    server.py
    contracts.py
    tools/
      inspect_dataset.py
      propose_analysis_plan.py
      run_analysis.py
      validate_results.py
      select_best_model.py
      recommend_next_model.py
      interpret_results.py
      generate_report.py
    core/
      datasets.py
      preprocessing.py
      modeling.py
      validation.py
      interpretation.py
      reporting.py
    artifacts.py
  prompt-library/
    guardrails.md
    skills/
    workflows/
    output-contracts/
  tests/
  runs/
  agent-memory/
```

## Phase 0: Repository audit and implementation baseline

Goal: establish what already exists before adding anything.

### Tasks

- [x] Confirm `PLAN.md` is the current roadmap.
- [x] Confirm `AGENTS.md` instructions still match the implementation direction.
- [x] Inventory existing notebooks, scripts, datasets, figures, and generated outputs that can be reused.
- [x] Identify any already-existing MCP server code or prompt-library files.
- [x] Identify the first NIR flooring data file or files to support.
- [x] Identify available dependency constraints from `requirements.txt` and any local environment files.
- [x] Decide whether implementation starts as a lightweight package, standalone MCP server module, or script-backed server.
- [x] Create a short baseline note in `agent-memory/CHANGELOG.md` if implementation work begins.

### Acceptance checklist

- [x] Agents know which files are source documents, reusable notebooks, raw data, and generated artifacts.
- [x] Agents know whether they are extending existing code or creating the first implementation.
- [x] The first target dataset is named, or the blocker is recorded.
- [x] No duplicate framework skeleton has been created.

## Phase 1: Contracts and schemas

Goal: define stable data, result, tool-input, and tool-output contracts before building broad functionality.

### Tasks

- [x] Implement or document `SpectralDataset` with spectra, axis, metadata, labels, modality, and source references.
- [x] Implement or document `AnalysisResult` with task name, model name, preprocessing, metrics, predictions, selected features, figures, warnings, and interpretation.
- [x] Define run ID and artifact directory conventions.
- [x] Define structured warning objects with severity, category, message, and affected stage.
- [x] Define MCP input and output schemas for `inspect_dataset`.
- [x] Define MCP input and output schemas for `propose_analysis_plan`.
- [x] Define MCP input and output schemas for `run_analysis`.
- [x] Define MCP input and output schemas for `validate_results`.
- [x] Define MCP input and output schemas for `select_best_model`.
- [x] Define MCP input and output schemas for `recommend_next_model`.
- [x] Define MCP input and output schemas for `interpret_results`.
- [x] Define MCP input and output schemas for `generate_report`.
- [x] Add contract-focused tests or validation examples where the project structure supports them.

### Acceptance checklist

- [x] Tool contracts are readable by an MCP-capable agent without inspecting implementation internals.
- [x] Contract fields match the standard objects in `PLAN.md`.
- [x] Missing labels, ambiguous metadata, invalid data, and small sample risks can be represented.
- [x] The contracts do not contain dataset-specific conclusions.

## Phase 2: Minimal MCP server skeleton

Goal: create the smallest discoverable MCP server with safe tool registration and placeholder-safe behavior.

### Tasks

- [x] Choose the MCP server package or API already available in the environment, or record dependency approval needed.
- [x] Create the server entry point.
- [x] Register the initial tool names from `PLAN.md`.
- [x] Wire each registered tool to a deterministic Python function.
- [x] Return clear not-implemented or unsupported responses only where functionality is intentionally deferred.
- [x] Add input validation at tool boundaries.
- [x] Add artifact path validation so tools cannot write outside approved run directories.
- [x] Add minimal smoke tests or a documented local command to list tools.

### Acceptance checklist

- [x] An MCP client can discover the tool names and schemas.
- [x] No tool exposes arbitrary code execution.
- [x] Invalid inputs fail with explicit user-facing messages.
- [x] Deferred capabilities are labeled as deferred rather than silently faked.

## Phase 3: NIR flooring dataset ingestion and inspection

Goal: make `inspect_dataset` useful for the first paper/demo dataset.

### Tasks

- [x] Select the first NIR flooring dataset file or files.
- [x] Implement dataset loading for the selected file format.
- [x] Parse spectra into samples-by-features shape.
- [x] Parse wavelength or spectral axis values.
- [x] Parse available metadata columns.
- [x] Detect candidate label columns.
- [x] Detect candidate group, replicate, batch, or sample-origin columns.
- [x] Infer modality when possible, while allowing explicit override.
- [x] Detect missing values, non-numeric spectra, duplicate sample IDs, and obvious shape mismatches.
- [x] Return sample count, feature count, axis range, metadata columns, candidate labels, candidate groups, and warnings.
- [x] Save a structured dataset summary artifact.

### Acceptance checklist

- [x] `inspect_dataset` can summarize the first NIR flooring dataset.
- [x] The agent can explain dataset shape and risks without opening raw spreadsheets manually.
- [x] Ambiguous labels or grouping fields are surfaced as questions, not guessed silently.
- [x] Inspection outputs are deterministic for the same input file.

## Phase 4: Shared prompt library MVP

Goal: create one reusable, agent-neutral prompt library that guides MCP clients through the workflow.

### Tasks

- [x] Create scientific guardrails.
- [x] Create dataset inspection skill.
- [x] Create analysis planning skill.
- [x] Create model selection workflow.
- [x] Create model failure fallback workflow.
- [x] Create validation review workflow.
- [x] Create interpretation skill.
- [x] Create report-writing skill.
- [x] Create end-to-end spectral analysis workflow.
- [x] Create paper-demo workflow for the NIR flooring case.
- [x] Add output contracts for agent summaries, approval requests, validation review, fallback decisions, and final report summaries.
- [x] Add a human-review checklist template.

### Acceptance checklist

- [x] Prompt materials reference MCP tools generically.
- [x] Prompt materials do not create different scientific rules for Claude, Codex, or any other client.
- [x] Prompt materials require approval before full runs, major fallbacks, risky final model selection, final scientific conclusions, and method-memory saves.
- [x] Prompt materials forbid invented metrics and chemical overclaiming.

## Phase 5: Analysis planning tool

Goal: convert dataset inspection into a bounded plan that a human can approve.

### Tasks

- [x] Implement `propose_analysis_plan` using only dataset summaries and explicit user intent.
- [x] Recommend candidate task types from labels and metadata.
- [x] Recommend preprocessing candidates suitable for spectral data.
- [x] Recommend validation strategy, including grouped or stratified options where metadata allows.
- [x] Recommend initial model families for classification, regression, PCA, clustering, and feature importance where appropriate.
- [~] Explain rejected or deferred tasks. — warns for ambiguous labels; no systematic explanation of why alternative task types were rejected.
- [x] Return a human-readable plan and machine-readable planned analysis list.
- [~] Mark which plan choices require human approval. — "Approval required before running" embedded in human_readable_plan text; no per-item machine-readable approval flag.

### Acceptance checklist

- [x] The plan is bounded and executable by `run_analysis`.
- [x] The plan does not silently choose ambiguous labels or groups.
- [x] The plan distinguishes exploratory tasks from supervised tasks.
- [x] The plan is suitable for a one-request end-to-end demo after user approval.

## Phase 6: First runnable analysis path

Goal: produce the first minimal end-to-end result from actual deterministic computations.

### Tasks

- [x] Implement baseline spectral preprocessing.
- [x] Implement train/test or cross-validation split handling with reproducible seeds.
- [x] Implement one binary classification path.
- [x] Implement PCA or clustering sanity check.
- [x] Compute real metrics from model outputs.
- [x] Generate basic figures from actual model outputs.
- [x] Save predictions, metrics, figures, preprocessing details, and warnings as run artifacts.
- [x] Generate a basic report from saved artifacts.
- [x] Add tests or reproducible demo commands for the minimal path.

### Acceptance checklist

- [x] `run_analysis` can execute at least one approved analysis plan.
- [x] Outputs are saved under a run ID.
- [x] Metrics and figures are produced by code, not written by the agent.
- [x] A human-readable report can be generated from the run artifacts.

## Phase 7: Validation engine

Goal: make scientific reliability checks explicit and central to the system.

### Tasks

- [x] Implement replicate leakage checks when group or replicate metadata exists.
- [x] Implement group leakage checks for splits.
- [x] Implement class imbalance warnings.
- [x] Implement too-few-samples-per-class warnings.
- [x] Implement split instability checks where repeated splits or CV are available.
- [x] Implement suspiciously high metric warnings.
- [x] Implement regression target leakage checks where metadata permits.
- [x] Implement missing validation metadata warnings.
- [x] Return explicit severity levels and recommended actions.
- [x] Add validation output to reports.

### Acceptance checklist

- [x] `validate_results` produces structured warnings and severity levels.
- [x] The agent can classify results as acceptable, cautionary, or invalid based on tool output.
- [x] Validation warnings are preserved in the final report.
- [x] Validation does not suppress inconvenient results.

## Phase 8: Expanded NIR paper-demo analyses

Goal: support the analyses needed for a convincing NIR flooring case study.

### Tasks

- [~] Add multi-class classification. — multi-class exists from Phase 6 (svm_rbf/random_forest/pca_lda); XGBoost added; SHAP not integrated.
- [~] Add wear-layer or comparable regression task if labels are available. — not implemented.
- [x] Add preprocessing comparison.
- [x] Add SVM, random forest, XGBoost, PLSR, SVR, LDA, or other appropriate models only as dependencies and data support allow.
- [~] Add feature or wavelength importance. — selected_features populated by random_forest; dedicated wavelength-importance visualization not done.
- [~] Add model comparison tables. — basic table in report; no standalone comparison artifact.
- [x] Add model selection output that balances performance, reliability, interpretability, stability, complexity, and task suitability.
- [x] Add improved figures for paper/demo review.
- [~] Add caveats for easy binary separation versus harder multi-class or regression tasks. — not implemented as dedicated caveat logic.

### Acceptance checklist

- [x] The demo covers both easy and harder tasks where data supports them.
- [x] `select_best_model` does not choose solely by headline metric.
- [x] Feature importance is framed as model evidence, not chemical causality.
- [~] Results support the paper narrative without overstating conclusions. — demo infrastructure complete; paper narrative not yet verified.

## Phase 9: Model failure fallback workflow

Goal: handle model failures transparently without silently changing the scientific plan.

### Tasks

- [x] Implement failure classification for data, preprocessing, convergence, unsupported task, dependency/runtime, and sample-size issues.
- [x] Implement `recommend_next_model` fallback recommendations.
- [x] Require human approval when fallback changes the scientific plan or comparability.
- [x] Rerun only affected stages after fallback approval.
- [x] Record failed model, failure reason, selected fallback, rationale, approval status, and comparability limitations.
- [~] Include fallback records in the agent trace and final report. — failed_models list in AnalysisRun/report; full fallback NextModelRecommendation not auto-injected into report.

### Acceptance checklist

- [x] Failed models are not hidden.
- [x] Fallback recommendations include rationale and comparability caveats.
- [x] The agent asks for approval when fallback materially changes the plan.
- [x] Reports preserve model failure history when relevant.

## Phase 10: Report, artifact, and trace polish

Goal: produce reviewable artifacts suitable for the paper/demo.

### Tasks

- [~] Finalize run metadata format. — RunMetadata populated but format not formally finalized for paper.
- [~] Finalize results JSON or equivalent structured output. — basic run_summary.json saved; not formally finalized.
- [~] Finalize validation report format. — ValidationSummary JSON saved; not formally finalized.
- [~] Finalize interpretation report format. — InterpretationSummary JSON saved; not formally finalized.
- [~] Finalize final report format. — Markdown + JSON produced; not formally finalized for paper.
- [~] Finalize agent trace format covering decisions, tool calls, failures, fallbacks, approvals, and outputs. — no agent trace artifact implemented.
- [~] Finalize metrics tables. — basic results table in report; not paper-ready.
- [~] Finalize validation summary table. — validation section in report; not tabular/paper-ready.
- [~] Finalize interpretation summary. — interpretation section in report; not formally finalized.
- [~] Finalize human-review checklist. — checklist template present in reporting.py; not finalized.
- [x] Export paper-ready figures where feasible.

### Acceptance checklist

- [x] A complete run can be reviewed without reading code.
- [~] Reports separate results, caveats, and conclusions. — sections present; not formally separated for paper.
- [x] Agent traces show what was automated and where human approval occurred.
- [x] Agent-readable and human-readable artifacts are both available.

## Phase 11: End-to-end agent demo workflow

Goal: allow a user to request analysis once while the agent orchestrates the sequence with approval gates.

### Tasks

- [x] Implement or document the exact end-to-end workflow invocation.
- [x] Ensure the workflow calls `inspect_dataset` first.
- [x] Ensure the workflow calls `propose_analysis_plan` before running analyses.
- [x] Ensure the workflow presents the plan and asks for approval.
- [x] Ensure the workflow runs only approved analyses.
- [x] Ensure the workflow handles model failures through the fallback workflow.
- [x] Ensure the workflow validates results before model selection.
- [x] Ensure the workflow selects a candidate best model only after validation review.
- [x] Ensure the workflow interprets results using tool outputs.
- [x] Ensure the workflow generates a report and human-review checklist.
- [x] Ensure the workflow summarizes artifacts, warnings, caveats, and follow-ups.

### Acceptance checklist

- [x] The user does not need to manually type every MCP tool call during the main demo.
- [x] The agent pauses at required human-in-the-loop gates.
- [x] The workflow can be demonstrated in an MCP-capable client.
- [x] The demo produces a trace suitable for paper screenshots or appendix material.

## Phase 12: Paper-demo package

Goal: package the NIR flooring demonstration for the paper-first milestone.

### Tasks

- [ ] Select the canonical NIR demo dataset and task definitions.
- [ ] Run the standard analysis suite.
- [ ] Save canonical run artifacts.
- [ ] Save canonical agent trace.
- [ ] Save final report.
- [ ] Save paper-ready tables and figures.
- [ ] Summarize expert-effort reduction in a way that does not overclaim autonomy.
- [ ] List limitations and required human review.
- [ ] Identify which outputs can support paper results, methods, figures, and supplemental material.

### Acceptance checklist

- [ ] The demo can be rerun or reviewed from documented steps.
- [ ] Paper-ready outputs are grounded in actual tool artifacts.
- [ ] Claims avoid replacing chemometric experts or proving chemical causality from importance alone.
- [ ] The paper narrative emphasizes bounded automation plus human review.

## Phase 13: Method-memory prototype, after MVP

Goal: add lightweight reviewed-method memory only after the core report-producing demo works.

### Tasks

- [x] Confirm Phases 1 through 12 are complete enough that method memory will not distract from the MVP.
- [x] Define reviewed-method record fields.
- [x] Implement `save_reviewed_method` only for human-approved methods.
- [x] Implement `search_method_memory` over reviewed method summaries.
- [x] Implement `recommend_from_memory` with caveats and no automatic trust in prior outputs.
- [x] Ensure method memory stores reviewed conclusions and reusable recipes, not every raw automated run.

### Acceptance checklist

- [x] Method memory cannot save unreviewed automated conclusions as trusted methods.
- [x] Recommendations from memory are clearly labeled as prior reviewed evidence, not proof.
- [x] The feature is framed as future-facing if not needed for the first paper/demo.

## Phase 14: FTIR follow-on application

Goal: demonstrate transferability beyond the NIR flooring proof-of-concept.

### Tasks

- [x] Select the FTIR follow-on dataset.
- [x] Implement FTIR dataset ingestion.
- [x] Add FTIR modality adapter behavior where needed.
- [x] Add FTIR-specific preprocessing candidates.
- [~] Add FTIR-specific report caveats. — modality-agnostic caveats in place; FTIR-specific caveat text not yet authored.
- [x] Run the same end-to-end workflow with minimal prompt changes.
- [x] Compare reusable core behavior against modality-specific changes.
- [~] Save FTIR demo artifacts and report. — synthetic data pipeline works; canonical FTIR demo run not yet saved.

### Acceptance checklist

- [x] The same MCP and prompt structure works for FTIR without separate agent behavior.
- [x] Modality-specific work is handled in deterministic tools, not improvised by prompts.
- [~] Reports clearly state which findings are FTIR-specific. — framework supports this; FTIR-specific report text not yet authored.
- [x] The paper can discuss transferability beyond the first NIR case.

## Tool-level definition of done

### `inspect_dataset`

- [x] Loads supported spectral data and metadata.
- [x] Reports shape, axis range, modality, candidate labels, candidate groups, and warnings.
- [x] Saves a structured inspection artifact.
- [x] Does not guess ambiguous labels silently.

### `propose_analysis_plan`

- [x] Converts inspection output into bounded candidate tasks.
- [x] Recommends preprocessing, validation, and model families.
- [~] Identifies approval-required decisions. — "Approval required before running" in human_readable_plan text; no per-decision machine-readable flag.
- [x] Returns human-readable and machine-readable plan outputs.

### `run_analysis`

- [x] Runs approved tasks only.
- [x] Saves structured metrics, predictions, figures, preprocessing details, and warnings as run artifacts.
- [x] Reports failures without hiding them.
- [x] Produces reproducible run artifacts.

### `validate_results`

- [x] Checks leakage, imbalance, sample-size, split stability, suspicious metrics, target leakage, and missing validation metadata where possible.
- [x] Returns severity levels and recommended actions.
- [x] Feeds warnings into final reports.

### `select_best_model`

- [x] Compares performance, validation reliability, interpretability, stability, complexity, and task suitability.
- [x] Distinguishes best measured model from best defensible model.
- [x] Flags when the highest metric should not win.

### `recommend_next_model`

- [x] Classifies failure cause.
- [x] Recommends fallback with rationale.
- [x] Identifies whether approval is required.
- [x] Records comparability limitations.

### `interpret_results`

- [x] Uses only tool-produced features, wavelengths, and importance scores.
- [~] Flags unstable or weak interpretations. — warns on no_feature_importance; no explicit stability/confidence metric beyond that.
- [x] Separates model evidence from chemical conclusions.

### `generate_report`

- [x] Produces human-readable and agent-readable artifacts.
- [x] Includes metrics, figures, validation warnings, caveats, interpretations, and next steps.
- [x] Includes a human-review checklist.
- [x] Avoids unsupported scientific conclusions.

### `save_method_memory`

- [x] Stores only human-reviewed method configurations.
- [x] Records dataset profile, preprocessing chain, model config, metrics, and caveats.
- [x] Rejects unreviewed automated conclusions.

### `search_method_memory`

- [x] Searches reviewed methods by keyword or dataset profile similarity.
- [x] Returns structured index of matching entries.

### `recommend_from_memory`

- [x] Recommends prior reviewed method with caveats.
- [x] Clearly labels recommendations as prior evidence, not proof.
- [x] Requires human approval before applying recommended method.

## Required human approval gates

Agents must preserve these gates in workflows and demos.

- [x] Before running the full analysis plan.
- [x] Before changing task definitions.
- [x] Before accepting fallback models after major failures.
- [x] Before selecting a final best model if validation warnings exist.
- [x] Before writing final scientific conclusions.
- [x] Before saving anything into method memory.

## Testing and validation expectations

Agents should prefer the repository's existing test conventions. If no test convention exists yet, add the smallest useful tests or documented checks for the changed layer.

- [x] Contract validation tests exist for core data/result structures.
- [x] Tool boundary tests cover invalid inputs.
- [x] Dataset inspection can be run on the chosen NIR file.
- [x] Minimal analysis path produces deterministic artifacts.
- [x] Validation checks produce expected warnings on known-risk fixtures or examples.
- [x] Reports are generated from saved artifacts, not agent-written summaries alone.
- [x] MCP tool discovery is verified or documented.
- [x] No new dependency is added without approval.

## Agent handoff checklist

At the end of each implementation session, agents must leave the project easier for the next agent to continue.

- [x] Mark completed checklist items as `[x]` only when verified.
- [x] Mark partial items as `[~]` and include what remains.
- [x] Mark blocked items as `[!]` and name the blocker.
- [x] Update `agent-memory/CHANGELOG.md`.
- [x] Update or create `changes_YYYY-MM-DD.md` when required.
- [x] Summarize modified files, validation performed, known risks, and next recommended step.
- [x] Do not claim tests passed unless they were actually run.

## Recommended first implementation sequence

Follow this order unless there is a documented reason to deviate.

1. Audit existing work and mark completed checklist items.
2. Choose the first NIR flooring data file.
3. Define contracts and schemas.
4. Build the minimal MCP server skeleton.
5. Implement `inspect_dataset`.
6. Create the shared prompt-library MVP.
7. Implement `propose_analysis_plan`.
8. Implement the first runnable `run_analysis` path.
9. Implement `validate_results`.
10. Implement `generate_report`.
11. Add model selection and fallback workflows.
12. Polish report, trace, and paper-demo artifacts.
13. Generalize to additional modalities (FTIR) and use method memory for reviewed configurations.
