# Plan: MCP-Based Agentic Chemometrics Prototype

## Status

Planning document for an MCP-first, paper-first agentic chemometrics prototype. This document does not describe implemented functionality yet.

## Guiding direction

Optimize for a paper-first prototype with a strong agent demo. Reusability matters, but it should not slow down the first paper/demo milestone.

The system should be an MCP-based chemometrics tool layer that can be used by Claude, Codex, or another MCP-capable agent. It should not be a framework that requires both Claude and Codex, and it should not maintain separate behavior for each client.

## Core thesis

Build an MCP-based agentic chemometrics prototype where an MCP-capable agent orchestrates bounded, deterministic chemometric tools for spectral method development. The system should reduce routine expert effort while preserving human review as the gatekeeper for scientific conclusions.

## Architecture

```text
MCP-capable agent client
  -> shared chemometrics skills, workflows, and guardrails
    -> chemometrics MCP server
      -> deterministic chemometrics core
        -> run artifacts
          -> human-reviewable report
```

## Responsibility split

### Agent client

The agent client may be Claude, Codex, or another MCP-capable interface. It should:

- Read the shared prompt library.
- Start the end-to-end workflow from a user request.
- Call MCP tools in the recommended sequence.
- Explain the proposed analysis plan.
- Ask for human approval at required decision gates.
- Summarize results, warnings, and caveats.
- Avoid making unsupported scientific conclusions.

### MCP server

The MCP server should own deterministic scientific operations:

- Dataset loading and inspection.
- Metadata harmonization.
- Preprocessing execution.
- Model training and evaluation.
- Cross-validation and split handling.
- Leakage and reliability checks.
- Feature and wavelength interpretation outputs.
- Report and artifact generation.

### Shared prompt library

The prompt library should define agent behavior independent of client. It should include:

- Skills.
- Workflows.
- Scientific guardrails.
- Output contracts.
- Model selection rules.
- Model failure fallback rules.
- Human-in-the-loop requirements.

## Scope

### In scope

- MCP server exposing deterministic chemometrics tools.
- Agent-neutral prompt library usable by Claude, Codex, or other MCP-capable clients.
- NIR flooring proof-of-concept as the first paper/demo case.
- FTIR follow-on application after the NIR MVP to demonstrate transfer beyond the first modality.
- Standard data and result contracts based on `SpectralDataset` and `AnalysisResult`.
- End-to-end guided workflow so users do not need to manually type every tool call.
- Human-in-the-loop approval gates before consequential scientific decisions.
- Deterministic scientific tool layer for ingestion, preprocessing, modeling, validation, interpretation, and reporting.
- Human-reviewable reports with metrics, warnings, figures, interpretation, caveats, and recommended next steps.
- Agent trace artifacts showing decisions, tool calls, failures, fallbacks, approvals, and outputs.
- Reusable structure where it supports the paper/demo and FTIR follow-on.

### Out of scope for the MVP

- Full general-purpose chemometrics package.
- Full support for all spectroscopy modalities.
- Fully autonomous scientific conclusions.
- Dataset-specific conclusions embedded in prompts.
- Separate prompt libraries for Claude and Codex.
- Cloud deployment.
- Arbitrary Python or shell execution through MCP tools.
- Method-memory automation before the core report-producing demo works.

## Standard data contract

Each analysis module should operate on a common dataset object:

```python
SpectralDataset(
    X,              # spectra as samples x features
    axis,           # wavelengths, wavenumbers, m/z values, 2-theta values, etc.
    metadata,       # parsed sample metadata
    labels,         # optional task labels
    modality,       # NIR, FTIR, Raman, pXRD, XRF, LIBS, MS, etc.
)
```

## Standard result contract

Each analysis module should return a common result object:

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

## Proposed MCP tool surface

### `inspect_dataset`

Purpose:

- Load spectral data and metadata.
- Detect sample count, feature count, and axis range.
- Detect candidate label columns.
- Detect possible group or replicate columns.
- Identify modality if possible.
- Return warnings about missing labels, small sample size, ambiguous metadata, or invalid data.

### `propose_analysis_plan`

Purpose:

- Convert dataset summary into a bounded analysis plan.
- Recommend task types.
- Recommend preprocessing candidates.
- Recommend validation strategy.
- Recommend initial model families.
- Return a human-readable plan for approval.

### `run_analysis`

Purpose:

- Execute approved analyses.
- Save structured run outputs.
- Return run ID, result summaries, artifact locations, and failures.

Initial tasks:

- Binary classification.
- Multi-class classification.
- Regression.
- PCA or clustering.
- Feature or wavelength importance.

### `validate_results`

Purpose:

- Check scientific reliability risks.

Validation checks:

- Replicate leakage.
- Group leakage.
- Class imbalance.
- Too few samples per class.
- Split instability.
- Suspiciously high metrics.
- Regression target leakage.
- Missing validation metadata.

### `select_best_model`

Purpose:

- Compare candidate models and recommend the most defensible model for the task.
- Avoid choosing a model based only on the highest headline metric.
- Consider validation reliability, interpretability, stability, complexity, and task suitability.

### `recommend_next_model`

Purpose:

- Recommend the next model to try when a model fails or is invalidated.
- Classify the failure cause.
- Return fallback rationale and whether human approval is required.

### `interpret_results`

Purpose:

- Summarize feature or wavelength importance.
- Compare evidence across models.
- Separate model evidence from chemical conclusions.
- Identify unstable or weak interpretations.

### `generate_report`

Purpose:

- Produce final human-reviewable report.
- Include metrics, figures, validation warnings, caveats, and next-step recommendations.
- Write both human-readable and agent-readable artifacts.

### Later method-memory tools

Potential later tools:

- `save_reviewed_method`
- `search_method_memory`
- `recommend_from_memory`

Method memory should store human-reviewed conclusions and reusable recipes, not every raw automated output.

## MVP workflow

The normal user experience should be end-to-end.

```text
User: Analyze this spectral dataset.

Agent:
  1. Starts the shared end-to-end chemometrics workflow.
  2. Calls inspect_dataset.
  3. Calls propose_analysis_plan.
  4. Presents the proposed plan.
  5. Asks for human approval.
  6. Runs approved analyses.
  7. Handles model failures through the fallback workflow.
  8. Validates results.
  9. Selects a candidate best model.
  10. Interprets results.
  11. Generates a report.
  12. Presents artifacts, caveats, and a human-review checklist.
```

The user should not need to manually type each MCP tool call during the main demo. The agent should orchestrate the workflow while preserving approval gates.

## Human-in-the-loop gates

The workflow should pause for approval:

- Before running the full analysis plan.
- Before changing task definitions.
- Before accepting fallback models after major failures.
- Before selecting a final best model if validation warnings exist.
- Before writing final scientific conclusions.
- Before saving anything into method memory.

## Shared prompt library

### Guiding principle

There should be one shared prompt library for all MCP clients. Client-specific setup notes may exist, but the scientific behavior should be shared.

### Required components

#### Scientific guardrails

Rules should include:

- Do not invent metrics.
- Do not infer chemical causality from model importance alone.
- Always report validation strategy.
- Always surface leakage risks.
- Always separate exploratory findings from validated conclusions.
- Always require human review before final scientific conclusions.
- Prefer MCP tool outputs over agent intuition.
- Do not alter raw results to fit the narrative.
- Do not silently switch models after failure.
- Explain why fallback models are selected.
- Distinguish best measured model from best scientifically defensible model.

#### Dataset inspection skill

Expected behavior:

- Call `inspect_dataset`.
- Summarize shape, modality, axis, labels, metadata quality, and risks.
- Ask clarifying questions if labels or task definitions are ambiguous.

#### Analysis planning skill

Expected behavior:

- Call `propose_analysis_plan`.
- Explain recommended tasks.
- Explain rejected or deferred tasks.
- Ask for user approval before running.

#### Model selection workflow

Expected behavior:

- Compare models by task type.
- Compare validation design.
- Compare primary and secondary metrics.
- Compare split stability.
- Compare interpretability.
- Compare model complexity against performance gain.
- Flag when the highest-performing model should not be selected.
- Recommend the best defensible model.

The best model is not automatically the model with the highest metric. It is the model with the best defensible balance of performance, validation reliability, interpretability, and task suitability.

#### Model failure fallback workflow

Expected behavior:

- Identify the failed model.
- Classify the failure as a data issue, preprocessing issue, convergence issue, unsupported task, dependency/runtime issue, or sample-size issue.
- Recommend the next model to try.
- Ask for approval if the fallback changes the scientific plan.
- Rerun only affected stages.
- Record failed model, failure reason, selected fallback, rationale, approval status, and comparability limitations.

Example fallback logic:

- If XGBoost fails due to dependency or runtime issues, try random forest or SVM.
- If SVM fails due to scaling or convergence, verify preprocessing and scaling, then try LDA, PLS-DA, or random forest.
- If regression fails due to small sample size, try simpler baselines and report limited reliability.
- If PCA or clustering fails due to missing or invalid numeric spectra, return to ingestion and preprocessing.
- If feature importance fails for model-specific reasons, use permutation importance if valid.
- If all supervised models are weak, recommend unsupervised exploration and data or label review.

#### Validation review workflow

Expected behavior:

- Call `validate_results`.
- Review leakage, grouping, class imbalance, split stability, suspicious metrics, and sample-size warnings.
- Classify results as acceptable, cautionary, or invalid.
- Recommend rerun strategy if validation is weak.

#### Interpretation skill

Expected behavior:

- Call `interpret_results`.
- Mention wavelengths or features only if produced by tools.
- Flag unstable or inconsistent importance.
- Avoid chemical overclaiming.

#### Report-writing skill

Expected behavior:

- Call `generate_report`.
- Summarize generated artifacts.
- Present a human-review checklist.
- Highlight caveats and follow-ups.

#### End-to-end spectral analysis workflow

Steps:

1. Inspect dataset.
2. Propose plan.
3. Ask approval.
4. Run analyses.
5. Handle failures and fallback choices.
6. Validate results.
7. Select candidate best model.
8. Interpret results.
9. Generate report.
10. Summarize artifacts.
11. Ask for human review.

#### Paper-demo workflow

Steps:

1. Use the NIR flooring dataset.
2. Produce an agent plan.
3. Run the standard analysis suite.
4. Save an agent trace.
5. Generate final report.
6. Summarize expert-effort reduction.
7. Produce paper-ready tables and figures.

#### FTIR follow-on workflow

Steps:

1. Reuse the same MCP and prompt structure on FTIR data.
2. Identify FTIR-specific preprocessing needs.
3. Compare what transfers cleanly from NIR.
4. Identify what requires modality adapter changes.
5. Generate FTIR run artifacts and report.

## Standard artifacts

Each completed run should produce:

- Agent plan.
- Run metadata.
- Results JSON or equivalent structured output.
- Validation report.
- Interpretation report.
- Figures.
- Tables.
- Final report.
- Agent trace.
- Human-review checklist.

## Phase breakdown

### Phase 1: Planning and contracts

Goal: define the conceptual foundation before implementation.

Deliverables:

- Final project plan.
- Data contract specification.
- Result contract specification.
- MCP tool contract specification.
- Prompt library specification.

Validation:

- The plan clearly distinguishes the MCP server, deterministic chemometrics core, shared prompt library, run artifacts, and paper/demo outputs.

### Phase 2: Minimal MCP server skeleton

Goal: create the smallest MCP server with stable tool contracts.

Deliverables:

- MCP server entry point.
- Tool registration.
- Dataset and run ID conventions.
- Artifact directory conventions.
- Initial tool schemas.

Validation:

- MCP clients can discover tools.
- Tool schemas are understandable from Claude, Codex, or another MCP client.
- No arbitrary code execution is exposed.

### Phase 3: NIR flooring dataset ingestion

Goal: load the current flooring spectral dataset into the standard contract.

Deliverables:

- Dataset loader.
- Metadata parser.
- Axis or wavelength handling.
- Label candidate detection.
- Group or replicate candidate detection.
- Dataset summary output.

Validation:

- `inspect_dataset` returns sample count, feature count, metadata columns, candidate labels, and warnings.
- The agent can explain the dataset without manually inspecting raw files.

### Phase 4: Shared prompt library MVP

Goal: create agent-neutral skills and workflows usable by Claude, Codex, or another MCP client.

Deliverables:

- Scientific guardrails.
- Dataset inspection skill.
- Analysis planning skill.
- Model selection workflow.
- Model failure fallback workflow.
- Validation review workflow.
- Interpretation skill.
- Report-writing skill.
- End-to-end spectral analysis workflow.
- Paper-demo workflow.

Validation:

- Same prompt materials can guide multiple MCP clients.
- Prompts reference MCP tools generically.
- Prompts do not contain dataset-specific conclusions.
- Prompts require human review before final scientific claims.

### Phase 5: First runnable analysis path

Goal: run a minimal end-to-end chemometric analysis.

Deliverables:

- Preprocessing baseline.
- Binary classification task.
- PCA or clustering sanity check.
- Metrics output.
- Basic figures.
- Basic report.

Validation:

- An agent can start the end-to-end workflow from one user request.
- A human-readable report is produced from actual outputs.
- Human approval gates are present.

### Phase 6: Validation engine

Goal: make scientific reliability checks a central contribution.

Deliverables:

- Group or replicate leakage checks.
- Stratified and grouped split recommendations.
- Class imbalance warnings.
- Small-sample warnings.
- Split stability checks.
- Suspicious metric warnings.

Validation:

- `validate_results` produces explicit warnings and severity levels.
- The report includes validation caveats.
- The agent can explain whether results are acceptable, cautionary, or invalid.

### Phase 7: Expanded paper-demo analyses

Goal: support the analyses needed for a convincing NIR flooring case study.

Deliverables:

- Multi-class classification.
- Wear-layer regression.
- Preprocessing comparison.
- Feature importance.
- Model comparison table.
- Model selection output.
- Improved figures.

Validation:

- Demo covers easy and hard tasks.
- Pipeline recognizes that binary separation is easy.
- Pipeline escalates to harder tasks and caveats.
- Results support the paper narrative.

### Phase 8: Report and trace polish

Goal: produce paper/demo-quality artifacts.

Deliverables:

- Final report format.
- Agent trace format.
- Metrics tables.
- Validation summary table.
- Interpretation summary.
- Human-review checklist.
- Paper-ready figure export.

Validation:

- A complete run can be reviewed without reading code.
- Agent trace shows what was automated.
- Report separates results, caveats, and conclusions.

### Phase 9: Method-memory prototype

Goal: add a lightweight reviewed-method memory interface after the main demo works.

Deliverables:

- Store reviewed method summaries.
- Search previous reviewed runs.
- Recommend preprocessing, model, and validation choices based on prior reviewed results.

Validation:

- Method memory stores only reviewed conclusions.
- It does not automatically trust raw automated outputs.
- It is clearly framed as future-facing in the paper.

### Phase 10: FTIR follow-on application

Goal: apply the MCP server and shared prompt workflows to FTIR data after the NIR flooring MVP works.

Deliverables:

- FTIR dataset ingestion path.
- FTIR modality adapter.
- FTIR preprocessing candidates.
- FTIR-specific report caveats.
- Comparison of reusable core versus modality-specific changes.
- FTIR demo run artifacts.

Validation:

- Same end-to-end workflow can run on FTIR with minimal prompt changes.
- Modality-specific preprocessing is handled by tools, not improvised by the agent.
- Report clearly states which conclusions are FTIR-specific.
- Paper can discuss transferability beyond NIR.

## Paper framing

### Proposed thesis

We present an MCP-based agentic chemometrics prototype that lets MCP-capable AI agents orchestrate bounded, deterministic spectral analysis tools for method-development tasks, reducing routine expert effort while preserving human review as the gatekeeper for scientific conclusions.

### Main contributions

- MCP tool layer for controlled chemometric operations.
- Agent-neutral prompt library for reliable MCP-client orchestration.
- End-to-end guided workflow with human-in-the-loop gates.
- Validation-first workflow that flags leakage, imbalance, and robustness risks.
- Model selection and fallback logic that prioritizes defensible science over raw metrics.
- Human-reviewable report generation from structured artifacts.
- NIR flooring case study as the first paper/demo.
- FTIR follow-on case to demonstrate transferability.

### Claims to avoid

- The system replaces chemometric experts.
- LLMs independently discover definitive chemistry.
- Feature importance proves causal chemical mechanisms.
- The MVP generalizes fully to all spectral modalities.
- The system is a complete chemometrics platform.

## Risk assessment

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| MCP demo becomes more engineering than science | Medium | High | Keep NIR flooring paper workflow as the primary success criterion. |
| Agent overclaims results | High | High | Use shared scientific guardrails and report templates. |
| Tool surface becomes too broad | Medium | Medium | Start with the core tools only. |
| Prompt library fragments across clients | Medium | Medium | Maintain one agent-neutral library. |
| Validation checks reveal weak existing results | Medium | Medium | Treat this as a strength: the system catches scientific risk. |
| Method memory distracts from MVP | Medium | Medium | Defer until after the report-producing demo works. |
| Reusability slows the prototype | Medium | High | Use clean contracts, but avoid full package overengineering. |
| End-to-end automation reduces user control | Medium | High | Add explicit approval gates for consequential decisions. |
| Model fallback changes comparability | Medium | Medium | Record fallback rationale and comparability limitations. |

## Open questions

- Which exact flooring data files should the MVP loader target first?
- Which metadata column best represents true sample groups or replicates?
- Should the first classification baseline be SVM/LDA, random forest, or XGBoost?
- Should wear-layer regression use SVR first, or should PLSR be included early because it is chemometrically familiar?
- Should the shared prompt library mimic an existing skills/workflows format or stay as generic markdown first?
- Should the final report be markdown-only first, or markdown plus HTML/PDF later?
- Which MCP client should be used for initial paper screenshots and traces?
- Which FTIR dataset should be used for the follow-on case?

## Recommended next implementation steps

1. Finalize tool contracts for the MCP server.
2. Choose the first NIR flooring data files for ingestion.
3. Define the shared prompt library folder structure.
4. Implement the smallest MCP server skeleton.
5. Implement `inspect_dataset` for the NIR flooring dataset.
6. Create the first end-to-end workflow prompt.
7. Build the first report-producing analysis path.
