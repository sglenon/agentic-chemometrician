# Workflow: Model Failure Fallback

## Purpose

Handle a model failure transparently without silently changing the scientific
plan. Use `recommend_next_model` to classify the failure and obtain a reasoned
fallback recommendation.

## Trigger

A model in `run_analysis` returns a failure result, an exception, or a warning
indicating the model could not be trained or evaluated.

## Steps

### Step 1 — Identify and report the failure

- Record the failed model name.
- Extract the failure message from the `AnalysisResult` warnings or error field.
- Report the failure to the user immediately. Do not hide it.

### Step 2 — Classify the failure

Call `recommend_next_model` with:
- `failed_model`: the model name.
- `failure_reason`: the error or warning message from the tool.
- `candidate_models`: optional list of remaining models in the approved plan.

The tool will classify the failure into one of:
- **data**: insufficient or malformed input data.
- **preprocessing**: scaling, centering, or normalization issue.
- **convergence**: optimisation did not converge.
- **unsupported_task**: model does not support this task type.
- **dependency_runtime**: missing package or environment error.
- **sample_size**: too few samples for the model requirements.

### Step 3 — Evaluate the fallback recommendation

Read the `NextModelRecommendation` payload:
- Note the `failure_classification`.
- Note the `fallback_model` and `rationale`.
- Note whether `requires_human_approval=True`.

### Step 4 — Human approval gate (when required)

If `requires_human_approval=True`:
- **[HUMAN APPROVAL GATE]** Present the failure, classification, recommended fallback,
  rationale, and any comparability limitations.
- Ask the user to approve or reject the fallback before re-running.
- Do not re-run automatically.

If `requires_human_approval=False` and the fallback is a minor substitution:
- Inform the user of the fallback choice and rationale before proceeding.
- Proceed only after notification, not silently.

### Step 5 — Record the fallback

In the agent trace and final report, record:
- Failed model name.
- Failure classification.
- Selected fallback model.
- Rationale.
- Approval status (approved / not required / pending).
- Any comparability limitations (e.g. different regularization, different
  feature space, loss of comparability with earlier runs).

### Step 6 — Re-run affected stages only

Rerun only the stages that depend on the failed model. Do not re-run
the entire analysis plan.

## Fallback heuristics (reference only)

These are starting-point suggestions. The `recommend_next_model` tool provides
the authoritative recommendation.

| Failed model | Failure class | Suggested fallback |
|---|---|---|
| XGBoost | dependency_runtime | Random forest or SVM |
| SVM | convergence / preprocessing | LDA, PLS-DA, or random forest (after checking scaling) |
| Regression model | sample_size | Simpler baseline; report limited reliability |
| PCA / clustering | data | Return to ingestion and preprocessing |
| Feature importance | model-specific | Permutation importance if valid |
| All supervised models weak | — | Unsupervised exploration; review labels and data |

## Prohibited behaviors

- Do not hide a model failure from the user or the report.
- Do not auto-accept a fallback that changes the scientific plan without approval.
- Do not restart the full analysis plan when only one model failed.
- Do not invent a fallback rationale; cite the `recommend_next_model` output.
