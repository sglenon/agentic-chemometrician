# Workflow: Validation Review

## Purpose

Evaluate the scientific reliability of analysis results using the
`validate_results` MCP tool. Classify results as acceptable, cautionary, or
invalid before model selection or interpretation.

## Steps

### Step 1 — Call validate_results

Call `validate_results` with:
- The `results` list from `run_analysis`.
- The `dataset` if group/replicate metadata is available for leakage checks.
- The `analysis_run` for full context if available.

### Step 2 — Read the ValidationSummary

Read the payload:
- `passed`: overall pass/fail.
- `checks`: individual check results.
- `warnings`: list of `ValidationWarning` objects with severity and category.

### Step 3 — Classify the results

| Condition | Classification |
|-----------|----------------|
| No warnings, passed=True | **Acceptable** — proceed to model selection |
| One or more `warning` severity warnings | **Cautionary** — proceed with caveats; note warnings in all subsequent summaries |
| Any `error` severity warning (e.g. leakage detected, invalid split) | **Invalid** — do not select or interpret; recommend rerun with corrected strategy |

### Step 4 — Report to the user

Present for each warning:
- Warning code and message.
- Severity (info / warning / error).
- Affected stage.
- Recommended action.

Do not filter out warnings. Suppressing inconvenient validation findings is
prohibited.

### Step 5 — Recommended actions by warning type

| Warning code | Recommended action |
|---|---|
| `replicate_leakage` | Rerun with grouped split; do not proceed to model selection |
| `group_leakage` | Rerun with grouped CV; flag comparability risk |
| `class_imbalance` | Use stratified split; report balanced vs. macro metrics |
| `small_sample_per_class` | Report that conclusions may be unreliable; increase sample size if possible |
| `split_instability` | Increase CV folds; report confidence intervals |
| `suspicious_high_metric` | Investigate potential leakage before accepting results |
| `target_leakage` | Halt; identify and remove leaking features |
| `missing_validation_metadata` | Note which checks could not be performed |

### Step 6 — Proceed or halt

- **Acceptable**: proceed to `select_best_model`.
- **Cautionary**: proceed to `select_best_model` but carry all warnings through to
  the report; require human approval at model selection.
- **Invalid**: do not proceed. Report the issue, recommend corrective action, and
  await human instruction.

## Prohibited behaviors

- Do not omit validation warnings from subsequent summaries.
- Do not classify `error`-severity warnings as cautionary.
- Do not proceed to model selection after an invalid classification without
  explicit human instruction.
