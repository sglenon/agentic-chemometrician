# Workflow: Model Selection

## Purpose

Compare candidate models and identify the most defensible model for the task
using the `select_best_model` MCP tool.

The best model is not automatically the model with the highest metric. It is
the model with the best defensible balance of:
- Performance on the primary metric.
- Validation reliability (no active leakage or imbalance warnings).
- Interpretability for the paper/demo narrative.
- Stability across folds or splits.
- Complexity relative to the sample size.
- Suitability to the task type.

## Steps

1. Confirm `validate_results` has been called and results are not classified
   as invalid. Do not select a model from invalid results.

2. Call `select_best_model` with:
   - The list of `AnalysisResult` objects.
   - The `ValidationSummary` from `validate_results`.
   - The task name if filtering by task type.

3. Read the `ModelSelectionRecommendation` payload.

4. Present the comparison to the user:
   - List candidate models with their primary metrics.
   - Explain the selection rationale from the tool output.
   - Highlight any gap between the highest-metric model and the recommended model.
   - Note whether the recommended model requires human approval.

5. If `requires_human_approval=True` in the recommendation:
   - **[HUMAN APPROVAL GATE]** Do not confirm model selection until approved.
   - Present the specific reasons why approval is required (e.g. validation warnings).

6. Record the selected model and rationale for the report and agent trace.

## Prohibited behaviors

- Do not automatically select the highest-metric model without checking
  validation reliability.
- Do not suppress the `requires_human_approval` flag.
- Do not select a model if `validate_results` has not been called.
