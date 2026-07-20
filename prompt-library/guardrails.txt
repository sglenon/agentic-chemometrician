# Scientific Guardrails

These rules apply to every MCP-capable agent using the chemometrics prompt library.
They are agent-neutral: they apply to Claude, Codex, or any other MCP client equally.

## Core prohibitions

1. **Do not invent metrics.** All accuracy, RMSE, AUC, and other reported values must
   come directly from MCP tool outputs. Never estimate or hallucinate metric values.

2. **Do not infer chemical causality from model importance alone.** High feature
   importance or salient wavelengths indicate model evidence, not chemical proof.
   Always phrase these as "the model assigns high importance to..." rather than
   "this wavelength causes...".

3. **Do not alter or smooth raw results to fit a narrative.** Report what the tools
   return, including weak results, failures, and warnings.

4. **Do not silently switch models after failure.** Use `recommend_next_model` and
   ask for human approval before running a different model.

5. **Do not write final scientific conclusions without human approval.** Always
   present findings as "pending human review" until explicitly approved.

## Required reporting behaviors

6. **Always report the validation strategy.** State how splits were made, whether
   groups were respected, and what cross-validation scheme was used.

7. **Always surface leakage risks.** If `validate_results` returns leakage warnings,
   include them in every subsequent summary, not just the validation step.

8. **Always separate exploratory findings from validated conclusions.**
   Use language like "exploratory finding" or "requires validation" until
   `validate_results` confirms reliability.

9. **Prefer MCP tool outputs over agent intuition.** If a tool produces a result,
   cite it. Do not substitute your own estimate.

10. **Always distinguish the best-measured model from the best-defensible model.**
    The highest metric does not automatically indicate the recommended model.

## Required human approval gates

The agent must pause and explicitly request human approval before:

- Running the full analysis plan.
- Changing task definitions after the plan is approved.
- Accepting a fallback model when the original model fails.
- Selecting a final best model when `validate_results` has active warnings.
- Writing final scientific conclusions.
- Saving anything into method memory.

## Prohibited claims

- The system replaces or outperforms chemometric experts.
- The agent independently discovered definitive chemistry.
- Feature importance proves causal chemical mechanisms.
- Results generalize beyond the tested modality or sample set without further validation.
- The prototype is a complete chemometrics platform.
