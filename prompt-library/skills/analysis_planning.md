# Skill: Analysis Planning

## Purpose

Convert a `DatasetInspection` summary into a bounded analysis plan using the
`propose_analysis_plan` MCP tool. Explain the plan to the user and obtain
approval before any analysis runs.

## When to use

After `inspect_dataset` has completed successfully and label/group columns
have been confirmed.

## Steps

1. Call `propose_analysis_plan` with:
   - The `DatasetInspection` payload from the inspection step.
   - The user's stated intent if provided (e.g. "classify floor materials",
     "predict wear layer thickness").
   - A `task_hint` if the user has specified a task type.

2. Read the `AnalysisPlan` payload from the response.

3. Present the plan to the user:
   - List the recommended task types and explain why each was recommended.
   - List any tasks that were rejected or deferred, and explain why.
   - List the recommended preprocessing candidates.
   - State the recommended validation strategy and why
     (e.g. grouped split due to replicate metadata).
   - List the initial model families proposed.
   - Identify which choices require human approval.

4. If `ok=False`, report the error. Do not fabricate a plan.

5. Ask for explicit human approval before proceeding to `run_analysis`.
   This is a required human-in-the-loop gate.

6. If the user modifies the plan:
   - Record the changes.
   - Flag whether the modification changes comparability or risk level.
   - Ask for re-confirmation before running.

## Output to user

Present the plan as a numbered list:
1. Tasks to run (with brief rationale for each).
2. Preprocessing steps (with brief rationale).
3. Validation strategy.
4. Models to try (with brief rationale).
5. Decisions that need approval.

End with a clear approval request:
> "Do you approve this analysis plan? You may modify any part before confirming."

## Guardrails

- Do not run any analysis before the user approves the plan.
- Do not invent task recommendations not supported by the inspection output.
- Do not assume a task is valid if `propose_analysis_plan` returns a warning
  about missing labels or insufficient sample size.
