# Workflow: End-to-End Spectral Analysis

## Purpose

Orchestrate a complete spectral analysis from a single user request, pausing
at required human approval gates. The user should not need to manually invoke
every MCP tool.

## Trigger

The user provides a spectral dataset file and a goal such as:
> "Analyse this NIR dataset to classify floor material types."

## Prerequisites

- `inspect_dataset` is available and functional.
- `propose_analysis_plan`, `run_analysis`, `validate_results`, `select_best_model`,
  `interpret_results`, and `generate_report` are available (or will skip with a
  clear deferred notice).

## Workflow steps

### Step 1 — Inspect dataset

Apply the **Dataset Inspection** skill.

- Call `inspect_dataset`.
- Summarise shape, modality, labels, groups, warnings.
- Resolve any ambiguous label columns before proceeding.

### Step 2 — Propose analysis plan

Apply the **Analysis Planning** skill.

- Call `propose_analysis_plan`.
- Present the proposed plan.
- **[HUMAN APPROVAL GATE]** Ask for approval before proceeding.

### Step 3 — Run approved analyses

- Call `run_analysis` with the approved plan.
- Report progress; surface any model failures immediately.
- If a model fails, enter the **Model Failure Fallback** workflow.

### Step 4 — Validate results

Apply the **Validation Review** workflow.

- Call `validate_results`.
- Classify results as acceptable, cautionary, or invalid.
- Surface all warnings. Do not suppress inconvenient findings.

### Step 5 — Select candidate best model

- Call `select_best_model`.
- If `validate_results` returned active warnings:
  - **[HUMAN APPROVAL GATE]** Present the warnings and ask for approval before
    accepting a model selection.
- Distinguish the best-measured model from the best-defensible model.

### Step 6 — Interpret results

Apply the **Interpretation** skill.

- Call `interpret_results`.
- Present importance and evidence with appropriate uncertainty language.
- Do not claim chemical causality.

### Step 7 — Generate report

Apply the **Report Writing** skill.

- Call `generate_report`.
- Present the human-review checklist.
- **[HUMAN APPROVAL GATE]** Ask the user to complete the review checklist before
  accepting final conclusions.

### Step 8 — Summarise artifacts

Present to the user:
- Run ID and artifact directory.
- Report file location.
- Agent trace location.
- Any open warnings, caveats, or follow-up questions.
- A reminder that final scientific conclusions require human sign-off.

## Required approval gates (summary)

| Gate | When |
|------|------|
| Plan approval | Before running analysis (Step 2) |
| Model selection approval | If validation warnings are active (Step 5) |
| Report review | Before accepting final conclusions (Step 7) |

## Deferred tool handling

If any tool returns `ok=False` with a deferred message:
- Inform the user that this step is not yet implemented.
- State the expected implementation phase from the error message.
- Offer to proceed with available steps and document the gap.
- Do not fabricate results for the deferred step.
