# Skill: Dataset Inspection

## Purpose

Load and summarise a spectral dataset using the `inspect_dataset` MCP tool.
Surface shape, modality, axis range, candidate labels, candidate groups, and
data-quality risks so the user can make an informed decision before analysis.

## When to use

Use this skill at the start of every analysis session, before calling
`propose_analysis_plan` or `run_analysis`.

## Steps

1. Call `inspect_dataset` with the dataset file path.
   - If `modality_override` is known (e.g. "NIR", "FTIR"), provide it.
   - If the user has named a label column, provide it as `label_column`.

2. Read the `DatasetInspection` payload from the response.

3. Summarise the result for the user:
   - Sample count and feature count.
   - Spectral axis range and inferred or confirmed modality.
   - Candidate label columns (what supervised tasks may be possible).
   - Candidate group or replicate columns (relevant to split strategy).
   - Any warnings from the tool (missing values, shape mismatch, duplicate IDs).

4. If `ok=False`, report the error and stop. Do not proceed to planning.

5. If `warnings` contains `ambiguous_label_columns`:
   - List the candidate columns.
   - Ask the user which column to use as the primary label before proceeding.
   - Do not guess silently.

6. If `warnings` contains `no_label_columns`:
   - Inform the user that supervised tasks (classification, regression) are not
     available without label data.
   - Suggest unsupervised exploration (PCA, clustering) as an alternative.

7. If `warnings` contains `small_sample`:
   - Highlight the sample count and explain why results will be unreliable.
   - Do not suppress this warning in summaries.

## Output to user

Provide a concise summary including:
- Dataset shape.
- Spectral modality and axis range.
- Available label and group columns.
- Active warnings with severity.
- Recommended next step (typically: approve label selection, then proceed to planning).

## Guardrails

- Do not invent shape, label, or modality information not returned by the tool.
- Do not auto-select a label if multiple candidates exist; ask the user.
- Do not proceed to `propose_analysis_plan` without confirmed label selection
  if any supervised task will be planned.
