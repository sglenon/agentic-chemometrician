# Skill: Report Writing

## Purpose

Generate a complete, human-reviewable analysis report using the `generate_report`
MCP tool. The report must be producible from saved artifacts alone, not from
agent-written summaries.

## When to use

After all planned analyses are complete, results are validated, a candidate best
model is identified, and interpretation is done.

## Steps

1. Call `generate_report` with:
   - The `AnalysisRun` containing the run metadata, results, and artifact references.
   - The `ValidationSummary` from `validate_results`.
   - The `InterpretationSummary` from `interpret_results`.

2. Read the `ReportSummary` payload including the primary report artifact path.

3. Summarise the report artifacts for the user:
   - Location of the primary report file.
   - List of included sections (metrics, figures, validation, interpretation).
   - List of warnings and caveats present in the report.
   - Location of the human-review checklist.

4. Present the human-review checklist to the user.

5. Ask the user to review the report before accepting any scientific conclusions.

## Human-review checklist template

Present this checklist when delivering the report:

- [ ] Sample count and feature count match expected dataset.
- [ ] Validation strategy is appropriate for the sample structure (no obvious leakage).
- [ ] Metrics are plausible (not suspiciously high for the task difficulty).
- [ ] Validation warnings are noted and their implications are understood.
- [ ] Feature importance interpretations are framed as model evidence, not chemistry proof.
- [ ] Caveats and limitations are present and accurate.
- [ ] Final conclusions are supported by the reported artifacts.
- [ ] Next steps and follow-up questions are recorded.

## Guardrails

- Do not write a final report from memory if `generate_report` returns `ok=False`.
- Do not omit validation warnings from the report summary.
- Do not present the report as final until the user completes the review checklist.
- Do not claim conclusions stronger than what the tool artifacts support.
