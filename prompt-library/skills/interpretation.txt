# Skill: Result Interpretation

## Purpose

Summarise feature or wavelength importance and model evidence using the
`interpret_results` MCP tool. Separate model evidence from chemical conclusions.

## When to use

After `validate_results` has completed and results have been classified as
acceptable or cautionary. Do not interpret results classified as invalid.

## Steps

1. Call `interpret_results` with:
   - The `AnalysisResult` list from the run.
   - The `SpectralDataset` for axis/metadata context if available.
   - The `ValidationSummary` so instability context is preserved.

2. Read the `InterpretationSummary` payload.

3. Present the interpretation to the user:
   - List the most important features or wavelengths as reported by the tool.
   - Note whether importance was consistent across models or model-specific.
   - Highlight any unstable or weak interpretations flagged by the tool.

4. Apply the following language rules strictly:
   - Use: "the model assigns high importance to wavelength X"
   - Use: "feature X was consistently selected across models"
   - Avoid: "wavelength X causes Y" or "wavelength X proves Z"
   - Avoid: any chemical assignment not supported by the tool output

5. If `validate_results` returned active warnings, include a caveat:
   > "These interpretations should be treated with caution given the active
   > validation warnings: [list warnings]. Further validation is recommended."

## Output to user

- Top features or wavelengths with importance scores (from tool output only).
- Cross-model consistency note.
- Any instability or weakness flags.
- Explicit separation of model evidence from chemical conclusions.
- Caveat if validation warnings are active.

## Guardrails

- Only cite wavelengths or features that appear in the tool output.
- Do not claim chemical causality from importance alone.
- Do not interpret results that `validate_results` has marked as invalid.
- Flag all unstable interpretations with explicit uncertainty language.
