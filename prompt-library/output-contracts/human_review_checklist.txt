# Output Contract: Human Review Checklist

This checklist must be presented to the user before any final scientific
conclusions are accepted from a completed analysis run.

## Checklist template

```markdown
## Human Review Checklist — Run [run_id]

### Dataset
- [ ] Sample count matches expected: [N] samples.
- [ ] Feature count is correct: [F] features.
- [ ] Spectral axis range is plausible for the modality: [axis_min]–[axis_max] [units].
- [ ] Label column used is correct: [label_column].

### Validation
- [ ] Validation strategy is appropriate (no obvious leakage from replicate or group structure).
- [ ] Class balance is acceptable for the task.
- [ ] Split stability was checked if multiple folds were used.
- [ ] All validation warnings have been reviewed: [list warnings or "none"].

### Results
- [ ] Metrics are plausible (not suspiciously high for the task difficulty).
- [ ] Results are consistent with the expected difficulty of this task.
- [ ] Any model failures were reviewed and fallback rationale is acceptable.
- [ ] The selected best model is defensible (not chosen solely by headline metric).

### Interpretation
- [ ] Feature importance is framed as model evidence, not chemical causality.
- [ ] No unsupported chemical conclusions are present.
- [ ] Unstable or weak interpretations are flagged.

### Report
- [ ] Caveats and limitations are present and accurate.
- [ ] The report separates results, caveats, and conclusions.
- [ ] Artifact paths and run ID are recorded for reproducibility.

### Sign-off
- [ ] I have reviewed the above items and accept these results as a basis for
      further analysis / paper inclusion / method development.

**Reviewer**: _______________
**Date**: _______________
**Notes**: _______________
```

## Rules

- This checklist must be included in every final report.
- The agent must present this checklist and ask the user to complete it before
  accepting final conclusions.
- The agent must not mark any checklist item as complete on the user's behalf.
