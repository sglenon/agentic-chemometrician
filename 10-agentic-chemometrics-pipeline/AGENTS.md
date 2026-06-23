# Agent Instructions: Agentic Chemometrics Pipeline

These instructions apply to work inside this folder and its descendants.

## Project priority

Optimize work in this order:

1. Paper-first prototype.
2. Strong MCP-based agent demo.
3. Reusable architecture where it supports the first two goals.

Do not overbuild a general chemometrics framework before the NIR flooring demo produces reliable, reviewable artifacts.

## Core architecture

The intended system has three layers:

```text
MCP-capable agent client
  -> shared chemometrics prompt library
    -> deterministic chemometrics MCP tools
```

The agent client may be Claude, Codex, or another MCP-capable tool. Do not create separate scientific behavior for Claude versus Codex unless explicitly requested.

## Scientific boundaries

- Keep numerical analysis inside deterministic Python/MCP tools.
- Do not let prompts invent metrics, figures, model outputs, or validation results.
- Do not claim chemical causality from feature importance alone.
- Always distinguish exploratory findings from validated conclusions.
- Preserve human review as the final gate for scientific claims.

## Human-in-the-loop requirements

End-to-end workflows should reduce manual command burden, but must pause for approval:

- Before running the full analysis plan.
- Before changing task definitions.
- Before accepting fallback models after major failures.
- Before selecting a final best model if validation warnings exist.
- Before writing final scientific conclusions.
- Before saving anything into method memory.

## Prompt library rules

The prompt library should be agent-neutral and reusable across MCP clients. It should include:

- Scientific guardrails.
- Skills.
- Workflows.
- Output contracts.
- Model selection rules.
- Model failure fallback rules.
- Human-review checklists.

Do not embed dataset-specific conclusions in reusable prompts.

## Model selection and fallback rules

- The best model is not automatically the model with the highest metric.
- Prefer the most defensible balance of performance, validation reliability, interpretability, stability, and task suitability.
- If a model fails, classify the failure before choosing a fallback.
- Record the failed model, failure reason, selected fallback, rationale, approval status, and comparability limitations.

## Agent-memory changelog

Maintain a changelog in `agent-memory/CHANGELOG.md` for meaningful agent work in this folder.

For every meaningful change, append an entry with:

- Date.
- Request summary.
- Files created or modified.
- Key decisions.
- Validation performed or not performed.
- Known risks and follow-ups.

If the `agent-memory/` folder or changelog does not exist, create it before completing the task.

## Change discipline

- Make surgical changes only.
- Inspect nearby files before editing.
- Do not refactor unrelated notebooks or modules.
- Do not add dependencies without explicit approval.
- Prefer small, reviewable phases.
- Keep generated artifacts separate from source documents where possible.

## Current planning source

Use `PLAN.md` as the current implementation roadmap for this folder.
