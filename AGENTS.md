# Agentic Chemometrician repository instructions

## Scientist analysis requests

When a user asks to analyze spectral or chemometric data, use the connected
`chemometrics` MCP as the analysis implementation. Read and follow
`skills/chemometrics-project/SKILL.md`.

- Begin with `list_chemometrics_capabilities`, then use the folder-first
  `create_project` → manifest review → plan → human approval → run → report
  workflow.
- Do not create a new Python/R analysis script, fit an unapproved model, or
  substitute a generic coding workflow for the MCP.
- Do not approve an analysis plan on the scientist's behalf.
- Return the generated run-local dashboard, report, figures, and tables.
  Generate the optional notebook only when requested.
- A generated notebook verifies and displays persisted evidence; it must not
  become a second analysis pipeline.
- If the MCP tools are unavailable, stop and report the connection problem
  instead of rebuilding the analysis in code.

## Product-development requests

The restrictions above apply to scientific data-analysis requests, not to
explicit requests to develop, test, document, or maintain this repository.
For product work, modify the implementation normally and run relevant tests.
