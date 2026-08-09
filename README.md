# Agentic Chemometrician

A local MCP server for scientist-supervised spectral analysis. Connect it to
Claude, Codex, or another MCP-capable agent. Describe your experiment in chat.
The server inventories data, proposes a reviewable analysis plan, runs
deterministic chemometric tools after you approve the plan, and returns
evidence-bound results.

## What it does

- Reads spectral files (CSV, TXT, ASC, XY, Excel, JCAMP) from a local folder.
- Reconstructs wavenumber axes from JCAMP headers when the file carries
  `FIRSTX`, `LASTX`, `DELTAX`, and `NPOINTS` metadata.
- Hashes all inputs and stores derived results separately in
  `chemometrics-output/`. It never modifies source files.
- Requires your explicit approval before any computation runs.
- Produces an offline HTML dashboard, SVG figures, and CSV tables for every
  completed run.
- Exposes a `chemometrics` CLI for terminal workflows in addition to the MCP
  server.

Supported analysis types: descriptive FTIR/NIR spectral comparison, exploratory
PCA, group-aware supervised screening, constrained reference-mixture
quantification, guarded Job's plots, and qualitative PXRD/MS reference matching.

The server does not certify purity, chemical identity, or LOD/LOQ. Unsupported
requests produce an abstention and a recommended next experiment.

## Repo structure

```text
src/
  chemometrics_contracts/   data contracts (SpectralDataset, AnalysisResult, ...)
  chemometrics_mcp/         MCP server, CLI, and analysis engine
    core/                   deterministic analysis logic (no MCP imports)
      task_packs/           per-modality task implementations (ftir_nir, pxrd, ...)
    tools/                  MCP tool wrappers (thin layer over core/)
    server.py               MCP server entry point
    cli.py                  CLI entry point
    artifacts.py            run artifact path management
tests/                      pytest test suite
prompt-library/             agent-neutral guardrails, skills, and workflows
skills/
  chemometrics-project/     SKILL.md for skill-capable MCP clients
docs/                       acceptance and evaluation documentation
scripts/                    one-off data-export and analysis scripts
ftir-purity-dataset/        reference dataset documentation
archive/                    exploratory notebooks 00-08 (historical)
runs/                       generated run artifacts (gitignored in practice)
```

## Installation

Requires Python 3.10 or newer.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/chemometrics doctor
```

The `doctor` command returns `{"ok": true, ...}` when the installation is valid.

On Windows, use `.venv/Scripts/` instead of `.venv/bin/`.

To install development and optional modeling dependencies:

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

Note: `xgboost` is pinned to `2.1.4` for SHAP compatibility. On macOS you may
also need `libomp` (`brew install libomp`).

## Connect to an MCP client

The MCP server starts as a stdio process. Use the absolute path to the installed
executable reported by `chemometrics doctor`.

### Claude Desktop

Edit `claude_desktop_config.json` (Settings > Developer > Edit Config):

```json
{
  "mcpServers": {
    "chemometrics": {
      "command": "/ABSOLUTE/PATH/TO/.venv/bin/chemometrics-mcp"
    }
  }
}
```

Quit Claude Desktop completely and reopen it. Verify with:

> Use `list_chemometrics_capabilities` and tell me whether the chemometrics MCP
> is ready. Do not start an analysis.

### Claude Code

```bash
claude mcp add chemometrics -- /ABSOLUTE/PATH/TO/.venv/bin/chemometrics-mcp
claude mcp list
```

### Codex desktop app

Open Settings > MCP servers > Add server. Choose STDIO and enter the absolute
executable path. Restart and verify with `list_chemometrics_capabilities`.

You can also add it from a terminal:

```bash
codex mcp add chemometrics -- /ABSOLUTE/PATH/TO/.venv/bin/chemometrics-mcp
```

### Troubleshooting

1. Run `chemometrics doctor` with the absolute path to confirm the install.
2. Confirm the MCP configuration uses an absolute path (not a relative one).
3. Restart the desktop application completely after changing settings.
4. On macOS, Claude Desktop logs are at `~/Library/Logs/Claude`.

## CLI usage

The `chemometrics` command runs the full project workflow from a terminal.

```bash
# Inventory a source folder and create a project
.venv/bin/chemometrics init ./my-spectra --output ./my-project

# Show project status
.venv/bin/chemometrics show ./my-project

# Propose an analysis plan
.venv/bin/chemometrics plan ./my-project --objective "compare batches"

# For supervised tasks, specify task kind and target
.venv/bin/chemometrics plan ./my-project \
  --objective "model concentration" \
  --task-kind regression \
  --target concentration

# Approve the plan
.venv/bin/chemometrics approve ./my-project PLAN_ID --by scientist@example.org

# Run the approved plan
.venv/bin/chemometrics run ./my-project PLAN_ID --approval-id APPROVAL_ID

# Check run status
.venv/bin/chemometrics status ./my-project RUN_ID

# Generate the report
.venv/bin/chemometrics report ./my-project RUN_ID

# Include a reproducibility notebook
.venv/bin/chemometrics report ./my-project RUN_ID --notebook
```

## Key concepts

### task_kinds

Pass `task_kinds` to `plan_project_analysis` to request multiple analyses in one
composite run. For example:

```json
["unsupervised_exploration", "mixture_quantification"]
```

A composite run executes each task and merges the results into a single
dashboard.

### manifest_hints

Place a `manifest_hints.json` file in the source folder before calling
`create_project`. This file pre-declares sample roles and compositions. It
removes the need to answer those questions interactively. Example:

```json
{
  "<filename-stem-or-glob>": {
    "role": "reference|sample|calibration",
    "reference_name": "Compound A",
    "composition": {"ComponentA": 1.0}
  }
}
```

When the file is absent, behavior is unchanged.

### preprocess_spectra tool

`preprocess_spectra` applies preprocessing steps to spectra without creating a
project or requiring approval. It returns before/after signals and inline SVG
figures. Use it for quick spectral inspection outside the full workflow.

### CHEMOMETRICS_ALLOWED_ROOTS

By default, the server has the filesystem authority of the process that starts
it. Set `CHEMOMETRICS_ALLOWED_ROOTS` to one or more directories (separated by
the OS path separator) to restrict which source and output folders the server
can access.

## MCP tools

| Tool | Purpose |
| --- | --- |
| `create_project` | Inventory a local folder, hash inputs, parse supported files, and draft a manifest |
| `get_project` | Return compact manifest and readiness status |
| `update_project_manifest` | Record modality, units, roles, preparation hierarchy, and metadata |
| `plan_project_analysis` | Produce a hash-stable task and pipeline plan; accepts `task_kinds` for composite runs |
| `approve_project_plan` | Bind scientist approval to the exact stored plan hash |
| `run_project_analysis` | Execute the approved plan and persist run-local evidence |
| `get_project_run` | Return compact terminal status and issues |
| `generate_project_report` | Validate evidence and return the report, offline dashboard, figures, tables, and optional notebook |
| `list_chemometrics_capabilities` | Describe supported task packs, inputs, metrics, and claim ceilings |
| `preprocess_spectra` | Apply preprocessing steps to spectra without a project; returns before/after signals and SVG figures |

## Run output

Each completed run writes to `chemometrics-output/runs/<RUN_ID>/`:

```text
dashboard.html          offline self-contained HTML dashboard
evidence.json           persisted evidence and artifact hashes
figures/                SVG figures (overlay, PCA scores, scree, loadings, ...)
tables/
  sample-assignments.csv
  issues.csv
  metrics.csv
```

Task-specific figures are added when supported: PCA scores scatter, scree plot,
and loadings; an F-statistic discrimination plot when two or more labeled groups
exist; residuals (regression) and confusion matrix (classification); a
per-estimator leaderboard when `consensus` is requested; and a
`mixture-consensus.svg` comparing `constrained-nnls` and `pls2-compositional`
for mixture tasks.

## Development and testing

Run the test suite with:

```bash
.venv/bin/python -m pytest
```

The `dev` extras install pytest and optional modeling libraries (xgboost, shap,
lime, matplotlib, statsmodels, nbformat, langchain).

An agent-facing skill is at `skills/chemometrics-project/SKILL.md`. Install it
in a skill-capable MCP client to make the approval gate and scientific guardrails
part of the agent workflow.
