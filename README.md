# Agentic Chemometrician

**A local spectral-analysis assistant for scientists.** Point Claude, Codex, or
OpenCode at a folder of experimental data and describe the scientific question
in chat. The assistant inventories and cleans the data, proposes a reviewable
analysis plan, runs deterministic chemometric tools after your approval, and
presents the evidence without taking scientific judgment away from you.

Technically, it is a local Model Context Protocol (MCP) server with
human-in-the-loop controls for preprocessing, modeling, validation,
interpretation, and report generation.

## Start here: analyze your first experiment

You do not need to write analysis code or learn the MCP commands. After a
one-time connection, you work by chatting with Claude, Codex, or OpenCode in
ordinary scientific language.

### 1. Put the experiment in a folder

Create one folder containing the files that belong to the study. It can contain
CSV, TXT, ASC, XY, Excel, or JCAMP files, and the files do not need to be
perfectly named.

For example:

```text
FTIR synthesis 24 July/
├── product_A.csv
├── precursor_B.csv
├── precursor_C.csv
└── notes.txt
```

Reference spectra, simulated patterns, side-product candidates, blanks, and
sample metadata can go in the same folder. The MCP inventories and hashes the
source files but never rewrites them. It stores derived results in a separate
`chemometrics-output` folder.

### 2. Connect the MCP once

The MCP runs locally on the same computer as your spectra. It requires Python
3.10 or newer.

If Codex, Claude Code, or another coding agent can use a terminal, open this
repository in that agent and paste:

> Set up Agentic Chemometrician for local use. Create a `.venv`, install this
> repository in editable mode, run the `chemometrics doctor` health check, and
> give me the absolute path to the installed `chemometrics-mcp` executable. Do
> not analyze or modify any experiment files yet.

To install it manually on macOS or Linux, run these commands from this
repository:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/chemometrics doctor
```

On Windows, the executable is under `.venv/Scripts/` instead of `.venv/bin/`.
The final health-check response should contain `"ok": true`.

This section is specifically for the installed desktop or coding applications.
A browser-only Claude.ai or ChatGPT conversation cannot start a local program
on your computer.

#### Codex desktop app

The current OpenAI documentation calls this **Codex in the ChatGPT desktop
app**. It shares MCP settings with the Codex CLI and IDE extension.

1. Open this repository as your workspace.
2. Open **Settings → MCP servers → Add server**.
3. Name the server `chemometrics` and choose **STDIO**.
4. For the command, enter the absolute executable path reported during setup,
   for example:

   ```text
   /Users/your-name/agentic-chemometrician/.venv/bin/chemometrics-mcp
   ```

5. Save it and select **Restart**.
6. In a new Codex chat, ask:

   > Use `list_chemometrics_capabilities` and tell me whether the chemometrics
   > MCP is ready. Do not start an analysis.

You can also connect it from a terminal:

```bash
codex mcp add chemometrics -- /ABSOLUTE/PATH/TO/.venv/bin/chemometrics-mcp
```

See the current [Codex MCP setup
guide](https://learn.chatgpt.com/docs/extend/mcp) if the settings screen has
changed.

#### Claude Desktop

This repository connects to Claude Desktop as a local MCP server; it does not
need to be uploaded or exposed to the internet.

1. In Claude Desktop, open **Settings → Developer → Edit Config**.
2. Add the server below to `claude_desktop_config.json`. Replace the example
   command with the absolute executable path reported during setup. If the file
   already contains other servers, preserve them and add only the
   `chemometrics` entry.

   ```json
   {
     "mcpServers": {
       "chemometrics": {
         "command": "/ABSOLUTE/PATH/TO/.venv/bin/chemometrics-mcp"
       }
     }
   }
   ```

3. Save the file and completely quit Claude Desktop—not just its window—then
   reopen it.
4. Select the **+** button beside the chat box and open **Connectors**.
   `chemometrics` should appear with its tools.
5. In a new chat, ask:

   > Use `list_chemometrics_capabilities` and tell me whether the chemometrics
   > MCP is ready. Do not start an analysis.

On a managed Team or Enterprise computer, an administrator can disable local
MCP servers. If the Developer option is missing, ask the administrator whether
local MCP is allowed. For current UI details and logs, see the [Claude Desktop
local MCP guide](https://modelcontextprotocol.io/docs/develop/connect-local-servers).
On Windows, use an absolute command such as
`C:/Users/your-name/agentic-chemometrician/.venv/Scripts/chemometrics-mcp.exe`.

#### Claude Code

From this repository, run:

```bash
claude mcp add chemometrics -- /ABSOLUTE/PATH/TO/.venv/bin/chemometrics-mcp
claude mcp list
```

#### OpenCode

Add this local server to `opencode.jsonc`, replacing the command with its
absolute path:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "chemometrics": {
      "type": "local",
      "command": [
        "/ABSOLUTE/PATH/TO/.venv/bin/chemometrics-mcp"
      ],
      "enabled": true,
      "timeout": 30000
    }
  }
}
```

Then restart OpenCode and ask it to call
`list_chemometrics_capabilities`. See the current [OpenCode MCP
guide](https://opencode.ai/docs/mcp-servers/) if its configuration format has
changed.

#### If the tools do not appear

1. Run the health check using its absolute path:

   ```bash
   /ABSOLUTE/PATH/TO/.venv/bin/chemometrics doctor
   ```

2. Confirm that the MCP configuration uses an absolute executable path.
3. Completely restart the desktop application after changing its settings.
4. In Codex, check **Settings → MCP servers** or type `/mcp`. In Claude
   Desktop, check **+ → Connectors**.
5. Claude Desktop records MCP errors under `~/Library/Logs/Claude` on macOS
   and `%APPDATA%\Claude\logs` on Windows. A missing executable or invalid JSON
   path will be shown there.

### 3. Describe the experiment in chat

Give the agent the folder path, experiment type, what the samples are, and the
question you want answered. A useful first prompt is:

> Use the chemometrics MCP to analyze the folder
> `/Users/me/Desktop/FTIR synthesis 24 July`. These are FTIR spectra from a
> synthesis of Product A using Precursors B and C. Compare the product with
> both precursors and show the important similarities and differences. Present
> possible explanations separately from measured findings. Do not claim
> identity or purity. Inventory the folder first and ask me only for scientific
> information that is genuinely missing.

The folder path should be the real absolute path on your computer. On macOS,
you can drag a folder from Finder into Terminal to reveal its path.

### 4. Review what the agent understood

The agent first inventories the files. It may ask you to confirm facts such as:

- Which spectrum belongs to the product, precursor, reference, blank, or
  replicate
- Whether an axis is wavelength, wavenumber, two-theta, or m/z
- Whether the signal is absorbance, percent transmittance, or intensity
- Which measurements came from independent preparations versus repeated scans
- Which metadata field is the target for classification or regression

These questions prevent scientifically unsafe guessing. If you do not know an
answer, say so; the MCP will narrow the analysis or explain what evidence is
missing.

### 5. Approve the analysis plan

Before computation, the agent presents a compact plan containing the proposed
task, preprocessing candidates, comparisons or models, validation split, claim
limits, and any blockers.

You can correct it conversationally:

> `sample_03` is a reference, not a product. Its x-axis is wavenumber in
> cm⁻¹. Please update the plan.

When the plan is correct, say:

> I have reviewed the sample assignments, units, preprocessing, and validation
> strategy. I approve this plan. Run it and generate the report.

The agent cannot approve a plan on your behalf. Approval is bound to that exact
plan, so a changed plan must be shown again.

### 6. Read the scientist-facing result

The final response should lead with:

- What was measured and compared
- Applied cleaning and preprocessing
- Plots, metrics, matched peaks, or model performance
- Warnings, uncertainty, and evidence limitations
- Possible explanations kept separate from observed findings
- The minimum next experiment when the requested conclusion is not supported

Detailed manifests, plans, evidence hashes, and run records remain in
`chemometrics-output` for reproducibility. The agent should summarize these
rather than making you read JSON files.

Every completed or scientifically blocked run receives its own offline
dashboard, figures, and exportable tables:

```text
chemometrics-output/
└── runs/
    └── RUN_ID/
        ├── dashboard.html
        ├── evidence.json
        ├── figures/
        │   └── source-measurement-overlay.svg
        └── tables/
            ├── sample-assignments.csv
            ├── issues.csv
            └── metrics.csv
```

Task-specific outputs—such as PCA scores, held-out predictions, Job’s-plot
responses, matched PXRD/MS peaks, or mixture-screening coefficients—are added
when supported by the approved analysis. The HTML dashboard is self-contained,
works offline, and only renders persisted evidence; opening it does not rerun
the analysis.

When you want a notebook as well, say:

> Generate the final report and include the reproducibility notebook.

The optional notebook verifies persisted artifact hashes and displays the
dashboard. It does not refit a model or create a second analysis pipeline.

### Prompt starters

| Experiment | Example request |
| --- | --- |
| FTIR | “Compare my product against both precursors. Show retained, shifted, missing, and new bands without making an identity claim.” |
| NIR | “Explore whether the wood and vinyl groups differ. Start with PCA, then use group-aware classification only if the sample hierarchy supports it.” |
| UV–Vis | “Compare the absorption profiles of the product and precursors. My axis is wavelength in nm and the signal is absorbance.” |
| Job’s method | “Analyze this Job’s-plot series for a descriptive stoichiometric maximum. Check the design controls before reporting a ratio.” |
| PXRD | “Compare the experimental pattern with the simulated crystal pattern and possible side-product references using my approved two-theta tolerance.” |
| Mass spectrometry | “Match the experimental peaks against these reference lists using a 5 ppm tolerance. Report mass errors and alternatives, not compound identity.” |

## Advanced: terminal workflow

Most users can stay in chat. The equivalent command-line workflow is:

```bash
.venv/bin/chemometrics init ./my-spectra --output ./my-project
.venv/bin/chemometrics show ./my-project
.venv/bin/chemometrics plan ./my-project --objective "compare batches"
# Supervised and Job's-method tasks also declare the exact target metadata key:
.venv/bin/chemometrics plan ./my-project --objective "model concentration" \
  --task-kind regression --target concentration
.venv/bin/chemometrics approve ./my-project PLAN_ID --by scientist@example.org
.venv/bin/chemometrics run ./my-project PLAN_ID --approval-id APPROVAL_ID
.venv/bin/chemometrics status ./my-project RUN_ID
.venv/bin/chemometrics report ./my-project RUN_ID
# Add an evidence-verification notebook:
.venv/bin/chemometrics report ./my-project RUN_ID --notebook
```

An agent-facing skill is included at
`skills/chemometrics-project/SKILL.md`. Install or reference that folder in a
skill-capable client to make the approval gate, task routing, and scientific
claim guardrails part of the agent workflow.

This is a local-user tool: by default it has the same filesystem authority as
the process that starts it. For clients you do not fully trust, set
`CHEMOMETRICS_ALLOWED_ROOTS` to one or more approved directories separated by
the operating-system path separator. Source and output folders outside that
allowlist are then rejected.

## MCP tools

The folder-first v2 tools are the recommended product surface:

| Tool | Purpose |
| --- | --- |
| `create_project` | Inventory a local folder, hash inputs, parse supported files, and draft a manifest |
| `get_project` | Return compact manifest/readiness status |
| `update_project_manifest` | Record explicit modality, units, roles, preparation hierarchy, and metadata |
| `plan_project_analysis` | Produce a bounded, hash-stable task and pipeline plan |
| `approve_project_plan` | Bind scientist approval to the exact stored plan hash |
| `run_project_analysis` | Execute the approved plan and persist run-local evidence |
| `get_project_run` | Return compact terminal status and issues |
| `generate_project_report` | Validate evidence and return the report, offline dashboard, figures, tables, and optional notebook |
| `list_chemometrics_capabilities` | Describe supported task packs, inputs, metrics, and claim ceilings |

The original single-dataset tools remain available as a compatibility surface:

| Tool | Purpose |
| --- | --- |
| `inspect_dataset` | Load spectral data, detect metadata, candidate labels, modality, and data quality warnings |
| `propose_analysis_plan` | Convert a dataset summary into a bounded, human-readable analysis plan for approval |
| `run_analysis` | Execute approved preprocessing and modeling tasks; save structured run artifacts |
| `validate_results` | Check for replicate leakage, group leakage, class imbalance, split instability, and suspicious metrics |
| `select_best_model` | Recommend the most defensible model by balancing performance, validation reliability, interpretability, and task suitability |
| `recommend_next_model` | Classify a model failure and return a fallback recommendation with rationale |
| `interpret_results` | Summarize feature and wavelength importance across models; flag unstable or overclaimed interpretations |
| `generate_report` | Produce a human-reviewable report with metrics, figures, validation warnings, caveats, and next-step recommendations |

## Scientific scope

Version 0.2 is ready for scientist-supervised data inventory, explicit unit and
sample-hierarchy resolution, descriptive spectral comparison, exploratory
FTIR/NIR PCA, group-aware supervised screening, constrained reference-mixture
screening, guarded Job's plots, qualitative PXRD/MS reference matching, and
hash-validated local reports.

It does not certify purity, chemical/phase identity, a validated analytical
method, binding mechanism, or LOD/LOQ. Unsupported requests produce an
abstention and a recommended next experiment. PCA/modeling requires explicitly
aligned axes; the system does not silently interpolate quantitative datasets.

The server is backed by a **shared, agent-neutral prompt library** (`prompt-library/`)
that defines scientific guardrails, skills, and workflows independent of client. The
workflow includes **human-in-the-loop approval gates** before consequential scientific
decisions — analysis plan, fallback model selection, final model selection when
validation warnings exist, and scientific conclusions.

## Project structure

```
src/
  chemometrics_contracts/   # Data contracts (SpectralDataset, AnalysisResult, ...)
  chemometrics_mcp/         # MCP server + tools + core analysis engine
    core/                   # Deterministic analysis logic (no MCP imports)
    tools/                  # MCP tool implementations (thin wrappers over core)
    server.py               # MCP server entry point
    artifacts.py            # Run artifact path management
tests/                      # Test suite (pytest / unittest)
prompt-library/             # Agent-neutral prompts: guardrails, skills, workflows
agent-memory/               # Changelog and trace of agent work
runs/                       # Generated run artifacts
ftir-purity-dataset/        # Reference dataset documentation
```

## The dataset

- **Source workbook:** `2026-05-21_22-59_Method Dev_wavelength_range_1450-2450.xlsx`
  (repo root), two sheets — `Spectra Metadata` and `Spectra P100001492`.
- **Instrument:** portable NIR (trinamiX), wavelength range 1454–2446 nm, 4 nm step,
  249 channels.
- **Samples:** 146 spectra joined to metadata, collected 2025-06-24 → 2025-09-09.
- **Material families:** hardwood species (fir, mahogany, oak, particle board, pine,
  poplar) and vinyl flooring brands / wear-layer grades (Home Decorators, LifeProof,
  TrafficMaster; 6/12/22 mil wear layers).

## Legacy references

| Location | Contents |
| --- | --- |
| `09-can-llms-be-used-for-chemometrics/` | Early self-contained agentic prototype (predecessor to this pipeline) |
| `archive/` | Notebook-based exploratory studies 00–08 and the original analysis notebook |

## Setup details

Developers who need the test and optional modeling dependencies can run:

```bash
.venv/bin/python -m pip install -e ".[dev]"
```

> **Note:** `xgboost` is pinned to `2.1.4` for SHAP compatibility (optional dev dep).
> On macOS you may also need `libomp` (`brew install libomp`).
