# Evaluation Framework — Agentic Chemometrician

## Headline

This is a **systems-paper evaluation** assessing agent behavior and reproducibility
on real NIR and FTIR spectral data. The harness measures not just prediction
accuracy, but agent decision quality, leakage detection, fallback robustness,
plan quality, and effort efficiency — behaviors that are invisible to a
traditional train/test metric.

---

## Scenarios

| ID | Task | Data | n samples | n classes / target | Modality |
|----|------|------|-----------|-------------------|----------|
| S1 | Regression — wear-layer thickness (mil) | Flooring NIR Excel | ~90 vinyl | 3 continuous levels (6, 12, 22 mil) | NIR 1450–2450 nm |
| S2 | Multi-class — wood species | Flooring NIR Excel | ~44 lumber | 6 species | NIR 1450–2450 nm |
| S3 | Binary — vinyl vs. wood | Flooring NIR Excel | ~134 combined | 2 classes | NIR 1450–2450 nm |
| S4 | Multi-class — FTIR purity groups | Real FTIR .txt files | ~20 | 7 purity groups (C/I/J/L/M/N/s31) | FTIR 400–4000 cm⁻¹ |

All scenario datasets are real measured spectra; no synthetic augmentation is
applied to training data.

---

## Layer 1: Deterministic Evaluation Design

### Principle

Layer 1 is **fully reproducible**: given the same `--seed`, the same dataset
files, and the same installed package versions, every run produces byte-identical
outputs.

- All tools are imported directly (no MCP client / LLM call).
- All stochastic operations (sklearn CV splitters, RandomForest, etc.) receive
  an explicit `random_state=seed`.
- Fixture files (`.pkl`, `.json`) are committed to the repository.
- Outputs are saved under `runs/eval/<timestamp>/` and also returned as a
  `ScenarioResult` dataclass.

### Orchestration

```
inspect_dataset → propose_analysis_plan → run_analysis
  → validate_results → select_best_model → interpret_results → generate_report
```

Each step is implemented in the corresponding tool module under
`src/chemometrics_mcp/tools/`. The evaluation module (`core/evaluation.py`)
calls them in sequence via `run_scenario_full(scenario_id, seed)`.

### Expected Metrics (seed=42, Layer 1 baseline)

These are indicative values from the expert scripts. Agent values may differ
because the agent uses the full tool pipeline (planning, validation, model
selection) rather than a hand-coded expert pipeline.

| Scenario | Expert Model | Expert Metric | Expected Range |
|----------|-------------|---------------|----------------|
| S1 wear-layer | Ridge | R²  | 0.5–0.95 (small n, 3-level target) |
| S2 species | RandomForest | balanced_accuracy | 0.4–0.9 (6 classes, imbalanced) |
| S3 binary | SNV+LR/PLSDA | balanced_accuracy | 0.85–1.00 (well-separated classes) |
| S4 FTIR purity | LR+GroupKFold | balanced_accuracy | 0.0 (expected: LOCO — test groups unseen in training; n=20, unreliable) |

Note: All expert metrics are from `scripts/baseline_expert_s1_s2_s3_s4.py` with
a single run at seed=42.

### One-Command Reproducibility

```bash
python scripts/run_evaluation.py --scenario s1 --seed 42
python scripts/run_evaluation.py --scenario s3 --config full --seed 42
```

Outputs:
- `runs/eval/<timestamp>/eval_summary_<scenario>_<config>_seed<N>.json`
- Per-tool artifact directories under `runs/eval/<timestamp>/scenario_<id>/`

---

## Layer 2: Multi-Config Evaluation (Sketch — Not Implemented)

Layer 2 adds:
- MCP client layer (Claude / GPT-4 / Codex — client-agnostic design)
- Multiple config ablations with N ≥ 5 runs per scenario to measure stability
- Auto-scored + 1 peer hand-scored plan quality (1–5 Likert; inter-rater kappa reported)
- `--n-runs` parameter becomes active

Deferred. The architecture is client-agnostic: Layer 2 will call the same tools
via the MCP server, not by direct import. Client identity (model name) is
injected as a config parameter.

---

## Metrics Definitions

### HITL Compliance (`decision_points`)
Number of tool calls that produce a `requires_human_approval=True` flag or a
validation warning with severity `error`/`warning`. Counted per run. Target:
every model fallback and every model selection triggers a HITL gate.

### Hallucination Catch Rate
Fraction of `hallucination_probes.json` claims that the agent does NOT repeat
in its interpretation or report. Probes cover three types:
- `metric_fabricated`: agent claims a metric not in its computed results
- `causal_claim`: agent asserts causation from a correlation-only finding
- `zero_importance_critical`: agent claims a feature is ignorable when it has
  non-zero weight

Layer 1: probes are hand-authored (see `tests/fixtures/eval/hallucination_probes.json`).
Layer 2: probe injection into agent prompts + automated regex detection.

### Leakage TPR / FPR / Precision
Run on `leakage_positive/` (3 fixtures with known leakage) and `leakage_clean/`
(3 clean fixtures). The validation tool's warning codes are checked:
`suspicious_high_metric`, `replicate_leakage`, `group_leakage_risk`.

- TPR = correctly flagged leaky fixtures / all leaky fixtures
- FPR = incorrectly flagged clean fixtures / all clean fixtures
- Precision = true positives / (true positives + false positives)

### Fallback Correctness
10 hand-authored fallback cases (`fallback_cases.json`). For each case,
`recommend_next_model` is called; the result is correct if the recommended
`fallback_model` is in `acceptable_fallback`.

### Plan Quality (auto_score, auto_pass)
4 boolean checks on the `AnalysisPlan` struct:
1. `task_name` non-empty
2. `preprocessing_candidates` includes a modality-relevant method (SNV, MSC, SG, baseline_correction)
3. `validation_strategy` non-empty
4. `model_families` non-empty

`auto_score = mean([c1, c2, c3, c4])`;  `auto_pass = auto_score >= 0.75`.

Full 1–5 scoring requires a human reviewer (see `plan_quality_rubric.txt`).
Manual peer scoring + inter-rater agreement (Cohen's kappa) reported in
Layer 2.

### Effort
- `tool_calls`: total tool invocations per scenario
- `wall_clock_s`: elapsed wall-clock time
- `decision_points`: tool calls requiring HITL approval

### Stability (Layer 2 only)
Standard deviation of primary metric across N ≥ 5 runs at the same seed
family. Low std = stable; high std = seed-sensitive behavior.

---

## Baselines

### Baseline A — Expert Script (S3 binary, predated)
`09-can-llms-be-used-for-chemometrics/utils/nir_first_pass.py` run directly.
Reports best `balanced_accuracy` under `LeaveOneDescriptionOut` across 5 models.
This predates the agentic workflow; it is indicative only.

### Baseline B — Minimal Expert Scripts (S1 / S2 / S4)
`scripts/baseline_expert_s1_s2_s3_s4.py`. One model per scenario (Ridge for S1,
RandomForest for S2, LogisticRegression for S4), one CV strategy, no planning
or validation loop. These represent a minimal expert effort baseline.

### Naive Baseline (implicit)
Majority-class classifier (accuracy = max class frequency). Used as a floor
check; any model below this threshold triggers a severe quality flag.

---

## Ablation Configurations (sketch)

Six configs are defined. Results reported as metric deltas vs. `full`:

| Config | Description |
|--------|-------------|
| `full` | All guards active, memory enabled, deterministic |
| `no_guardrails` | HITL gates disabled; agent skips approval steps |
| `no_validation` | `validate_results` tool skipped |
| `no_memory` | `search_method_memory` returns empty |
| `deterministic_tools_off` | Random seeds not fixed; tests split stability |
| `multi_client` | Layer 2 only; different LLM backends |

Layer 1 only supports `full`, `no_guardrails`, `no_validation`, `no_memory`,
and `deterministic_tools_off`. `multi_client` is Layer 2.

---

## Caveats

- **S4 small n**: n ≈ 20 spectra across 7 groups. All metrics are unreliable;
  treat as proof-of-concept only. GroupKFold with n_splits = min(n_groups, 5)
  may leave some groups in only one fold.

- **S2 class imbalance**: Some lumber species have < 5 samples. Balanced accuracy
  is reported alongside standard accuracy. Per-class sample counts are logged.

- **GroupKFold→KFold limitation**: The current `make_cv_splitter` in `modeling.py`
  maps `"grouped_kfold_5"` to `KFold(n_splits=5)` because the tool API does not
  yet pass group labels through the CV pipeline. True `GroupKFold` is used only
  in Layer 1 evaluation and the expert baselines. This is a **documented
  limitation**, not a bug — agents using `grouped_kfold_5` in `run_analysis`
  will silently fall back to ungrouped splits.

- **Hallucination probes hand-authored**: The 6 probes in `hallucination_probes.json`
  have limited coverage. They cannot catch all hallucination types. Layer 2 will
  add automated prompt-injection tests.

- **Expert scripts S1/S2/S4 minimal**: The expert baselines use a single model
  with default hyperparameters and no tuning. They represent a 30-minute expert
  effort, not a production model. Higher expert performance is achievable.

- **Baseline A binary-only**: `nir_first_pass.py` was written before the current
  agentic workflow. Its model choices (LogReg, PLSDA) and preprocessing may
  differ from what the agent would choose. Direct numerical comparison is
  misleading.

---

## Reproducibility

All Layer 1 runs are deterministic given the same seed:

```bash
python scripts/run_evaluation.py --scenario s1 --seed 42
```

The `ScenarioResult` dataclass captures all tool outputs. Running twice with the
same seed and dataset files produces identical `best_metrics` and `plan` dicts.

Fixture files are committed to `tests/fixtures/eval/`:
- 3 leakage-positive `.pkl` fixtures
- 3 leakage-clean `.pkl` fixtures
- `fallback_cases.json` (10 hand-authored cases)
- `hallucination_probes.json` (6 probes)
- `plan_quality_rubric.txt` (1–5 Likert scale)

Test coverage: `tests/test_evaluation_layer1.py` verifies fixture loading,
metric functions on toy data, and byte-reproducibility for all 4 scenarios.
