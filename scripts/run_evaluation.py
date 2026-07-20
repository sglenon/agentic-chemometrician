"""Layer 1 evaluation harness entry point.

Usage
-----
    python scripts/run_evaluation.py --scenario s1 --seed 42
    python scripts/run_evaluation.py --scenario s3 --config full
    python scripts/run_evaluation.py --scenario s4 --seed 42 --ftir-dir PATH

Layer 1 (deterministic, seeded, no LLM):
    Imports tools directly; all random operations use --seed.
    Outputs saved to runs/eval/<timestamp>/.

Layer 2 (MCP client, multi-config):
    Not implemented — deferred to a future phase.
    --n-runs is parsed but ignored in Layer 1 (deterministic = single run).
    --config multi_client is skipped in Layer 1.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

# Project root
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Ensure src/ is importable
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

_VALID_SCENARIOS = ["s1", "s2", "s3", "s4"]
_VALID_CONFIGS = [
    "full",
    "no_guardrails",
    "no_validation",
    "no_memory",
    "deterministic_tools_off",
    "multi_client",  # Layer 2 only — skipped in Layer 1
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_table(rows: list[list[str]], headers: list[str]) -> None:
    """Print a simple ASCII table."""
    col_widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_row = "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, col_widths)) + " |"
    print(sep)
    print(header_row)
    print(sep)
    for row in rows:
        print("| " + " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)) + " |")
    print(sep)


def _format_metrics(metrics: dict) -> str:
    if not metrics:
        return "—"
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return "  ".join(parts[:3])  # truncate to 3 metrics for display


# ---------------------------------------------------------------------------
# Layer 1: run a single scenario
# ---------------------------------------------------------------------------

def run_layer1_scenario(
    scenario_id: str,
    config: str,
    seed: int,
    runs_root: Path,
    workbook_path: Path | None,
    ftir_dir: Path | None,
    fixture_dir: Path | None,
) -> dict:
    """Run Layer 1 evaluation for one scenario. Returns summary dict."""
    from chemometrics_mcp.core.evaluation import (
        ScenarioResult,
        compute_effort_metrics,
        compute_fallback_correctness,
        compute_leakage_detection,
        compute_plan_quality,
        run_scenario_full,
    )

    print(f"\n{'='*60}")
    print(f"Layer 1 Evaluation — Scenario {scenario_id.upper()}  config={config}  seed={seed}")
    print(f"{'='*60}")

    # -------------------------------------------------------------------
    # 1. Run main scenario
    # -------------------------------------------------------------------
    print(f"[1/4] Running scenario {scenario_id.upper()} ...")
    result = run_scenario_full(
        scenario_id=scenario_id,
        seed=seed,
        runs_root=runs_root / f"scenario_{scenario_id}",
        workbook_path=workbook_path,
        ftir_dir=ftir_dir,
    )

    status = "OK" if result.ok else f"FAILED: {result.error}"
    print(f"      status={status}  tool_calls={result.tool_calls}  wall_clock={result.wall_clock_s:.2f}s")

    # -------------------------------------------------------------------
    # 2. Leakage detection
    # -------------------------------------------------------------------
    if fixture_dir is not None and fixture_dir.exists():
        print(f"[2/4] Computing leakage detection ...")
        try:
            tpr, fpr, precision = compute_leakage_detection(config, fixture_dir)
            leakage_metrics = {"tpr": round(tpr, 3), "fpr": round(fpr, 3), "precision": round(precision, 3)}
        except Exception as exc:  # noqa: BLE001
            leakage_metrics = {"error": str(exc)}
    else:
        print(f"[2/4] Skipping leakage detection (no fixture_dir).")
        leakage_metrics = {"skipped": True}

    # -------------------------------------------------------------------
    # 3. Fallback correctness
    # -------------------------------------------------------------------
    fallback_path = None
    if fixture_dir is not None:
        fallback_path = fixture_dir / "fallback_cases.json"

    if fallback_path and fallback_path.exists():
        print(f"[3/4] Computing fallback correctness ...")
        try:
            fb_rate = compute_fallback_correctness(fallback_path)
            fallback_metrics = {"correctness_rate": round(fb_rate, 3)}
        except Exception as exc:  # noqa: BLE001
            fallback_metrics = {"error": str(exc)}
    else:
        print(f"[3/4] Skipping fallback correctness (no fallback_cases.json).")
        fallback_metrics = {"skipped": True}

    # -------------------------------------------------------------------
    # 4. Expert baseline comparison
    # -------------------------------------------------------------------
    print(f"[4/4] Loading expert baseline for {scenario_id.upper()} ...")
    expert_metrics: dict = {}
    try:
        sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
        if scenario_id in ("s1", "s2", "s3") and workbook_path and workbook_path.exists():
            from baseline_expert_s1_s2_s3_s4 import (
                run_s1_baseline,
                run_s2_baseline,
                run_s3_baseline,
            )
            if scenario_id == "s1":
                bline = run_s1_baseline(workbook_path, seed=seed)
            elif scenario_id == "s2":
                bline = run_s2_baseline(workbook_path, seed=seed)
            else:
                bline = run_s3_baseline(workbook_path, seed=seed)
            expert_metrics = bline.get("metrics") or bline.get("best_metrics") or {}
        elif scenario_id == "s4" and ftir_dir and ftir_dir.exists():
            from baseline_expert_s1_s2_s3_s4 import run_s4_baseline
            bline = run_s4_baseline(ftir_dir, seed=seed)
            expert_metrics = bline.get("metrics", {})
        else:
            print("      (skipped — dataset path not provided)")
    except Exception as exc:  # noqa: BLE001
        expert_metrics = {"error": str(exc)}

    # -------------------------------------------------------------------
    # Effort
    # -------------------------------------------------------------------
    tool_sequence = [
        "inspect_dataset",
        "propose_analysis_plan",
        "run_analysis",
        "validate_results",
        "select_best_model",
        "interpret_results",
        "generate_report",
    ]
    effort = compute_effort_metrics(tool_sequence, result.wall_clock_s)

    # -------------------------------------------------------------------
    # Plan auto-score
    # -------------------------------------------------------------------
    if result.plan:
        plan_auto_score = result.plan_auto_score
        plan_auto_pass = result.plan_auto_pass
    else:
        plan_auto_score = 0.0
        plan_auto_pass = False

    # -------------------------------------------------------------------
    # Print results table
    # -------------------------------------------------------------------
    print(f"\n--- Results: Scenario {scenario_id.upper()} ---")
    rows = [
        ["scenario", scenario_id.upper()],
        ["config", config],
        ["seed", str(seed)],
        ["status", status],
        ["best_model", str(result.best_model)],
        ["agent_metrics", _format_metrics(result.best_metrics)],
        ["expert_metrics", _format_metrics(expert_metrics)],
        ["plan_auto_score", f"{plan_auto_score:.2f}"],
        ["plan_auto_pass", str(plan_auto_pass)],
        ["tool_calls", str(effort["tool_calls"])],
        ["decision_points", str(result.decision_points)],
        ["wall_clock_s", f"{result.wall_clock_s:.2f}"],
        ["leakage_tpr", str(leakage_metrics.get("tpr", "—"))],
        ["leakage_fpr", str(leakage_metrics.get("fpr", "—"))],
        ["fallback_correctness", str(fallback_metrics.get("correctness_rate", "—"))],
    ]
    _print_table(rows, ["Metric", "Value"])

    return {
        "scenario_id": scenario_id,
        "config": config,
        "seed": seed,
        "status": "ok" if result.ok else "failed",
        "error": result.error,
        "best_model": result.best_model,
        "agent_metrics": result.best_metrics,
        "expert_metrics": expert_metrics,
        "plan_auto_score": plan_auto_score,
        "plan_auto_pass": plan_auto_pass,
        "effort": effort,
        "decision_points": result.decision_points,
        "leakage": leakage_metrics,
        "fallback": fallback_metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Layer 1 evaluation harness (deterministic, seeded, no LLM).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_evaluation.py --scenario s1 --seed 42
  python scripts/run_evaluation.py --scenario s3 --config full --seed 42
  python scripts/run_evaluation.py --scenario s4 --seed 42 \\
      --ftir-dir ftir-purity-dataset/fwdftirjune262026
        """,
    )
    parser.add_argument("--scenario", required=True, choices=_VALID_SCENARIOS,
                        help="Evaluation scenario: s1=wear-layer, s2=species, s3=binary, s4=FTIR")
    parser.add_argument("--config", default="full", choices=_VALID_CONFIGS,
                        help="Evaluation configuration. multi_client is Layer 2 only (skipped).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for all stochastic operations (default: 42).")
    parser.add_argument("--n-runs", type=int, default=1,
                        help="Number of runs (Layer 2 only; ignored in Layer 1).")
    parser.add_argument("--workbook", type=Path, default=None,
                        help="Path to NIR Excel workbook (S1/S2/S3). Auto-detected if omitted.")
    parser.add_argument("--ftir-dir", type=Path, default=None,
                        help="Path to FTIR .txt directory (S4). Auto-detected if omitted.")
    parser.add_argument("--fixture-dir", type=Path, default=None,
                        help="Path to tests/fixtures/eval/ directory. Auto-detected if omitted.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: runs/eval/<timestamp>/).")
    args = parser.parse_args()

    # Guard: multi_client is Layer 2 only
    if args.config == "multi_client":
        print("WARNING: multi_client config is Layer 2 only — skipping in Layer 1.", file=sys.stderr)
        print("STATUS: SKIPPED")
        return 0

    # Guard: n_runs > 1 is ignored
    if args.n_runs > 1:
        print(f"NOTE: --n-runs={args.n_runs} ignored in Layer 1 (deterministic single run).", file=sys.stderr)

    # Resolve paths
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    runs_root = args.out_dir or (_PROJECT_ROOT / "runs" / "eval" / ts)
    runs_root.mkdir(parents=True, exist_ok=True)

    workbook_path = args.workbook
    if workbook_path is None:
        auto_wb = _PROJECT_ROOT / "2026-05-21_22-59_Method Dev_wavelength_range_1450-2450.xlsx"
        if auto_wb.exists():
            workbook_path = auto_wb

    ftir_dir = args.ftir_dir
    if ftir_dir is None:
        auto_ftir = _PROJECT_ROOT / "ftir-purity-dataset" / "fwdftirjune262026"
        if auto_ftir.exists():
            ftir_dir = auto_ftir

    fixture_dir = args.fixture_dir
    if fixture_dir is None:
        auto_fix = _PROJECT_ROOT / "tests" / "fixtures" / "eval"
        if auto_fix.exists():
            fixture_dir = auto_fix

    print(f"Layer 1 Evaluation Harness")
    print(f"  scenario  : {args.scenario}")
    print(f"  config    : {args.config}")
    print(f"  seed      : {args.seed}")
    print(f"  runs_root : {runs_root}")
    print(f"  workbook  : {workbook_path or '(not found)'}")
    print(f"  ftir_dir  : {ftir_dir or '(not found)'}")
    print(f"  fixture_dir: {fixture_dir or '(not found)'}")

    summary = run_layer1_scenario(
        scenario_id=args.scenario,
        config=args.config,
        seed=args.seed,
        runs_root=runs_root,
        workbook_path=workbook_path,
        ftir_dir=ftir_dir,
        fixture_dir=fixture_dir,
    )

    # Save summary JSON
    summary_path = runs_root / f"eval_summary_{args.scenario}_{args.config}_seed{args.seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nSummary saved to: {summary_path}")

    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
