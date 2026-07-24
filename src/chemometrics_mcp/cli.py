"""JSON-first scientist CLI and MCP stdio entry point."""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


def _json(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, default=str))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chemometrics")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    init = commands.add_parser("init"); init.add_argument("source"); init.add_argument("--output"); init.add_argument("--project-id")
    show = commands.add_parser("show"); show.add_argument("output")
    plan = commands.add_parser("plan"); plan.add_argument("output"); plan.add_argument("--objective", required=True); plan.add_argument("--task-kind"); plan.add_argument("--target"); plan.add_argument("--claim-level", default="exploratory"); plan.add_argument("--max-pipelines", type=int); plan.add_argument("--options-json")
    approve = commands.add_parser("approve"); approve.add_argument("output"); approve.add_argument("plan_id"); approve.add_argument("--by", required=True); approve.add_argument("--notes")
    run = commands.add_parser("run"); run.add_argument("output"); run.add_argument("plan_id"); run.add_argument("--approval-id", required=True); run.add_argument("--run-id")
    status = commands.add_parser("status"); status.add_argument("output"); status.add_argument("run_id")
    report = commands.add_parser("report"); report.add_argument("output"); report.add_argument("run_id"); report.add_argument("--notebook", action="store_true")
    return parser


def _run_record(output: str, run_id: str) -> dict[str, Any]:
    from chemometrics_mcp.core.project_store import ProjectStore
    store = ProjectStore(output)
    for path in (store.output_root / "runs").glob("*.json"):
        data = store.read_json(path.relative_to(store.output_root))
        if data.get("run_id") == run_id:
            return data
    raise FileNotFoundError(f"Run not found: {run_id}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "doctor":
            with tempfile.TemporaryDirectory() as directory:
                probe = Path(directory) / "write-probe"; probe.write_text("ok"); writable = probe.read_text() == "ok"
            try:
                version = importlib.metadata.version(
                    "agentic-chemometrician"
                )
            except importlib.metadata.PackageNotFoundError:
                version = "uninstalled"
            _json({"ok": writable, "python": sys.version.split()[0], "package": "chemometrics_mcp", "package_version": version, "temp_writable": writable})
        elif args.command == "init":
            from chemometrics_mcp.tools.project_workflow import create_project
            _json(create_project(args.source, args.output, args.project_id))
        elif args.command == "show":
            from chemometrics_mcp.tools.project_workflow import get_project
            _json(get_project(args.output))
        elif args.command == "plan":
            from chemometrics_mcp.tools.project_workflow import plan_project_analysis
            options = (
                json.loads(args.options_json)
                if args.options_json
                else None
            )
            if options is not None and not isinstance(options, dict):
                raise ValueError("--options-json must decode to a JSON object")
            _json(
                plan_project_analysis(
                    args.output,
                    args.objective,
                    args.task_kind,
                    args.claim_level,
                    compute_budget=(
                        {"max_pipelines": args.max_pipelines}
                        if args.max_pipelines is not None
                        else None
                    ),
                    target=args.target,
                    analysis_options=options,
                )
            )
        elif args.command == "approve":
            from chemometrics_mcp.tools.project_workflow import approve_project_plan
            _json(approve_project_plan(args.output, args.plan_id, args.by, args.notes))
        elif args.command == "run":
            from chemometrics_mcp.core.run_service import run_project_analysis
            _json(run_project_analysis(args.output, args.plan_id, args.approval_id, args.run_id))
        elif args.command == "status":
            _json(_run_record(args.output, args.run_id))
        elif args.command == "report":
            from chemometrics_mcp.core.project_reporting import (
                generate_report_for_run,
            )
            _json(
                generate_report_for_run(
                    args.output,
                    args.run_id,
                    include_notebook=args.notebook,
                )
            )
        return 0
    except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def mcp_main() -> int:
    """Start the existing MCP stdio server without importing it at install time."""
    from chemometrics_mcp.server import main as server_main
    asyncio.run(server_main())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
