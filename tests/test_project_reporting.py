import hashlib

import pytest

from chemometrics_mcp.core.project_store import ProjectStore
from chemometrics_mcp.core.project_reporting import (
    generate_evidence_report,
    generate_report_for_run,
    validate_local_evidence,
)


def _run(tmp_path):
    evidence = tmp_path / "runs" / "run-1" / "metrics.json"; evidence.parent.mkdir(parents=True, exist_ok=True); evidence.write_text("{}")
    return {"project_id": "p", "run_id": "run-1", "manifest_hash": "a" * 64, "plan_hash": "b" * 64,
            "artifacts": [{"uri": "runs/run-1/metrics.json", "sha256": hashlib.sha256(b"{}").hexdigest()}],
            "results": [{"task_id": "task", "pipeline_id": "pipe", "metrics": {"rmse": .2}}],
            "counts": {"scan_count": 12, "preparation_count": 3}}


def test_report_builds_local_evidence_ledger_and_abstains_from_lod(tmp_path):
    report = generate_evidence_report(_run(tmp_path), tmp_path)
    assert report["evidence_ledger"][0]["run_id"] == "run-1"
    assert "LOD/LOQ not reported" in report["markdown"]
    assert "scans: 12" in report["markdown"]


def test_invalid_hash_and_cross_run_evidence_are_rejected(tmp_path):
    run = _run(tmp_path); run["artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_local_evidence(run, tmp_path)
    run = _run(tmp_path); run["artifacts"][0]["run_id"] = "other"
    with pytest.raises(ValueError, match="cross-run"):
        validate_local_evidence(run, tmp_path)


def test_report_for_run_loads_exact_local_record_and_persists(tmp_path):
    store = ProjectStore(tmp_path)
    store.write_json("project.json", {"project_id": "p"})
    run = _run(tmp_path)
    store.write_json("runs/run-1.json", run)
    report = generate_report_for_run(tmp_path, "run-1")
    assert report["report_path"] == "reports/run-1.json"
    assert store.read_json("reports/run-1.json")["run_id"] == "run-1"
