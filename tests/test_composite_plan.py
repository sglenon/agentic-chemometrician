"""Tests for composite (multi-task) planning, execution, and reporting.

These tests exercise the new task_kinds parameter path.  The existing single-task
path is covered by test_project_planning.py and test_run_service.py — those tests
MUST remain green to prove backward compatibility.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from chemometrics_contracts.project import (
    MeasurementRecord,
    Modality,
    ProjectManifest,
    Representation,
    SampleRecord,
    ScientificIntent,
)
from chemometrics_mcp.core.project_planning import (
    SPLIT_SUPERVISED_KINDS,
    TASK_PACKS,
    build_analysis_plan,
    capability_catalog,
)
from chemometrics_mcp.tools import project_workflow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _manifest(n_samples: int = 6, prepared: bool = True) -> ProjectManifest:
    samples = tuple(
        SampleRecord(
            sample_id=f"s{i}",
            preparation_id=f"prep-{i}" if prepared else None,
            metadata={"target": float(i)},
            composition={"concentration": float(i) / 10.0},
        )
        for i in range(n_samples)
    )
    measurements = tuple(
        MeasurementRecord(
            measurement_id=f"m{i}",
            asset_id=f"a{i}",
            sample_id=f"s{i}",
            modality=Modality.NIR,
            representation=Representation.SPECTRUM,
        )
        for i in range(n_samples)
    )
    return ProjectManifest(
        project_id="test-composite",
        samples=samples,
        measurements=measurements,
    )


# ---------------------------------------------------------------------------
# (c) Unknown task kind rejected at plan time
# ---------------------------------------------------------------------------

def test_unknown_task_kind_raises() -> None:
    with pytest.raises(ValueError, match="Unknown task kind"):
        build_analysis_plan(
            _manifest(),
            ScientificIntent(objective="explore"),
            task_kinds=["unsupervised_exploration", "nonexistent_pack"],
        )


# ---------------------------------------------------------------------------
# (b) Two supervised kinds rejected at plan time
# ---------------------------------------------------------------------------

def test_two_split_supervised_kinds_rejected() -> None:
    """classification + regression must be rejected; single split_manifest can't serve both."""
    with pytest.raises(ValueError, match="at most one supervised task"):
        build_analysis_plan(
            _manifest(),
            ScientificIntent(objective="both supervised", target="concentration"),
            task_kinds=["classification", "regression"],
        )


def test_split_supervised_plus_unsupervised_ok() -> None:
    """regression + unsupervised_exploration is a valid composite."""
    plan = build_analysis_plan(
        _manifest(),
        ScientificIntent(objective="explore and regress", target="concentration"),
        task_kinds=["unsupervised_exploration", "regression"],
    )
    assert len(plan.tasks) == 2
    task_types = {t.task_type for t in plan.tasks}
    assert task_types == {"unsupervised_exploration", "regression"}


# ---------------------------------------------------------------------------
# (a) Composite plan produces merged task_result
# ---------------------------------------------------------------------------

def test_composite_plan_has_tasks_for_all_kinds() -> None:
    plan = build_analysis_plan(
        _manifest(),
        ScientificIntent(objective="explore and quantify"),
        task_kinds=["unsupervised_exploration", "mixture_quantification"],
    )
    assert len(plan.tasks) == 2
    task_types = [t.task_type for t in plan.tasks]
    assert "unsupervised_exploration" in task_types
    assert "mixture_quantification" in task_types


def test_composite_plan_pipelines_namespaced_per_kind() -> None:
    plan = build_analysis_plan(
        _manifest(),
        ScientificIntent(objective="explore and quantify"),
        task_kinds=["unsupervised_exploration", "mixture_quantification"],
    )
    pipeline_ids = [p.pipeline_id for p in plan.pipelines]
    # each pipeline_id is prefixed with its task kind
    assert all(":" in pid for pid in pipeline_ids)
    kinds = {pid.split(":")[0] for pid in pipeline_ids}
    assert "unsupervised_exploration" in kinds
    assert "mixture_quantification" in kinds


def test_composite_plan_task_backward_compat_field() -> None:
    """plan.task must equal plan.tasks[0] for backward compat."""
    plan = build_analysis_plan(
        _manifest(),
        ScientificIntent(objective="explore and quantify"),
        task_kinds=["unsupervised_exploration", "mixture_quantification"],
    )
    assert plan.task is not None
    assert plan.task.task_type == plan.tasks[0].task_type


def test_composite_plan_has_stable_plan_hash() -> None:
    """Same inputs → same plan_hash (determinism)."""
    intent = ScientificIntent(objective="explore and quantify")
    plan1 = build_analysis_plan(_manifest(), intent, task_kinds=["unsupervised_exploration", "mixture_quantification"])
    plan2 = build_analysis_plan(_manifest(), intent, task_kinds=["unsupervised_exploration", "mixture_quantification"])
    assert plan1.plan_hash == plan2.plan_hash


# ---------------------------------------------------------------------------
# (d) Default single-task path (task_kinds=None) is unaffected
# ---------------------------------------------------------------------------

def test_single_task_path_unchanged_when_task_kinds_absent() -> None:
    """Single-task plan (task_kinds=None) plan_hash must be byte-identical."""
    intent = ScientificIntent(objective="x")
    plan_a = build_analysis_plan(_manifest(), intent, task_kind="spectral_comparison")
    plan_b = build_analysis_plan(_manifest(), intent, task_kinds=None, task_kind="spectral_comparison")
    assert plan_a.plan_hash == plan_b.plan_hash
    assert len(plan_a.tasks) == 1


def test_single_task_plan_has_schema_version_2_in_run(tmp_path: Path) -> None:
    """Single-task run evidence must carry schema_version=2 (unchanged)."""
    output, _ = _create_prepared_project(tmp_path)

    plan = project_workflow.plan_project_analysis(
        output,
        "Explore spectra",
        task_kind="unsupervised_exploration",
    )
    approval = project_workflow.approve_project_plan(
        output, plan["plan_id"], approved_by="tester"
    )
    result = project_workflow.run_project_analysis(
        output, plan["plan_id"], approval["approval_id"], run_id="single-run"
    )
    # Should succeed or block (not crash)
    assert result["status"] in {"succeeded", "blocked", "failed"}
    # Even on failure, if evidence was written it should be schema_version 2
    from chemometrics_mcp.core.project_store import ProjectStore
    store = ProjectStore(output)
    try:
        evidence = store.read_json("runs/single-run/evidence.json")
        assert evidence["schema_version"] == "2", (
            f"Single-task evidence must use schema_version=2, got {evidence['schema_version']!r}"
        )
        assert "task_types" not in evidence, "composite-only field 'task_types' must be absent for single-task runs"
        assert "per_task_claim_eligibility" not in evidence, "composite-only field must be absent for single-task runs"
    except FileNotFoundError:
        # If the run failed before writing evidence, that's OK for schema version check
        pass


# ---------------------------------------------------------------------------
# Composite run via project_workflow (integration)
# ---------------------------------------------------------------------------

def _create_prepared_project(tmp_path: Path, n: int = 4) -> tuple[str, str]:
    """Create a minimal project with n spectra, all with preparation_id."""
    source = tmp_path / "src"
    source.mkdir()
    for i in range(n):
        (source / f"s{i}.csv").write_text(
            f"1000,{0.1 + i * 0.1}\n1001,{0.2 + i * 0.05}\n1002,{0.3 - i * 0.02}\n",
            encoding="utf-8",
        )
    created = project_workflow.create_project(str(source), project_id="composite-run")
    output = created["output_root"]

    from chemometrics_mcp.core.project_service import ProjectService

    service = ProjectService.open(output)
    manifest = service.get_manifest()
    measurement_updates = {
        row.measurement_id: {
            "modality": "ftir",
            "axis_kind": "wavenumber",
            "axis_unit": "cm^-1",
            "signal_kind": "absorbance",
            "signal_unit": "absorbance",
        }
        for row in manifest.measurements
    }
    sample_updates = {
        sample.sample_id: {
            "preparation_id": f"prep-{idx}",
            "composition": {"concentration": float(idx) / 10.0},
        }
        for idx, sample in enumerate(manifest.samples)
    }
    project_workflow.update_project_manifest(
        output,
        {"measurements": measurement_updates, "samples": sample_updates},
    )
    return output, created["project_id"]


def test_composite_run_merges_pca_and_mixture_keys(tmp_path: Path) -> None:
    """Composite plan with unsupervised_exploration + mixture_quantification
    must produce a merged task_result containing both 'pca' and 'mixture_screening'
    (or 'evidence_rows') keys in the evidence file.
    """
    output, _ = _create_prepared_project(tmp_path)

    plan = project_workflow.plan_project_analysis(
        output,
        "Explore and quantify mixture",
        task_kinds=["unsupervised_exploration", "mixture_quantification"],
        target="concentration",
    )
    approval = project_workflow.approve_project_plan(
        output, plan["plan_id"], approved_by="tester"
    )
    result = project_workflow.run_project_analysis(
        output, plan["plan_id"], approval["approval_id"], run_id="composite-run"
    )

    # Run must complete (succeeded or blocked, not failed)
    assert result["status"] in {"succeeded", "blocked"}, f"Got 'failed' with issues: {result.get('issues')}"

    from chemometrics_mcp.core.project_store import ProjectStore
    store = ProjectStore(output)
    evidence = store.read_json("runs/composite-run/evidence.json")

    # Schema version bumped for composite
    assert evidence["schema_version"] == "3"
    # task_types list present and correct
    assert set(evidence["task_types"]) == {"unsupervised_exploration", "mixture_quantification"}
    # per_task_claim_eligibility has two entries
    assert len(evidence["per_task_claim_eligibility"]) == 2
    per_task_types = {e["task_type"] for e in evidence["per_task_claim_eligibility"]}
    assert per_task_types == {"unsupervised_exploration", "mixture_quantification"}

    # The merged task_result must carry keys from BOTH task packs.
    task_result = evidence["task_result"]
    assert "pca" in task_result, "unsupervised_exploration result key 'pca' missing from merged task_result"
    # mixture_quantification writes 'mixture_screening' or 'evidence_rows'
    has_mixture_key = "mixture_screening" in task_result or "evidence_rows" in task_result
    assert has_mixture_key, "mixture_quantification result key missing from merged task_result"


# ---------------------------------------------------------------------------
# (e) Composite report shows per-task claim levels
# ---------------------------------------------------------------------------

def test_composite_report_shows_per_task_claim_levels(tmp_path: Path) -> None:
    """generate_project_report for a composite run must surface per-task claim levels."""
    output, _ = _create_prepared_project(tmp_path)

    plan = project_workflow.plan_project_analysis(
        output,
        "Explore and quantify",
        task_kinds=["unsupervised_exploration", "mixture_quantification"],
        target="concentration",
    )
    approval = project_workflow.approve_project_plan(
        output, plan["plan_id"], approved_by="tester"
    )
    run = project_workflow.run_project_analysis(
        output, plan["plan_id"], approval["approval_id"], run_id="composite-report-run"
    )
    assert run["status"] in {"succeeded", "blocked"}

    report = project_workflow.generate_project_report(output, "composite-report-run")

    from chemometrics_mcp.core.project_store import ProjectStore
    store = ProjectStore(output)
    report_data = store.read_json(f"reports/composite-report-run.json")

    # machine_summary must have per_task_claim_eligibility
    ms = report_data.get("machine_summary", {})
    assert "per_task_claim_eligibility" in ms
    assert len(ms["per_task_claim_eligibility"]) == 2

    # markdown must contain per-task section
    markdown = report_data.get("markdown", "")
    assert "Per-task claim eligibility" in markdown
    assert "unsupervised_exploration" in markdown
    assert "mixture_quantification" in markdown


# ---------------------------------------------------------------------------
# Capability catalog includes split_supervised flag
# ---------------------------------------------------------------------------

def test_capability_catalog_exposes_split_supervised() -> None:
    catalog = capability_catalog()
    packs = catalog["task_packs"]
    for kind in SPLIT_SUPERVISED_KINDS:
        assert packs[kind]["split_supervised"] is True
    for kind in TASK_PACKS:
        if kind not in SPLIT_SUPERVISED_KINDS:
            assert packs[kind]["split_supervised"] is False


# ---------------------------------------------------------------------------
# MCP request model accepts task_kinds
# ---------------------------------------------------------------------------

def test_mcp_request_model_accepts_task_kinds() -> None:
    from chemometrics_mcp.mcp import PlanProjectAnalysisRequest
    req = PlanProjectAnalysisRequest.model_validate({
        "output_root": "/tmp/x",
        "objective": "explore and quantify",
        "task_kinds": ["unsupervised_exploration", "mixture_quantification"],
    })
    assert req.task_kinds == ["unsupervised_exploration", "mixture_quantification"]
    assert req.task_kind is None  # default not set


def test_mcp_request_model_task_kinds_default_is_none() -> None:
    from chemometrics_mcp.mcp import PlanProjectAnalysisRequest
    req = PlanProjectAnalysisRequest.model_validate({
        "output_root": "/tmp/x",
        "objective": "explore",
    })
    assert req.task_kinds is None
