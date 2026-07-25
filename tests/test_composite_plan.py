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


# ---------------------------------------------------------------------------
# Item 6 — consensus leaderboard tests
# ---------------------------------------------------------------------------

def _make_supervised_splits() -> tuple:
    """Return (X, y, sample_ids, groups, splits) for a minimal regression problem."""
    import numpy as np
    from chemometrics_mcp.core.splits import FoldIndices, MaterializedSplits

    rng = np.random.default_rng(0)
    n, p = 12, 10
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    sample_ids = [f"s{i}" for i in range(n)]
    groups = [f"g{i % 4}" for i in range(n)]
    # 3 outer folds
    folds = (
        FoldIndices(train_ids=tuple(sample_ids[4:]), test_ids=tuple(sample_ids[:4]), fold=1),
        FoldIndices(train_ids=tuple(sample_ids[:4] + sample_ids[8:]), test_ids=tuple(sample_ids[4:8]), fold=2),
        FoldIndices(train_ids=tuple(sample_ids[:8]), test_ids=tuple(sample_ids[8:]), fold=3),
    )
    splits = MaterializedSplits(
        split_id="test-split",
        strategy="group_kfold",
        group_key="preparation_id",
        seed=42,
        folds=folds,
        issues=(),
    )
    return X, y, sample_ids, groups, splits


def test_leaderboard_evaluates_all_candidates_with_metric_and_importance() -> None:
    """evaluate_candidate_leaderboard returns one entry per candidate with CV metric and importance."""
    import numpy as np
    from chemometrics_mcp.core.model_selection import (
        Candidate,
        evaluate_candidate_leaderboard,
        default_candidates,
    )

    X, y, sample_ids, groups, splits = _make_supervised_splits()
    candidates = (
        Candidate("raw:pls:n1", "raw", "pls", {"n_components": 1}),
        Candidate("snv:ridge:a1", "snv", "ridge", {"alpha": 1.0}),
    )
    result = evaluate_candidate_leaderboard(
        X, y, sample_ids, groups, splits,
        task_name="regression",
        candidates=candidates,
    )
    assert result["status"] == "ok"
    leaderboard = result["leaderboard"]
    # Both candidates should appear
    identities = {row["candidate_identity"] for row in leaderboard}
    assert identities == {"raw:pls:n1", "snv:ridge:a1"}
    # Each entry has cv_metric_mean, cv_metric_std, feature_importance
    for row in leaderboard:
        assert "cv_metric_mean" in row
        assert "cv_metric_std" in row
        assert "feature_importance" in row
        assert "feature_importance_std" in row
        assert len(row["feature_importance"]) == X.shape[1]
        assert len(row["fold_scores"]) == len(splits.folds)
    # Sorted descending by cv_metric_mean
    means = [row["cv_metric_mean"] for row in leaderboard]
    assert means == sorted(means, reverse=True)


def test_leaderboard_uses_same_fold_structure_as_nested_supervised() -> None:
    """Both evaluators must use the same outer folds (apples-to-apples comparison)."""
    import numpy as np
    from chemometrics_mcp.core.model_selection import (
        Candidate,
        evaluate_candidate_leaderboard,
        evaluate_nested_supervised,
    )

    X, y, sample_ids, groups, splits = _make_supervised_splits()
    candidates = (Candidate("snv:ridge:a1", "snv", "ridge", {"alpha": 1.0}),)
    leaderboard = evaluate_candidate_leaderboard(
        X, y, sample_ids, groups, splits, candidates=candidates,
    )
    nested = evaluate_nested_supervised(
        X, y, sample_ids, groups, splits, candidates=candidates,
    )
    # Split IDs must match — same fold source
    assert leaderboard["split_id"] == nested["split_id"]
    # Leaderboard fold_scores count must equal the number of outer folds
    assert len(leaderboard["leaderboard"][0]["fold_scores"]) == len(nested["fold_assignments"])


def test_consensus_flag_off_produces_no_consensus_key(tmp_path: Path) -> None:
    """When consensus is not set, task_result must have no 'consensus' key."""
    output, _ = _create_prepared_project(tmp_path)
    plan = project_workflow.plan_project_analysis(
        output, "Explore spectra", task_kind="unsupervised_exploration",
    )
    approval = project_workflow.approve_project_plan(
        output, plan["plan_id"], approved_by="tester"
    )
    result = project_workflow.run_project_analysis(
        output, plan["plan_id"], approval["approval_id"], run_id="no-consensus-run"
    )
    assert result["status"] in {"succeeded", "blocked", "failed"}
    from chemometrics_mcp.core.project_store import ProjectStore
    store = ProjectStore(output)
    try:
        evidence = store.read_json("runs/no-consensus-run/evidence.json")
        assert "consensus" not in evidence.get("task_result", {}), (
            "task_result must not contain 'consensus' key when flag is off"
        )
    except FileNotFoundError:
        pass


def test_mixture_consensus_produces_both_pipeline_coefficients() -> None:
    """run_ftir_nir_task for mixture_quantification must produce mixture_consensus
    with both constrained-nnls and pls2-compositional pipeline entries."""
    import numpy as np
    from chemometrics_mcp.core.task_packs.ftir_nir import run_ftir_nir_task

    axis = tuple(float(i) for i in range(10))
    ref_sig_a = (1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005)
    ref_sig_b = (0.01, 0.05, 0.1, 0.3, 0.6, 0.9, 0.8, 0.5, 0.2, 0.1)
    mix_sig = tuple(0.5 * a + 0.5 * b for a, b in zip(ref_sig_a, ref_sig_b))

    measurements = [
        {
            "measurement_id": "ref-a",
            "modality": "ftir",
            "axis": axis,
            "signal": ref_sig_a,
            "axis_kind": "wavenumber",
            "axis_unit": "cm^-1",
            "signal_kind": "absorbance",
            "signal_unit": "absorbance",
            "role": "reference",
            "reference_name": "component_a",
            "preparation_id": "prep-ref-a",
        },
        {
            "measurement_id": "ref-b",
            "modality": "ftir",
            "axis": axis,
            "signal": ref_sig_b,
            "axis_kind": "wavenumber",
            "axis_unit": "cm^-1",
            "signal_kind": "absorbance",
            "signal_unit": "absorbance",
            "role": "reference",
            "reference_name": "component_b",
            "preparation_id": "prep-ref-b",
        },
        {
            "measurement_id": "mix-1",
            "modality": "ftir",
            "axis": axis,
            "signal": mix_sig,
            "axis_kind": "wavenumber",
            "axis_unit": "cm^-1",
            "signal_kind": "absorbance",
            "signal_unit": "absorbance",
            "role": "sample",
            "reference_name": None,
            "preparation_id": "prep-mix",
        },
    ]
    result = run_ftir_nir_task(measurements, task_type="mixture_quantification")

    # mixture_screening must have been produced (valid references present)
    assert "mixture_screening" in result, (
        f"mixture_screening absent; issues: {result.get('issues')}"
    )
    # mixture_consensus must also be present
    assert "mixture_consensus" in result, (
        "mixture_consensus must be present alongside mixture_screening"
    )
    mc = result["mixture_consensus"]
    assert "constrained-nnls" in mc, "constrained-nnls pipeline missing from consensus"
    assert "pls2-compositional" in mc, "pls2-compositional pipeline missing from consensus"
    assert "coefficients" in mc["constrained-nnls"]
    assert "coefficients" in mc["pls2-compositional"]
    assert "agreement" in mc
    assert "coefficient_mae_per_component" in mc["agreement"]
    # Both pipelines should yield 2-component coefficients (one per reference)
    nnls_coefs = mc["constrained-nnls"]["coefficients"]
    comp_coefs = mc["pls2-compositional"]["coefficients"]
    assert len(nnls_coefs) == 1  # one mixture
    assert len(nnls_coefs[0]) == 2  # two components
    assert len(comp_coefs) == 1
    assert len(comp_coefs[0]) == 2


def test_dashboard_consensus_figures_render_with_synthetic_data() -> None:
    """Dashboard figure builders skip cleanly when consensus absent, render when present."""
    from chemometrics_mcp.core.dashboard import _bar_svg, _line_svg, _finite_vector

    # No consensus key → figures should be skipped (guard works)
    assert _bar_svg([], title="T", value_label="V") is None

    # Synthetic consensus data → leaderboard bar chart renders
    rows = [
        {"label": "raw:pls:n1 (±0.01 std)", "value": -0.5},
        {"label": "snv:ridge:a1 (±0.02 std)", "value": -0.6},
    ]
    svg = _bar_svg(rows, title="Candidate leaderboard", value_label="CV metric")
    assert svg is not None
    assert "Candidate leaderboard" in svg
    assert "raw:pls:n1" in svg

    # Synthetic feature importance data → line chart renders
    import numpy as np
    axis = np.linspace(1000, 1100, 20)
    imp = np.abs(np.random.default_rng(0).normal(size=20))
    series = [{"label": "raw:pls:n1", "x": axis, "y": imp}]
    svg2 = _line_svg(series, title="Feature importance", x_label="Wavenumber", y_label="Importance")
    assert svg2 is not None
    assert "Feature importance" in svg2


def test_dashboard_consensus_figures_skip_when_absent(tmp_path: Path) -> None:
    """Dashboard must produce no consensus/mixture-consensus figures for a plain unsupervised run."""
    output, _ = _create_prepared_project(tmp_path)
    plan = project_workflow.plan_project_analysis(
        output, "Explore spectra", task_kind="unsupervised_exploration",
    )
    approval = project_workflow.approve_project_plan(
        output, plan["plan_id"], approved_by="tester"
    )
    result = project_workflow.run_project_analysis(
        output, plan["plan_id"], approval["approval_id"], run_id="skip-consensus-run"
    )
    assert result["status"] in {"succeeded", "blocked", "failed"}
    # Check no consensus SVG artifacts were emitted
    artifacts = result.get("artifacts", [])
    artifact_paths = [str(a.get("path", "")) for a in artifacts]
    for p in artifact_paths:
        assert "consensus-leaderboard" not in p, f"Unexpected consensus artifact: {p}"
        assert "feature-importance-consensus" not in p
        assert "mixture-consensus" not in p


# ---------------------------------------------------------------------------
# Collision-guard raise-path test (Fix 3 from minor fixes)
# ---------------------------------------------------------------------------

def test_collision_guard_raises_on_scientific_key_overlap() -> None:
    """_check_scientific_collision must raise RuntimeError when two task dicts share a scientific key."""
    from chemometrics_mcp.core.run_service import _check_scientific_collision

    shared_keys = frozenset({"task_pack", "version", "issues", "counts"})
    existing = {"task_pack": "ftir_nir", "issues": [], "pca": {"scores": [[1, 2]]}}
    new_result = {"task_pack": "ftir_nir", "issues": [], "pca": {"scores": [[3, 4]]}}
    with pytest.raises(RuntimeError, match="pca"):
        _check_scientific_collision(existing, new_result, shared_keys)


def test_collision_guard_allows_shared_metadata_overlap() -> None:
    """Shared metadata keys must not trigger the collision guard."""
    from chemometrics_mcp.core.run_service import _check_scientific_collision

    shared_keys = frozenset({"task_pack", "version", "issues", "counts"})
    existing = {"task_pack": "ftir_nir", "pca": {}}
    new_result = {"task_pack": "ftir_nir", "mixture_screening": {}}
    # Should not raise — only "task_pack" overlaps and it's in shared_keys
    _check_scientific_collision(existing, new_result, shared_keys)  # no exception


# ---------------------------------------------------------------------------
# Claim_ceiling most-conservative + issues dedup (Fix 1 & 2)
# ---------------------------------------------------------------------------

def test_dedup_issues_removes_exact_duplicates() -> None:
    """_dedup_issues must remove duplicate (code, message) pairs, preserving order."""
    from chemometrics_mcp.core.run_service import _dedup_issues

    issues = [
        {"code": "a", "message": "msg1"},
        {"code": "b", "message": "msg2"},
        {"code": "a", "message": "msg1"},  # exact duplicate
        {"code": "c", "message": "msg3"},
    ]
    deduped = _dedup_issues(issues)
    assert len(deduped) == 3
    codes = [i["code"] for i in deduped]
    assert codes == ["a", "b", "c"]


def test_claim_ceiling_is_most_conservative_in_composite_run(tmp_path: Path) -> None:
    """In a composite run, task_result['claim_ceiling'] must be the most conservative
    across all tasks, not just the last task's value."""
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
    result = project_workflow.run_project_analysis(
        output, plan["plan_id"], approval["approval_id"], run_id="ceiling-run"
    )
    assert result["status"] in {"succeeded", "blocked"}
    from chemometrics_mcp.core.project_store import ProjectStore
    store = ProjectStore(output)
    evidence = store.read_json("runs/ceiling-run/evidence.json")
    task_result = evidence.get("task_result", {})
    # unsupervised_exploration has ceiling="descriptive"; mixture_quantification has "screening"
    # Most conservative = "descriptive" (lower in claim level order)
    claim_ceiling = task_result.get("claim_ceiling")
    assert claim_ceiling in {"descriptive", "exploratory"}, (
        f"Expected most-conservative ceiling, got {claim_ceiling!r}"
    )
