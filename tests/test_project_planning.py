from __future__ import annotations

import pytest

from chemometrics_contracts.project import MeasurementRecord, Modality, ProjectManifest, Representation, ScientificIntent
from chemometrics_mcp.core.project_planning import TASK_PACKS, approve_plan, build_analysis_plan, capability_catalog, verify_plan_approval


def manifest(modality: Modality = Modality.NIR, prepared: bool = True) -> ProjectManifest:
    from chemometrics_contracts.project import SampleRecord
    return ProjectManifest(project_id="p", samples=(SampleRecord(sample_id="s", preparation_id="prep" if prepared else None),), measurements=(MeasurementRecord(measurement_id="m", asset_id="a", sample_id="s", modality=modality, representation=Representation.SPECTRUM),))


@pytest.mark.parametrize("kind", TASK_PACKS)
def test_explicit_routes_are_deterministic_and_bounded(kind: str) -> None:
    route_modality = (
        Modality.UV_VIS
        if kind == "uvvis_job_plot"
        else Modality.PXRD
        if kind == "pxrd_reference_matching"
        else Modality.MASS_SPECTROMETRY
        if kind == "ms_peak_matching"
        else Modality.NIR
    )
    plan = build_analysis_plan(manifest(route_modality), ScientificIntent(objective="x"), kind, compute_budget=1)
    assert plan.tasks[0].task_type == kind and len(plan.pipelines) == 1
    assert plan.plan_hash == build_analysis_plan(manifest(route_modality), ScientificIntent(objective="x"), kind, compute_budget=1).plan_hash


def test_compatibility_hierarchy_and_label_group_safety() -> None:
    bad = build_analysis_plan(manifest(Modality.NIR), ScientificIntent(objective="x"), "uvvis_job_plot")
    assert any(item.code == "modality_task_incompatible" for item in bad.issues)
    plan = build_analysis_plan(manifest(prepared=True), ScientificIntent(objective="classify", target="label"), "classification")
    assert plan.split_manifest.group_key == "preparation_id"
    assert plan.tasks[0].metadata["group_key"] != "label"
    assert plan.tasks[0].metadata["target_is_group"] is False


def test_blocked_approval_tamper_detection_and_catalog() -> None:
    blocked = build_analysis_plan(manifest(prepared=False), ScientificIntent(objective="x"), "regression")
    with pytest.raises(ValueError): approve_plan(blocked, "reviewer")
    clean = build_analysis_plan(manifest(), ScientificIntent(objective="compare"), "spectral_comparison")
    approval = approve_plan(clean, "reviewer")
    assert verify_plan_approval(clean, approval)
    assert not verify_plan_approval(clean.model_copy(update={"pipelines": ()}), approval)
    assert "classification" in capability_catalog()["task_packs"]


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_compute_budget_is_rejected(budget) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        build_analysis_plan(
            manifest(),
            ScientificIntent(objective="compare"),
            "spectral_comparison",
            compute_budget=budget,
        )


def test_supervised_plan_materializes_exact_group_folds() -> None:
    from chemometrics_contracts.project import SampleRecord

    samples = tuple(
        SampleRecord(
            sample_id=f"s{i}",
            preparation_id=f"prep-{i}",
            metadata={"target": float(i)},
        )
        for i in range(6)
    )
    project = ProjectManifest(
        project_id="grouped",
        samples=samples,
        measurements=(
            MeasurementRecord(
                measurement_id="m",
                asset_id="a",
                sample_id="s0",
                modality=Modality.NIR,
            ),
        ),
    )
    plan = build_analysis_plan(
        project,
        ScientificIntent(
            objective="model target", target="target"
        ),
        "regression",
    )
    assert plan.split_manifest.strategy == "group_kfold"
    assert plan.split_manifest.metadata["row_indices_declared"] is True
    declared = {
        sample_id
        for fold in plan.split_manifest.metadata["folds"]
        for sample_id in (*fold["train_ids"], *fold["test_ids"])
    }
    assert declared == {sample.sample_id for sample in samples}
