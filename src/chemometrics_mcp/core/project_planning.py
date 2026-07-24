"""Deterministic, conservative task planning for project manifests."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from chemometrics_contracts.project import (
    ProjectAnalysisPlan, AnalysisTaskSpec, ClaimLevel, PipelineSpec, PlanApproval,
    ProjectManifest, ScientificIntent, SplitManifest, ValidationIssue, WarningLevel,
)
from chemometrics_mcp.core.claims import evaluate_claim_eligibility
from chemometrics_mcp.core.project_store import data_hash
from chemometrics_mcp.core.splits import materialize_splits


TASK_PACKS: dict[str, dict[str, Any]] = {
    "spectral_comparison": {"modalities": (), "required": (), "metrics": ("spectral_angle", "cosine_similarity", "matched_peak_fraction"), "ceiling": ClaimLevel.DESCRIPTIVE, "pipelines": ("raw-comparison", "normalized-comparison")},
    "unsupervised_exploration": {"modalities": (), "required": (), "metrics": ("explained_variance",), "ceiling": ClaimLevel.EXPLORATORY, "pipelines": ("pca", "pca-snv")},
    "classification": {"modalities": (), "required": ("preparation_id",), "metrics": ("balanced_accuracy", "f1"), "ceiling": ClaimLevel.SCREENING, "pipelines": ("logistic", "svm-rbf")},
    "regression": {"modalities": (), "required": ("preparation_id",), "metrics": ("rmse", "r2"), "ceiling": ClaimLevel.QUANTITATIVE_METHOD_CANDIDATE, "pipelines": ("plsr", "svr-rbf")},
    "mixture_quantification": {"modalities": (), "required": ("preparation_id",), "metrics": ("rmse", "bias", "closure_error"), "ceiling": ClaimLevel.QUANTITATIVE_METHOD_CANDIDATE, "pipelines": ("constrained-nnls", "pls2-compositional")},
    "uvvis_job_plot": {"modalities": ("uv_vis",), "required": (), "metrics": ("job_maximum", "stoichiometric_ratio"), "ceiling": ClaimLevel.DESCRIPTIVE, "pipelines": ("uvvis-job-plot",)},
    "pxrd_reference_matching": {"modalities": ("pxrd",), "required": (), "metrics": ("matched_peak_fraction", "unmatched_peak_count", "pattern_similarity"), "ceiling": ClaimLevel.SCREENING, "pipelines": ("pxrd-reference-match",)},
    "ms_peak_matching": {"modalities": ("mass_spectrometry",), "required": (), "metrics": ("matched_peak_fraction", "mass_error_ppm"), "ceiling": ClaimLevel.SCREENING, "pipelines": ("ms-peak-match",)},
}


class PlannedAnalysisPlan(ProjectAnalysisPlan):
    """Analysis plan with the fields required by the planning service."""
    tasks: tuple[AnalysisTaskSpec, ...] = ()
    manifest_hash: str
    plan_hash: str | None = None


class ProjectPlanApproval(PlanApproval):
    plan_hash: str


def _issue(code: str, message: str, *, blocker: bool = True) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, level=WarningLevel.BLOCKER if blocker else WarningLevel.ADVISORY, stage="planning")


def _manifest_hash(manifest: ProjectManifest) -> str:
    return manifest.manifest_hash or data_hash(manifest.model_copy(update={"manifest_hash": None}).canonical_dict())


def _route(intent: ScientificIntent) -> str:
    text = " ".join(filter(None, (intent.objective, intent.target, intent.intended_use))).lower()
    for needle, kind in (("job plot", "uvvis_job_plot"), ("job's", "uvvis_job_plot"), ("pxrd", "pxrd_reference_matching"), ("xrd", "pxrd_reference_matching"), ("mass spect", "ms_peak_matching"), ("mixture", "mixture_quantification"), ("purity", "mixture_quantification"), ("quantif", "mixture_quantification"), ("classif", "classification"), ("regress", "regression"), ("explor", "unsupervised_exploration")):
        if needle in text:
            return kind
    return "spectral_comparison"


def _pipeline_specs(kind: str, budget: int | None) -> tuple[PipelineSpec, ...]:
    names = TASK_PACKS[kind]["pipelines"]
    if budget is not None:
        if (
            isinstance(budget, bool)
            or int(budget) != budget
            or int(budget) < 1
        ):
            raise ValueError("compute budget must be a positive integer")
        limit = min(len(names), int(budget))
    else:
        limit = len(names)
    return tuple(PipelineSpec(pipeline_id=f"{kind}:{name}", model_family=name, description=f"Bounded default for {kind}") for name in names[:limit])


def build_analysis_plan(manifest: ProjectManifest, intent: ScientificIntent, task_kind: str | None = None, compute_budget: int | None = None) -> PlannedAnalysisPlan:
    kind = task_kind or _route(intent)
    if kind not in TASK_PACKS:
        raise ValueError(f"Unknown task kind: {kind}")
    pack = TASK_PACKS[kind]
    issues = list(manifest.unresolved_issues)
    present_modalities = {measurement.modality.value for measurement in manifest.measurements}
    allowed = set(pack["modalities"])
    if allowed and present_modalities and not present_modalities <= allowed:
        issues.append(_issue("modality_task_incompatible", f"{kind} supports only {sorted(allowed)}; manifest has {sorted(present_modalities)}"))
    samples = tuple(manifest.samples)
    supervised = kind in {"classification", "regression", "mixture_quantification"}
    group_key = None
    if supervised:
        if not samples or any(not sample.preparation_id for sample in samples):
            issues.append(_issue("missing_preparation_hierarchy", "Supervised and quantitative tasks require explicit preparation_id for every sample."))
        else:
            group_key = "preparation_id"
        if not intent.target:
            issues.append(
                _issue(
                    "missing_supervised_target",
                    "Supervised and quantitative tasks require an explicit target metadata/composition key.",
                )
            )
    task = AnalysisTaskSpec(task_id=f"task:{kind}", task_type=kind, target=intent.target, metric_names=pack["metrics"], scientific_intent=intent, metadata={"group_key": group_key, "target_is_group": False, "analysis_options": dict(intent.metadata)})
    split = SplitManifest(
        split_id=f"split:{kind}",
        strategy="unspecified",
        group_key=group_key,
        metadata={
            "requires_group_safe": supervised,
            "row_indices_declared": False,
        },
    )
    if supervised and group_key:
        sample_ids = [sample.sample_id for sample in samples]
        groups = [sample.preparation_id for sample in samples]
        targets: list[Any] | None = None
        if intent.target:
            targets = []
            for sample in samples:
                if intent.target in sample.composition:
                    targets.append(sample.composition[intent.target])
                elif intent.target in sample.metadata:
                    targets.append(sample.metadata[intent.target])
                else:
                    issues.append(
                        _issue(
                            "supervised_target_missing",
                            f"Target key {intent.target!r} is missing from sample {sample.sample_id!r}.",
                        )
                    )
                    targets = None
                    break
        unique_groups = len(set(groups))
        materialized = materialize_splits(
            sample_ids,
            groups,
            strategy="group_kfold",
            n_splits=min(5, unique_groups),
            seed=42,
            y=targets,
            task_name=kind,
        )
        for split_issue in materialized.issues:
            issues.append(
                ValidationIssue(
                    code=str(split_issue["code"]),
                    message=str(split_issue["message"]),
                    level=WarningLevel(str(split_issue["level"])),
                    stage=str(split_issue.get("stage", "splits")),
                    details={},
                )
            )
        split = SplitManifest(
            split_id=materialized.split_id,
            strategy=materialized.strategy,
            seed=materialized.seed,
            group_key=group_key,
            metadata={
                "requires_group_safe": True,
                "row_indices_declared": True,
                "folds": [
                    {
                        "fold": fold.fold,
                        "train_ids": list(fold.train_ids),
                        "test_ids": list(fold.test_ids),
                    }
                    for fold in materialized.folds
                ],
                "issues": list(materialized.issues),
            },
        )
    # Invoke the shared gate so a requested scientific ceiling is never silently exceeded.
    requested_index = list(ClaimLevel).index(intent.claim_level)
    ceiling_index = list(ClaimLevel).index(pack["ceiling"])
    requested_claim = list(ClaimLevel)[min(requested_index, ceiling_index)]
    gate = evaluate_claim_eligibility(requested_claim.value, design_metadata={"group_safe_validation": bool(group_key), "independent_preparations": len({sample.preparation_id for sample in samples if sample.preparation_id})})
    for gate_issue in gate.issues:
        issues.append(ValidationIssue(code=gate_issue.code, message=gate_issue.message, level=WarningLevel(gate_issue.level), stage=gate_issue.stage, details=dict(gate_issue.details)))
    manifest_hash = _manifest_hash(manifest)
    task = task.model_copy(
        update={
            "metadata": {
                **dict(task.metadata),
                "requested_claim_level": intent.claim_level.value,
                "task_claim_ceiling": pack["ceiling"].value,
                "planned_claim_level": gate.claim_level,
            }
        }
    )
    draft = PlannedAnalysisPlan(
        plan_id=f"plan:{manifest.project_id}:{kind}:{manifest_hash[:10]}",
        project_id=manifest.project_id,
        task=task,
        tasks=(task,),
        pipelines=_pipeline_specs(kind, compute_budget),
        split_manifest=split,
        issues=tuple(issues),
        manifest_hash=manifest_hash,
        compute_budget={"max_pipelines": compute_budget} if compute_budget is not None else {},
    )
    plan_hash = data_hash(draft.model_dump(mode="json", exclude={"plan_hash"}))
    return draft.model_copy(update={"plan_hash": plan_hash})


def approve_plan(plan: PlannedAnalysisPlan, approved_by: str, notes: str | None = None) -> ProjectPlanApproval:
    if any(issue.level == WarningLevel.BLOCKER for issue in plan.issues):
        raise ValueError("Cannot approve a plan with blocker issues")
    if not plan.plan_hash:
        raise ValueError("Plan must have a plan_hash")
    approved_at = datetime.now(timezone.utc).isoformat()
    approval_id = f"approval:{data_hash({'plan_hash': plan.plan_hash, 'approved_by': approved_by, 'approved_at': approved_at})[:16]}"
    return ProjectPlanApproval(
        approval_id=approval_id,
        project_id=plan.project_id,
        plan_id=plan.plan_id,
        plan_hash=plan.plan_hash,
        approved=True,
        approver_id=approved_by,
        approved_at=approved_at,
        notes=notes,
    )


def verify_plan_approval(plan: PlannedAnalysisPlan, approval: ProjectPlanApproval) -> bool:
    expected_hash = data_hash(plan.model_dump(mode="json", exclude={"plan_hash"}))
    return bool(plan.plan_hash == expected_hash and approval.approved and approval.plan_id == plan.plan_id and approval.plan_hash == expected_hash)


def capability_catalog() -> dict[str, Any]:
    return {"task_packs": {name: {"allowed_modalities": list(pack["modalities"]), "required_manifest_fields": list(pack["required"]), "primary_metrics": list(pack["metrics"]), "requested_claim_ceiling": pack["ceiling"].value, "max_default_pipelines": len(pack["pipelines"])} for name, pack in TASK_PACKS.items()}}
