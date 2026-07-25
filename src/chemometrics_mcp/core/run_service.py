"""Execution boundary for immutable, scientist-approved project plans."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from chemometrics_contracts.project import (
    ProjectAnalysisPlan,
    ClaimLevel,
    PlanApproval,
    ProjectManifest,
    WarningLevel,
)
from chemometrics_mcp.core.claims import (
    GateIssue,
    evaluate_claim_eligibility,
    evaluate_detection_limit_eligibility,
)
from chemometrics_mcp.core.project_service import ProjectService
from chemometrics_mcp.core.project_store import (
    ProjectStore,
    data_hash,
    sha256_file,
    slugify_project_id,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _issues(items: Any) -> list[dict[str, Any]]:
    return [_json(item) for item in (items or ())]


def _blocker(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "level": "blocker",
        "stage": "execution",
        "details": details,
    }


def _has_blocker(items: Sequence[Mapping[str, Any]]) -> bool:
    return any(str(item.get("level")) == "blocker" for item in items)


def _check_scientific_collision(
    existing: dict[str, Any],
    new: dict[str, Any],
    shared_keys: frozenset[str],
) -> None:
    """Raise RuntimeError if two task result dicts share a scientific result key.

    Shared metadata keys (task_pack, counts, etc.) are explicitly allowed to
    overlap — only task-specific scientific result keys must be disjoint.
    """
    collisions = (set(existing.keys()) & set(new.keys())) - shared_keys
    if collisions:
        raise RuntimeError(
            f"Task-specific scientific result key collision detected "
            f"between tasks: {sorted(collisions)}. "
            f"Each task pack must write to distinct scientific result "
            f"keys (pca, mixture_screening, evaluation, etc.)."
        )


def _dedup_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate issues by (code, message) identity; preserve original order."""
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for issue in issues:
        key = (str(issue.get("code", "")), str(issue.get("message", "")))
        if key not in seen:
            seen.add(key)
            result.append(issue)
    return result


def _package_version() -> str:
    try:
        return importlib.metadata.version("agentic-chemometrician")
    except importlib.metadata.PackageNotFoundError:
        return "uninstalled"


_METRIC_KEYS = {
    "accuracy",
    "balanced_accuracy",
    "bias",
    "closure_error",
    "condition_number",
    "correlation",
    "cosine_similarity",
    "descriptive_stoichiometric_ratio",
    "explained_variance",
    "intensity_cosine",
    "mae",
    "match_fraction",
    "matched_peak_fraction",
    "mean_position_error_degrees",
    "overlap_fraction",
    "r2",
    "rank",
    "residual_std",
    "rmse",
    "spectral_angle",
}


def _extract_metrics(
    value: Any, path: tuple[str, ...] = (), in_metrics: bool = False
) -> dict[str, Any]:
    """Extract bounded scalar/short-vector metrics, never raw spectra."""
    rows: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            token = str(key)
            if token in {
                "aggregated_predictions",
                "predictions",
                "reconstructed_spectra",
                "residuals",
                "matches",
                "unmatched_left",
                "unmatched_right",
            }:
                continue
            rows.update(
                _extract_metrics(
                    item,
                    path + (token,),
                    in_metrics or token in {"metrics", "scan", "group"},
                )
            )
        return rows
    if isinstance(value, (list, tuple)) and len(value) <= 100:
        key = path[-1] if path else ""
        metric = (
            in_metrics or key in _METRIC_KEYS or key.endswith("_error_ppm")
        )
        if metric and all(
            item is None
            or isinstance(item, (int, float, np.integer, np.floating))
            for item in value
        ):
            rows[".".join(path)] = _json(value)
            return rows
        for index, item in enumerate(value):
            if isinstance(item, Mapping):
                rows.update(
                    _extract_metrics(
                        item, path + (str(index),), in_metrics
                    )
                )
        return rows
    key = path[-1] if path else ""
    metric = in_metrics or key in _METRIC_KEYS or key.endswith("_error_ppm")
    if metric and (
        value is None
        or isinstance(value, (int, float, np.integer, np.floating))
    ):
        rows[".".join(path)] = _json(value)
    return rows


class ProjectRunService:
    """Execute one stored plan and persist a complete local evidence chain."""

    def __init__(self, output_root: str | Path):
        self.store = ProjectStore(output_root)
        self.project = ProjectService.open(output_root)

    def run_project_analysis(
        self,
        plan_id: str,
        approval_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        if run_id is None:
            run_id = f"run-{uuid.uuid4().hex}"
        elif slugify_project_id(run_id) != run_id:
            raise ValueError("run_id must be a safe lowercase slug")
        started_at = _now()
        pending = {
            "schema_version": "2",
            "run_id": run_id,
            "status": "pending",
            "plan_id": plan_id,
            "approval_id": approval_id,
            "started_at": started_at,
            "issues": [],
        }
        self.store.save_run(pending, run_id)
        try:
            self.store.save_run({**pending, "status": "running"}, run_id)
            plan = self._find_model(
                "plans", ProjectAnalysisPlan, plan_id, "plan_id"
            )
            manifest = self.project.get_manifest()
            blockers = self._preflight(plan, manifest, approval_id)
            base = self._base_record(
                run_id, started_at, plan, manifest, approval_id
            )
            if blockers:
                return self._terminal(base, "blocked", blockers)

            measurements = self._materialize(manifest)

            # Collect the ordered list of tasks to execute.  The contract has
            # both plan.task (singular, backward-compat) and plan.tasks (tuple).
            # For composite plans, plan.tasks has >1 element.
            tasks_to_run = list(plan.tasks) if plan.tasks else (
                [plan.task] if plan.task else []
            )
            if not tasks_to_run:
                return self._terminal(
                    base,
                    "blocked",
                    [_blocker("missing_task", "Plan has no executable task.")],
                )

            is_composite = len(tasks_to_run) > 1

            # Execute each task and merge results.
            #
            # Multiple task packs that share the same underlying implementation
            # (e.g. both unsupervised_exploration and mixture_quantification call
            # run_ftir_nir_task) will produce a set of shared metadata keys that
            # are identical from both calls because they operate on the same
            # measurements.  These are safe to overwrite with the last task's
            # value.  Only the TASK-SPECIFIC SCIENTIFIC RESULT KEYS (pca,
            # mixture_screening, evaluation, etc.) must never collide — two
            # tasks producing the same named scientific result would indicate a
            # bug in task-pack key namespacing and must fail loudly.
            _SHARED_METADATA_KEYS: frozenset[str] = frozenset({
                "task_pack",
                "version",
                "task_type",
                "claim_ceiling",
                "measurement_provenance",
                "counts",
                "evidence_rows",
                "issues",
                # supervised path shares these
                "split_id",
                "target",
                "signal_semantics",
            })
            per_task_results: list[tuple[Any, dict[str, Any]]] = []
            task_result: dict[str, Any] = {}
            for task in tasks_to_run:
                # Filter pipeline families to those namespaced for this task kind.
                task_pipeline_families = [
                    item.model_family
                    for item in plan.pipelines
                    if item.model_family
                    and item.pipeline_id.startswith(f"{task.task_type}:")
                ]
                # Fallback: if no kind-namespaced pipelines found (e.g. legacy
                # single-task plan without kind prefix), use all families.
                if not task_pipeline_families:
                    task_pipeline_families = [
                        item.model_family
                        for item in plan.pipelines
                        if item.model_family
                    ]
                single_result = self._execute(
                    task.task_type,
                    measurements,
                    target=task.target,
                    pipeline_families=task_pipeline_families,
                    task_options=dict(
                        task.metadata.get("analysis_options", {})
                    ),
                    split_manifest=plan.split_manifest,
                )
                per_task_results.append((task, single_result))

                # Collision check via extracted helper: only on task-specific
                # scientific result keys.  Shared metadata keys (task_pack, counts,
                # etc.) are safe to overwrite.
                _check_scientific_collision(task_result, single_result, _SHARED_METADATA_KEYS)
                # Merge: concatenate + deduplicate issues, union all other keys
                # (last task wins for shared metadata keys, distinct scientific
                # keys accumulate).  Issues are deduplicated so that tasks sharing
                # the same underlying measurements don't repeat identical advisories.
                merged_issues = _dedup_issues(
                    list(task_result.pop("issues", []))
                    + list(single_result.get("issues", []))
                )
                task_result.update(single_result)
                task_result["issues"] = merged_issues

            # Fix: claim_ceiling is task-type-dependent (e.g. "descriptive" for
            # PCA, "screening" for mixture).  After the merge loop the field
            # holds only the LAST task's value, which misrepresents the merged
            # result.  Set it to the most-conservative value across all tasks so
            # no caller is misled.  The authoritative per-task breakdown is always
            # in per_task_claim_eligibility.
            if is_composite and "claim_ceiling" in task_result:
                _claim_order = [level.value for level in ClaimLevel]
                collected_ceilings = [
                    str(r.get("claim_ceiling", "descriptive"))
                    for _, r in per_task_results
                    if "claim_ceiling" in r
                ]
                if collected_ceilings:
                    task_result["claim_ceiling"] = min(
                        collected_ceilings,
                        key=lambda v: _claim_order.index(v) if v in _claim_order else 0,
                    )

            result_issues = _issues(task_result.get("issues", ()))
            if (
                isinstance(task_result.get("evaluation"), Mapping)
                and task_result["evaluation"].get("status") != "ok"
            ):
                result_issues.append(
                    _blocker(
                        "model_evaluation_blocked",
                        "Nested group-aware model evaluation could not produce a defensible result.",
                        warnings=task_result["evaluation"].get(
                            "warnings", []
                        ),
                    )
                )

            counts = {
                "scan_count": len(measurements),
                "preparation_count": (
                    len(
                        {
                            row["preparation_id"]
                            for row in measurements
                            if row.get("preparation_id")
                        }
                    )
                    if all(row.get("preparation_id") for row in measurements)
                    else None
                ),
            }

            # Compute per-task claim eligibility.  Single-task plans produce one
            # entry; composite plans produce one entry per executed task.
            per_task_claim: list[dict[str, Any]] = []
            for task, single_result in per_task_results:
                planned_claim = str(
                    task.metadata.get("planned_claim_level", "descriptive")
                )
                try:
                    requested_claim = ClaimLevel(planned_claim).value
                except ValueError:
                    requested_claim = ClaimLevel.DESCRIPTIVE.value
                task_issues = _issues(single_result.get("issues", ()))
                task_gate = evaluate_claim_eligibility(
                    requested_claim,
                    issues=tuple(
                        GateIssue(
                            code=str(item.get("code", "execution_issue")),
                            message=str(item.get("message", "Execution issue.")),
                            level=(
                                str(item.get("level"))
                                if str(item.get("level"))
                                in {"blocker", "advisory", "information"}
                                else "advisory"
                            ),
                            stage=str(item.get("stage", "execution")),
                            details=dict(item.get("details", {})),
                        )
                        for item in task_issues
                    ),
                    design_metadata={
                        "independent_preparations": counts["preparation_count"] or 0,
                        "group_safe_validation": task.task_type
                        in {"classification", "regression"},
                        "fold_safe_pipeline": task.task_type
                        in {"classification", "regression"},
                        "calibration_coverage": False,
                    },
                )
                per_task_claim.append({
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "claim_level": task_gate.claim_level,
                    "eligible": task_gate.can_execute and not _has_blocker(task_issues),
                    "reasons": [item.message for item in task_gate.issues],
                })

            # Overall claim eligibility = most conservative across all tasks.
            claim_level_order = [level.value for level in ClaimLevel]
            min_claim = min(
                per_task_claim,
                key=lambda e: claim_level_order.index(e["claim_level"])
                if e["claim_level"] in claim_level_order
                else 0,
            )
            overall_gate_claim_level = min_claim["claim_level"]
            overall_eligible = all(e["eligible"] for e in per_task_claim)

            # Use the first task for backward-compat single-task fields.
            primary_task = tasks_to_run[0]

            detection = evaluate_detection_limit_eligibility(
                primary_task.metadata.get("detection_limit_design", {})
            )
            environment = {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "package_version": _package_version(),
            }
            environment["environment_hash"] = data_hash(environment)

            # Schema version: "2" for single-task (byte-compatible), "3" for composite.
            schema_version = "3" if is_composite else "2"
            evidence: dict[str, Any] = {
                "schema_version": schema_version,
                "project_id": manifest.project_id,
                "run_id": run_id,
                # Backward-compat single fields (first/only task).
                "task_id": primary_task.task_id,
                "task_type": primary_task.task_type,
                "manifest_hash": manifest.manifest_hash,
                "plan_hash": plan.plan_hash,
                "pipeline_ids": [
                    item.pipeline_id for item in plan.pipelines
                ],
                "split_id": (
                    plan.split_manifest.split_id
                    if plan.split_manifest
                    else task_result.get("split_id")
                ),
                "counts": counts,
                "task_result": _json(task_result),
                "issues": result_issues,
                "claim_eligibility": {
                    "claim_level": overall_gate_claim_level,
                    "eligible": overall_eligible and not _has_blocker(result_issues),
                    "reasons": [r for e in per_task_claim for r in e.get("reasons", [])],
                },
                "detection_limit_eligibility": detection.as_dict(),
                "data_hashes": {
                    row["measurement_id"]: data_hash(
                        {
                            "axis": row["axis"].tolist(),
                            "signal": row["signal"].tolist(),
                        }
                    )
                    for row in measurements
                },
                "environment": environment,
            }
            # Composite-only fields (not present for single-task to stay byte-compatible).
            if is_composite:
                evidence["task_types"] = [t.task_type for t in tasks_to_run]
                evidence["per_task_claim_eligibility"] = per_task_claim

            evidence_path = f"runs/{run_id}/evidence.json"
            written = self.store.write_json(evidence_path, evidence)
            artifact = {
                "kind": "analysis_evidence",
                "path": evidence_path,
                "sha256": sha256_file(written),
                "project_id": manifest.project_id,
                "run_id": run_id,
                "manifest_hash": manifest.manifest_hash,
                "plan_hash": plan.plan_hash,
            }

            if is_composite:
                # Composite: compute _report_fields per task using its own result slice,
                # then merge into a unified report with per-task claim levels.
                report = self._composite_report_fields(
                    per_task_results,
                    plan,
                    counts,
                    per_task_claim,
                    detection.as_dict(),
                    evidence_path,
                )
            else:
                # Single-task: unchanged _report_fields call.
                task = primary_task
                report = self._report_fields(
                    task.task_id,
                    task.task_type,
                    task_result,
                    counts,
                    overall_gate_claim_level,
                    detection.as_dict(),
                    evidence_path,
                    [item.pipeline_id for item in plan.pipelines],
                )
            base.update(report)
            base["artifacts"] = [artifact]
            status = "blocked" if _has_blocker(result_issues) else "succeeded"
            return self._terminal(base, status, result_issues)
        except Exception as exc:
            # Data/scientific exceptions become an auditable terminal record.
            failed = {
                **pending,
                "project_id": self._project_id_or_none(),
            }
            return self._terminal(
                failed,
                "failed",
                [_blocker("run_failure", str(exc))],
            )

    def _project_id_or_none(self) -> str | None:
        try:
            return self.project.get_manifest().project_id
        except Exception:
            return None

    def _find_model(
        self,
        directory: str,
        model: Any,
        identifier: str,
        field: str,
    ) -> Any:
        for path in sorted((self.store.output_root / directory).glob("*.json")):
            try:
                payload = self.store.read_json(
                    path.relative_to(self.store.output_root)
                )
            except (ValueError, OSError):
                continue
            if payload.get(field) == identifier:
                return model.model_validate(payload)
        raise ValueError(f"Stored {field} not found: {identifier}")

    def _preflight(
        self,
        plan: ProjectAnalysisPlan,
        manifest: ProjectManifest,
        approval_id: str | None,
    ) -> list[dict[str, Any]]:
        blockers = [
            issue
            for issue in _issues(plan.issues)
            + _issues(manifest.unresolved_issues)
            if issue.get("level") == WarningLevel.BLOCKER.value
        ]
        expected = data_hash(
            plan.model_dump(mode="json", exclude={"plan_hash"})
        )
        if plan.plan_hash != expected:
            blockers.append(
                _blocker(
                    "plan_hash_mismatch",
                    "Stored plan hash does not match its contents.",
                )
            )
        if plan.project_id not in (None, manifest.project_id):
            blockers.append(
                _blocker(
                    "plan_project_mismatch",
                    "Plan belongs to another project.",
                )
            )
        if plan.manifest_hash != manifest.manifest_hash:
            blockers.append(
                _blocker(
                    "manifest_hash_mismatch",
                    "Plan does not match the current manifest.",
                )
            )
        for pipeline in plan.pipelines:
            if pipeline.transformations or pipeline.hyperparameters:
                blockers.append(
                    _blocker(
                        "unsupported_pipeline_detail",
                        "Custom transformations or hyperparameters are not silently ignored by this runner.",
                        pipeline_id=pipeline.pipeline_id,
                    )
                )
        if plan.approval_required:
            if not approval_id:
                blockers.append(
                    _blocker(
                        "approval_required",
                        "A matching plan approval is required.",
                    )
                )
            else:
                try:
                    approval = self._find_model(
                        "approvals",
                        PlanApproval,
                        approval_id,
                        "approval_id",
                    )
                except ValueError:
                    blockers.append(
                        _blocker(
                            "approval_not_found",
                            "The supplied approval ID was not found in this project.",
                        )
                    )
                    approval = None
                if (
                    approval is not None
                    and (
                        not approval.approved
                        or approval.plan_id != plan.plan_id
                        or approval.plan_hash != plan.plan_hash
                        or approval.project_id
                        not in (None, manifest.project_id)
                    )
                ):
                    blockers.append(
                        _blocker(
                            "approval_mismatch",
                            "Approval does not match the project and stored plan hash.",
                        )
                    )
        return blockers

    def _base_record(
        self,
        run_id: str,
        started_at: str,
        plan: ProjectAnalysisPlan,
        manifest: ProjectManifest,
        approval_id: str | None,
    ) -> dict[str, Any]:
        task = plan.task or (plan.tasks[0] if plan.tasks else None)
        return {
            "schema_version": "2",
            "project_id": manifest.project_id,
            "run_id": run_id,
            "status": "running",
            "plan_id": plan.plan_id,
            "plan_hash": plan.plan_hash,
            "manifest_hash": manifest.manifest_hash,
            "approval_id": approval_id,
            "task_kind": task.task_type if task else None,
            "started_at": started_at,
            "issues": [],
        }

    def _materialize(
        self, manifest: ProjectManifest
    ) -> list[dict[str, Any]]:
        samples = {sample.sample_id: sample for sample in manifest.samples}
        result: list[dict[str, Any]] = []
        for measurement in manifest.measurements:
            arrays = self.project.load_measurement(measurement.measurement_id)
            sample = samples[measurement.sample_id]
            record = {
                **measurement.model_dump(mode="json"),
                "sample": sample.model_dump(mode="json"),
                "axis": arrays["axis"],
                "signal": arrays["signal"],
                "preparation_id": sample.preparation_id,
                "technical_replicate_id": sample.technical_replicate_id,
                "physical_state": sample.physical_state,
                "reference_name": measurement.metadata.get(
                    "reference_name",
                    sample.metadata.get("reference_name"),
                ),
            }
            result.append(record)
        return result

    def _execute(
        self,
        task_type: str,
        measurements: list[dict[str, Any]],
        target: str | None,
        pipeline_families: Sequence[str],
        task_options: Mapping[str, Any],
        split_manifest: Any,
    ) -> dict[str, Any]:
        modalities = {str(row.get("modality")) for row in measurements}
        if task_type in {"classification", "regression"}:
            return self._supervised(
                task_type,
                measurements,
                target,
                pipeline_families,
                split_manifest,
                task_options=task_options,
            )
        if task_type == "uvvis_job_plot":
            return self._uvvis_job(measurements, target, task_options)
        if task_type == "pxrd_reference_matching":
            from chemometrics_mcp.core.task_packs.pxrd import (
                compare_pxrd_references,
            )

            references = [
                row
                for row in measurements
                if row.get("role")
                in {
                    "reference",
                    "simulated_reference",
                    "side_product_candidate",
                }
            ]
            experimental = next(
                (
                    row
                    for row in measurements
                    if row.get("role")
                    not in {
                        "reference",
                        "simulated_reference",
                        "side_product_candidate",
                    }
                ),
                None,
            )
            if experimental is None or not references:
                return {
                    "issues": [
                        _blocker(
                            "explicit_reference_roles_required",
                            "PXRD matching requires an experimental pattern and explicit reference roles.",
                        )
                    ]
                }
            result = compare_pxrd_references(
                experimental,
                references,
                two_theta_tolerance=float(
                    task_options.get("two_theta_tolerance", 0.1)
                ),
                normalization=task_options.get("normalization"),
                peak_prominence=task_options.get("peak_prominence"),
            )
            if not task_options.get("calibration_confirmed", False):
                result.setdefault("issues", []).append(
                    {
                        "code": "pxrd_calibration_unconfirmed",
                        "message": "Instrument position calibration/geometry was not confirmed; ranking remains qualitative.",
                        "level": "advisory",
                        "stage": "pxrd",
                        "details": {},
                    }
                )
            return result
        if task_type == "ms_peak_matching":
            from chemometrics_mcp.core.task_packs.mass_spec import (
                run_mass_spec_task,
            )

            references = [
                row
                for row in measurements
                if row.get("role")
                in {"reference", "side_product_candidate"}
            ]
            experimental = next(
                (
                    row
                    for row in measurements
                    if row.get("role")
                    not in {"reference", "side_product_candidate"}
                ),
                None,
            )
            if experimental is None or not references:
                return {
                    "issues": [
                        _blocker(
                            "explicit_reference_roles_required",
                            "MS matching requires an experimental peak list and explicit reference roles.",
                        )
                    ]
                }
            tolerance = task_options.get("mass_tolerance")
            unit = task_options.get("mass_tolerance_unit")
            if tolerance is None or unit not in {"da", "ppm"}:
                return {
                    "issues": [
                        _blocker(
                            "mass_tolerance_required",
                            "MS matching requires an approved mass_tolerance and mass_tolerance_unit ('da' or 'ppm') based on instrument calibration.",
                        )
                    ]
                }
            return run_mass_spec_task(
                experimental,
                references,
                tolerance=float(tolerance),
                tolerance_unit=str(unit),
                precursor_hypotheses=task_options.get(
                    "precursor_hypotheses"
                ),
                calibration_metadata=task_options.get(
                    "calibration_metadata"
                ),
            )
        if task_type == "spectral_comparison" and modalities == {"uv_vis"}:
            from chemometrics_mcp.core.task_packs.uvvis import (
                compare_role_tagged_spectra,
            )

            return compare_role_tagged_spectra(measurements)
        from chemometrics_mcp.core.task_packs.ftir_nir import (
            run_ftir_nir_task,
        )

        return run_ftir_nir_task(
            measurements,
            task_type=task_type,
            peak_tolerance=float(task_options.get("peak_tolerance", 5.0)),
        )

    def _uvvis_job(
        self,
        rows: list[dict[str, Any]],
        target: str | None,
        options: Mapping[str, Any],
    ) -> dict[str, Any]:
        from chemometrics_mcp.core.task_packs.uvvis import analyze_jobs_method

        if not target:
            return {
                "issues": [
                    _blocker(
                        "job_fraction_target_required",
                        "Job's method requires the explicit sample composition/metadata key containing mole fraction.",
                    )
                ]
            }
        values: list[float] = []
        for row in rows:
            sample = row["sample"]
            value = sample["composition"].get(
                target, sample["metadata"].get(target)
            )
            if value is None:
                return {
                    "issues": [
                        _blocker(
                            "job_fraction_missing",
                            f"Mole-fraction key {target!r} is missing from a sample.",
                        )
                    ]
                }
            values.append(float(value))
        axis = rows[0]["axis"]
        if any(not np.array_equal(axis, row["axis"]) for row in rows[1:]):
            return {
                "issues": [
                    _blocker(
                        "job_axis_alignment_required",
                        "Job's method spectra require exact aligned wavelength axes.",
                    )
                ]
            }
        if any(row.get("modality") != "uv_vis" for row in rows):
            return {
                "issues": [
                    _blocker(
                        "job_uvvis_modality_required",
                        "Job's method task requires UV-Vis measurements.",
                    )
                ]
            }
        semantics = {
            (row.get("signal_kind"), row.get("signal_unit")) for row in rows
        }
        if len(semantics) != 1:
            return {
                "issues": [
                    _blocker(
                        "job_signal_semantics_mismatch",
                        "Job's method spectra require one matching signal kind and unit.",
                    )
                ]
            }
        signal_kind, signal_unit = next(iter(semantics))
        groups = (
            [row["preparation_id"] for row in rows]
            if all(row.get("preparation_id") for row in rows)
            and len({row["preparation_id"] for row in rows}) >= 2
            else None
        )
        return analyze_jobs_method(
            values,
            [row["signal"] for row in rows],
            axis=axis,
            axis_unit=rows[0].get("axis_unit"),
            wavelength=options.get("wavelength"),
            selection_bounds=(
                tuple(options["selection_bounds"])
                if options.get("selection_bounds") is not None
                else None
            ),
            replicate_groups=groups,
            bootstrap_iterations=int(
                options.get("bootstrap_iterations", 0)
            ),
            design_metadata=options.get("design_metadata"),
            signal_kind=signal_kind,
            signal_unit=signal_unit,
        )

    def _supervised(
        self,
        task_type: str,
        rows: list[dict[str, Any]],
        target: str | None,
        pipeline_families: Sequence[str],
        split_manifest: Any,
        task_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not rows or any(not row["preparation_id"] for row in rows):
            return {
                "issues": [
                    _blocker(
                        "missing_preparation_hierarchy",
                        "Supervised tasks require preparation_id for every measurement.",
                    )
                ]
            }
        if len({row["sample_id"] for row in rows}) != len(rows):
            return {
                "issues": [
                    _blocker(
                        "duplicate_sample_ids",
                        "Every modeled spectrum requires a unique sample_id.",
                    )
                ]
            }
        axis = rows[0]["axis"]
        if any(not np.array_equal(axis, row["axis"]) for row in rows[1:]):
            return {
                "issues": [
                    _blocker(
                        "supervised_axis_alignment_required",
                        "Supervised tasks require exact aligned axes.",
                    )
                ]
            }
        if not target:
            return {
                "issues": [
                    _blocker(
                        "supervised_target_required",
                        "A supervised target key must be explicit in the plan.",
                    )
                ]
            }
        values = []
        signals = []
        resolved_semantics: set[tuple[str, str | None]] = set()
        from chemometrics_mcp.core.units import (
            percent_transmittance_to_absorbance,
            validate_signal,
        )

        for row in rows:
            sample = row["sample"]
            if target in sample["composition"]:
                values.append(sample["composition"][target])
            elif target in sample["metadata"]:
                values.append(sample["metadata"][target])
            else:
                return {
                    "issues": [
                        _blocker(
                            "supervised_target_missing",
                            f"Explicit target key {target!r} is missing from a sample.",
                        )
                    ]
                }
            check = validate_signal(
                row["signal"],
                row.get("signal_kind"),
                row.get("signal_unit"),
                quantitative=True,
            )
            if not check.is_valid:
                return {
                    "issues": [
                        _blocker(
                            "supervised_signal_invalid",
                            "A modeled signal has invalid or unknown semantics.",
                            measurement_id=row["measurement_id"],
                        )
                    ]
                }
            signal = np.asarray(row["signal"], dtype=float)
            kind, unit = check.normalized_kind, check.normalized_unit
            if kind == "percent_transmittance":
                conversion = percent_transmittance_to_absorbance(signal)
                if not conversion.is_valid:
                    return {
                        "issues": [
                            _blocker(
                                "supervised_transmittance_conversion_failed",
                                "Percent transmittance could not be converted to absorbance.",
                            )
                        ]
                    }
                signal = np.asarray(conversion.values)
                kind, unit = "absorbance", "absorbance"
            resolved_semantics.add((str(kind), unit))
            signals.append(signal)
        if len(resolved_semantics) != 1:
            return {
                "issues": [
                    _blocker(
                        "supervised_signal_semantics_mismatch",
                        "All modeled spectra require matching signal semantics.",
                    )
                ]
            }
        from chemometrics_mcp.core.model_selection import (
            Candidate,
            evaluate_nested_supervised,
        )
        from chemometrics_mcp.core.splits import (
            FoldIndices,
            MaterializedSplits,
        )

        sample_ids = [row["sample_id"] for row in rows]
        groups = [row["preparation_id"] for row in rows]
        if (
            split_manifest is None
            or split_manifest.strategy != "group_kfold"
            or not split_manifest.metadata.get("row_indices_declared")
        ):
            return {
                "issues": [
                    _blocker(
                        "approved_split_manifest_required",
                        "Supervised execution requires materialized folds in the approved plan.",
                    )
                ]
            }
        splits = MaterializedSplits(
            split_id=split_manifest.split_id,
            strategy=split_manifest.strategy,
            group_key=split_manifest.group_key,
            seed=int(split_manifest.seed or 42),
            folds=tuple(
                FoldIndices(
                    train_ids=tuple(item["train_ids"]),
                    test_ids=tuple(item["test_ids"]),
                    fold=int(item["fold"]),
                )
                for item in split_manifest.metadata.get("folds", ())
            ),
            issues=tuple(
                dict(item)
                for item in split_manifest.metadata.get("issues", ())
            ),
        )
        declared_ids = {
            item
            for fold in splits.folds
            for item in (*fold.train_ids, *fold.test_ids)
        }
        if declared_ids != set(sample_ids):
            return {
                "issues": [
                    _blocker(
                        "approved_split_sample_mismatch",
                        "Approved split IDs do not exactly match the modeled samples.",
                    )
                ]
            }
        candidates = self._approved_candidates(
            task_type, pipeline_families
        )
        evaluation = evaluate_nested_supervised(
            np.asarray(signals),
            values,
            sample_ids,
            groups,
            splits,
            task_name=task_type,
            candidates=candidates,
        )
        issues = list(splits.issues)
        result: dict[str, Any] = {
            "task_pack": "supervised",
            "claim_ceiling": "screening",
            "issues": issues,
            "evaluation": evaluation,
            "split_id": splits.split_id,
            "target": target,
            "signal_semantics": list(resolved_semantics)[0],
        }

        # Optional consensus leaderboard: run ALL candidates on the same folds
        # and compute permutation-importance, providing multi-method cross-validation.
        # Activated by analysis_options.consensus = true; purely additive — the
        # primary nested-selection `evaluation` key is completely unaffected.
        opts = dict(task_options) if task_options else {}
        if opts.get("consensus", False):
            from chemometrics_mcp.core.model_selection import evaluate_candidate_leaderboard
            axis_list = (
                np.asarray(rows[0]["axis"]).tolist()
                if rows
                else []
            )
            leaderboard_result = evaluate_candidate_leaderboard(
                np.asarray(signals),
                values,
                sample_ids,
                groups,
                splits,
                task_name=task_type,
                candidates=candidates,
            )
            result["consensus"] = {
                "status": leaderboard_result.get("status", "blocked"),
                "leaderboard": leaderboard_result.get("leaderboard", []),
                "feature_importance": [
                    {
                        "candidate_identity": row["candidate_identity"],
                        "importance": row["feature_importance"],
                        "importance_std": row["feature_importance_std"],
                        "axis": axis_list,
                    }
                    for row in leaderboard_result.get("leaderboard", [])
                ],
                "warnings": leaderboard_result.get("warnings", []),
            }

        return result

    @staticmethod
    def _approved_candidates(
        task_type: str, families: Sequence[str]
    ) -> tuple[Any, ...]:
        from chemometrics_mcp.core.model_selection import Candidate

        requested = set(families)
        candidates: list[Candidate] = []
        if task_type == "classification":
            if not requested or requested & {"logistic", "pls-da"}:
                candidates.append(
                    Candidate(
                        "raw:logistic:c1",
                        "raw",
                        "logistic",
                        {"C": 1.0},
                    )
                )
            if not requested or requested & {"svm", "svm-rbf"}:
                candidates.append(
                    Candidate(
                        "snv:svm:c1",
                        "snv",
                        "svm",
                        {"C": 1.0},
                    )
                )
        else:
            if not requested or requested & {"pls", "plsr"}:
                candidates.extend(
                    [
                        Candidate(
                            "raw:pls:n1",
                            "raw",
                            "pls",
                            {"n_components": 1},
                        ),
                        Candidate(
                            "raw:pls:n2",
                            "raw",
                            "pls",
                            {"n_components": 2},
                        ),
                    ]
                )
            if not requested or requested & {"svr", "svr-rbf"}:
                candidates.append(
                    Candidate(
                        "snv:svr:c1",
                        "snv",
                        "svr",
                        {"C": 1.0, "epsilon": 0.1},
                    )
                )
        if not candidates:
            raise ValueError(
                f"Approved pipeline families are unsupported: {sorted(requested)}"
            )
        return tuple(candidates)

    @staticmethod
    def _composite_report_fields(
        per_task_results: list[tuple[Any, dict[str, Any]]],
        plan: Any,
        counts: Mapping[str, Any],
        per_task_claim: list[dict[str, Any]],
        detection: Mapping[str, Any],
        evidence_path: str,
    ) -> dict[str, Any]:
        """Merge per-task _report_fields results into one composite report dict."""
        merged: dict[str, Any] = {
            "counts": dict(counts),
            "detection_limit_eligibility": dict(detection),
            "observed_spectral_evidence": [],
            "model_evidence": [],
            "tentative_explanations": [],
            "unsupported_claims": [],
            "blockers": [],
            "limitations": [],
            "next_experiments": [],
            "findings": [],
            "results": [],
        }
        # Overall claim eligibility: most conservative across tasks.
        claim_level_order = [level.value for level in ClaimLevel]
        min_entry = min(
            per_task_claim,
            key=lambda e: claim_level_order.index(e["claim_level"])
            if e["claim_level"] in claim_level_order
            else 0,
        )
        merged["claim_eligibility"] = {
            "claim_level": min_entry["claim_level"],
            "eligible": all(e["eligible"] for e in per_task_claim),
        }
        # Per-task breakdown for scientist-facing report.
        merged["per_task_claim_eligibility"] = per_task_claim

        seen_unsupported: set[str] = set()
        seen_next: set[str] = set()

        for (task, single_result), task_claim in zip(per_task_results, per_task_claim):
            # Collect pipeline ids for this task by kind prefix.
            task_pipeline_ids = [
                item.pipeline_id for item in plan.pipelines
                if item.pipeline_id.startswith(f"{task.task_type}:")
            ]
            if not task_pipeline_ids:
                task_pipeline_ids = [item.pipeline_id for item in plan.pipelines]

            single = ProjectRunService._report_fields(
                task.task_id,
                task.task_type,
                single_result,
                counts,
                task_claim["claim_level"],
                detection,
                evidence_path,
                task_pipeline_ids,
            )
            merged["observed_spectral_evidence"].extend(
                single.get("observed_spectral_evidence", [])
            )
            merged["model_evidence"].extend(single.get("model_evidence", []))
            merged["tentative_explanations"].extend(
                single.get("tentative_explanations", [])
            )
            # Deduplicate unsupported_claims and next_experiments across tasks.
            for item in single.get("unsupported_claims", []):
                if item not in seen_unsupported:
                    seen_unsupported.add(item)
                    merged["unsupported_claims"].append(item)
            merged["blockers"].extend(single.get("blockers", []))
            merged["limitations"].extend(single.get("limitations", []))
            for item in single.get("next_experiments", []):
                if item not in seen_next:
                    seen_next.add(item)
                    merged["next_experiments"].append(item)
            merged["findings"].extend(single.get("findings", []))
            merged["results"].extend(single.get("results", []))

        return merged

    @staticmethod
    def _report_fields(
        task_id: str,
        task_type: str,
        result: Mapping[str, Any],
        counts: Mapping[str, Any],
        claim_level: str,
        detection: Mapping[str, Any],
        evidence_pointer: str,
        pipeline_ids: Sequence[str],
    ) -> dict[str, Any]:
        blockers = [
            item
            for item in _issues(result.get("issues", ()))
            if item.get("level") == "blocker"
        ]
        observed = []
        model = []
        if task_type in {
            "spectral_comparison",
            "pxrd_reference_matching",
            "ms_peak_matching",
            "uvvis_job_plot",
        }:
            observed.append(
                "Task-pack observations were computed from explicit local spectra; numerical details are in the run evidence."
            )
        if task_type in {"classification", "regression"}:
            model.append(
                "Nested preparation-group evaluation was run over the approved bounded pipeline families."
            )
        if task_type in {"pca", "unsupervised_exploration"}:
            pca = result.get("pca")
            if pca and pca.get("explained_variance_ratio"):
                evr = pca["explained_variance_ratio"]
                cumulative = sum(evr)
                parts = ", ".join(
                    f"PC{i + 1} explains {v * 100:.1f}% of variance"
                    for i, v in enumerate(evr)
                )
                observed.append(
                    f"{parts} ({cumulative * 100:.1f}% cumulative across {len(evr)} component{'s' if len(evr) != 1 else ''})."
                )
        if task_type in {"mixture", "mixture_quantification"}:
            ms = result.get("mixture_screening")
            if ms:
                ref_names = ms.get("provenance", {}).get("reference_names") or []
                mix_ids = ms.get("mixture_measurement_ids", [])
                coefficients = ms.get("coefficients", [])
                rmse_list = ms.get("rmse", [])
                closure_list = ms.get("closure_error", [])
                for i, (coeffs, mid) in enumerate(zip(coefficients, mix_ids)):
                    parts = ", ".join(
                        f"{name}: {c * 100:.1f}%"
                        for name, c in zip(ref_names, coeffs)
                    )
                    rmse_str = f"{rmse_list[i]:.4g}" if i < len(rmse_list) else "n/a"
                    closure_str = f"{closure_list[i]:.4g}" if i < len(closure_list) else "n/a"
                    observed.append(
                        f"Sample {mid}: constrained coefficients — {parts}; RMSE={rmse_str}, closure_error={closure_str}."
                    )
        limitations = [
            item.get("message", str(item))
            for item in _issues(result.get("issues", ()))
        ]
        unsupported = [
            "Similarity or peak matching does not establish chemical identity or purity."
        ]
        if detection.get("estimable") is not True:
            unsupported.append(
                "LOD/LOQ are not estimable from the declared experimental design."
            )
        next_experiments = []
        if counts.get("preparation_count") is None:
            next_experiments.append(
                "Declare preparation and technical-replicate hierarchy."
            )
        if detection.get("estimable") is not True:
            next_experiments.append(
                "Collect blanks, low-level standards, independent replicates, and declare a detection-limit method before requesting LOD/LOQ."
            )
        findings = [
            {
                "name": "task_completion",
                "text": "See local analysis evidence.",
                "evidence_pointer": evidence_pointer,
            }
        ]
        metrics = _extract_metrics(result)
        return {
            "counts": dict(counts),
            "claim_eligibility": {
                "claim_level": claim_level,
                "eligible": not blockers,
            },
            "detection_limit_eligibility": dict(detection),
            "observed_spectral_evidence": observed,
            "model_evidence": model,
            "tentative_explanations": [],
            "unsupported_claims": unsupported,
            "blockers": [
                item.get("message", str(item)) for item in blockers
            ],
            "limitations": limitations,
            "next_experiments": next_experiments,
            "findings": findings,
            "results": (
                [
                    {
                        "task_id": task_id,
                        "pipeline_id": (
                            "nested-selection"
                            if task_type in {"classification", "regression"}
                            else (
                                pipeline_ids[0]
                                if len(pipeline_ids) == 1
                                else "task-pack"
                            )
                        ),
                        "pipeline_candidates": list(pipeline_ids),
                        "metrics": metrics,
                        "evidence_pointer": evidence_pointer,
                    }
                ]
                if metrics
                else []
            ),
        }

    def _terminal(
        self,
        base: Mapping[str, Any],
        status: str,
        issues: list[dict[str, Any]],
    ) -> dict[str, Any]:
        record = {
            **dict(base),
            "status": status,
            "completed_at": _now(),
            "issues": _json(issues),
        }
        try:
            from chemometrics_mcp.core.dashboard import (
                render_run_dashboard,
            )

            rendered = render_run_dashboard(
                self.store, record, project=self.project
            )
            if rendered:
                record["artifacts"] = [
                    *_json(record.get("artifacts", [])),
                    *_json(rendered),
                ]
        except (OSError, TypeError, ValueError) as exc:
            # A rendering problem must remain visible without invalidating
            # already-computed scientific evidence.
            render_issue = {
                "code": "scientist_artifact_render_failed",
                "message": f"Scientist dashboard rendering failed: {exc}",
                "level": "advisory",
                "stage": "reporting",
                "details": {},
            }
            record["issues"] = [*record["issues"], render_issue]
        self.store.save_run(record, str(record["run_id"]))
        return {
            "project_id": record.get("project_id"),
            "run_id": record["run_id"],
            "status": status,
            "plan_id": record.get("plan_id"),
            "manifest_hash": record.get("manifest_hash"),
            "plan_hash": record.get("plan_hash"),
            "issues": _json(issues),
            "artifacts": _json(record.get("artifacts", [])),
        }


def run_project_analysis(
    output_root: str | Path,
    plan_id: str,
    approval_id: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    return ProjectRunService(output_root).run_project_analysis(
        plan_id, approval_id, run_id
    )
