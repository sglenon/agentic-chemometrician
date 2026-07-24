"""Deterministic adversarial acceptance checks for compact project evidence."""
from __future__ import annotations

from typing import Any, Mapping


def _items(value: Any) -> list[Mapping[str, Any]]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _issue_codes(value: Mapping[str, Any]) -> set[str]:
    return {str(item.get("code")) for item in _items(value.get("issues"))}


def evaluate_scenarios(project: Mapping[str, Any], run: Mapping[str, Any]) -> dict[str, Any]:
    """Check that known unsafe inputs are blocked or explicitly abstained from.

    Scenario success means the unsafe condition is recognized, not that the
    input itself is scientifically acceptable.
    """
    measurements = _items(project.get("measurements"))
    samples = _items(project.get("samples"))
    issues = _issue_codes(project) | _issue_codes(run)
    task = str(run.get("task_kind", run.get("task", ""))).lower()
    outcomes: dict[str, bool] = {}
    percent_invalid = any(str(row.get("signal_kind", "")).lower() == "percent_transmittance" and any(float(value) < 0 or float(value) > 100 for value in row.get("signal", ())) for row in measurements)
    outcomes["invalid_percent_transmittance"] = not percent_invalid or bool(issues & {"invalid_percent_transmittance", "percent_transmittance_out_of_range"})
    supervised = any(word in task for word in ("regression", "classification", "quantif"))
    missing_preps = supervised and (not samples or any(not row.get("preparation_id") for row in samples))
    outcomes["missing_preparation_groups"] = not missing_preps or bool(issues & {"missing_preparation_id", "missing_preparation_hierarchy"})
    groups_equal_class = bool(run.get("group_equals_class", False)) or any(row.get("group") == row.get("class") for row in _items(run.get("rows")) if row.get("group") is not None)
    outcomes["group_equals_class"] = not groups_equal_class or bool(issues & {"group_equals_class", "group_leakage"})
    mixture = run.get("mixture", {})
    coefficients = mixture.get("coefficients", ()) if isinstance(mixture, Mapping) else ()
    negative = any(float(value) < 0 for row in coefficients for value in (row if isinstance(row, (list, tuple)) else [row]))
    nonclosure = bool(mixture.get("sum_to_one", False)) and any(abs(sum(row) - 1) > 1e-6 for row in coefficients if isinstance(row, (list, tuple)))
    outcomes["nonclosure_or_negative_mixture"] = not (negative or nonclosure) or bool(issues & {"negative_mixture_input", "composition_not_closed", "closure_violation"})
    incompatible_state = bool(run.get("reference_state_mismatch", False)) or any(row.get("reference_physical_state") and row.get("sample_physical_state") and row["reference_physical_state"] != row["sample_physical_state"] for row in measurements)
    outcomes["incompatible_reference_state"] = not incompatible_state or "reference_physical_state_mismatch" in issues
    lod = run.get("detection_limit_eligibility", {})
    lod_requested = bool(run.get("lod_requested", False) or run.get("lod") is not None or run.get("loq") is not None)
    lod_missing = lod_requested and lod.get("estimable") is not True
    outcomes["missing_low_level_lod_design"] = not lod_missing or bool(issues & {"lod_not_estimable", "missing_low_level_standards"}) or run.get("lod") is None
    manifest_equal = bool(project.get("manifest_hash")) and project.get("manifest_hash") == run.get("manifest_hash")
    plan_equal = (not project.get("plan_hash")) or project.get("plan_hash") == run.get("plan_hash")
    outcomes["plan_manifest_hash_tampering"] = manifest_equal and plan_equal
    cross_run = any(item.get("run_id") not in (None, run.get("run_id")) for item in _items(run.get("artifacts")))
    outcomes["cross_run_artifact_reference"] = not cross_run or bool(issues & {"cross_run_artifact_reference", "cross_run_evidence"})
    passed = [name for name, ok in outcomes.items() if ok]
    failed = [name for name, ok in outcomes.items() if not ok]
    unsupported = list(run.get("unsupported_claims", ()))
    if lod_missing and run.get("lod") is not None:
        unsupported.append("LOD reported without estimable low-level design")
    manual = run.get("manual_transforms", run.get("manual_transformations", ()))
    manual_count = len(manual) if isinstance(manual, (list, tuple)) else int(bool(manual))
    return {"passed_scenario_ids": passed, "failed_scenario_ids": failed,
            "unsupported_claim_count": len(unsupported), "manual_transform_count": manual_count,
            "reproducibility": {"manifest_hash_equal": manifest_equal, "plan_hash_equal": plan_equal, "hashes_equal": manifest_equal and plan_equal},
            "readiness": not failed and not unsupported}


evaluate_adversarial_scenarios = evaluate_scenarios
