from chemometrics_mcp.core.acceptance import evaluate_scenarios


def test_adversarial_conditions_pass_when_explicitly_abstained_or_blocked():
    project = {"manifest_hash": "m", "plan_hash": "p", "samples": [{"sample_id": "s"}],
               "measurements": [{"signal_kind": "percent_transmittance", "signal": [120], "group": "a", "class": "a"}],
               "issues": [{"code": "percent_transmittance_out_of_range"}, {"code": "missing_preparation_id"}, {"code": "group_leakage"}, {"code": "closure_violation"}, {"code": "reference_physical_state_mismatch"}, {"code": "cross_run_evidence"}]}
    run = {"run_id": "r", "manifest_hash": "m", "plan_hash": "p", "task_kind": "regression", "lod_requested": True,
           "detection_limit_eligibility": {"estimable": False}, "mixture": {"sum_to_one": True, "coefficients": [[-1, 1.5]]},
           "artifacts": [{"run_id": "other"}], "manual_transforms": ["crop"]}
    result = evaluate_scenarios(project, run)
    assert not result["failed_scenario_ids"]
    assert result["manual_transform_count"] == 1
    assert result["readiness"]


def test_hash_tampering_and_unsupported_lod_fail_readiness():
    result = evaluate_scenarios({"manifest_hash": "expected", "plan_hash": "p"}, {"run_id": "r", "manifest_hash": "tampered", "plan_hash": "p", "lod": 1, "lod_requested": True, "detection_limit_eligibility": {"estimable": False}})
    assert "plan_manifest_hash_tampering" in result["failed_scenario_ids"]
    assert result["unsupported_claim_count"] == 1
    assert not result["readiness"]
