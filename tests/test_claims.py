from chemometrics_mcp.core.claims import (
    GateIssue,
    deduplicate_issues,
    evaluate_claim_eligibility,
    evaluate_detection_limit_eligibility,
    regression_targets_are_not_classes,
    targets_are_classes,
)


COMPLETE_CANDIDATE = {
    "independent_preparations": 5,
    "group_safe_validation": True,
    "fold_safe_pipeline": True,
    "calibration_coverage": True,
    "independent_levels": True,
    "endpoint_coverage": True,
    "batch_representation": True,
    "bias_acceptable": True,
    "applicability_domain": True,
}


def test_no_warning_allows_validated_claim_with_complete_design():
    design = {**COMPLETE_CANDIDATE, "external_test": True, "user_validation_criteria": True}
    summary = evaluate_claim_eligibility("validated_method", [], design)
    assert summary.claim_level == "validated_method"
    assert summary.can_execute and summary.can_select


def test_blocker_caps_claim_at_descriptive():
    issue = GateIssue("leakage", "Leakage detected", "blocker", "validation")
    summary = evaluate_claim_eligibility("validated_method", [issue], COMPLETE_CANDIDATE)
    assert summary.claim_level == "descriptive"
    assert not summary.can_execute
    assert summary.blockers == (issue,)


def test_advisory_caps_claim_at_screening_and_deduplicates():
    one = GateIssue("small_n", "Small N", "advisory", "design", {"scope": "batch-a", "n": None})
    two = GateIssue("small_n", "Another message", "advisory", "design", {"scope": "batch-a", "n": 3})
    merged = deduplicate_issues([one, two])
    assert len(merged) == 1
    assert merged[0].details["n"] == 3
    summary = evaluate_claim_eligibility("validated_method", merged, COMPLETE_CANDIDATE)
    assert summary.claim_level == "screening"


def test_candidate_thresholds_are_required():
    summary = evaluate_claim_eligibility("quantitative_method_candidate", [], {**COMPLETE_CANDIDATE, "independent_preparations": 4})
    assert summary.claim_level == "screening"
    assert "independent_preparations>=5" in summary.advisories[0].details["missing_requirements"]


def test_detection_limit_abstains_without_experimental_requirements():
    result = evaluate_detection_limit_eligibility({"rmse": 0.01, "blanks": True})
    assert not result.estimable
    assert result.status == "not_estimable"
    assert set(result.missing_requirements) == {"low_level_standards", "independent_replicates", "declared_method"}


def test_detection_limit_is_eligible_with_design_evidence():
    result = evaluate_detection_limit_eligibility({
        "blanks": True, "low_level_standards": True,
        "independent_replicates": True, "declared_method": True,
        "rmse": 999,
    })
    assert result.estimable
    assert result.status == "estimable"


def test_regression_targets_are_not_classes():
    assert regression_targets_are_not_classes("quantitative_regression")
    assert not targets_are_classes("quantitative_regression")
    assert targets_are_classes("binary_classification")
