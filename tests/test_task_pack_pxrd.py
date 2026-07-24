from chemometrics_mcp.core.task_packs.pxrd import compare_pxrd_references


def _record(role, axis=(10, 20, 30, 40, 50), signal=(0, 2, 0, 1, 0), **extra):
    return {"axis": axis, "signal": signal, "axis_kind": "two_theta", "axis_unit": "degree", "signal_kind": "diffraction_intensity", "signal_unit": "a.u.", "role": role, **extra}


def test_pxrd_ranks_references_using_global_one_to_one_peak_matches():
    result = compare_pxrd_references(_record("sample"), [_record("reference", candidate_id="far", axis=(10, 21, 30, 40, 50)), _record("simulated_reference", candidate_id="near")], two_theta_tolerance=.2, normalization="baseline_max")
    assert result["ranked_results"][0]["candidate_id"] == "near"
    assert result["claim_ceiling"] == "screening"
    assert result["provenance"]["phase_identity_inferred"] is False


def test_pxrd_rejects_missing_semantics_without_guessing():
    bad = _record("sample"); bad.pop("axis_unit")
    result = compare_pxrd_references(bad, [_record("reference")])
    assert result["ranked_results"] == []
    assert any(item["code"] == "pxrd_axis_unit_required" for item in result["issues"])


def test_side_product_reference_is_review_hypothesis_only():
    candidate = _record("reference", candidate_id="side")
    candidate["role"] = "side_product_candidate"
    result = compare_pxrd_references(_record("sample"), [candidate])
    assert result["ranked_results"][0]["scientist_review_required"] is True


def test_negative_candidate_intensity_is_retained_as_invalid_evidence():
    bad = _record("reference", candidate_id="bad", signal=(0, -1, 0, 1, 0))
    result = compare_pxrd_references(_record("sample"), [bad])
    assert result["ranked_results"][0]["candidate_id"] == "bad"
    assert any(
        issue["code"] == "invalid_reference_pattern"
        for issue in result["issues"]
    )
