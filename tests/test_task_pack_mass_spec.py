from __future__ import annotations

from chemometrics_mcp.core.task_packs.mass_spec import run_mass_spec_task


def peak_list(role="reference", masses=(100., 150.), intensities=(10., 5.), **extra):
    return {"candidate_id": role, "role": role, "axis_kind": "mass_to_charge", "axis": masses, "signal_kind": "intensity", "signal_unit": "a.u.", "signal": intensities, **extra}


def test_global_matching_and_da_ppm_errors_are_recorded() -> None:
    result = run_mass_spec_task(peak_list("sample", (100.01, 150.02)), [peak_list("reference", (100., 150.))], tolerance=.03)
    row = result["evidence_rows"][0]
    assert len(row["matches"]) == 2 and row["matches"][0]["signed_error_da"] > 0
    assert "abs(observed-reference)" in result["tolerance"]["formula"]


def test_all_candidates_are_ranked_and_explicit_semantics_required() -> None:
    result = run_mass_spec_task(peak_list("sample"), [peak_list("reference"), peak_list("side_product_candidate", (200.,))], tolerance=20, tolerance_unit="ppm", precursor_hypotheses=[{"adduct": "[M+H]+"}])
    assert len(result["evidence_rows"]) == 2 and result["precursor_hypotheses"][0]["adduct"] == "[M+H]+"
    invalid = run_mass_spec_task({"axis": [1], "signal": [1]}, [], tolerance=1)
    assert any(item["code"] == "mz_axis_required" for item in invalid["issues"])


def test_invalid_candidate_role_is_retained_and_ppm_assignment_is_global() -> None:
    result = run_mass_spec_task(
        peak_list("sample", (100.0009, 200.001)),
        [peak_list("simulated_reference", (100.0, 200.0))],
        tolerance=10,
        tolerance_unit="ppm",
    )
    assert result["evidence_rows"][0]["role"] == "simulated_reference"
    assert any(
        issue["code"] == "reference_role_required" for issue in result["issues"]
    )
