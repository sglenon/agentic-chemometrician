from chemometrics_mcp.core.task_packs.uvvis import analyze_jobs_method, compare_role_tagged_spectra


def test_role_tagged_comparison_is_overlap_only_and_descriptive() -> None:
    result = compare_role_tagged_spectra([
        {"id": "product", "role": "product", "axis": [400, 500, 600], "signal": [1, 2, 1], "axis_kind": "wavelength", "axis_unit": "nm", "signal_kind": "absorbance", "signal_unit": "absorbance"},
        {"id": "precursor", "role": "precursor", "axis": [500, 600, 700], "signal": [1, 1, 2], "axis_kind": "wavelength", "axis_unit": "nm", "signal_kind": "absorbance", "signal_unit": "absorbance"},
    ])
    comparison = result["evidence"]["comparisons"][0]["comparison"]
    assert comparison["overlap"] == (500.0, 600.0)
    assert result["claim_ceiling"] == "descriptive"


def test_unknown_units_block_role_comparison() -> None:
    result = compare_role_tagged_spectra([
        {"role": "complex", "axis": [1, 2], "signal": [1, 2], "axis_kind": "wavelength", "axis_unit": "mystery", "signal_kind": "absorbance", "signal_unit": "absorbance"},
        {"role": "reference", "axis": [1, 2], "signal": [1, 2], "axis_kind": "wavelength", "axis_unit": "nm", "signal_kind": "absorbance", "signal_unit": "absorbance"},
    ])
    assert result["issues"][0]["level"] == "blocker"


def test_jobs_explicit_and_bounded_selection_and_bootstrap_gate() -> None:
    fractions = [0, .25, .5, .75, 1]
    spectra = [[1, 1, 1], [1, 2, 1], [1, 4, 1], [1, 2, 1], [1, 1, 1]]
    selected = analyze_jobs_method(fractions, spectra, axis=[400, 500, 600], axis_unit="nm", selection_bounds=(450, 550), signal_kind="absorbance", signal_unit="absorbance")
    assert selected["evidence"]["selected_wavelength"] == 500.0
    assert selected["evidence"]["observed_maximum"]["mole_fraction"] == .5
    blocked = analyze_jobs_method(fractions, [1, 2, 3, 2, 1], bootstrap_iterations=10)
    assert blocked["provenance"]["status"] == "blocked"


def test_jobs_requires_order_and_coverage_and_is_descriptive() -> None:
    bad = analyze_jobs_method([.2, .1, .8], [1, 2, 1])
    assert bad["issues"][0]["code"] == "invalid_fraction_range_or_order"
    result = analyze_jobs_method([0, .5, 1], [1, 3, 1])
    assert "descriptive_stoichiometric_ratio" not in result["evidence"]
    designed = analyze_jobs_method(
        [0, .5, 1],
        [1, 3, 1],
        design_metadata={
            "blank_corrected": True,
            "component_controls": True,
            "constant_total_concentration": True,
            "response_definition": "blank-corrected delta absorbance",
            "validated_response_method": True,
        },
    )
    assert designed["evidence"]["descriptive_stoichiometric_ratio"] == 1.0
    assert result["claim_ceiling"] == "descriptive"


def test_jobs_replicates_are_aggregated_by_fraction() -> None:
    result = analyze_jobs_method(
        [0, 0.5, 0.5, 1],
        [0, 2, 4, 0],
        design_metadata={
            "blank_corrected": True,
            "component_controls": True,
            "constant_total_concentration": True,
            "response_definition": "delta absorbance",
            "validated_response_method": True,
        },
    )
    assert result["evidence"]["observed_maximum"]["response"] == 3.0
    assert result["evidence"]["observed_maximum"]["replicate_count"] == 2
    assert result["evidence"]["job_plot_points"] == [
        {"mole_fraction": 0.0, "response": 0.0, "replicate_count": 1},
        {"mole_fraction": 0.5, "response": 3.0, "replicate_count": 2},
        {"mole_fraction": 1.0, "response": 0.0, "replicate_count": 1},
    ]
