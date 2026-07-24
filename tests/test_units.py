from chemometrics_mcp.core.units import (
    percent_transmittance_to_absorbance, quarantine_for_display,
    sort_and_deduplicate_axis, validate_axis, validate_signal,
)


def test_valid_percent_transmittance_conversion() -> None:
    result = percent_transmittance_to_absorbance([100, 10])
    assert result.is_valid
    assert result.values == (0.0, 1.0)
    assert result.record and result.record.action == "percent_transmittance_to_absorbance"


def test_invalid_transmittance_blocks_without_clipping() -> None:
    result = percent_transmittance_to_absorbance([100, 0, 101])
    assert not result.is_valid
    assert result.values == (100.0, 0.0, 101.0)
    assert any(issue.code == "percent_transmittance_out_of_range" for issue in result.issues)


def test_unknown_semantics_blocks_quantitative_but_can_be_displayed() -> None:
    assert not validate_signal([1], "mystery", None).is_valid
    advisory = validate_signal([1], "mystery", None, quantitative=False)
    assert advisory.is_valid and advisory.issues[0].severity == "advisory"
    quarantined = quarantine_for_display([1, 2], advisory.issues)
    assert quarantined.quarantined and quarantined.raw_values == (1.0, 2.0)


def test_axis_reports_duplicates_and_non_monotonic_without_rewriting() -> None:
    check = validate_axis([1, 3, 3, 2], "wavelength", "nm")
    assert {issue.code for issue in check.issues} >= {"axis_duplicates", "axis_not_monotonic"}
    axis, values, record = sort_and_deduplicate_axis([3, 1, 3, 2], [30, 10, 31, 20])
    assert axis == (1.0, 2.0, 3.0) and values == (10.0, 20.0, 30.0)
    assert record.action == "sort_and_deduplicate_axis"


def test_signal_preservation_and_explicit_reflectance_bounds() -> None:
    source = [-1, 101]
    assert any(issue.code == "reflectance_out_of_range" for issue in validate_signal(source, "reflectance", "%").issues)
    unknown_unit = validate_signal(source, "reflectance", "instrument_units")
    assert not unknown_unit.is_valid
    assert any(issue.code == "signal_unit_incompatible" for issue in unknown_unit.issues)
    assert unknown_unit.raw_values == (-1.0, 101.0)


def test_transmittance_semantics_are_resolved_only_from_explicit_unit() -> None:
    percent = validate_signal([100, 50], "transmittance", "%")
    fraction = validate_signal([1.0, 0.5], "transmittance", "fraction")
    assert percent.normalized_kind == "percent_transmittance"
    assert fraction.normalized_kind == "transmittance"
    assert percent.is_valid and fraction.is_valid
