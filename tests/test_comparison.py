import numpy as np
import pytest

from chemometrics_mcp.core.comparison import (
    canonicalize_spectrum, compare_continuous_spectra, compare_peak_lists, detect_peaks,
)


def test_continuous_comparison_uses_overlap_only_and_does_not_mutate_raw() -> None:
    left_axis, left_signal = np.array([2., 0., 1., 1.]), np.array([2., 0., 1., 3.])
    result = compare_continuous_spectra(left_axis, left_signal, [1, 2, 3], [2, 4, 6])
    assert result.overlap == (1.0, 2.0) and result.n_overlap_points == 2
    assert result.metrics["rmse"] is not None
    assert np.array_equal(left_axis, [2., 0., 1., 1.])
    axis, signal = canonicalize_spectrum(left_axis, left_signal)
    assert np.array_equal(axis, [0, 1, 2]) and np.array_equal(signal, [0, 2, 2])


def test_no_extrapolation_and_zero_signal_warnings() -> None:
    no_overlap = compare_continuous_spectra([0, 1], [1, 2], [3, 4], [2, 3])
    assert no_overlap.n_overlap_points == 0 and "extrapolation" in no_overlap.warnings[0]
    zero = compare_continuous_spectra([0, 1, 2], [0, 0, 0], [0, 1, 2], [1, 2, 3], normalization="max")
    assert zero.metrics["cosine_similarity"] is None
    assert any("zero signal" in warning for warning in zero.warnings)


def test_peak_detection_and_one_to_one_tolerance_matching() -> None:
    assert detect_peaks([0, 1, 2, 3, 4], [0, 3, 0, 2, 0]) == ((1.0, 3.0), (3.0, 2.0))
    result = compare_peak_lists([(1, 10), (2, 9)], [(1.05, 8), (1.1, 7), (3, 5)], tolerance=.11)
    assert len(result.matches) == 1 and result.match_fraction == 1 / 3
    assert len(result.unmatched_left) == 1 and len(result.unmatched_right) == 2
    assert result.to_dict()["tolerance"] == .11


def test_peak_matching_maximizes_cardinality_before_distance() -> None:
    # Greedy nearest-neighbour matching would consume 1.0 -> 0.9 first and
    # leave only one match; the global one-to-one solution retains both.
    result = compare_peak_lists([0.0, 1.0], [0.9, 1.9], tolerance=1.0)
    assert len(result.matches) == 2


def test_invalid_values_are_rejected() -> None:
    with pytest.raises(ValueError):
        compare_continuous_spectra([0, np.nan], [1, 2], [0, 1], [1, 2])
    with pytest.raises(ValueError):
        compare_peak_lists([1], [2], tolerance=-1)
