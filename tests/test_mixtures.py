import numpy as np
import pytest

from chemometrics_mcp.core.mixtures import estimate_mixtures


REFS = [[1, 0, 0], [0, 1, 0]]


def test_nonnegative_reference_estimation_and_diagnostics():
    result = estimate_mixtures(REFS, [[0.25, 0.75, 0]], sum_to_one=True)
    assert np.allclose(result["coefficients"], [[0.25, 0.75]])
    assert result["rmse"][0] < 1e-10
    assert result["closure_error"][0] < 1e-10
    assert result["raw_coefficients"] == result["coefficients"]
    assert result["constraint_correction_norm"][0] < 1e-10
    assert "lod" not in result and "purity" not in result


def test_shape_and_finite_validation():
    with pytest.raises(ValueError):
        estimate_mixtures([[1, 0]], [[1, 0, 2]])
    with pytest.raises(ValueError):
        estimate_mixtures([[np.nan, 0]], [[1, 0]])


def test_rank_warning_and_bootstrap_requires_independent_groups():
    result = estimate_mixtures([[1, 0], [2, 0]], [[1, 0]], bootstrap_iterations=10)
    assert any(issue["code"] == "rank_deficient_references" for issue in result["issues"])
    assert result["bootstrap_uncertainty"] == {"available": False, "reason": "independent_replicate_groups_required"}


def test_bounded_bootstrap_with_explicit_groups():
    mixtures = [[.2, .8, 0], [.25, .75, 0], [.8, .2, 0], [.75, .25, 0]]
    result = estimate_mixtures(REFS, mixtures, sum_to_one=True, replicate_groups=["a", "a", "b", "b"], bootstrap_iterations=20)
    assert result["bootstrap_uncertainty"]["available"]
    assert result["bootstrap_uncertainty"]["iterations"] == 20


def test_constrained_fit_retains_raw_prediction_and_single_bootstrap_is_finite():
    result = estimate_mixtures(
        REFS,
        [[0.2, 0.2, 0.0], [0.8, 0.1, 0.0]],
        sum_to_one=True,
        replicate_groups=["a", "b"],
        bootstrap_iterations=1,
    )
    assert not np.allclose(result["raw_coefficients"], result["coefficients"])
    assert np.isfinite(
        result["bootstrap_uncertainty"]["mean_coefficient_std"]
    ).all()
