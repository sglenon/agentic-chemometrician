import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline

from chemometrics_mcp.core.pipelines import (
    AreaNormalizer, MSCTransformer, SNVTransformer, SavgolDerivativeTransformer, build_pipeline,
    make_preprocessor, validate_fold_safe_pipeline,
)


def test_msc_reference_is_training_fold_only() -> None:
    train = np.array([[1., 2., 3.], [2., 4., 6.]])
    held_out = np.array([[1000., 2000., 3000.]])
    fitted = MSCTransformer().fit(train)
    assert np.array_equal(fitted.reference_, train.mean(axis=0))
    assert not np.array_equal(fitted.reference_, np.vstack([train, held_out]).mean(axis=0))


def test_sample_local_snv_does_not_depend_on_other_rows() -> None:
    row = np.array([[1., 2., 4.]])
    alone = SNVTransformer().fit_transform(row)
    together = SNVTransformer().fit_transform(np.vstack([row, [100., 200., 300.]]))
    assert np.allclose(alone[0], together[0])


def test_sklearn_clone_and_cross_validation_compatibility() -> None:
    X = np.arange(40, dtype=float).reshape(10, 4) + np.array([0, 1, 2, 4])
    y = np.arange(10, dtype=float)
    pipeline = build_pipeline("snv", LinearRegression())
    assert isinstance(clone(pipeline), Pipeline)
    prediction = cross_val_predict(pipeline, X, y, cv=KFold(2))
    assert prediction.shape == y.shape


def test_short_axis_derivative_and_non_estimator_are_rejected() -> None:
    with pytest.raises(ValueError):
        SavgolDerivativeTransformer().fit(np.ones((2, 2)))
    with pytest.raises(TypeError):
        validate_fold_safe_pipeline(Pipeline([("bad", object()), ("model", LinearRegression())]))


def test_provenance_and_factory() -> None:
    transformer = make_preprocessor("sg_2nd_deriv").fit(np.arange(14, dtype=float).reshape(2, 7))
    provenance = transformer.get_provenance()
    assert provenance["transform_scope"] == "sample_local"
    assert provenance["fitted_state"]["window_length"] == 7
    assert make_preprocessor("msc").transform_scope == "training_fold"


def test_area_normalization_is_independent_of_axis_orientation() -> None:
    X = np.array([[1.0, 2.0, 1.0]])
    ascending = AreaNormalizer(axis=[1.0, 2.0, 3.0]).fit_transform(X)
    descending = AreaNormalizer(axis=[3.0, 2.0, 1.0]).fit_transform(X)
    assert np.allclose(ascending, descending)


def test_derivative_validates_and_uses_axis_spacing() -> None:
    X = np.array([[0.0, 1.0, 4.0, 9.0, 16.0]])
    first = SavgolDerivativeTransformer(
        order=1, window_length=5, polyorder=2, delta=1.0
    ).fit_transform(X)
    doubled_spacing = SavgolDerivativeTransformer(
        order=1, window_length=5, polyorder=2, delta=2.0
    ).fit_transform(X)
    assert np.allclose(first, doubled_spacing * 2)
    with pytest.raises(ValueError):
        SavgolDerivativeTransformer(delta=0).fit(X)
