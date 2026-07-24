"""Fold-safe sklearn preprocessing primitives for spectra.

Transforms that operate on individual spectra never learn across rows.  MSC is
the sole stateful transform here and learns its reference only in ``fit``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import savgol_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


def _as_matrix(X: Any, *, n_features: int | None = None) -> np.ndarray:
    array = np.asarray(X, dtype=float)
    if array.ndim != 2 or not array.size:
        raise ValueError("X must be a non-empty 2D numeric array")
    if n_features is not None and array.shape[1] != n_features:
        raise ValueError("X has a different number of features than fit data")
    if not np.isfinite(array).all():
        raise ValueError("X must contain only finite values")
    return array


class _SpectralTransformer(BaseEstimator, TransformerMixin):
    transform_scope = "sample_local"

    def fit(self, X: Any, y: Any = None):
        matrix = _as_matrix(X)
        self.n_features_in_ = matrix.shape[1]
        return self

    def _matrix_for_transform(self, X: Any) -> np.ndarray:
        if not hasattr(self, "n_features_in_"):
            raise ValueError("Transformer has not been fitted")
        return _as_matrix(X, n_features=self.n_features_in_)

    def get_provenance(self) -> dict[str, Any]:
        fitted = {"n_features": getattr(self, "n_features_in_", None)}
        return {
            "transformer": type(self).__name__,
            "transform_scope": self.transform_scope,
            "parameters": self.get_params(deep=False),
            "fitted_state": fitted,
        }


class IdentityTransformer(_SpectralTransformer):
    """An explicit raw/no-op preprocessing step."""

    def transform(self, X: Any) -> np.ndarray:
        return self._matrix_for_transform(X).copy()


class SNVTransformer(_SpectralTransformer):
    """Standard normal variate, independently applied to each row."""

    def transform(self, X: Any) -> np.ndarray:
        matrix = self._matrix_for_transform(X)
        mean = matrix.mean(axis=1, keepdims=True)
        scale = matrix.std(axis=1, keepdims=True)
        if np.any(scale == 0):
            raise ValueError("SNV cannot normalize a constant spectrum")
        return (matrix - mean) / scale


class SavgolDerivativeTransformer(_SpectralTransformer):
    """Sample-local Savitzky--Golay first or second derivative."""

    def __init__(self, order: int = 1, window_length: int | None = None,
                 polyorder: int = 2, delta: float = 1.0):
        self.order = order
        self.window_length = window_length
        self.polyorder = polyorder
        self.delta = delta

    def fit(self, X: Any, y: Any = None):
        super().fit(X, y)
        if self.order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        if self.polyorder < self.order:
            raise ValueError("polyorder must be at least derivative order")
        if not np.isfinite(self.delta) or self.delta <= 0:
            raise ValueError("delta must be a positive finite axis spacing")
        n_features = self.n_features_in_
        if self.window_length is None:
            window = min(11, n_features if n_features % 2 else n_features - 1)
        else:
            window = self.window_length
        if not isinstance(window, (int, np.integer)) or window <= self.polyorder or window % 2 == 0 or window > n_features:
            raise ValueError("Savgol window must be odd, exceed polyorder, and fit the spectral axis")
        self.window_length_ = int(window)
        return self

    def transform(self, X: Any) -> np.ndarray:
        matrix = self._matrix_for_transform(X)
        return savgol_filter(
            matrix,
            self.window_length_,
            self.polyorder,
            deriv=self.order,
            delta=self.delta,
            axis=1,
        )

    def get_provenance(self) -> dict[str, Any]:
        result = super().get_provenance()
        result["fitted_state"]["window_length"] = getattr(self, "window_length_", None)
        return result


class AreaNormalizer(_SpectralTransformer):
    """Normalize each spectrum by its own trapezoidal area."""

    def __init__(self, axis: Any = None):
        self.axis = axis

    def fit(self, X: Any, y: Any = None):
        super().fit(X, y)
        if self.axis is not None:
            axis = np.asarray(self.axis, dtype=float)
            if axis.ndim != 1 or len(axis) != self.n_features_in_ or not np.isfinite(axis).all():
                raise ValueError("axis must be finite, one-dimensional, and match X features")
            self.axis_ = axis.copy()
        return self

    def transform(self, X: Any) -> np.ndarray:
        matrix = self._matrix_for_transform(X)
        # Axis orientation must not invert the spectrum during normalization.
        areas = np.abs(
            np.trapezoid(matrix, x=getattr(self, "axis_", None), axis=1)
        )
        if np.any(areas == 0):
            raise ValueError("Area normalization cannot normalize a zero-area spectrum")
        return matrix / areas[:, None]


def _als_baseline(row: np.ndarray, lam: float, p: float, max_iter: int) -> np.ndarray:
    n = len(row)
    if n < 3:
        raise ValueError("Baseline correction requires at least three spectral features")
    difference = diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
    penalty = lam * difference.T @ difference
    weights = np.ones(n)
    baseline = row.copy()
    for _ in range(max_iter):
        baseline = spsolve(diags(weights, 0) + penalty, weights * row)
        weights = p * (row > baseline) + (1 - p) * (row <= baseline)
    return np.asarray(baseline)


class BaselineCorrector(_SpectralTransformer):
    """ALS baseline correction, independently applied to each sample."""

    def __init__(self, lam: float = 1e5, p: float = 0.01, max_iter: int = 20):
        self.lam = lam
        self.p = p
        self.max_iter = max_iter

    def fit(self, X: Any, y: Any = None):
        super().fit(X, y)
        if self.lam <= 0 or not 0 < self.p < 1 or self.max_iter < 1:
            raise ValueError("lam must be positive, p must be in (0, 1), and max_iter must be positive")
        return self

    def transform(self, X: Any) -> np.ndarray:
        matrix = self._matrix_for_transform(X)
        return np.vstack([row - _als_baseline(row, self.lam, self.p, self.max_iter) for row in matrix])


class MSCTransformer(_SpectralTransformer):
    """Multiplicative scatter correction using a training-fold reference."""
    transform_scope = "training_fold"

    def fit(self, X: Any, y: Any = None):
        super().fit(X, y)
        self.reference_ = _as_matrix(X).mean(axis=0)
        if np.std(self.reference_) == 0:
            raise ValueError("MSC reference spectrum must not be constant")
        return self

    def transform(self, X: Any) -> np.ndarray:
        matrix = self._matrix_for_transform(X)
        corrected = np.empty_like(matrix, dtype=float)
        for index, row in enumerate(matrix):
            slope, intercept = np.polyfit(self.reference_, row, 1)
            if not np.isfinite(slope) or slope == 0:
                raise ValueError("MSC encountered a zero or non-finite slope")
            corrected[index] = (row - intercept) / slope
        return corrected

    def get_provenance(self) -> dict[str, Any]:
        result = super().get_provenance()
        result["fitted_state"]["reference_spectrum"] = (
            self.reference_.tolist() if hasattr(self, "reference_") else None
        )
        return result


def make_preprocessor(name: str, **params: Any) -> _SpectralTransformer:
    """Build one supported preprocessing transformer by its stable short name."""
    factories = {
        "raw": IdentityTransformer,
        "snv": SNVTransformer,
        "sg_1st_deriv": lambda **kwargs: SavgolDerivativeTransformer(order=1, **kwargs),
        "sg_2nd_deriv": lambda **kwargs: SavgolDerivativeTransformer(order=2, **kwargs),
        "area_normalization": AreaNormalizer,
        "baseline_correction": BaselineCorrector,
        "msc": MSCTransformer,
    }
    try:
        return factories[name](**params)
    except KeyError as exc:
        raise ValueError(f"Unknown preprocessor: {name!r}") from exc


def build_pipeline(preprocessor: str | _SpectralTransformer, estimator: BaseEstimator) -> Pipeline:
    transformer = make_preprocessor(preprocessor) if isinstance(preprocessor, str) else preprocessor
    if not isinstance(transformer, BaseEstimator) or not hasattr(transformer, "transform"):
        raise TypeError("preprocessor must be an sklearn-compatible transformer")
    if not isinstance(estimator, BaseEstimator):
        raise TypeError("estimator must be an sklearn BaseEstimator")
    pipeline = Pipeline([("preprocessor", transformer), ("estimator", estimator)])
    validate_fold_safe_pipeline(pipeline)
    return pipeline


def validate_fold_safe_pipeline(pipeline: Pipeline) -> bool:
    """Require sklearn estimators for every preprocessing pipeline step."""
    if not isinstance(pipeline, Pipeline):
        raise TypeError("pipeline must be sklearn.pipeline.Pipeline")
    for name, step in pipeline.steps:
        if not isinstance(step, BaseEstimator):
            raise TypeError(f"Pipeline step {name!r} is not an sklearn estimator")
        if name != pipeline.steps[-1][0] and (not hasattr(step, "fit") or not hasattr(step, "transform")):
            raise TypeError(f"Pipeline transformer {name!r} must implement fit and transform")
    return True
