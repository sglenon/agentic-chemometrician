"""Spectral preprocessing methods for chemometrics analysis.

All functions are pure numpy/scipy — no matplotlib dependency.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from scipy.sparse import diags, eye as speye
from scipy.sparse.linalg import spsolve


def _adaptive_sg_window(n_features: int) -> int:
    window = min(11, n_features - 1)
    window = max(5, window)
    if window % 2 == 0:
        window -= 1
    return window


def _als_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.01, max_iter: int = 20) -> np.ndarray:
    L = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    H = lam * D.T @ D
    w = np.ones(L)
    z = y.copy()
    for _ in range(max_iter):
        W = diags(w, 0)
        Z = W + H
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def apply(X: np.ndarray, method: str, *, axis: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Apply a named preprocessing method to X (n_samples × n_features).

    Parameters
    ----------
    X:
        2D array of shape (n_samples, n_features).
    method:
        One of ``"raw"``, ``"snv"``, ``"msc"``, ``"sg_1st_deriv"``, ``"sg_2nd_deriv"``,
        ``"baseline_correction"``, ``"area_normalization"``.
    axis:
        Optional 1D array of axis values (e.g. wavenumbers) for spacing-aware methods.

    Returns
    -------
    tuple of (X_processed, details_dict).

    Raises
    ------
    ValueError
        If X is not 2D or if method is unknown.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")

    n, p = X.shape

    if method == "raw":
        details: dict = {
            "method": method,
            "shape_in": [n, p],
            "shape_out": [n, p],
        }
        return X.copy(), details

    if method == "snv":
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        # Avoid division by zero: replace zero std with 1
        std = np.where(std == 0, 1.0, std)
        X_out = (X - mean) / std
        details = {
            "method": method,
            "shape_in": [n, p],
            "shape_out": list(X_out.shape),
        }
        return X_out, details

    if method == "msc":
        mean_spectrum = X.mean(axis=0)
        X_out = np.empty_like(X, dtype=float)
        for i in range(n):
            # OLS fit: spectrum_i = a * mean_spectrum + b
            coeffs = np.polyfit(mean_spectrum, X[i], deg=1)
            a, b = coeffs[0], coeffs[1]
            X_out[i] = (X[i] - b) / a if a != 0 else X[i] - b
        details = {
            "method": method,
            "shape_in": [n, p],
            "shape_out": list(X_out.shape),
            "reference_mean_shape": [p],
        }
        return X_out, details

    if method == "sg_1st_deriv":
        window_length = _adaptive_sg_window(p)
        polyorder = 2
        deriv = 1
        X_out = savgol_filter(X, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)
        details = {
            "method": method,
            "shape_in": [n, p],
            "shape_out": list(X_out.shape),
            "window_length": window_length,
            "polyorder": polyorder,
            "deriv": deriv,
        }
        return X_out, details

    if method == "sg_2nd_deriv":
        window_length = _adaptive_sg_window(p)
        polyorder = 2
        deriv = 2
        X_out = savgol_filter(X, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)
        details = {
            "method": method,
            "shape_in": [n, p],
            "shape_out": list(X_out.shape),
            "window_length": window_length,
            "polyorder": polyorder,
            "deriv": deriv,
        }
        return X_out, details

    if method == "baseline_correction":
        X_out = np.empty_like(X, dtype=float)
        for i in range(n):
            baseline = _als_baseline(X[i])
            X_out[i] = X[i] - baseline
        details = {
            "method": method,
            "shape_in": [n, p],
            "shape_out": list(X_out.shape),
            "als_lam": 1e5,
            "als_p": 0.01,
            "als_max_iter": 20,
        }
        return X_out, details

    if method == "area_normalization":
        if axis is not None:
            dx = np.diff(axis)
            areas = np.array([np.trapezoid(X[i], x=axis) for i in range(n)])
        else:
            areas = np.array([np.trapezoid(X[i]) for i in range(n)])
        areas = np.where(areas == 0, 1.0, areas)
        X_out = X / areas[:, np.newaxis]
        details = {
            "method": method,
            "shape_in": [n, p],
            "shape_out": list(X_out.shape),
            "used_axis": axis is not None,
        }
        return X_out, details

    if method == "standard_scaler":
        # Column-wise z-score normalization (fit on full X — acceptable for exploratory runs).
        # Note: fitting on all samples before CV splits is a minor data-leakage risk;
        # acceptable for chemometrics exploratory analysis but flagged as a caveat.
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_out = scaler.fit_transform(X)
        details = {
            "method": method,
            "shape_in": [n, p],
            "shape_out": list(X_out.shape),
            "note": "StandardScaler fit on full X — minor leakage risk in CV",
        }
        return X_out, details

    if method == "robust_scaler":
        # Column-wise scaling using IQR (robust to outlier wavelengths).
        # Same caveat as standard_scaler.
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_out = scaler.fit_transform(X)
        details = {
            "method": method,
            "shape_in": [n, p],
            "shape_out": list(X_out.shape),
            "note": "RobustScaler fit on full X — minor leakage risk in CV",
        }
        return X_out, details

    raise ValueError(f"Unknown preprocessing method: {method!r}")
