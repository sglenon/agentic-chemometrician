"""Spectral preprocessing methods for chemometrics analysis.

All functions are pure numpy/scipy — no matplotlib dependency.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def apply(X: np.ndarray, method: str) -> tuple[np.ndarray, dict]:
    """Apply a named preprocessing method to X (n_samples × n_features).

    Parameters
    ----------
    X:
        2D array of shape (n_samples, n_features).
    method:
        One of ``"raw"``, ``"snv"``, ``"msc"``, ``"sg_1st_deriv"``, ``"sg_2nd_deriv"``.

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
        window_length = 11
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
        window_length = 11
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

    raise ValueError(f"Unknown preprocessing method: {method!r}")
