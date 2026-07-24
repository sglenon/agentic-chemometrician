"""Conservative explicit-reference spectral mixture estimation.

The functions here estimate non-negative reference coefficients.  They do not
turn those coefficients into purity, LOD, or LOQ claims.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize, nnls


def _matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or not array.size:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _issue(code: str, message: str, level: str = "advisory", **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "level": level, "details": details}


def _fit(reference: np.ndarray, mixture: np.ndarray, closure: bool) -> np.ndarray:
    design = reference.T
    if not closure:
        return nnls(design, mixture)[0]
    initial = nnls(design, mixture)[0]
    initial = initial / initial.sum() if initial.sum() else np.full(reference.shape[0], 1 / reference.shape[0])
    result = minimize(
        lambda coefficients: float(np.sum((design @ coefficients - mixture) ** 2)), initial,
        method="SLSQP", bounds=[(0.0, None)] * reference.shape[0],
        constraints={"type": "eq", "fun": lambda coefficients: float(coefficients.sum() - 1.0)},
    )
    if not result.success:
        raise ValueError(f"sum-to-one constrained fit failed: {result.message}")
    return result.x


def _bootstrap(reference: np.ndarray, mixtures: np.ndarray, groups: Sequence[Any] | None,
               closure: bool, iterations: int, seed: int) -> dict[str, Any]:
    if not iterations:
        return {"available": False, "reason": "not_requested"}
    if groups is None:
        return {"available": False, "reason": "independent_replicate_groups_required"}
    labels = np.asarray(groups, dtype=object)
    if labels.ndim != 1 or len(labels) != len(mixtures):
        raise ValueError("replicate_groups must provide one explicit group per mixture")
    unique = np.unique(labels)
    if len(unique) < 2:
        return {"available": False, "reason": "at_least_two_independent_groups_required"}
    count = min(int(iterations), 1_000)
    if count < 1:
        raise ValueError("bootstrap_iterations must be non-negative")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(count):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([np.flatnonzero(labels == group) for group in selected])
        coefficients = np.asarray([_fit(reference, mixture, closure) for mixture in mixtures[rows]])
        estimates.append(coefficients.mean(axis=0))
    return {
        "available": True,
        "iterations": count,
        "group_count": int(len(unique)),
        # ddof=0 remains finite for the explicitly supported one-iteration
        # diagnostic and avoids non-JSON NaN values in persisted evidence.
        "mean_coefficient_std": np.std(
            np.asarray(estimates), axis=0, ddof=0
        ).tolist(),
    }


def estimate_mixtures(
    reference_spectra: Any,
    mixture_spectra: Any,
    *,
    sum_to_one: bool = False,
    reference_names: Sequence[str] | None = None,
    reference_roles: Sequence[str] | None = None,
    replicate_groups: Sequence[Any] | None = None,
    bootstrap_iterations: int = 0,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Estimate coefficients for already axis-aligned mixtures and references.

    ``reference_spectra`` has shape ``(components, features)`` and
    ``mixture_spectra`` has shape ``(mixtures, features)`` (a single mixture is
    accepted as one row).  Coefficients have no purity interpretation unless a
    caller supplies a separately justified scientific model.
    """
    reference = _matrix(reference_spectra, "reference_spectra")
    mixtures = _matrix(mixture_spectra, "mixture_spectra")
    if reference.shape[1] != mixtures.shape[1]:
        raise ValueError("reference_spectra and mixture_spectra must share feature count/aligned axis")
    if reference_names is not None and len(reference_names) != reference.shape[0]:
        raise ValueError("reference_names must match the number of reference components")
    if reference_roles is not None and len(reference_roles) != reference.shape[0]:
        raise ValueError("reference_roles must match the number of reference components")
    if bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations must be non-negative")

    raw_coefficients = np.asarray(
        [_fit(reference, mixture, False) for mixture in mixtures]
    )
    coefficients = (
        np.asarray([_fit(reference, mixture, True) for mixture in mixtures])
        if sum_to_one
        else raw_coefficients.copy()
    )
    reconstructed = coefficients @ reference
    residuals = mixtures - reconstructed
    rmse = np.sqrt(np.mean(residuals ** 2, axis=1))
    closure_error = np.abs(coefficients.sum(axis=1) - 1.0)
    rank = int(np.linalg.matrix_rank(reference))
    condition = float(np.linalg.cond(reference))
    issues: list[dict[str, Any]] = []
    if rank < reference.shape[0]:
        issues.append(_issue("rank_deficient_references", "Reference spectra are rank deficient; component coefficients are not uniquely identifiable.", "advisory", rank=rank, components=reference.shape[0]))
    if not np.isfinite(condition) or condition > 1e8:
        issues.append(_issue("ill_conditioned_references", "Reference spectra are poorly conditioned; coefficient estimates may be unstable.", "advisory", condition_number=condition))
    if not sum_to_one:
        issues.append(_issue("closure_not_enforced", "Coefficient sums were not constrained to one; interpret them as fit weights.", "information"))
    uncertainty = _bootstrap(reference, mixtures, replicate_groups, sum_to_one, bootstrap_iterations, random_seed)
    provenance = {
        "method": "nonnegative_least_squares" if not sum_to_one else "nonnegative_sum_to_one_least_squares",
        "reference_component_count": int(reference.shape[0]), "feature_count": int(reference.shape[1]),
        "mixture_count": int(mixtures.shape[0]), "sum_to_one": sum_to_one,
        "reference_names": list(reference_names) if reference_names is not None else None,
        "reference_roles": list(reference_roles) if reference_roles is not None else None,
        "purity_claim_supported": False,
    }
    return {
        "coefficients": coefficients.tolist(),
        "raw_coefficients": raw_coefficients.tolist(),
        "constraint_correction_norm": np.linalg.norm(
            coefficients - raw_coefficients, axis=1
        ).tolist(),
        "reconstructed_spectra": reconstructed.tolist(),
        "residuals": residuals.tolist(), "rmse": rmse.tolist(), "closure_error": closure_error.tolist(),
        "condition_number": condition, "rank": rank, "issues": issues,
        "bootstrap_uncertainty": uncertainty, "provenance": provenance,
    }
