"""Deterministic, modality-neutral comparison primitives for spectral data."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks


@dataclass(frozen=True)
class ComparisonResult:
    metrics: dict[str, float | None]
    overlap: tuple[float, float] | None
    overlap_fraction: float
    n_overlap_points: int
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PeakMatch:
    left_position: float
    right_position: float
    distance: float
    left_intensity: float | None = None
    right_intensity: float | None = None


@dataclass(frozen=True)
class PeakComparisonResult:
    matches: tuple[PeakMatch, ...]
    unmatched_left: tuple[tuple[float, float | None], ...]
    unmatched_right: tuple[tuple[float, float | None], ...]
    match_fraction: float
    tolerance: float
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _vectors(axis: Iterable[float], signal: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(tuple(axis), dtype=float), np.asarray(tuple(signal), dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or not len(x):
        raise ValueError("axis and signal must be non-empty, equal-length 1D arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("axis and signal must contain only finite values")
    return x, y


def canonicalize_spectrum(axis: Iterable[float], signal: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted coordinates, averaging duplicate signals without mutating input."""
    x, y = _vectors(axis, signal)
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]
    unique, inverse = np.unique(x, return_inverse=True)
    summed = np.bincount(inverse, weights=y)
    counts = np.bincount(inverse)
    return unique, summed / counts


def _normalize(signal: np.ndarray, axis: np.ndarray, method: str | None) -> tuple[np.ndarray, dict[str, Any], str | None]:
    if method is None or method == "none":
        return signal.copy(), {"normalization": None}, None
    if method == "area":
        scale = float(np.trapezoid(np.abs(signal), axis))
    elif method == "max":
        scale = float(np.max(np.abs(signal)))
    else:
        raise ValueError("normalization must be None, 'area', or 'max'")
    if scale == 0:
        return signal.copy(), {"normalization": method, "scale": scale}, "zero signal; normalization skipped"
    return signal / scale, {"normalization": method, "scale": scale}, None


def compare_continuous_spectra(
    left_axis: Iterable[float], left_signal: Iterable[float],
    right_axis: Iterable[float], right_signal: Iterable[float], *,
    normalization: str | None = None, min_overlap_fraction: float = 0.05,
) -> ComparisonResult:
    """Compare two continuous spectra on their shared axis interval only."""
    if not 0 <= min_overlap_fraction <= 1:
        raise ValueError("min_overlap_fraction must be between zero and one")
    lx, ly = canonicalize_spectrum(left_axis, left_signal)
    rx, ry = canonicalize_spectrum(right_axis, right_signal)
    low, high = max(lx[0], rx[0]), min(lx[-1], rx[-1])
    left_span, right_span = lx[-1] - lx[0], rx[-1] - rx[0]
    common_span = max(0.0, high - low)
    denominator = min(left_span, right_span)
    overlap_fraction = float(common_span / denominator) if denominator > 0 else 0.0
    warnings: list[str] = []
    provenance: dict[str, Any] = {"canonicalized": True, "interpolation": "linear_overlap_only"}
    if high < low or common_span == 0:
        warnings.append("insufficient overlap; no extrapolation was performed")
        return ComparisonResult({}, None, overlap_fraction, 0, provenance, tuple(warnings))
    grid = np.unique(np.concatenate((lx[(lx >= low) & (lx <= high)], rx[(rx >= low) & (rx <= high)])))
    if len(grid) < 2:
        warnings.append("insufficient overlap points for continuous metrics")
        return ComparisonResult({}, (float(low), float(high)), overlap_fraction, len(grid), provenance, tuple(warnings))
    # np.interp is bounded here by construction: grid is wholly inside both axes.
    left = np.interp(grid, lx, ly)
    right = np.interp(grid, rx, ry)
    left, left_provenance, left_warning = _normalize(left, grid, normalization)
    right, right_provenance, right_warning = _normalize(right, grid, normalization)
    provenance["left"] = left_provenance
    provenance["right"] = right_provenance
    warnings.extend(item for item in (left_warning, right_warning) if item)
    if overlap_fraction < min_overlap_fraction:
        warnings.append("overlap fraction is below the requested minimum")
    lnorm, rnorm = float(np.linalg.norm(left)), float(np.linalg.norm(right))
    metrics: dict[str, float | None] = {"cosine_similarity": None, "spectral_angle": None, "correlation": None, "rmse": None}
    if not lnorm or not rnorm:
        warnings.append("zero signal prevents cosine similarity and spectral angle")
    else:
        cosine = float(np.clip(np.dot(left, right) / (lnorm * rnorm), -1.0, 1.0))
        metrics["cosine_similarity"] = cosine
        metrics["spectral_angle"] = float(np.arccos(cosine))
    if np.std(left) == 0 or np.std(right) == 0:
        warnings.append("zero-variance signal prevents correlation")
    else:
        metrics["correlation"] = float(np.corrcoef(left, right)[0, 1])
    metrics["rmse"] = float(np.sqrt(np.mean((left - right) ** 2)))
    return ComparisonResult(metrics, (float(low), float(high)), overlap_fraction, len(grid), provenance, tuple(warnings))


def detect_peaks(axis: Iterable[float], signal: Iterable[float], *, prominence: float | None = None,
                 max_peaks: int = 1000) -> tuple[tuple[float, float], ...]:
    """Detect bounded local maxima after canonicalizing a continuous spectrum."""
    if max_peaks < 1:
        raise ValueError("max_peaks must be positive")
    x, y = canonicalize_spectrum(axis, signal)
    if len(x) < 3:
        return ()
    if prominence is None:
        # A robust, deterministic threshold that remains permissive for clean data.
        prominence = max(0.0, float(np.median(np.abs(y - np.median(y))) * 0.5))
    indices, properties = find_peaks(y, prominence=prominence)
    if len(indices) > max_peaks:
        ranks = np.argsort(properties["prominences"])[-max_peaks:]
        indices = np.sort(indices[ranks])
    return tuple((float(x[index]), float(y[index])) for index in indices)


def _peak_pairs(peaks: Sequence[float] | Sequence[Sequence[float]]) -> tuple[tuple[float, float | None], ...]:
    result: list[tuple[float, float | None]] = []
    for item in peaks:
        if np.isscalar(item):
            position, intensity = float(item), None
        else:
            if len(item) not in (1, 2):
                raise ValueError("each peak must be a position or (position, intensity)")
            position, intensity = float(item[0]), float(item[1]) if len(item) == 2 else None
        if not np.isfinite(position) or (intensity is not None and not np.isfinite(intensity)):
            raise ValueError("peak values must be finite")
        result.append((position, intensity))
    return tuple(sorted(result, key=lambda peak: peak[0]))


def compare_peak_lists(left_peaks: Sequence[float] | Sequence[Sequence[float]],
                       right_peaks: Sequence[float] | Sequence[Sequence[float]], *,
                       tolerance: float) -> PeakComparisonResult:
    """One-to-one match peak positions within an explicit tolerance."""
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite non-negative number")
    left, right = _peak_pairs(left_peaks), _peak_pairs(right_peaks)
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[PeakMatch] = []
    if left and right:
        # Padded assignment makes each unmatched peak cost ``unmatched_cost``.
        # Since every valid match costs less than two unmatched assignments,
        # the solution first maximizes cardinality, then minimizes mass/axis
        # error. A nearest-first greedy pass can lose valid matches.
        n_left, n_right = len(left), len(right)
        unmatched_cost = max(float(tolerance), 1.0) + 1.0
        cost = np.zeros((n_left + n_right, n_right + n_left), dtype=float)
        cost[:n_left, :n_right] = unmatched_cost * 4.0
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                distance = abs(a[0] - b[0])
                if distance <= tolerance:
                    cost[i, j] = distance
        cost[:n_left, n_right:] = unmatched_cost
        cost[n_left:, :n_right] = unmatched_cost
        rows, columns = linear_sum_assignment(cost)
        for i, j in zip(rows, columns):
            if i >= n_left or j >= n_right:
                continue
            distance = abs(left[i][0] - right[j][0])
            if distance > tolerance:
                continue
            used_left.add(i)
            used_right.add(j)
            matches.append(PeakMatch(left[i][0], right[j][0], float(distance), left[i][1], right[j][1]))
        matches.sort(key=lambda item: (item.left_position, item.right_position))
    warnings: list[str] = []
    if not left or not right:
        warnings.append("one or both peak lists are empty")
    fraction = len(matches) / max(len(left), len(right)) if left or right else 0.0
    return PeakComparisonResult(tuple(matches), tuple(p for i, p in enumerate(left) if i not in used_left),
                                tuple(p for i, p in enumerate(right) if i not in used_right), fraction, float(tolerance), tuple(warnings))


# Concise aliases for callers that distinguish representation before dispatch.
compare_spectra = compare_continuous_spectra
match_peaks = compare_peak_lists
