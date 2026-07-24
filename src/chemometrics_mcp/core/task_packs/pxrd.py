"""Conservative PXRD reference comparison on explicit two-theta coordinates."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from chemometrics_mcp.core.comparison import canonicalize_spectrum, compare_continuous_spectra, compare_peak_lists, detect_peaks


_SIGNALS = {"diffraction_intensity", "counts", "intensity"}
_REFERENCE_ROLES = {"reference", "simulated_reference", "side_product_candidate"}


def _issue(code: str, message: str, level: str = "blocker", **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "level": level, "details": details}


def _value(record: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return record.get(key, default)


def _validate(record: Mapping[str, Any], *, reference: bool) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if str(_value(record, "axis_kind", "")).lower().replace(" ", "_") not in {"two_theta", "2theta"}:
        issues.append(_issue("pxrd_axis_kind_required", "PXRD comparison requires axis_kind='two_theta'."))
    if str(_value(record, "axis_unit", "")).lower() not in {"degree", "degrees", "deg"}:
        issues.append(_issue("pxrd_axis_unit_required", "PXRD comparison requires two-theta coordinates in degrees."))
    if str(_value(record, "signal_kind", "")).lower().replace(" ", "_") not in _SIGNALS:
        issues.append(_issue("pxrd_signal_semantics_required", "PXRD comparison requires diffraction intensity, counts, or intensity semantics."))
    signal_unit = str(_value(record, "signal_unit", "")).strip().lower()
    if signal_unit not in {
        "count",
        "counts",
        "a.u.",
        "au",
        "arbitrary unit",
        "arbitrary units",
    }:
        issues.append(_issue("pxrd_signal_unit_required", "PXRD comparison requires explicit counts or arbitrary-intensity units."))
    role = str(_value(record, "role", "")).lower()
    if reference and role not in _REFERENCE_ROLES:
        issues.append(_issue("pxrd_reference_role_required", "Candidate patterns must have explicit reference, simulated_reference, or side_product_candidate role."))
    if not reference and role not in {"sample", "unknown"}:
        issues.append(_issue("pxrd_experimental_role_invalid", "Experimental PXRD pattern must be a sample/unknown measurement, not a declared reference."))
    return issues


def _prepared(record: Mapping[str, Any], normalization: str | None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    axis = np.asarray(_value(record, "axis"), dtype=float)
    signal = np.asarray(_value(record, "signal"), dtype=float)
    if axis.ndim != 1 or signal.ndim != 1 or len(axis) != len(signal) or not len(axis):
        raise ValueError("axis and signal must be non-empty equal-length vectors")
    if not np.isfinite(axis).all() or not np.isfinite(signal).all():
        raise ValueError("axis and signal must be finite")
    if np.any(signal < 0):
        raise ValueError("PXRD intensities must be non-negative before optional baseline normalization")
    x, y = canonicalize_spectrum(axis, signal)
    provenance: dict[str, Any] = {"canonicalized": True, "normalization": normalization}
    if normalization is None or normalization == "none":
        return x, y, provenance
    if normalization == "baseline_max":
        y = y - np.min(y)
        scale = float(np.max(y))
    elif normalization == "max":
        scale = float(np.max(np.abs(y)))
    else:
        raise ValueError("normalization must be None, 'max', or 'baseline_max'")
    provenance["scale"] = scale
    if scale:
        y = y / scale
    else:
        provenance["normalization_skipped"] = "zero signal"
    return x, y, provenance


def _peaks(record: Mapping[str, Any], axis: np.ndarray, signal: np.ndarray, prominence: float | None) -> tuple[tuple[float, float], ...]:
    supplied = _value(record, "peaks")
    if supplied is not None:
        return tuple((float(item[0]), float(item[1]) if len(item) > 1 else 1.0) for item in supplied)
    return detect_peaks(axis, signal, prominence=prominence)


def compare_pxrd_references(
    experimental: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], *,
    two_theta_tolerance: float = 0.1, normalization: str | None = None,
    peak_prominence: float | None = None,
) -> dict[str, Any]:
    """Compare one experimental pattern against explicitly declared references.

    Ranking is only a qualitative screening aid. It never identifies a phase or
    establishes phase purity; side-product matches remain review hypotheses.
    """
    if not np.isfinite(two_theta_tolerance) or two_theta_tolerance < 0:
        raise ValueError("two_theta_tolerance must be finite and non-negative")
    if not candidates:
        raise ValueError("at least one candidate reference is required")
    issues = _validate(experimental, reference=False)
    for candidate in candidates:
        issues.extend(_validate(candidate, reference=True))
    if any(item["level"] == "blocker" for item in issues):
        return {"ranked_results": [], "issues": issues, "claim_ceiling": "screening",
                "provenance": {"task": "pxrd_reference_comparison", "executed": False}}
    try:
        exp_axis, exp_signal, exp_provenance = _prepared(
            experimental, normalization
        )
    except (TypeError, ValueError) as exc:
        return {
            "ranked_results": [],
            "issues": issues
            + [_issue("invalid_experimental_pattern", str(exc))],
            "claim_ceiling": "screening",
            "provenance": {
                "task": "pxrd_reference_comparison",
                "executed": False,
            },
        }
    exp_peaks = _peaks(experimental, exp_axis, exp_signal, peak_prominence)
    results = []
    for index, candidate in enumerate(candidates):
        candidate_id = str(
            candidate.get(
                "candidate_id", candidate.get("measurement_id", index)
            )
        )
        try:
            ref_axis, ref_signal, ref_provenance = _prepared(
                candidate, normalization
            )
        except (TypeError, ValueError) as exc:
            issues.append(
                _issue(
                    "invalid_reference_pattern",
                    str(exc),
                    candidate_id=candidate_id,
                )
            )
            results.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_role": str(candidate.get("role", "")),
                    "matched_peaks": [],
                    "unmatched_experimental_peaks": [
                        list(item) for item in exp_peaks
                    ],
                    "unmatched_reference_peaks": [],
                    "matched_peak_fraction": 0.0,
                    "mean_position_error_degrees": None,
                    "whole_pattern_metrics": {},
                    "overlap_fraction": 0.0,
                    "overlap_range_degrees": None,
                    "issues": ["invalid_reference_pattern"],
                }
            )
            continue
        ref_peaks = _peaks(candidate, ref_axis, ref_signal, peak_prominence)
        peak_result = compare_peak_lists(exp_peaks, ref_peaks, tolerance=two_theta_tolerance)
        whole = compare_continuous_spectra(exp_axis, exp_signal, ref_axis, ref_signal, normalization=None)
        position_errors = [match.distance for match in peak_result.matches]
        role = str(candidate.get("role", ""))
        experimental_state = experimental.get("physical_state")
        reference_state = candidate.get("physical_state")
        if (
            experimental_state
            and reference_state
            and experimental_state != reference_state
        ):
            issues.append(
                _issue(
                    "pxrd_physical_state_mismatch",
                    "Experimental and reference specimen physical states differ; ranking is not directly comparable.",
                    "advisory",
                    candidate_id=candidate_id,
                )
            )
        acquisition_keys = (
            "xray_wavelength",
            "geometry",
            "position_calibration",
        )
        if any(
            not experimental.get(key) or not candidate.get(key)
            for key in acquisition_keys
        ):
            issues.append(
                _issue(
                    "pxrd_acquisition_comparability_unconfirmed",
                    "X-ray wavelength, geometry, or position calibration metadata are incomplete.",
                    "advisory",
                    candidate_id=candidate_id,
                )
            )
        evidence = {
            "candidate_id": candidate_id,
            "candidate_role": role, "matched_peaks": [{"experimental_two_theta": match.left_position, "reference_two_theta": match.right_position, "position_error_degrees": match.distance} for match in peak_result.matches],
            "unmatched_experimental_peaks": [list(item) for item in peak_result.unmatched_left],
            "unmatched_reference_peaks": [list(item) for item in peak_result.unmatched_right],
            "matched_peak_fraction": peak_result.match_fraction,
            "mean_position_error_degrees": float(np.mean(position_errors)) if position_errors else None,
            "whole_pattern_metrics": whole.metrics, "overlap_fraction": whole.overlap_fraction,
            "overlap_range_degrees": list(whole.overlap) if whole.overlap else None,
            "provenance": {
                "experimental": exp_provenance,
                "reference": ref_provenance,
                "experimental_peak_source": "provided"
                if experimental.get("peaks") is not None
                else "detected",
                "reference_peak_source": "provided"
                if candidate.get("peaks") is not None
                else "detected",
                "tolerance_degrees": two_theta_tolerance,
            },
        }
        for warning in (*whole.warnings, *peak_result.warnings):
            issues.append(
                _issue(
                    "pxrd_comparison_warning",
                    warning,
                    "advisory",
                    candidate_id=candidate_id,
                )
            )
        if role == "side_product_candidate":
            evidence["scientist_review_required"] = True
        results.append(evidence)
    results.sort(key=lambda row: (-row["matched_peak_fraction"], -(row["whole_pattern_metrics"].get("cosine_similarity") or -1), row["candidate_id"]))
    return {"ranked_results": results, "issues": issues, "claim_ceiling": "screening",
            "provenance": {"task": "pxrd_reference_comparison", "axis_kind": "two_theta", "axis_unit": "degree", "phase_identity_inferred": False, "phase_purity_inferred": False}}


# A compact spelling useful to task routers.
compare_pxrd = compare_pxrd_references
