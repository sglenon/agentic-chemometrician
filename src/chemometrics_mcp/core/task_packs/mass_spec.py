"""Conservative, evidence-preserving centroided mass-spectrometry matching."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def _issue(code: str, message: str, level: str = "blocker", **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "level": level, "stage": "mass_spec", "details": details}


def _peaks(item: Mapping[str, Any]) -> tuple[list[tuple[float, float | None]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    axis = item.get("mz", item.get("axis", ()))
    intensity = item.get("intensity", item.get("signal"))
    if str(item.get("axis_kind", "")).lower().replace("/", "_") not in {"m_z", "mass_to_charge", "mz"}:
        issues.append(_issue("mz_axis_required", "An explicit m/z axis_kind is required."))
    semantic = str(item.get("signal_kind", "")).lower().replace(" ", "_")
    if semantic not in {"intensity", "counts", "relative_abundance"}:
        issues.append(_issue("incompatible_intensity_semantics", "Signal semantics must be intensity, counts, or relative_abundance."))
    unit = str(item.get("signal_unit", "")).strip().lower()
    compatible_units = {
        "intensity": {"a.u.", "au", "arbitrary unit", "arbitrary units"},
        "counts": {"count", "counts"},
        "relative_abundance": {"%", "percent", "fraction"},
    }
    if semantic in compatible_units and unit not in compatible_units[semantic]:
        issues.append(
            _issue(
                "incompatible_intensity_unit",
                f"Signal unit is missing or incompatible with {semantic}.",
            )
        )
    if intensity is None: intensity = [None] * len(axis)
    if len(axis) != len(intensity) or not axis:
        issues.append(_issue("invalid_peak_list", "m/z and intensity arrays must be non-empty and equal length."))
        return [], issues
    try:
        pairs = [(float(mass), None if value is None else float(value)) for mass, value in zip(axis, intensity)]
    except (TypeError, ValueError):
        return [], issues + [_issue("invalid_peak_list", "Peak values must be numeric.")]
    if any(not np.isfinite(mass) or mass <= 0 or (value is not None and (not np.isfinite(value) or value < 0)) for mass, value in pairs):
        issues.append(_issue("invalid_peak_values", "m/z must be positive and intensities/counts must be finite and nonnegative."))
    return sorted(pairs), issues


def _match(experimental: list[tuple[float, float | None]], reference: list[tuple[float, float | None]], tolerance: float, unit: str) -> dict[str, Any]:
    n, m = len(experimental), len(reference)
    valid = np.zeros((n, m), dtype=bool)
    da = np.zeros((n, m), dtype=float)
    ppm = np.zeros((n, m), dtype=float)
    for i, (observed, _) in enumerate(experimental):
        for j, (expected, _) in enumerate(reference):
            da[i, j] = observed - expected
            ppm[i, j] = da[i, j] / expected * 1_000_000 if expected else float("inf")
            valid[i, j] = abs(da[i, j]) <= tolerance if unit == "da" else abs(ppm[i, j]) <= tolerance
    cost = np.full((n + m, n + m), max(tolerance, 1.0) * 4, dtype=float)
    selected_error = np.abs(da) if unit == "da" else np.abs(ppm)
    if n and m:
        cost[:n, :m] = np.where(
            valid, selected_error, max(tolerance, 1.0) * 4
        )
    cost[:n, m:] = max(tolerance, 1.0)
    cost[n:, :m] = max(tolerance, 1.0)
    cost[n:, m:] = 0.0
    rows, cols = linear_sum_assignment(cost)
    matches = []
    used_e, used_r = set(), set()
    for i, j in zip(rows, cols):
        if i < n and j < m and valid[i, j]:
            used_e.add(i); used_r.add(j)
            matches.append({"experimental_mz": experimental[i][0], "reference_mz": reference[j][0], "experimental_intensity": experimental[i][1], "reference_intensity": reference[j][1], "signed_error_da": float(da[i, j]), "absolute_error_da": float(abs(da[i, j])), "signed_error_ppm": float(ppm[i, j]), "absolute_error_ppm": float(abs(ppm[i, j]))})
    cosine = None
    if matches and all(row["experimental_intensity"] is not None and row["reference_intensity"] is not None for row in matches):
        left = np.asarray([row["experimental_intensity"] for row in matches], dtype=float); right = np.asarray([row["reference_intensity"] for row in matches], dtype=float)
        if np.linalg.norm(left) and np.linalg.norm(right): cosine = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
    return {"matches": matches, "unmatched_experimental": [{"mz": mass, "intensity": value} for i, (mass, value) in enumerate(experimental) if i not in used_e], "unmatched_reference": [{"mz": mass, "intensity": value} for j, (mass, value) in enumerate(reference) if j not in used_r], "match_fraction": len(matches) / max(n, m) if n or m else 0.0, "intensity_cosine": cosine}


def run_mass_spec_task(experimental: Mapping[str, Any], references: Sequence[Mapping[str, Any]], *, tolerance: float, tolerance_unit: str = "da", precursor_hypotheses: Sequence[Mapping[str, Any]] | None = None, calibration_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not np.isfinite(tolerance) or tolerance <= 0 or tolerance_unit not in {"da", "ppm"}:
        return {"task_pack": "mass_spec", "version": "1", "claim_ceiling": "screening", "evidence_rows": [], "issues": [_issue("invalid_tolerance", "Tolerance must be positive and expressed as 'da' or 'ppm'.")]}
    experiment, issues = _peaks(experimental)
    if not calibration_metadata or not calibration_metadata.get(
        "mass_error_basis"
    ):
        issues.append(
            _issue(
                "mass_calibration_unconfirmed",
                "Mass-error tolerance lacks explicit calibration/error-model provenance; ranking remains qualitative.",
                "advisory",
            )
        )
    if str(experimental.get("role", "")).lower() not in {
        "sample",
        "product",
        "unknown",
    }:
        issues.append(
            _issue(
                "experimental_role_required",
                "Experimental peaks must be explicitly role-tagged sample, product, or unknown.",
            )
        )
    if not references:
        issues.append(
            _issue(
                "mass_spec_references_required",
                "At least one explicit reference or side-product candidate is required.",
            )
        )
    if any(issue["level"] == "blocker" for issue in issues):
        return {
            "task_pack": "mass_spec",
            "version": "1",
            "claim_ceiling": "screening",
            "evidence_rows": [],
            "issues": issues,
            "claim_limitations": [
                "No compound identity, structure, purity, fragmentation-mechanism, or formula claim is supported."
            ],
        }
    rows = []
    for candidate in references:
        role = str(candidate.get("role", "")).lower()
        candidate_peaks, candidate_issues = _peaks(candidate)
        issues.extend(candidate_issues)
        if role not in {"reference", "side_product_candidate"}:
            role_issue = _issue("reference_role_required", "Each candidate must be explicitly role-tagged reference or side_product_candidate.", candidate_id=candidate.get("candidate_id"))
            issues.append(role_issue)
            rows.append(
                {
                    "candidate_id": str(
                        candidate.get(
                            "candidate_id",
                            candidate.get("measurement_id", "unknown"),
                        )
                    ),
                    "role": role,
                    "matches": [],
                    "unmatched_experimental": [
                        {"mz": mass, "intensity": value}
                        for mass, value in experiment
                    ],
                    "unmatched_reference": [
                        {"mz": mass, "intensity": value}
                        for mass, value in candidate_peaks
                    ],
                    "match_fraction": 0.0,
                    "intensity_cosine": None,
                    "issues": [role_issue],
                }
            )
            continue
        if candidate_issues:
            rows.append({"candidate_id": str(candidate.get("candidate_id", candidate.get("measurement_id", "unknown"))), "role": role, "matches": [], "unmatched_experimental": [{"mz": mass, "intensity": value} for mass, value in experiment], "unmatched_reference": [], "match_fraction": 0.0, "intensity_cosine": None, "issues": candidate_issues})
            continue
        if experiment and candidate_peaks and not candidate_issues:
            match = _match(experiment, candidate_peaks, tolerance, tolerance_unit)
            rows.append({"candidate_id": str(candidate.get("candidate_id", candidate.get("measurement_id", "unknown"))), "role": role, **match})
    rows.sort(key=lambda row: (-row["match_fraction"], -(row["intensity_cosine"] if row["intensity_cosine"] is not None else -1), row["candidate_id"]))
    return {"task_pack": "mass_spec", "version": "1", "claim_ceiling": "screening", "tolerance": {"value": tolerance, "unit": tolerance_unit, "formula": "abs(observed-reference) <= Da tolerance" if tolerance_unit == "da" else "abs((observed-reference)/reference*1e6) <= ppm tolerance"}, "precursor_hypotheses": list(precursor_hypotheses or ()), "evidence_rows": rows, "issues": issues, "claim_limitations": ["No compound identity, structure, purity, fragmentation-mechanism, or formula claim is supported."]}
