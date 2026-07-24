"""Conservative, descriptive-only UV--Vis task helpers."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from chemometrics_mcp.core.comparison import compare_continuous_spectra
from chemometrics_mcp.core.units import validate_signal


def _issue(code: str, message: str, level: str = "blocker") -> dict[str, str]:
    return {"code": code, "message": message, "level": level}


def _axis_unit(unit: Any) -> str | None:
    token = str(unit or "").strip().lower().replace(" ", "")
    return {"nm": "nm", "nanometer": "nm", "nanometers": "nm"}.get(token)


def _signal_unit(unit: Any) -> str | None:
    token = str(unit or "").strip().lower()
    return {
        "abs": "absorbance",
        "absorbance": "absorbance",
        "a.u.": "arbitrary_units",
        "au": "arbitrary_units",
        "arbitrary units": "arbitrary_units",
    }.get(token)


def _signal_kind(kind: Any) -> str | None:
    token = str(kind or "").strip().lower().replace(" ", "_")
    return {
        "abs": "absorbance",
        "absorbance": "absorbance",
        "intensity": "intensity",
    }.get(token)


def compare_role_tagged_spectra(measurements: Sequence[Mapping[str, Any]], *,
                                normalization: str | None = None) -> dict[str, Any]:
    """Compare product/complex spectra to explicitly role-tagged controls.

    Measurements are mappings with ``axis``, ``signal``, ``role``, ``axis_unit``
    and ``signal_unit``.  No filename or axis-range semantics are inferred.
    """
    issues: list[dict[str, str]] = []
    targets = [row for row in measurements if str(row.get("role", "")).lower() in {"product", "complex"}]
    controls = [row for row in measurements if str(row.get("role", "")).lower() in {"precursor", "reference"}]
    if not targets:
        issues.append(_issue("missing_product_or_complex", "An explicitly role-tagged product or complex spectrum is required."))
    if not controls:
        issues.append(_issue("missing_precursor_or_reference", "An explicitly role-tagged precursor or reference spectrum is required."))
    comparisons: list[dict[str, Any]] = []
    for target in targets:
        for control in controls:
            target_axis, control_axis = _axis_unit(target.get("axis_unit")), _axis_unit(control.get("axis_unit"))
            target_signal, control_signal = _signal_unit(target.get("signal_unit")), _signal_unit(control.get("signal_unit"))
            target_kind, control_kind = _signal_kind(target.get("signal_kind")), _signal_kind(control.get("signal_kind"))
            if str(target.get("axis_kind", "")).lower().replace(" ", "_") != "wavelength" or str(control.get("axis_kind", "")).lower().replace(" ", "_") != "wavelength":
                issues.append(_issue("uvvis_wavelength_axis_required", "UV-Vis comparison requires explicit axis_kind='wavelength'."))
                continue
            if not target_axis or not control_axis or target_axis != control_axis:
                issues.append(_issue("incompatible_or_unknown_axis_unit", "Comparison requires matching explicit supported wavelength units."))
                continue
            valid_pairs = {("absorbance", "absorbance"), ("intensity", "arbitrary_units")}
            if (
                (target_kind, target_signal) not in valid_pairs
                or (control_kind, control_signal) not in valid_pairs
                or (target_kind, target_signal) != (control_kind, control_signal)
            ):
                issues.append(_issue("incompatible_or_unknown_signal_semantics", "Comparison requires matching explicit supported signal kinds and units."))
                continue
            try:
                result = compare_continuous_spectra(target["axis"], target["signal"], control["axis"], control["signal"], normalization=normalization)
            except (KeyError, TypeError, ValueError) as exc:
                issues.append(_issue("invalid_spectrum", str(exc)))
                continue
            comparisons.append({
                "target_id": str(target.get("measurement_id", target.get("id", "target"))),
                "control_id": str(control.get("measurement_id", control.get("id", "control"))),
                "comparison": result.to_dict(),
            })
    return {
        "provenance": {"normalization": normalization, "comparison": "overlap_only", "roles": {"targets": ["product", "complex"], "controls": ["precursor", "reference"]}},
        "evidence": {"comparisons": comparisons}, "issues": issues,
        "claim_ceiling": "descriptive",
    }


def analyze_jobs_method(
    mole_fractions: Sequence[float], response: Sequence[float] | Sequence[Sequence[float]], *,
    axis: Sequence[float] | None = None, axis_unit: str | None = None,
    wavelength: float | None = None, selection_bounds: tuple[float, float] | None = None,
    replicate_groups: Sequence[str | int] | None = None, bootstrap_iterations: int = 0,
    design_metadata: Mapping[str, Any] | None = None,
    signal_kind: str | None = None,
    signal_unit: str | None = None,
) -> dict[str, Any]:
    """Summarize a Job's plot as descriptive maximum-location evidence only."""
    issues: list[dict[str, str]] = []
    fractions = np.asarray(mole_fractions, dtype=float)
    values = np.asarray(response, dtype=float)
    unique_fractions = np.unique(fractions) if fractions.ndim == 1 else np.asarray([])
    if fractions.ndim != 1 or len(unique_fractions) < 3 or not np.isfinite(fractions).all():
        issues.append(_issue("invalid_mole_fractions", "At least three finite mole fractions are required."))
    elif np.any((fractions < 0) | (fractions > 1)) or np.any(np.diff(fractions) < 0):
        issues.append(_issue("invalid_fraction_range_or_order", "Mole fractions must be nondecreasing and within [0, 1]."))
    elif unique_fractions[0] > 0 or unique_fractions[-1] < 1:
        issues.append(_issue("incomplete_fraction_coverage", "Job's method requires endpoint coverage from 0 to 1."))
    if values.ndim not in (1, 2) or values.shape[0] != len(fractions) or not np.isfinite(values).all():
        issues.append(_issue("invalid_response", "Response must be finite and have one row per mole fraction."))
    if bootstrap_iterations < 0:
        issues.append(_issue("invalid_bootstrap_iterations", "bootstrap_iterations cannot be negative."))
    if values.ndim == 2:
        signal_check = validate_signal(
            values.reshape(-1),
            signal_kind,
            signal_unit,
            quantitative=True,
        )
        if (
            not signal_check.is_valid
            or signal_check.normalized_kind != "absorbance"
        ):
            issues.append(
                _issue(
                    "jobs_absorbance_semantics_required",
                    "Spectral Job's analysis requires explicit valid absorbance signal semantics.",
                )
            )
        if axis is None or _axis_unit(axis_unit) is None:
            issues.append(_issue("unknown_or_unsupported_axis_unit", "Spectral Job's analysis requires an explicit supported wavelength unit."))
        else:
            x = np.asarray(axis, dtype=float)
            if x.ndim != 1 or len(x) != values.shape[1] or not np.isfinite(x).all():
                issues.append(_issue("invalid_axis", "Axis must be finite, one-dimensional, and match spectra."))
    elif axis is not None or wavelength is not None:
        issues.append(_issue("axis_not_applicable", "A scalar response must not include an axis or wavelength." , "advisory"))
    if bootstrap_iterations and replicate_groups is None:
        issues.append(_issue("bootstrap_requires_independent_groups", "Bootstrap is allowed only with explicit independent replicate groups."))
    if replicate_groups is not None:
        if len(replicate_groups) != len(fractions) or len(set(replicate_groups)) < 2:
            issues.append(_issue("invalid_replicate_hierarchy", "Replicate groups must align with rows and contain at least two independent groups."))
    if any(item["level"] == "blocker" for item in issues):
        return {"provenance": {"method": "jobs_method", "status": "blocked"}, "evidence": {}, "issues": issues, "claim_ceiling": "descriptive"}

    selected_wavelength: float | None = None
    selection: str = "provided_response"
    aggregated_values = np.asarray(
        [
            np.mean(values[fractions == fraction], axis=0)
            for fraction in unique_fractions
        ]
    )
    if values.ndim == 2:
        x = np.asarray(axis, dtype=float)
        if wavelength is not None:
            index = int(np.argmin(np.abs(x - wavelength)))
            selected_wavelength, selection = float(x[index]), "explicit_wavelength_nearest_axis_point"
        else:
            low, high = selection_bounds if selection_bounds is not None else (float(x.min()), float(x.max()))
            if low > high:
                return {"provenance": {"method": "jobs_method"}, "evidence": {}, "issues": issues + [_issue("invalid_selection_bounds", "selection_bounds must be ordered.")], "claim_ceiling": "descriptive"}
            candidates = np.where((x >= low) & (x <= high))[0]
            if not len(candidates):
                return {"provenance": {"method": "jobs_method"}, "evidence": {}, "issues": issues + [_issue("empty_selection_bounds", "No axis values lie in selection_bounds.")], "claim_ceiling": "descriptive"}
            # Selection is transparent and bounded, based on the aggregate signal.
            index = int(candidates[np.argmax(np.mean(aggregated_values[:, candidates], axis=0))])
            selected_wavelength, selection = float(x[index]), "bounded_mean_signal_maximum"
        observed = aggregated_values[:, index]
    else:
        observed = aggregated_values
    maximum = int(np.argmax(observed))
    design = dict(design_metadata or {})
    required_design = (
        "blank_corrected",
        "component_controls",
        "constant_total_concentration",
        "response_definition",
        "validated_response_method",
    )
    missing_design = [
        name for name in required_design if not design.get(name)
    ]
    ratio = (
        float(unique_fractions[maximum] / (1 - unique_fractions[maximum]))
        if not missing_design and unique_fractions[maximum] not in (0, 1)
        else None
    )
    evidence: dict[str, Any] = {
        "observed_maximum": {"mole_fraction": float(unique_fractions[maximum]), "response": float(observed[maximum]), "index": maximum,
                             "replicate_count": int(np.sum(fractions == unique_fractions[maximum]))},
        "job_plot_points": [
            {
                "mole_fraction": float(fraction),
                "response": float(response_value),
                "replicate_count": int(np.sum(fractions == fraction)),
            }
            for fraction, response_value in zip(unique_fractions, observed)
        ],
    }
    if ratio is not None:
        evidence["descriptive_stoichiometric_ratio"] = ratio
    else:
        issues.append(
            _issue(
                "jobs_design_incomplete",
                "A stoichiometric ratio was not emitted because blank correction, component controls, constant-total design, or a validated response definition is missing.",
                "advisory",
            )
        )
    if selected_wavelength is not None:
        evidence["selected_wavelength"] = selected_wavelength
    if bootstrap_iterations:
        # Resampling independent group labels is intentionally only a stability
        # summary, not a confidence interval for a chemical mechanism.
        groups = np.asarray(replicate_groups)
        unique = np.unique(groups)
        rng = np.random.default_rng(0)
        boot_maxima = []
        for _ in range(bootstrap_iterations):
            chosen = rng.choice(unique, size=len(unique), replace=True)
            indices = np.concatenate([np.where(groups == group)[0] for group in chosen])
            boot_maxima.append(float(fractions[indices][np.argmax(observed[indices])]))
        evidence["bootstrap_maximum_fraction_interval"] = [float(np.quantile(boot_maxima, .025)), float(np.quantile(boot_maxima, .975))]
    issues.append(_issue("descriptive_only", "This result does not establish binding constants, species identity, mechanism, or validated stoichiometry.", "advisory"))
    return {
        "provenance": {"method": "jobs_method", "selection": selection, "selection_bounds": list(selection_bounds) if selection_bounds else None,
                       "bootstrap_iterations": bootstrap_iterations, "independent_replicate_groups": replicate_groups is not None,
                       "design_metadata": design},
        "evidence": evidence, "issues": issues, "claim_ceiling": "descriptive",
    }


# Short aliases for callers that use the task-pack action names.
compare_uvvis_roles = compare_role_tagged_spectra
jobs_method_analysis = analyze_jobs_method
