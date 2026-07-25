"""Evidence-first FTIR/NIR comparison task pack."""
from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.decomposition import PCA

from chemometrics_mcp.core.comparison import compare_continuous_spectra, compare_peak_lists, detect_peaks
from chemometrics_mcp.core.mixtures import estimate_mixtures
from chemometrics_mcp.core.units import percent_transmittance_to_absorbance, validate_axis, validate_signal

TASK_PACK = "ftir_nir"
VERSION = "1"


def _issue(code: str, message: str, level: str = "blocker", **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "level": level, "stage": "ftir_nir", "details": details}


def _prepared(measurement: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    ident = str(measurement.get("measurement_id", "unknown"))
    modality = str(measurement.get("modality", "")).strip().lower()
    if modality not in {"ftir", "nir"}:
        issues.append(
            _issue(
                "ftir_nir_modality_required",
                "This task pack requires an explicit FTIR or NIR modality.",
                measurement_id=ident,
            )
        )
    axis, signal = measurement.get("axis", ()), measurement.get("signal", ())
    axis_check = validate_axis(axis, measurement.get("axis_kind"), measurement.get("axis_unit"))
    signal_check = validate_signal(signal, measurement.get("signal_kind"), measurement.get("signal_unit"), quantitative=True)
    for item in axis_check.issues + signal_check.issues:
        issues.append(_issue(item.code, item.message, item.severity, measurement_id=ident))
    if issues or not axis_check.is_valid or not signal_check.is_valid or len(axis) != len(signal):
        if len(axis) != len(signal): issues.append(_issue("axis_signal_length_mismatch", "Axis and signal lengths must match.", measurement_id=ident))
        return None, issues
    converted = tuple(float(value) for value in signal)
    provenance: list[dict[str, Any]] = []
    kind, unit = signal_check.normalized_kind, signal_check.normalized_unit
    if kind == "percent_transmittance":
        if unit != "percent":
            return None, issues + [_issue("invalid_transmittance_unit", "Percent transmittance requires an explicit percent unit.", measurement_id=ident)]
        conversion = percent_transmittance_to_absorbance(converted)
        if not conversion.is_valid:
            return None, issues + [_issue(item.code, item.message, item.severity, measurement_id=ident) for item in conversion.issues]
        converted, kind, unit = conversion.values, "absorbance", "absorbance"
        provenance.append({"action": conversion.record.action, "details": conversion.record.details})
    return {"measurement": measurement, "measurement_id": ident, "axis": tuple(float(v) for v in axis), "signal": converted, "signal_kind": kind, "signal_unit": unit, "conversion_provenance": provenance}, issues


def run_ftir_nir_task(measurements: Sequence[Mapping[str, Any]], task_type: str = "spectral_comparison", *, peak_tolerance: float = 5.0) -> dict[str, Any]:
    """Run only descriptive/screening FTIR/NIR evidence operations.

    Input mappings and their raw arrays are read only; converted arrays exist only
    in the returned evidence/provenance.
    """
    issues: list[dict[str, Any]] = []
    prepared = []
    for measurement in measurements:
        item, item_issues = _prepared(measurement)
        issues.extend(item_issues)
        if item is not None: prepared.append(item)
    evidence: list[dict[str, Any]] = []
    for left, right in combinations(prepared, 2):
        if (
            left["signal_kind"],
            left["signal_unit"],
        ) != (
            right["signal_kind"],
            right["signal_unit"],
        ):
            issues.append(
                _issue(
                    "incompatible_signal_semantics",
                    "Pairwise comparison requires matching explicit signal kinds and units.",
                    left=left["measurement_id"],
                    right=right["measurement_id"],
                )
            )
            continue
        comparison = compare_continuous_spectra(left["axis"], left["signal"], right["axis"], right["signal"])
        left_peaks, right_peaks = detect_peaks(left["axis"], left["signal"]), detect_peaks(right["axis"], right["signal"])
        peak_match = compare_peak_lists(left_peaks, right_peaks, tolerance=peak_tolerance)
        evidence.append({"kind": "pairwise_comparison", "left_measurement_id": left["measurement_id"], "right_measurement_id": right["measurement_id"], "comparison": comparison.to_dict(), "peak_matches": peak_match.to_dict()})
        issues.extend(_issue("comparison_warning", warning, "advisory", left=left["measurement_id"], right=right["measurement_id"]) for warning in comparison.warnings)
    preparation_ids = [
        str(item["measurement"]["preparation_id"])
        for item in prepared
        if item["measurement"].get("preparation_id")
    ]
    if prepared and len(preparation_ids) != len(prepared):
        issues.append(
            _issue(
                "preparation_hierarchy_incomplete",
                "Independent preparation count is unavailable until preparation_id is declared for every spectrum.",
                "advisory",
            )
        )
    result: dict[str, Any] = {"task_pack": TASK_PACK, "version": VERSION, "task_type": task_type, "claim_ceiling": "descriptive" if task_type not in {"mixture", "mixture_quantification"} else "screening", "evidence_rows": evidence, "issues": issues, "measurement_provenance": [{"measurement_id": item["measurement_id"], "conversions": item["conversion_provenance"]} for item in prepared], "counts": {"scan_count": len(prepared), "preparation_count": len(set(preparation_ids)) if len(preparation_ids) == len(prepared) else None}}
    if task_type in {"pca", "unsupervised_exploration"}:
        if len(prepared) < 3:
            result["issues"].append(_issue("pca_requires_three_spectra", "Descriptive PCA requires at least three aligned spectra."))
        else:
            axes = [item["axis"] for item in prepared]
            if not all(axis == axes[0] for axis in axes[1:]):
                result["issues"].append(_issue("pca_alignment_required", "PCA requires explicitly aligned spectra.", "advisory"))
            else:
                n_components = min(2, len(prepared), len(axes[0]))
                pca_model = PCA(n_components=n_components, random_state=0)
                signals_matrix = np.asarray([item["signal"] for item in prepared])
                scores = pca_model.fit_transform(signals_matrix)
                result["pca"] = {
                    "scores": scores.tolist(),
                    "label": "descriptive scan-level PCA; no chemical identity claim",
                    "scan_count": len(prepared),
                    "preparation_count": result["counts"]["preparation_count"],
                    "explained_variance_ratio": pca_model.explained_variance_ratio_.tolist(),
                    "components": pca_model.components_.tolist(),
                    "axis": list(axes[0]),
                    "signals": signals_matrix.tolist(),
                    "labels": [str(item["measurement"].get("role", "sample")) for item in prepared],
                }
    if task_type in {"mixture", "mixture_quantification"}:
        references = [item for item in prepared if str(item["measurement"].get("role", "")).lower() in {"reference", "calibration"} and item["measurement"].get("reference_name")]
        mixtures = [item for item in prepared if item not in references]
        if not references or not mixtures:
            result["issues"].append(_issue("explicit_references_required", "Mixture screening requires explicit reference roles and reference_name values."))
        elif not all(item["axis"] == references[0]["axis"] for item in references + mixtures):
            result["issues"].append(_issue("mixture_alignment_required", "Mixture screening requires explicitly aligned axes."))
        elif not all(
            (item["signal_kind"], item["signal_unit"])
            == (references[0]["signal_kind"], references[0]["signal_unit"])
            for item in references + mixtures
        ):
            result["issues"].append(
                _issue(
                    "mixture_signal_semantics_mismatch",
                    "Mixture screening requires matching signal kinds and units.",
                )
            )
        elif any(
            reference["measurement"].get("physical_state")
            and mixture["measurement"].get("physical_state")
            and reference["measurement"].get("physical_state")
            != mixture["measurement"].get("physical_state")
            for reference in references
            for mixture in mixtures
        ):
            result["issues"].append(
                _issue(
                    "reference_physical_state_mismatch",
                    "Reference and mixture physical states differ; reference coefficients are not comparable.",
                )
            )
        else:
            mixture_screening = estimate_mixtures(
                [item["signal"] for item in references],
                [item["signal"] for item in mixtures],
                sum_to_one=True,
                reference_names=[
                    str(item["measurement"]["reference_name"])
                    for item in references
                ],
                reference_roles=[
                    str(item["measurement"].get("role"))
                    for item in references
                ],
            )
            mixture_screening["reference_measurement_ids"] = [
                item["measurement_id"] for item in references
            ]
            mixture_screening["mixture_measurement_ids"] = [
                item["measurement_id"] for item in mixtures
            ]
            result["mixture_screening"] = mixture_screening
    return result
