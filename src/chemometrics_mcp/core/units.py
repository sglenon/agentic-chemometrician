"""Explicit unit and value checks for spectral data.

These helpers are intentionally conservative: unknown semantics are never
inferred and validation never clips, sorts, or otherwise changes measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class UnitIssue:
    code: str
    message: str
    severity: str = "blocker"

    @property
    def is_blocker(self) -> bool:
        return self.severity == "blocker"


@dataclass(frozen=True)
class UnitCheckResult:
    issues: tuple[UnitIssue, ...] = ()
    normalized_kind: str | None = None
    normalized_unit: str | None = None
    raw_values: tuple[float, ...] = ()
    quarantined: bool = False

    @property
    def is_valid(self) -> bool:
        return not any(issue.is_blocker for issue in self.issues)

    @property
    def blockers(self) -> tuple[UnitIssue, ...]:
        return tuple(issue for issue in self.issues if issue.is_blocker)


@dataclass(frozen=True)
class ConversionRecord:
    action: str
    source_kind: str | None = None
    source_unit: str | None = None
    target_kind: str | None = None
    target_unit: str | None = None
    details: str = ""


@dataclass(frozen=True)
class ConversionResult:
    values: tuple[float, ...]
    record: ConversionRecord | None = None
    issues: tuple[UnitIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not any(issue.is_blocker for issue in self.issues)


_AXIS_KINDS = {
    "wavenumber": "wavenumber", "wave number": "wavenumber", "frequency": "frequency",
    "wavelength": "wavelength", "time": "time", "raman_shift": "raman_shift",
    "raman shift": "raman_shift", "2theta": "two_theta", "two theta": "two_theta",
    "mass to charge": "mass_to_charge", "m/z": "mass_to_charge", "mz": "mass_to_charge",
}
_AXIS_UNITS = {
    "cm-1": "cm^-1", "cm^-1": "cm^-1", "1/cm": "cm^-1", "cm⁻¹": "cm^-1",
    "nm": "nm", "nanometer": "nm", "nanometers": "nm", "um": "um", "µm": "um",
    "micrometer": "um", "micrometers": "um", "s": "s", "sec": "s", "second": "s",
    "seconds": "s", "min": "min", "minute": "min", "minutes": "min", "degree": "degree",
    "degrees": "degree", "deg": "degree",
}
_SIGNAL_KINDS = {
    "absorbance": "absorbance", "abs": "absorbance", "a.u.": "intensity",
    "transmittance": "transmittance", "%t": "percent_transmittance",
    "percent transmittance": "percent_transmittance",
    "reflectance": "reflectance", "counts": "counts", "count": "counts",
    "relative abundance": "relative_abundance",
    "intensity": "intensity", "diffraction intensity": "diffraction_intensity",
}


def _token(value: str | None) -> str:
    return "" if value is None else " ".join(str(value).strip().lower().replace("_", " ").split())


def normalize_axis_unit(unit: str | None) -> str | None:
    """Normalize recognized axis-unit aliases; return ``None`` for unknowns."""
    return _AXIS_UNITS.get(_token(unit).replace(" ", "")) or _AXIS_UNITS.get(_token(unit))


def normalize_signal_kind(kind: str | None) -> str | None:
    """Normalize recognized signal semantic aliases; never infer a modality."""
    return _SIGNAL_KINDS.get(_token(kind))


def _values(values: Iterable[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def validate_axis(axis: Iterable[float], axis_kind: str | None, axis_unit: str | None) -> UnitCheckResult:
    """Validate an axis while leaving its order and duplicates intact."""
    raw = _values(axis)
    issues: list[UnitIssue] = []
    if not raw:
        issues.append(UnitIssue("axis_empty", "Axis contains no values."))
    array = np.asarray(raw, dtype=float)
    if raw and not np.isfinite(array).all():
        issues.append(UnitIssue("axis_nonfinite", "Axis contains non-finite values."))
    finite = array[np.isfinite(array)]
    if len(finite) > 1:
        delta = np.diff(finite)
        if np.any(delta == 0):
            issues.append(UnitIssue("axis_duplicates", "Axis contains duplicate coordinates."))
        if not (np.all(delta > 0) or np.all(delta < 0)):
            issues.append(UnitIssue("axis_not_monotonic", "Axis is not monotonic."))
    kind = _AXIS_KINDS.get(_token(axis_kind))
    unit = normalize_axis_unit(axis_unit)
    if kind is None:
        issues.append(UnitIssue("axis_kind_unknown", "Axis kind is unknown; modality was not guessed."))
    if unit is None:
        issues.append(UnitIssue("axis_unit_unknown", "Axis unit is unknown; it was not guessed."))
    return UnitCheckResult(tuple(issues), kind, unit, raw)


def _unit_class(unit: str | None) -> str | None:
    token = _token(unit)
    if token in {"%", "percent", "percentage", "%t"}:
        return "percent"
    if token in {"fraction", "unitless", "1", "ratio"}:
        return "fraction"
    if token in {"count", "counts"}:
        return "counts"
    if token in {"absorbance", "abs"}:
        return "absorbance"
    if token in {"a.u.", "au", "arb", "arbitrary unit", "arbitrary units"}:
        return "arbitrary_units"
    return None


def validate_signal(values: Iterable[float], signal_kind: str | None, signal_unit: str | None, quantitative: bool = True) -> UnitCheckResult:
    """Validate values without clipping or assuming unstated units."""
    raw = _values(values)
    issues: list[UnitIssue] = []
    array = np.asarray(raw, dtype=float)
    if not raw:
        issues.append(UnitIssue("signal_empty", "Signal contains no values."))
    if raw and not np.isfinite(array).all():
        issues.append(UnitIssue("signal_nonfinite", "Signal contains non-finite values."))
    kind = normalize_signal_kind(signal_kind)
    unit = _unit_class(signal_unit)
    if kind is None:
        severity = "blocker" if quantitative else "advisory"
        issues.append(UnitIssue("signal_kind_unknown", "Signal semantics are unknown and were not inferred.", severity))
    finite = array[np.isfinite(array)]
    # Resolve the otherwise ambiguous word "transmittance" from its explicit
    # ordinate unit; no value-range inference is used.
    if kind == "transmittance" and unit == "percent":
        kind = "percent_transmittance"
    compatible_units = {
        "percent_transmittance": {"percent"},
        "transmittance": {"fraction"},
        "reflectance": {"percent", "fraction"},
        "absorbance": {"absorbance", "fraction", "arbitrary_units"},
        "counts": {"counts"},
        "relative_abundance": {"percent", "fraction", "arbitrary_units"},
        "diffraction_intensity": {"counts", "arbitrary_units"},
        "intensity": {"counts", "arbitrary_units"},
    }
    if kind in compatible_units and unit not in compatible_units[kind]:
        severity = "blocker" if quantitative else "advisory"
        issues.append(
            UnitIssue(
                "signal_unit_incompatible",
                f"Signal unit is missing, unknown, or incompatible with {kind}.",
                severity,
            )
        )
    if kind == "percent_transmittance" and unit == "percent":
        if np.any((finite <= 0) | (finite > 100)):
            issues.append(UnitIssue("percent_transmittance_out_of_range", "Percent transmittance must satisfy 0 < value <= 100."))
    if kind == "transmittance" and unit == "fraction" and np.any(
        (finite <= 0) | (finite > 1)
    ):
        issues.append(
            UnitIssue(
                "transmittance_out_of_range",
                "Fraction transmittance must satisfy 0 < value <= 1.",
            )
        )
    if kind == "reflectance" and unit == "percent" and np.any((finite < 0) | (finite > 100)):
        issues.append(UnitIssue("reflectance_out_of_range", "Percent reflectance must be between 0 and 100."))
    if kind == "reflectance" and unit == "fraction" and np.any((finite < 0) | (finite > 1)):
        issues.append(UnitIssue("reflectance_out_of_range", "Fraction reflectance must be between 0 and 1."))
    if kind in {"counts", "relative_abundance", "diffraction_intensity"} and np.any(finite < 0):
        issues.append(UnitIssue("signal_negative", f"{kind} cannot contain negative values."))
    return UnitCheckResult(tuple(issues), kind, unit, raw)


def percent_transmittance_to_absorbance(values: Iterable[float]) -> ConversionResult:
    """Convert valid percent transmittance using ``-log10(%T / 100)``."""
    raw = _values(values)
    check = validate_signal(raw, "percent_transmittance", "%", quantitative=True)
    if not check.is_valid:
        return ConversionResult(raw, None, check.issues)
    converted = tuple(float(value) for value in -np.log10(np.asarray(raw) / 100.0))
    return ConversionResult(converted, ConversionRecord(
        action="percent_transmittance_to_absorbance", source_kind="percent_transmittance",
        source_unit="percent", target_kind="absorbance", target_unit="absorbance",
        details="Applied -log10(%T / 100).",
    ))


def quarantine_for_display(values: Iterable[float], issues: Iterable[UnitIssue] = ()) -> UnitCheckResult:
    """Mark raw data display-only without changing or attempting to repair it."""
    raw = _values(values)
    advisory = tuple(issues) + (UnitIssue("display_only", "Raw values are quarantined for display only.", "advisory"),)
    return UnitCheckResult(advisory, raw_values=raw, quarantined=True)


def sort_and_deduplicate_axis(axis: Iterable[float], values: Iterable[float] | None = None) -> tuple[tuple[float, ...], tuple[float, ...] | None, ConversionRecord]:
    """Explicitly sort an axis and retain the first value for each coordinate."""
    raw_axis = _values(axis)
    raw_values = _values(values) if values is not None else None
    if raw_values is not None and len(raw_axis) != len(raw_values):
        raise ValueError("axis and values must have the same length")
    order = sorted(range(len(raw_axis)), key=raw_axis.__getitem__)
    kept: list[int] = []
    seen: set[float] = set()
    for index in order:
        if raw_axis[index] not in seen:
            seen.add(raw_axis[index])
            kept.append(index)
    return (
        tuple(raw_axis[index] for index in kept),
        tuple(raw_values[index] for index in kept) if raw_values is not None else None,
        ConversionRecord("sort_and_deduplicate_axis", details="Sorted ascending and retained first value per duplicate coordinate."),
    )
