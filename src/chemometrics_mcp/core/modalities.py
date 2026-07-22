"""Modality profile registry for spectral data.

A single declarative record per spectroscopic modality centralises everything
that used to be scattered across the codebase as hardcoded ``if modality ==
"FTIR"`` branches: axis units, value-domain validity + out-of-range policy,
which preprocessing methods are appropriate, and how the modality is inferred
from an axis.

Adding a new *gridded* modality (UV-Vis, NMR, ...) is a single ``ModalityProfile``
entry in ``_REGISTRY`` — no edits to ``datasets``/``validation``/``planning``/
``reporting`` required. Sparse (peak-list) modalities such as MS need a
``representation="peaklist"`` resampler that does not yet exist; the field is
present so the distinction is explicit rather than silently assumed away.

Consumers
---------
- :func:`infer_modality` — replaces range-based modality guessing in ``datasets``.
- :func:`enforce_value_domain` — replaces the FTIR-only ``%T`` range check;
  clips / flags values per the profile's policy.
- :func:`preprocessing_candidates` — feeds the planner's method proposals.
- :func:`allowed_preprocessing` / :func:`modality_specific_preprocessing` —
  feed validation's modality/preprocessing consistency check and reporting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chemometrics_contracts import ValidationWarning

# Preprocessing appropriate for essentially any spectral modality.
_UNIVERSAL_PREPROCESSING: tuple[str, ...] = (
    "snv",
    "msc",
    "sg_1st_deriv",
    "sg_2nd_deriv",
)


@dataclass(frozen=True)
class ModalityProfile:
    """Declarative description of one spectroscopic modality.

    Attributes
    ----------
    name:
        Canonical modality name (upper-case), e.g. ``"FTIR"``.
    aliases:
        Alternative names that resolve to this profile (case-insensitive).
    axis_unit:
        Physical unit of the spectral axis (``"cm-1"``, ``"nm"``, ``"ppm"``,
        ``"m/z"``).
    axis_direction:
        ``"ascending"`` or ``"descending"`` — the conventional plotting order.
    axis_infer_range:
        ``(min, max)`` axis span used to auto-detect this modality from data,
        or ``None`` when the axis range does not uniquely identify the modality
        (e.g. Raman shift overlaps the FTIR wavenumber range, so Raman must be
        declared explicitly).
    representation:
        ``"gridded"`` for continuous spectra sampled on a shared axis (support
        linear interpolation to a common grid) or ``"peaklist"`` for sparse
        peak data such as MS (require binning/alignment, not interpolation).
    value_min, value_max:
        Valid intensity range; ``None`` means unbounded on that side.
    out_of_range_policy:
        What :func:`enforce_value_domain` does with values outside
        ``[value_min, value_max]``:

        - ``"clip"``      clip into range and warn (bounded modalities, %T).
        - ``"reject"``    leave values untouched but warn (intensity < 0, etc.).
        - ``"allow"``     no domain constraint (e.g. phased NMR real part).
    preprocessing_candidates:
        Preprocessing methods the planner should propose for this modality
        (excluding ``"raw"``, which is always available).
    """

    name: str
    aliases: tuple[str, ...]
    axis_unit: str
    axis_direction: str
    axis_infer_range: tuple[float, float] | None
    representation: str
    value_min: float | None
    value_max: float | None
    out_of_range_policy: str
    preprocessing_candidates: tuple[str, ...]

    @property
    def allowed_preprocessing(self) -> frozenset[str]:
        """Methods valid for this modality (``raw`` + its candidates)."""
        return frozenset({"raw", *self.preprocessing_candidates})

    @property
    def specific_preprocessing(self) -> frozenset[str]:
        """Candidates beyond the universal baseline (modality-characteristic)."""
        return frozenset(self.preprocessing_candidates) - frozenset(
            _UNIVERSAL_PREPROCESSING
        )


# --------------------------------------------------------------------------- #
# Registry. One entry per supported modality. Add gridded modalities freely.  #
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, ModalityProfile] = {
    "FTIR": ModalityProfile(
        name="FTIR",
        aliases=("ftir", "ir", "mid-ir", "mir"),
        axis_unit="cm-1",
        axis_direction="descending",
        axis_infer_range=(400.0, 4000.0),
        representation="gridded",
        value_min=0.0,
        value_max=100.0,  # %T
        out_of_range_policy="clip",
        preprocessing_candidates=(
            *_UNIVERSAL_PREPROCESSING,
            "baseline_correction",
            "area_normalization",
        ),
    ),
    "NIR": ModalityProfile(
        name="NIR",
        aliases=("nir", "near-ir"),
        axis_unit="nm",
        axis_direction="ascending",
        axis_infer_range=(700.0, 2500.0),
        representation="gridded",
        value_min=None,  # absorbance / reflectance — not %T-bounded
        value_max=None,
        out_of_range_policy="allow",
        preprocessing_candidates=_UNIVERSAL_PREPROCESSING,
    ),
    "RAMAN": ModalityProfile(
        name="RAMAN",
        aliases=("raman",),
        axis_unit="cm-1",
        # Raman shift overlaps the FTIR wavenumber range, so it cannot be
        # inferred from the axis span alone — require an explicit modality.
        axis_direction="ascending",
        axis_infer_range=None,
        representation="gridded",
        value_min=0.0,  # intensity is non-negative
        value_max=None,  # unbounded (arbitrary counts)
        out_of_range_policy="reject",
        preprocessing_candidates=(
            *_UNIVERSAL_PREPROCESSING,
            "baseline_correction",  # fluorescence baseline is a Raman staple
            "area_normalization",
        ),
    ),
}

# Alias -> canonical name lookup (canonical names included).
_ALIAS_INDEX: dict[str, str] = {}
for _profile in _REGISTRY.values():
    _ALIAS_INDEX[_profile.name.lower()] = _profile.name
    for _alias in _profile.aliases:
        _ALIAS_INDEX[_alias.lower()] = _profile.name


def get_profile(modality: str | None) -> ModalityProfile | None:
    """Resolve a modality name or alias to its profile, or ``None``."""
    if modality is None:
        return None
    canonical = _ALIAS_INDEX.get(str(modality).strip().lower())
    return _REGISTRY.get(canonical) if canonical else None


def infer_modality(axis_values: np.ndarray) -> str | None:
    """Infer the modality name from an axis span, or ``None`` if ambiguous.

    Only modalities with an ``axis_infer_range`` participate. Modalities whose
    axis range is not distinctive (Raman, MS, NMR) are never inferred and must
    be supplied explicitly by the caller.
    """
    axis_min = float(np.min(axis_values))
    axis_max = float(np.max(axis_values))
    best_name: str | None = None
    best_width = float("inf")
    for profile in _REGISTRY.values():
        rng = profile.axis_infer_range
        if rng is not None and rng[0] <= axis_min and axis_max <= rng[1]:
            # Prefer the narrowest enclosing range: NIR (700-2500 nm) sits
            # inside the FTIR wavenumber span (400-4000 cm-1), so the tighter
            # profile is the more specific match regardless of registry order.
            width = rng[1] - rng[0]
            if width < best_width:
                best_name, best_width = profile.name, width
    return best_name


def enforce_value_domain(
    X: np.ndarray, modality: str | None
) -> tuple[np.ndarray, ValidationWarning | None]:
    """Apply a modality's value-domain policy to a spectral matrix.

    Returns the (possibly clipped) matrix and a data-quality warning describing
    any action taken. Unknown modalities and ``policy="allow"`` are pass-through.
    """
    profile = get_profile(modality)
    if profile is None or profile.out_of_range_policy == "allow":
        return X, None
    if profile.value_min is None and profile.value_max is None:
        return X, None

    lo, hi = profile.value_min, profile.value_max
    below = X < lo if lo is not None else np.zeros_like(X, dtype=bool)
    above = X > hi if hi is not None else np.zeros_like(X, dtype=bool)
    n_out = int(np.sum(below | above))
    if n_out == 0:
        return X, None

    n_total = X.size
    rng_txt = f"[{'-inf' if lo is None else lo}, {'inf' if hi is None else hi}]"
    stats = f"(min={float(X.min()):.2f}, max={float(X.max()):.2f})"

    if profile.out_of_range_policy == "clip":
        X_new = np.clip(X, lo, hi)
        return X_new, ValidationWarning(
            code="value_domain_clipped",
            message=(
                f"{n_out}/{n_total} {profile.name} values fell outside the valid "
                f"{profile.axis_unit} intensity range {rng_txt} {stats} and were "
                "clipped. Un-ratioed single-beam data or instrument-edge "
                "artefacts suspected."
            ),
            category="data_quality",
            severity="warning",
            affected_stage="inspection",
        )

    # policy == "reject": flag but do not mutate.
    return X, ValidationWarning(
        code="value_domain_out_of_range",
        message=(
            f"{n_out}/{n_total} {profile.name} values fall outside the valid "
            f"range {rng_txt} {stats}. Values left unmodified; review data quality."
        ),
        category="data_quality",
        severity="warning",
        affected_stage="inspection",
    )


def universal_preprocessing() -> tuple[str, ...]:
    """Preprocessing methods appropriate for any spectral modality."""
    return _UNIVERSAL_PREPROCESSING


def preprocessing_candidates(modality: str | None) -> tuple[str, ...]:
    """Preprocessing methods the planner should propose for *modality*.

    Falls back to the universal baseline for unknown modalities.
    """
    profile = get_profile(modality)
    if profile is None:
        return _UNIVERSAL_PREPROCESSING
    return profile.preprocessing_candidates


def allowed_preprocessing(modality: str | None) -> frozenset[str] | None:
    """Methods valid for *modality*, or ``None`` if the modality is unknown."""
    profile = get_profile(modality)
    return profile.allowed_preprocessing if profile else None


def modality_specific_preprocessing(modality: str | None) -> frozenset[str]:
    """Methods characteristic of *modality* (beyond the universal baseline)."""
    profile = get_profile(modality)
    return profile.specific_preprocessing if profile else frozenset()
