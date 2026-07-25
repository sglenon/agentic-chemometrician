"""SVG figure rendering for the standalone preprocess_spectra tool.

Produces before/after overlay and derivative overlay figures from the
dict returned by :func:`chemometrics_mcp.tools.preprocess_spectra.preprocess_spectra`.
No project pipeline, ProjectStore, or disk I/O required — returns inline SVG strings.
"""
from __future__ import annotations

from typing import Any

from chemometrics_mcp.core.dashboard import _MAX_DISPLAY_SERIES, _line_svg

# Mapping from preprocessing step name → human-readable derivative order label.
_DERIV_STEP_LABEL: dict[str, str] = {
    "sg_1st_deriv": "1st-Derivative",
    "sg_2nd_deriv": "2nd-Derivative",
    "sg_3rd_deriv": "3rd-Derivative",
}

_DERIV_STEPS: frozenset[str] = frozenset(_DERIV_STEP_LABEL)


def render_preprocessing_figures(result: dict[str, Any]) -> dict[str, str]:
    """Render before/after overlay and (optional) derivative overlay SVGs.

    Parameters
    ----------
    result:
        Return value from :func:`preprocess_spectra`.  Expected keys:
        ``"before"``, ``"after"``, ``"steps"``.

    Returns
    -------
    dict mapping figure filename → SVG string.  Keys present:

    - ``"before-overlay.svg"`` — raw spectra, one line per sample.
    - ``"after-overlay.svg"`` — preprocessed spectra, one line per sample.
    - ``"derivative-overlay.svg"`` — only when a derivative step was applied;
      same data as ``"after-overlay.svg"`` but titled with the derivative order.

    Returns an empty dict when there are no samples or all series fail
    validation (non-finite values, empty arrays).
    """
    figures: dict[str, str] = {}

    before = result.get("before", {})
    after = result.get("after", {})
    steps: list[str] = list(result.get("steps", []))

    before_axis: list[float] = before.get("axis", [])
    after_axis: list[float] = after.get("axis", [])
    before_signals: dict[str, list[float]] = before.get("signals", {})
    after_signals: dict[str, list[float]] = after.get("signals", {})

    if not before_signals:
        return figures

    # Mirror the existing dashboard limit for multi-series overlays.
    sample_names = list(before_signals.keys())[:_MAX_DISPLAY_SERIES]

    # ------------------------------------------------------------------ before
    before_series = [
        {"label": name, "x": before_axis, "y": before_signals[name]}
        for name in sample_names
        if name in before_signals
    ]
    before_svg = _line_svg(
        before_series,
        title="Raw Spectra (Before Preprocessing)",
        x_label="Wavenumber / wavelength",
        y_label="Signal",
    )
    if before_svg:
        figures["before-overlay.svg"] = before_svg

    # ------------------------------------------------------------------- after
    after_sample_names = list(after_signals.keys())[:_MAX_DISPLAY_SERIES]
    after_series = [
        {"label": name, "x": after_axis, "y": after_signals[name]}
        for name in after_sample_names
        if name in after_signals
    ]

    step_label = " + ".join(steps) if steps else "preprocessed"
    after_svg = _line_svg(
        after_series,
        title=f"Preprocessed Spectra (After: {step_label})",
        x_label="Wavenumber / wavelength",
        y_label="Signal (transformed)",
    )
    if after_svg:
        figures["after-overlay.svg"] = after_svg

    # -------------------------------------------------------- derivative overlay
    # Collect derivative steps in pipeline order; use first for the title.
    deriv_steps = [s for s in steps if s in _DERIV_STEPS]
    if deriv_steps:
        order_label = _DERIV_STEP_LABEL[deriv_steps[0]]
        deriv_series = [
            {"label": name, "x": after_axis, "y": after_signals[name]}
            for name in after_sample_names
            if name in after_signals
        ]
        deriv_svg = _line_svg(
            deriv_series,
            title=f"{order_label} Spectra",
            x_label="Wavenumber / wavelength",
            y_label=f"{order_label} (a.u.)",
        )
        if deriv_svg:
            figures["derivative-overlay.svg"] = deriv_svg

    return figures
