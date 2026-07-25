"""Standalone spectral preprocessing tool — no project pipeline required.

Loads spectra from a file or directory, applies an ordered list of preprocessing
steps, and returns before/after arrays suitable for before-after overlay plots.
Does NOT create project records, update manifest hash chains, or persist
measurement data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from chemometrics_mcp.core.ingestion import ParserRegistry
from chemometrics_mcp.core import preprocessing as _pre
from chemometrics_mcp.core.preprocessing_dashboard import render_preprocessing_figures


def preprocess_spectra(
    source_path: str,
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Load spectra and apply an ordered preprocessing pipeline standalone.

    Parameters
    ----------
    source_path:
        Absolute path to a single spectrum file (txt, csv, tsv, asc, …) or a
        directory.  All supported files in the directory are ingested.
    steps:
        Ordered list of preprocessing step configs.  Each entry must have a
        ``"name"`` key whose value is a recognised preprocessing method name
        (``"raw"``, ``"snv"``, ``"msc"``, ``"sg_1st_deriv"``, ``"sg_2nd_deriv"``,
        ``"sg_3rd_deriv"``, ``"baseline_correction"``, ``"area_normalization"``,
        ``"region_select"``).  Extra keys in each entry are forwarded as
        per-step parameters.  For ``"region_select"`` the optional ``"min"`` and
        ``"max"`` keys set axis-unit bounds.

    Returns
    -------
    dict with keys:
    - ``steps``: list of step names actually applied.
    - ``before``: ``{"axis": [...], "signals": {sample_name: [...]}}`` — raw spectra.
    - ``after``: ``{"axis": [...], "signals": {sample_name: [...]}}`` — transformed spectra.
    - ``region`` (only when ``"region_select"`` was used): bounds and counts.
    - ``step_details``: list of details dicts from each preprocessing step.
    - ``n_samples``: number of spectra processed.
    - ``n_features_before``: number of spectral features before preprocessing.
    - ``n_features_after``: number of spectral features after preprocessing.
    - ``issues``: list of ingestion issue descriptions (non-fatal).
    - ``figures``: dict mapping figure filename to inline SVG string.
      Always includes ``"before-overlay.svg"`` and ``"after-overlay.svg"``
      (one line per sample, up to 24 samples).  Includes
      ``"derivative-overlay.svg"`` only when a Savitzky-Golay derivative step
      (``"sg_1st_deriv"``, ``"sg_2nd_deriv"``, or ``"sg_3rd_deriv"``) was
      applied; the title reflects the derivative order.
    """
    path = Path(source_path)
    registry = ParserRegistry()

    if path.is_dir():
        measurements, issues = registry.ingest_directory(path)
    elif path.is_file():
        measurements, issues = registry.parse(path)
    else:
        raise FileNotFoundError(f"source_path does not exist or is not accessible: {source_path!r}")

    if not measurements:
        issue_msgs = [i.message for i in issues]
        raise ValueError(
            f"No spectra could be loaded from {source_path!r}. "
            f"Issues: {issue_msgs}"
        )

    # Align all measurements onto a common axis.  Use the first measurement's
    # axis as the reference; samples that share an axis length are stacked.
    ref_axis = np.asarray(measurements[0].axis, dtype=float)
    names: list[str] = []
    rows: list[np.ndarray] = []
    skipped: list[str] = []
    for m in measurements:
        ax = np.asarray(m.axis, dtype=float)
        sig = np.asarray(m.signal, dtype=float)
        if ax.shape != ref_axis.shape:
            skipped.append(f"{m.measurement_name}: axis length mismatch ({len(ax)} vs {len(ref_axis)})")
            continue
        names.append(m.measurement_name)
        rows.append(sig)

    if not rows:
        raise ValueError(
            f"No spectra share a common axis length.  Skipped: {skipped}"
        )

    X = np.vstack(rows)
    current_axis: np.ndarray | None = ref_axis
    n_features_before = X.shape[1]

    # Snapshot before any transformation.
    before_signals = {name: X[i].tolist() for i, name in enumerate(names)}
    before_axis = ref_axis.tolist() if current_axis is not None else list(range(n_features_before))

    # Apply each step in order.
    step_details_list: list[dict[str, Any]] = []
    region_info: dict[str, Any] | None = None

    for step_cfg in steps:
        step_cfg = dict(step_cfg)  # shallow copy to avoid mutating caller input
        method = step_cfg.pop("name")
        step_params = step_cfg  # remaining keys are per-step params

        X_out, details = _pre.apply(X, method, axis=current_axis, **step_params)

        # If region_select changed the axis, update current_axis for subsequent steps.
        if method == "region_select" and "axis_out" in details:
            current_axis = details.pop("axis_out")
            region_info = {
                "min": details.get("min"),
                "max": details.get("max"),
                "n_kept": details.get("n_kept"),
                "n_total": details.get("n_total"),
            }
        else:
            details.pop("axis_out", None)  # defensive: remove if somehow present

        step_details_list.append(details)
        X = X_out

    after_signals = {name: X[i].tolist() for i, name in enumerate(names)}
    after_axis = current_axis.tolist() if current_axis is not None else list(range(X.shape[1]))

    result: dict[str, Any] = {
        "steps": [s.get("name", s) if isinstance(s, dict) else s for s in steps],
        # Re-extract names from original steps list for the response.
        "before": {"axis": before_axis, "signals": before_signals},
        "after": {"axis": after_axis, "signals": after_signals},
        "step_details": step_details_list,
        "n_samples": len(names),
        "n_features_before": n_features_before,
        "n_features_after": int(X.shape[1]),
        "issues": [i.message for i in issues] + skipped,
    }

    if region_info is not None:
        result["region"] = region_info

    # Fix steps list — we already popped "name" from dicts above; re-derive from
    # step_details_list which each contain a "method" key.
    result["steps"] = [d.get("method", "unknown") for d in step_details_list]

    # Render before/after overlay figures (and derivative overlay when applicable).
    result["figures"] = render_preprocessing_figures(result)

    return result
