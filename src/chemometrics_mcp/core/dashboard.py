"""Deterministic, offline scientist dashboards for persisted project runs.

The renderer consumes hash-bound run evidence and project measurements.  It
does not fit models, choose preprocessing, or alter scientific results.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.sax.saxutils import escape as xml_escape

import numpy as np

from chemometrics_mcp.core.project_service import ProjectService
from chemometrics_mcp.core.project_store import (
    ProjectStore,
    sha256_file,
    slugify_project_id,
)


_COLORS = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#7c3aed",
    "#d97706",
    "#0891b2",
    "#be185d",
    "#4d7c0f",
)
_MAX_DISPLAY_POINTS = 1_000
_MAX_DISPLAY_SERIES = 24
_MAX_TABLE_ROWS = 10_000


def _artifact(
    path: Path,
    store: ProjectStore,
    run: Mapping[str, Any],
    *,
    kind: str,
    media_type: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": path.relative_to(store.output_root).as_posix(),
        "sha256": sha256_file(path),
        "media_type": media_type,
        "project_id": run["project_id"],
        "run_id": run["run_id"],
        "manifest_hash": run["manifest_hash"],
        "plan_hash": run["plan_hash"],
    }


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    text = str(value)
    # Prevent formula execution when a scientist opens an exported CSV.
    if text.startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


def _write_csv(
    store: ProjectStore,
    relative: str,
    headers: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(headers), extrasaction="ignore")
    writer.writeheader()
    for index, row in enumerate(rows):
        if index >= _MAX_TABLE_ROWS:
            break
        writer.writerow({name: _cell(row.get(name)) for name in headers})
    return store.write_bytes(relative, output.getvalue().encode("utf-8"))


def _finite_vector(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        return None
    return array


def _downsample(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= _MAX_DISPLAY_POINTS:
        return x, y
    positions = np.linspace(0, len(x) - 1, _MAX_DISPLAY_POINTS, dtype=int)
    return x[positions], y[positions]


def _fmt(value: float) -> str:
    magnitude = abs(value)
    if magnitude and (magnitude >= 100_000 or magnitude < 0.001):
        return f"{value:.2e}"
    return f"{value:.4g}"


def _human_label(value: Any) -> str:
    text = str(value or "not available")
    acronyms = {
        "ftir": "FTIR",
        "mae": "MAE",
        "ms": "MS",
        "nir": "NIR",
        "pca": "PCA",
        "pls": "PLS",
        "pxrd": "PXRD",
        "r2": "R²",
        "rmse": "RMSE",
        "uv": "UV",
        "vis": "Vis",
    }
    words = text.replace("_", " ").replace("-", " ").strip().split()
    return " ".join(acronyms.get(word.lower(), word.title()) for word in words)


def _metric_label(value: Any) -> str:
    tokens = [token for token in str(value).split(".") if not token.isdigit()]
    if not tokens:
        return "Metric"
    label = _human_label(tokens[-1])
    for level in ("group", "scan"):
        if level in tokens[-4:]:
            return f"{_human_label(level)} {label}"
    return label


def _line_svg(
    series: Sequence[Mapping[str, Any]],
    *,
    title: str,
    x_label: str,
    y_label: str,
) -> str | None:
    prepared: list[tuple[str, np.ndarray, np.ndarray]] = []
    for item in series:
        x = _finite_vector(item.get("x"))
        y = _finite_vector(item.get("y"))
        if x is None or y is None or len(x) != len(y):
            continue
        x, y = _downsample(x, y)
        prepared.append((str(item.get("label", "spectrum")), x, y))
    if not prepared:
        return None
    width, height = 1_000, 560
    left, right, top, bottom = 92, 28, 55, 92
    plot_w, plot_h = width - left - right, height - top - bottom
    x_min = min(float(np.min(row[1])) for row in prepared)
    x_max = max(float(np.max(row[1])) for row in prepared)
    y_min = min(float(np.min(row[2])) for row in prepared)
    y_max = max(float(np.max(row[2])) for row in prepared)
    if x_min == x_max:
        x_max = x_min + 1
    if y_min == y_max:
        y_max = y_min + 1
    reverse_x = prepared[0][1][0] > prepared[0][1][-1]
    id_token = hashlib.sha256(title.encode("utf-8")).hexdigest()[:10]
    title_id, desc_id = f"title-{id_token}", f"desc-{id_token}"

    def sx(value: float) -> float:
        ratio = (value - x_min) / (x_max - x_min)
        if reverse_x:
            ratio = 1 - ratio
        return left + ratio * plot_w

    def sy(value: float) -> float:
        return top + (1 - (value - y_min) / (y_max - y_min)) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="{title_id} {desc_id}">',
        f'<title id="{title_id}">{xml_escape(title)}</title>',
        f'<desc id="{desc_id}">Overlay of {len(prepared)} spectra. '
        f"Horizontal axis: {xml_escape(x_label)}. Vertical axis: {xml_escape(y_label)}.</desc>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="30" font-family="system-ui,sans-serif" font-size="20" '
        f'font-weight="600" fill="#111827">{xml_escape(title)}</text>',
    ]
    for tick in range(6):
        ratio = tick / 5
        px = left + ratio * plot_w
        py = top + ratio * plot_h
        x_value = x_max - ratio * (x_max - x_min) if reverse_x else x_min + ratio * (x_max - x_min)
        y_value = y_max - ratio * (y_max - y_min)
        parts.extend(
            [
                f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{top + plot_h}" stroke="#e5e7eb"/>',
                f'<text x="{px:.2f}" y="{top + plot_h + 24}" text-anchor="middle" '
                f'font-family="system-ui,sans-serif" font-size="12" fill="#4b5563">{xml_escape(_fmt(x_value))}</text>',
                f'<line x1="{left}" y1="{py:.2f}" x2="{left + plot_w}" y2="{py:.2f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 12}" y="{py + 4:.2f}" text-anchor="end" '
                f'font-family="system-ui,sans-serif" font-size="12" fill="#4b5563">{xml_escape(_fmt(y_value))}</text>',
            ]
        )
    for index, (label, x, y) in enumerate(prepared):
        color = _COLORS[index % len(_COLORS)]
        points = " ".join(f"{sx(float(a)):.2f},{sy(float(b)):.2f}" for a, b in zip(x, y))
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.8" '
            f'stroke-linejoin="round" stroke-linecap="round"><title>{xml_escape(label)}</title></polyline>'
        )
    parts.extend(
        [
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
            f'<text x="{left + plot_w / 2:.2f}" y="{height - 25}" text-anchor="middle" '
            f'font-family="system-ui,sans-serif" font-size="14" fill="#111827">{xml_escape(x_label)}</text>',
            f'<text x="22" y="{top + plot_h / 2:.2f}" text-anchor="middle" '
            f'transform="rotate(-90 22 {top + plot_h / 2:.2f})" '
            f'font-family="system-ui,sans-serif" font-size="14" fill="#111827">{xml_escape(y_label)}</text>',
        ]
    )
    legend_y = height - 58
    cursor = left
    for index, (label, _, _) in enumerate(prepared[:8]):
        color = _COLORS[index % len(_COLORS)]
        shown = label if len(label) <= 24 else label[:21] + "…"
        parts.extend(
            [
                f'<line x1="{cursor}" y1="{legend_y}" x2="{cursor + 20}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{cursor + 26}" y="{legend_y + 4}" font-family="system-ui,sans-serif" '
                f'font-size="12" fill="#374151">{xml_escape(shown)}</text>',
            ]
        )
        cursor += min(210, 50 + len(shown) * 7)
        if cursor > width - 190:
            break
    parts.append("</svg>")
    return "".join(parts)


def _scatter_svg(
    points: Sequence[Mapping[str, Any]],
    *,
    title: str,
    x_label: str,
    y_label: str,
    identity_line: bool = False,
) -> str | None:
    prepared = []
    for point in points:
        try:
            x, y = float(point["x"]), float(point["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            prepared.append((x, y, str(point.get("label", "")), str(point.get("group", ""))))
    if not prepared:
        return None
    id_token = hashlib.sha256(title.encode("utf-8")).hexdigest()[:10]
    title_id, desc_id = f"title-{id_token}", f"desc-{id_token}"
    width, height = 820, 560
    left, right, top, bottom = 85, 32, 55, 82
    plot_w, plot_h = width - left - right, height - top - bottom
    x_values = [item[0] for item in prepared]
    y_values = [item[1] for item in prepared]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if identity_line:
        shared_min, shared_max = min(x_min, y_min), max(x_max, y_max)
        x_min = y_min = shared_min
        x_max = y_max = shared_max
    x_padding = (x_max - x_min) * 0.07 or 1
    y_padding = (y_max - y_min) * 0.07 or 1
    x_min, x_max = x_min - x_padding, x_max + x_padding
    y_min, y_max = y_min - y_padding, y_max + y_padding

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (1 - (value - y_min) / (y_max - y_min)) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="{title_id} {desc_id}">',
        f'<title id="{title_id}">{xml_escape(title)}</title>',
        f'<desc id="{desc_id}">Scatter plot with {len(prepared)} points. '
        f"Horizontal axis: {xml_escape(x_label)}. Vertical axis: {xml_escape(y_label)}.</desc>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="30" font-family="system-ui,sans-serif" font-size="20" '
        f'font-weight="600" fill="#111827">{xml_escape(title)}</text>',
    ]
    for tick in range(6):
        ratio = tick / 5
        px = left + ratio * plot_w
        py = top + ratio * plot_h
        xv = x_min + ratio * (x_max - x_min)
        yv = y_max - ratio * (y_max - y_min)
        parts.extend(
            [
                f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{top + plot_h}" stroke="#e5e7eb"/>',
                f'<text x="{px:.2f}" y="{top + plot_h + 24}" text-anchor="middle" '
                f'font-family="system-ui,sans-serif" font-size="12" fill="#4b5563">{xml_escape(_fmt(xv))}</text>',
                f'<line x1="{left}" y1="{py:.2f}" x2="{left + plot_w}" y2="{py:.2f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 12}" y="{py + 4:.2f}" text-anchor="end" '
                f'font-family="system-ui,sans-serif" font-size="12" fill="#4b5563">{xml_escape(_fmt(yv))}</text>',
            ]
        )
    if identity_line:
        low, high = max(x_min, y_min), min(x_max, y_max)
        parts.append(
            f'<line x1="{sx(low):.2f}" y1="{sy(low):.2f}" x2="{sx(high):.2f}" y2="{sy(high):.2f}" '
            'stroke="#6b7280" stroke-width="1.5" stroke-dasharray="6 5"/>'
        )
    groups = {item[3] for item in prepared}
    group_order = {value: index for index, value in enumerate(sorted(groups))}
    for x, y, label, group in prepared:
        color = _COLORS[group_order[group] % len(_COLORS)]
        parts.append(
            f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="5" fill="{color}" '
            f'stroke="#ffffff" stroke-width="1"><title>{xml_escape(label or f"{_fmt(x)}, {_fmt(y)}")}</title></circle>'
        )
    parts.extend(
        [
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
            f'<text x="{left + plot_w / 2:.2f}" y="{height - 24}" text-anchor="middle" '
            f'font-family="system-ui,sans-serif" font-size="14" fill="#111827">{xml_escape(x_label)}</text>',
            f'<text x="22" y="{top + plot_h / 2:.2f}" text-anchor="middle" '
            f'transform="rotate(-90 22 {top + plot_h / 2:.2f})" '
            f'font-family="system-ui,sans-serif" font-size="14" fill="#111827">{xml_escape(y_label)}</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def _bar_svg(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str,
    value_label: str,
) -> str | None:
    prepared: list[tuple[str, float]] = []
    for row in rows[:30]:
        try:
            value = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            prepared.append((str(row.get("label", "value")), value))
    if not prepared:
        return None
    id_token = hashlib.sha256(title.encode("utf-8")).hexdigest()[:10]
    title_id, desc_id = f"title-{id_token}", f"desc-{id_token}"
    width = 900
    row_height = 30
    height = 100 + row_height * len(prepared)
    left, right, top = 250, 55, 58
    plot_w = width - left - right
    low = min(0.0, min(value for _, value in prepared))
    high = max(0.0, max(value for _, value in prepared))
    if low == high:
        high = low + 1

    def sx(value: float) -> float:
        return left + (value - low) / (high - low) * plot_w

    zero = sx(0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="{title_id} {desc_id}">',
        f'<title id="{title_id}">{xml_escape(title)}</title>',
        f'<desc id="{desc_id}">Horizontal bar chart of {xml_escape(value_label)}.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="20" y="30" font-family="system-ui,sans-serif" font-size="20" '
        f'font-weight="600" fill="#111827">{xml_escape(title)}</text>',
        f'<line x1="{zero:.2f}" y1="{top - 8}" x2="{zero:.2f}" y2="{height - 28}" stroke="#6b7280"/>',
    ]
    for index, (label, value) in enumerate(prepared):
        y = top + index * row_height
        start, end = sorted((zero, sx(value)))
        parts.extend(
            [
                f'<text x="{left - 12}" y="{y + 16}" text-anchor="end" '
                f'font-family="system-ui,sans-serif" font-size="12" fill="#374151">{xml_escape(label[:34])}</text>',
                f'<rect x="{start:.2f}" y="{y}" width="{max(1, end - start):.2f}" height="20" '
                f'fill="{_COLORS[index % len(_COLORS)]}"><title>{xml_escape(label)}: {xml_escape(_fmt(value))}</title></rect>',
                f'<text x="{min(width - 5, end + 7):.2f}" y="{y + 15}" '
                f'font-family="system-ui,sans-serif" font-size="12" fill="#111827">{xml_escape(_fmt(value))}</text>',
            ]
        )
    parts.append("</svg>")
    return "".join(parts)


def _write_svg(store: ProjectStore, relative: str, svg: str) -> Path:
    return store.write_bytes(relative, (svg + "\n").encode("utf-8"))


def _flatten_metrics(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for result in run.get("results", ()):
        for name, value in result.get("metrics", {}).items():
            rows.append(
                {
                    "task_id": result.get("task_id"),
                    "pipeline_id": result.get("pipeline_id"),
                    "metric": name,
                    "value": value,
                }
            )
    return rows


def _sample_rows(manifest: Any) -> list[dict[str, Any]]:
    samples = {item.sample_id: item for item in manifest.samples}
    rows = []
    for measurement in manifest.measurements:
        sample = samples[measurement.sample_id]
        rows.append(
            {
                "measurement_id": measurement.measurement_id,
                "measurement_name": measurement.metadata.get("measurement_name"),
                "sample_id": measurement.sample_id,
                "role": measurement.role.value,
                "modality": measurement.modality.value,
                "axis_kind": measurement.axis_kind.value,
                "axis_unit": measurement.axis_unit,
                "signal_kind": measurement.signal_kind.value,
                "signal_unit": measurement.signal_unit,
                "preparation_id": sample.preparation_id,
                "technical_replicate_id": sample.technical_replicate_id,
                "batch_id": sample.batch_id,
                "physical_state": sample.physical_state,
            }
        )
    return rows


def _measurement_series(project: ProjectService, manifest: Any) -> list[dict[str, Any]]:
    sample_by_id = {item.sample_id: item for item in manifest.samples}
    rows = []
    for measurement in manifest.measurements[:_MAX_DISPLAY_SERIES]:
        arrays = project.load_measurement(measurement.measurement_id)
        sample = sample_by_id[measurement.sample_id]
        label = str(
            measurement.metadata.get("measurement_name")
            or sample.metadata.get("measurement_name")
            or measurement.measurement_id
        )
        rows.append(
            {
                "label": f"{label} [{measurement.role.value}]",
                "x": arrays["axis"],
                "y": arrays["signal"],
            }
        )
    return rows


def _analysis_evidence(
    store: ProjectStore, run: Mapping[str, Any]
) -> Mapping[str, Any]:
    for artifact in run.get("artifacts", ()):
        if artifact.get("kind") != "analysis_evidence":
            continue
        relative = str(artifact.get("path", ""))
        path = store.path_for(relative)
        if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
            raise ValueError("analysis evidence is missing or its hash does not match")
        payload = store.read_json(relative)
        for key in ("project_id", "run_id", "manifest_hash", "plan_hash"):
            if payload.get(key) != run.get(key):
                raise ValueError(f"analysis evidence {key} does not match run")
        return payload
    return {}


def _task_tables_and_figures(
    store: ProjectStore,
    run: Mapping[str, Any],
    task_result: Mapping[str, Any],
    *,
    run_prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[str, str]]]:
    artifacts: list[dict[str, Any]] = []
    dashboard_tables: list[dict[str, Any]] = []
    dashboard_figures: list[tuple[str, str]] = []

    def table(
        filename: str,
        headers: Sequence[str],
        rows: Sequence[Mapping[str, Any]],
        label: str,
    ) -> None:
        if not rows:
            return
        path = _write_csv(store, f"{run_prefix}/tables/{filename}", headers, rows)
        artifacts.append(
            _artifact(path, store, run, kind="scientist_table", media_type="text/csv")
        )
        dashboard_tables.append(
            {
                "label": label,
                "path": path.relative_to(store.output_root).as_posix(),
                "rows": rows[:100],
                "headers": list(headers),
            }
        )

    def figure(filename: str, svg: str | None, label: str) -> None:
        if not svg:
            return
        path = _write_svg(store, f"{run_prefix}/figures/{filename}", svg)
        artifacts.append(
            _artifact(path, store, run, kind="scientist_figure", media_type="image/svg+xml")
        )
        dashboard_figures.append((label, svg))

    comparisons = list(task_result.get("evidence_rows", ()))
    if not comparisons:
        comparisons = list(
            task_result.get("evidence", {}).get("comparisons", ())
            if isinstance(task_result.get("evidence"), Mapping)
            else ()
        )
    comparison_rows = []
    for row in comparisons:
        comparison = row.get("comparison", {})
        metrics = comparison.get("metrics", {})
        peak = row.get("peak_matches", {})
        if comparison or peak:
            comparison_rows.append(
                {
                    "left_id": row.get("left_measurement_id", row.get("target_id")),
                    "right_id": row.get("right_measurement_id", row.get("control_id")),
                    "cosine_similarity": metrics.get("cosine_similarity"),
                    "correlation": metrics.get("correlation"),
                    "rmse": metrics.get("rmse"),
                    "spectral_angle": metrics.get("spectral_angle"),
                    "overlap_fraction": comparison.get("overlap_fraction"),
                    "peak_match_fraction": peak.get("match_fraction"),
                }
            )
    table(
        "pairwise-comparisons.csv",
        (
            "left_id",
            "right_id",
            "cosine_similarity",
            "correlation",
            "rmse",
            "spectral_angle",
            "overlap_fraction",
            "peak_match_fraction",
        ),
        comparison_rows,
        "Pairwise comparisons",
    )
    figure(
        "pairwise-similarity.svg",
        _bar_svg(
            [
                {
                    "label": f"{row['left_id']} vs {row['right_id']}",
                    "value": row["cosine_similarity"],
                }
                for row in comparison_rows
                if row.get("cosine_similarity") is not None
            ],
            title="Pairwise spectral similarity",
            value_label="cosine similarity",
        ),
        "Pairwise spectral similarity",
    )

    pca = task_result.get("pca")
    if isinstance(pca, Mapping):
        scores = list(pca.get("scores", ()))
        pca_rows = [
            {
                "measurement_index": index,
                "pc1": row[0] if len(row) else None,
                "pc2": row[1] if len(row) > 1 else 0.0,
            }
            for index, row in enumerate(scores)
        ]
        table(
            "pca-scores.csv",
            ("measurement_index", "pc1", "pc2"),
            pca_rows,
            "PCA scores",
        )
        figure(
            "pca-scores.svg",
            _scatter_svg(
                [
                    {
                        "x": row["pc1"],
                        "y": row["pc2"],
                        "label": f"measurement {row['measurement_index']}",
                    }
                    for row in pca_rows
                ],
                title="Descriptive PCA scores",
                x_label="PC1 score",
                y_label="PC2 score",
            ),
            "Descriptive PCA scores",
        )

    evaluation = task_result.get("evaluation")
    if isinstance(evaluation, Mapping):
        predictions = list(evaluation.get("predictions", ()))
        table(
            "predictions.csv",
            ("sample_id", "group", "fold", "y_true", "y_pred"),
            predictions,
            "Held-out predictions",
        )
        numeric_points = []
        for row in predictions:
            try:
                numeric_points.append(
                    {
                        "x": float(row["y_true"]),
                        "y": float(row["y_pred"]),
                        "label": str(row.get("sample_id", "")),
                        "group": str(row.get("group", "")),
                    }
                )
            except (KeyError, TypeError, ValueError):
                numeric_points = []
                break
        figure(
            "predicted-vs-observed.svg",
            _scatter_svg(
                numeric_points,
                title="Held-out predicted versus observed",
                x_label="Observed target",
                y_label="Held-out prediction",
                identity_line=True,
            ),
            "Held-out predicted versus observed",
        )

    job_evidence = task_result.get("evidence")
    if isinstance(job_evidence, Mapping):
        job_points = list(job_evidence.get("job_plot_points", ()))
        table(
            "jobs-plot.csv",
            ("mole_fraction", "response", "replicate_count"),
            job_points,
            "Job’s-plot response",
        )
        figure(
            "jobs-plot.svg",
            _line_svg(
                [
                    {
                        "label": "aggregated response",
                        "x": [row.get("mole_fraction") for row in job_points],
                        "y": [row.get("response") for row in job_points],
                    }
                ],
                title="Job’s-plot response",
                x_label="Mole fraction",
                y_label="Declared response",
            ),
            "Job’s-plot response",
        )

    ranked = list(task_result.get("ranked_results", ()))
    pxrd_rows = [
        {
            "candidate_id": row.get("candidate_id"),
            "candidate_role": row.get("candidate_role"),
            "matched_peak_fraction": row.get("matched_peak_fraction"),
            "mean_position_error_degrees": row.get("mean_position_error_degrees"),
            "cosine_similarity": row.get("whole_pattern_metrics", {}).get(
                "cosine_similarity"
            ),
            "correlation": row.get("whole_pattern_metrics", {}).get("correlation"),
            "overlap_fraction": row.get("overlap_fraction"),
        }
        for row in ranked
    ]
    table(
        "pxrd-reference-ranking.csv",
        (
            "candidate_id",
            "candidate_role",
            "matched_peak_fraction",
            "mean_position_error_degrees",
            "cosine_similarity",
            "correlation",
            "overlap_fraction",
        ),
        pxrd_rows,
        "PXRD reference ranking",
    )
    pxrd_matches = [
        {
            "candidate_id": row.get("candidate_id"),
            **match,
        }
        for row in ranked
        for match in row.get("matched_peaks", ())
    ]
    table(
        "pxrd-matched-peaks.csv",
        (
            "candidate_id",
            "experimental_two_theta",
            "reference_two_theta",
            "position_error_degrees",
        ),
        pxrd_matches,
        "PXRD matched peaks",
    )
    figure(
        "pxrd-reference-ranking.svg",
        _bar_svg(
            [
                {
                    "label": row.get("candidate_id"),
                    "value": row.get("matched_peak_fraction"),
                }
                for row in pxrd_rows
            ],
            title="PXRD reference screening",
            value_label="matched peak fraction",
        ),
        "PXRD reference screening",
    )

    if task_result.get("task_pack") == "mass_spec":
        ms_rows = [
            {
                "candidate_id": row.get("candidate_id"),
                "role": row.get("role"),
                "match_fraction": row.get("match_fraction"),
                "intensity_cosine": row.get("intensity_cosine"),
                "matched_peak_count": len(row.get("matches", ())),
            }
            for row in task_result.get("evidence_rows", ())
        ]
        table(
            "ms-reference-ranking.csv",
            (
                "candidate_id",
                "role",
                "match_fraction",
                "intensity_cosine",
                "matched_peak_count",
            ),
            ms_rows,
            "Mass-spectrometry reference ranking",
        )
        ms_matches = [
            {"candidate_id": row.get("candidate_id"), **match}
            for row in task_result.get("evidence_rows", ())
            for match in row.get("matches", ())
        ]
        table(
            "ms-matched-peaks.csv",
            (
                "candidate_id",
                "experimental_mz",
                "reference_mz",
                "signed_error_da",
                "absolute_error_da",
                "signed_error_ppm",
                "absolute_error_ppm",
            ),
            ms_matches,
            "Mass-spectrometry matched peaks",
        )
        figure(
            "ms-reference-ranking.svg",
            _bar_svg(
                [
                    {
                        "label": row.get("candidate_id"),
                        "value": row.get("match_fraction"),
                    }
                    for row in ms_rows
                ],
                title="Mass-spectrometry reference screening",
                value_label="matched peak fraction",
            ),
            "Mass-spectrometry reference screening",
        )

    mixture = task_result.get("mixture_screening")
    if isinstance(mixture, Mapping):
        names = list(
            mixture.get("provenance", {}).get("reference_names") or ()
        )
        mixture_ids = list(mixture.get("mixture_measurement_ids") or ())
        coefficient_rows = []
        for mixture_index, values in enumerate(mixture.get("coefficients", ())):
            for component_index, value in enumerate(values):
                coefficient_rows.append(
                    {
                        "mixture_id": (
                            mixture_ids[mixture_index]
                            if mixture_index < len(mixture_ids)
                            else mixture_index
                        ),
                        "reference": (
                            names[component_index]
                            if component_index < len(names)
                            else component_index
                        ),
                        "coefficient": value,
                        "screening_only": True,
                    }
                )
        table(
            "mixture-screening-coefficients.csv",
            ("mixture_id", "reference", "coefficient", "screening_only"),
            coefficient_rows,
            "Mixture-screening coefficients",
        )
        figure(
            "mixture-screening-coefficients.svg",
            _bar_svg(
                [
                    {
                        "label": f"{row['mixture_id']} · {row['reference']}",
                        "value": row["coefficient"],
                    }
                    for row in coefficient_rows
                ],
                title="Constrained mixture-screening coefficients",
                value_label="screening coefficient",
            ),
            "Constrained mixture-screening coefficients",
        )

    transformations = []
    for row in task_result.get("measurement_provenance", ()):
        for conversion in row.get("conversions", ()):
            transformations.append(
                {
                    "measurement_id": row.get("measurement_id"),
                    "action": conversion.get("action"),
                    "details": conversion.get("details"),
                    "scope": "measurement",
                }
            )
    if isinstance(evaluation, Mapping):
        for selected in evaluation.get("selected_configs", ()):
            candidate = selected.get("candidate", {})
            transformations.append(
                {
                    "measurement_id": "",
                    "action": candidate.get("preprocessor", "raw"),
                    "details": {
                        "fold": selected.get("fold"),
                        "model_family": candidate.get("family"),
                        "parameters": candidate.get("parameters", {}),
                    },
                    "scope": "training_fold",
                }
            )
    table(
        "transformations.csv",
        ("measurement_id", "action", "details", "scope"),
        transformations,
        "Applied transformations",
    )
    return artifacts, dashboard_tables, dashboard_figures


def _list(items: Sequence[Any], fallback: str) -> str:
    values = list(items)
    if not values:
        values = [fallback]
    return "<ul>" + "".join(
        f"<li>{html.escape(str(item.get('message', item) if isinstance(item, Mapping) else item))}</li>"
        for item in values
    ) + "</ul>"


def _html_table(headers: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    head = "".join(f"<th>{html.escape(name.replace('_', ' ').title())}</th>" for name in headers)
    body = []
    for row in rows[:100]:
        body.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(_display_cell(row.get(name)))}</td>"
                for name in headers
            )
            + "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _display_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def _dashboard_html(
    run: Mapping[str, Any],
    *,
    sample_rows: Sequence[Mapping[str, Any]],
    figures: Sequence[tuple[str, str]],
    tables: Sequence[Mapping[str, Any]],
    artifacts: Sequence[Mapping[str, Any]],
) -> str:
    status = str(run.get("status", "unknown"))
    issues = list(run.get("issues", ()))
    limitations = list(run.get("blockers", ())) + list(run.get("limitations", ()))
    metrics = _flatten_metrics(run)
    display_metrics = [
        {
            "metric": _metric_label(row.get("metric")),
            "value": (
                _fmt(float(row["value"]))
                if isinstance(row.get("value"), (int, float))
                else row.get("value")
            ),
        }
        for row in metrics
    ]
    counts = run.get("counts", {})
    claim = run.get("claim_eligibility", {})
    report_sections = (
        ("Observed evidence", run.get("observed_spectral_evidence", ()), "No observed evidence was produced."),
        ("Model evidence", run.get("model_evidence", ()), "No model evidence was produced."),
        ("Possible explanations", run.get("tentative_explanations", ()), "No explanations were supplied."),
        ("Unsupported claims", run.get("unsupported_claims", ()), "No unsupported claims were listed."),
        ("Limitations", limitations, "No additional limitations were listed."),
        ("Recommended next experiment", run.get("next_experiments", ()), "No next experiment was proposed."),
    )
    nav = [
        ("overview", "Overview"),
        ("figures", "Figures"),
        ("findings", "Findings"),
        ("samples", "Samples"),
        ("tables", "Tables"),
        ("provenance", "Provenance"),
    ]
    style = """
:root{color-scheme:light dark;--bg:#f7f8fb;--panel:#fff;--text:#172033;--muted:#5f6b7a;--line:#dce2ea;--accent:#2457d6;--danger:#b42318;--warn:#9a6700;--ok:#067647}
@media(prefers-color-scheme:dark){:root{--bg:#10141c;--panel:#181e29;--text:#edf1f7;--muted:#aab4c3;--line:#303a49;--accent:#8fb0ff;--danger:#ff9b91;--warn:#ffd27a;--ok:#75dfb4}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1180px;margin:auto;padding:28px 22px 70px}header{padding:8px 0 20px;border-bottom:1px solid var(--line)}h1{font-size:30px;line-height:1.2;margin:0 0 8px}h2{font-size:21px;margin:0 0 14px}h3{font-size:16px;margin:22px 0 8px}p{margin:6px 0 12px}.muted{color:var(--muted)}nav{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}nav a,.download{color:var(--accent);text-decoration:none;border:1px solid var(--line);border-radius:8px;padding:7px 10px;background:var(--panel)}
section{margin:22px 0;padding:20px;background:var(--panel);border:1px solid var(--line);border-radius:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}.stat{padding:14px;border:1px solid var(--line);border-radius:9px}.stat b{display:block;font-size:23px}.badge{display:inline-block;padding:3px 9px;border-radius:999px;border:1px solid var(--line);font-weight:600}.succeeded{color:var(--ok)}.blocked,.failed{color:var(--danger)}.running,.pending{color:var(--warn)}
.figure{margin:18px 0}.figure svg{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-weight:600}code{overflow-wrap:anywhere}details{margin:12px 0}summary{cursor:pointer;font-weight:600}ul{padding-left:22px}.warning{border-left:4px solid var(--warn);padding-left:12px}.danger{border-left:4px solid var(--danger);padding-left:12px}
@media(max-width:600px){main{padding:18px 12px 45px}section{padding:15px}h1{font-size:25px}}
"""
    parts = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>Chemometrics run {html.escape(str(run['run_id']))}</title>",
        f"<style>{style}</style></head><body><main>",
        "<header>",
        f"<h1>Chemometrics run {html.escape(str(run['run_id']))}</h1>",
        f'<p><span class="badge {html.escape(status)}">{html.escape(status)}</span> '
        f'<span class="muted">Task: {html.escape(_human_label(run.get("task_kind")))}</span></p>',
        "<p class=\"muted\">Scientist-facing evidence dashboard. Possible explanations are separated from observations; final scientific judgment remains with the researcher.</p>",
        "</header>",
        "<nav aria-label=\"Dashboard sections\">",
        "".join(f'<a href="#{target}">{label}</a>' for target, label in nav),
        "</nav>",
        '<section id="overview"><h2>Overview</h2><div class="grid">',
        f'<div class="stat"><span class="muted">Measurements</span><b>{html.escape(str(counts.get("scan_count", len(sample_rows))))}</b></div>',
        f'<div class="stat"><span class="muted">Independent preparations</span><b>{html.escape(str(counts.get("preparation_count") if counts.get("preparation_count") is not None else "not declared"))}</b></div>',
        f'<div class="stat"><span class="muted">Claim level</span><b>{html.escape(str(claim.get("claim_level", "descriptive")))}</b></div>',
        f'<div class="stat"><span class="muted">Issues</span><b>{len(issues)}</b></div>',
        "</div>",
    ]
    if issues:
        parts.append(
            '<div class="warning"><h3>Run issues</h3>'
            + _list(issues, "No issues.")
            + "</div>"
        )
    if display_metrics:
        parts.extend(
            [
                "<h3>Reported metrics</h3>",
                _html_table(
                    ("metric", "value"),
                    display_metrics,
                ),
            ]
        )
    parts.append("</section>")
    parts.append('<section id="figures"><h2>Figures</h2>')
    if figures:
        for label, svg in figures:
            parts.append(
                f'<div class="figure"><h3>{html.escape(label)}</h3>{svg}</div>'
            )
    else:
        parts.append("<p>No plot-ready evidence was available for this run.</p>")
    parts.append("</section>")
    parts.append('<section id="findings"><h2>Findings and interpretation boundaries</h2>')
    for heading, items, fallback in report_sections:
        parts.append(f"<h3>{html.escape(heading)}</h3>{_list(list(items), fallback)}")
    parts.append("</section>")
    parts.extend(
        [
            '<section id="samples"><h2>Samples and measurement semantics</h2>',
            _html_table(
                (
                    "measurement_name",
                    "role",
                    "modality",
                    "axis_kind",
                    "axis_unit",
                    "signal_kind",
                    "signal_unit",
                    "preparation_id",
                    "technical_replicate_id",
                ),
                sample_rows,
            )
            if sample_rows
            else "<p>Sample inventory was unavailable.</p>",
            "</section>",
            '<section id="tables"><h2>Tables</h2>',
        ]
    )
    if tables:
        for table in tables:
            relative = Path(str(table["path"]))
            # Dashboard and tables share the run directory with tables below it.
            href = "tables/" + relative.name
            parts.extend(
                [
                    f"<details><summary>{html.escape(str(table['label']))}</summary>",
                    f'<p><a class="download" href="{html.escape(href)}" download>Download CSV</a></p>',
                    _html_table(table["headers"], table["rows"]),
                    "</details>",
                ]
            )
    else:
        parts.append("<p>No tabular outputs were generated.</p>")
    parts.extend(
        [
            "</section>",
            '<section id="provenance"><h2>Provenance</h2>',
            "<p>Every listed artifact is tied to this run and verified with SHA-256.</p>",
            "<dl>",
            f"<dt>Project</dt><dd><code>{html.escape(str(run.get('project_id')))}</code></dd>",
            f"<dt>Manifest hash</dt><dd><code>{html.escape(str(run.get('manifest_hash')))}</code></dd>",
            f"<dt>Plan hash</dt><dd><code>{html.escape(str(run.get('plan_hash')))}</code></dd>",
            f"<dt>Approval</dt><dd><code>{html.escape(str(run.get('approval_id') or 'not available'))}</code></dd>",
            "</dl>",
            _html_table(
                ("kind", "path", "sha256"),
                [
                    {
                        "kind": item.get("kind"),
                        "path": item.get("path"),
                        "sha256": item.get("sha256"),
                    }
                    for item in artifacts
                ],
            ),
            "</section></main></body></html>",
        ]
    )
    return "".join(parts)


def render_run_dashboard(
    store: ProjectStore,
    run: Mapping[str, Any],
    *,
    project: ProjectService | None = None,
) -> tuple[dict[str, Any], ...]:
    """Render run-local figures, CSV tables, and one offline HTML dashboard."""
    required = ("project_id", "run_id", "manifest_hash", "plan_hash")
    if any(not run.get(key) for key in required):
        return ()
    run_id = str(run["run_id"])
    if slugify_project_id(run_id) != run_id:
        raise ValueError("run_id must be a safe lowercase slug")
    project = project or ProjectService.open(store.output_root)
    manifest = project.get_manifest()
    if manifest.project_id != run["project_id"]:
        raise ValueError("dashboard project identity does not match run")
    if manifest.manifest_hash != run["manifest_hash"]:
        raise ValueError("dashboard requires the exact run manifest")
    evidence = _analysis_evidence(store, run)
    task_result = (
        evidence.get("task_result", {})
        if isinstance(evidence.get("task_result"), Mapping)
        else {}
    )
    run_prefix = f"runs/{run_id}"
    artifacts: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    figures: list[tuple[str, str]] = []
    sample_rows = _sample_rows(manifest)

    def add_table(
        filename: str,
        headers: Sequence[str],
        rows: Sequence[Mapping[str, Any]],
        label: str,
    ) -> None:
        path = _write_csv(store, f"{run_prefix}/tables/{filename}", headers, rows)
        artifacts.append(
            _artifact(path, store, run, kind="scientist_table", media_type="text/csv")
        )
        tables.append(
            {
                "label": label,
                "path": path.relative_to(store.output_root).as_posix(),
                "headers": list(headers),
                "rows": rows[:100],
            }
        )

    add_table(
        "sample-assignments.csv",
        (
            "measurement_id",
            "measurement_name",
            "sample_id",
            "role",
            "modality",
            "axis_kind",
            "axis_unit",
            "signal_kind",
            "signal_unit",
            "preparation_id",
            "technical_replicate_id",
            "batch_id",
            "physical_state",
        ),
        sample_rows,
        "Sample assignments",
    )
    issue_rows = [
        {
            "code": item.get("code"),
            "level": item.get("level"),
            "stage": item.get("stage"),
            "message": item.get("message"),
            "details": item.get("details", {}),
        }
        for item in run.get("issues", ())
        if isinstance(item, Mapping)
    ]
    add_table(
        "issues.csv",
        ("code", "level", "stage", "message", "details"),
        issue_rows,
        "Issues and warnings",
    )
    metric_rows = _flatten_metrics(run)
    add_table(
        "metrics.csv",
        ("task_id", "pipeline_id", "metric", "value"),
        metric_rows,
        "Reported metrics",
    )
    try:
        series = _measurement_series(project, manifest)
        first = manifest.measurements[0] if manifest.measurements else None
        raw_svg = _line_svg(
            series,
            title=(
                "Source measurement overlay"
                if len(manifest.measurements) <= _MAX_DISPLAY_SERIES
                else (
                    "Source measurement overlay "
                    f"(first {_MAX_DISPLAY_SERIES} of {len(manifest.measurements)})"
                )
            ),
            x_label=(
                f"{first.axis_kind.value} ({first.axis_unit or 'unit not declared'})"
                if first
                else "axis"
            ),
            y_label=(
                f"{first.signal_kind.value} ({first.signal_unit or 'unit not declared'})"
                if first
                else "signal"
            ),
        )
        if raw_svg:
            path = _write_svg(
                store, f"{run_prefix}/figures/source-measurement-overlay.svg", raw_svg
            )
            artifacts.append(
                _artifact(
                    path,
                    store,
                    run,
                    kind="scientist_figure",
                    media_type="image/svg+xml",
                )
            )
            figures.append(("Source measurement overlay", raw_svg))
    except (ValueError, FileNotFoundError):
        # A blocked run still receives its issue/sample dashboard even if raw
        # arrays are unavailable or fail their storage-integrity checks.
        pass
    task_artifacts, task_tables, task_figures = _task_tables_and_figures(
        store, run, task_result, run_prefix=run_prefix
    )
    artifacts.extend(task_artifacts)
    tables.extend(task_tables)
    figures.extend(task_figures)
    existing = list(run.get("artifacts", ()))
    dashboard = _dashboard_html(
        run,
        sample_rows=sample_rows,
        figures=figures,
        tables=tables,
        artifacts=existing + artifacts,
    )
    dashboard_path = store.write_bytes(
        f"{run_prefix}/dashboard.html", dashboard.encode("utf-8")
    )
    artifacts.append(
        _artifact(
            dashboard_path,
            store,
            run,
            kind="scientist_dashboard",
            media_type="text/html",
        )
    )
    return tuple(artifacts)


def render_reproducibility_notebook(
    store: ProjectStore, run: Mapping[str, Any]
) -> dict[str, Any]:
    """Create a notebook that verifies and displays persisted run artifacts."""
    run_id = str(run["run_id"])
    if slugify_project_id(run_id) != run_id:
        raise ValueError("run_id must be a safe lowercase slug")
    dashboard = next(
        (
            item
            for item in run.get("artifacts", ())
            if item.get("kind") == "scientist_dashboard"
        ),
        None,
    )
    if dashboard is None:
        raise ValueError("dashboard must exist before notebook generation")
    verifiable = [
        {
            "path": str(Path(str(item["path"])).relative_to(f"runs/{run_id}")),
            "sha256": item["sha256"],
            "kind": item.get("kind"),
        }
        for item in run.get("artifacts", ())
        if item.get("path")
        and item.get("sha256")
        and str(item["path"]).startswith(f"runs/{run_id}/")
    ]
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
            "chemometrics_run": {
                "project_id": run["project_id"],
                "run_id": run_id,
                "manifest_hash": run["manifest_hash"],
                "plan_hash": run["plan_hash"],
            },
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# Chemometrics run `{run_id}`\n",
                    "\n",
                    "This notebook verifies and displays persisted evidence. "
                    "It does not refit models, change preprocessing, or perform a second analysis.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import hashlib\n",
                    f"artifacts = {verifiable!r}\n",
                    "for artifact in artifacts:\n",
                    "    path = Path(artifact['path'])\n",
                    "    actual = hashlib.sha256(path.read_bytes()).hexdigest()\n",
                    "    assert actual == artifact['sha256'], f\"Hash mismatch: {path}\"\n",
                    "print(f\"Verified {len(artifacts)} persisted run artifacts.\")\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from IPython.display import HTML, display\n",
                    "display(HTML(Path('dashboard.html').read_text(encoding='utf-8')))\n",
                ],
            },
        ],
    }
    path = store.write_json(f"runs/{run_id}/analysis-notebook.ipynb", notebook)
    return _artifact(
        path,
        store,
        run,
        kind="reproducibility_notebook",
        media_type="application/x-ipynb+json",
    )
