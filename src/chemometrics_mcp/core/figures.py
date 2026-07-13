"""Render figure data dicts to publication-quality PNG/PDF images."""
from __future__ import annotations

import logging
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    import seaborn as _sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

log = logging.getLogger(__name__)

_FIG_WIDTH = 6.0
_FIG_HEIGHT = 4.0
_FIG_DPI = 150

if _HAS_MPL:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "font.family": "sans-serif",
    })


def _require_backend():
    if not _HAS_MPL:
        raise RuntimeError(
            "matplotlib is not installed. Install it with: pip install matplotlib"
        )


def _save(fig, output_path: Path, fmt: str, dpi: int) -> Path:
    save_kwargs = {"format": fmt, "dpi": dpi}
    if fmt != "pdf":
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(output_path, **save_kwargs)
    plt.close(fig)
    return output_path


def _render_confusion_matrix(fig_data: dict, output_path: Path, fmt: str, dpi: int) -> Path:
    cm = fig_data["confusion_matrix"]
    labels = fig_data.get("class_labels", [str(i) for i in range(len(cm))])
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))
    if _HAS_SNS:
        _sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, ax=ax, cmap="Blues")
    else:
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    return _save(fig, output_path, fmt, dpi)


def _render_feature_importances(fig_data: dict, output_path: Path, fmt: str, dpi: int) -> Path:
    importances = fig_data["feature_importances"]
    axis_vals = fig_data.get("axis")
    if axis_vals is not None:
        labels = [f"{v:.1f}" for v in axis_vals]
    else:
        labels = [str(i) for i in range(len(importances))]
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))
    y_pos = range(len(importances))
    ax.barh(y_pos, importances, color="#4C72B0", edgecolor="#333333", linewidth=0.5)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importances")
    fig.tight_layout()
    return _save(fig, output_path, fmt, dpi)


def _render_explained_variance(fig_data: dict, output_path: Path, fmt: str, dpi: int) -> Path:
    evr = fig_data["explained_variance_ratio"]
    cumulative = fig_data.get("cumulative_variance", [sum(evr[:i + 1]) for i in range(len(evr))])
    components = list(range(1, len(evr) + 1))
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))
    ax.bar(components, evr, color="#4C72B0", edgecolor="#333333", linewidth=0.5, label="Individual")
    ax.plot(components, cumulative, color="#DD4444", marker="o", markersize=4, linewidth=1.5, label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("Explained Variance")
    ax.legend(loc="center right")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return _save(fig, output_path, fmt, dpi)


def _render_predicted_vs_actual(fig_data: dict, output_path: Path, fmt: str, dpi: int) -> Path:
    pva = fig_data["predicted_vs_actual"]
    predicted = pva["predicted"]
    actual = pva["actual"]
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))
    ax.scatter(actual, predicted, s=20, alpha=0.7, color="#4C72B0", edgecolors="#333333", linewidths=0.3)
    all_vals = list(actual) + list(predicted)
    lo, hi = min(all_vals), max(all_vals)
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "--", color="#888888", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return _save(fig, output_path, fmt, dpi)


def _render_cv_accuracy(fig_data: dict, output_path: Path, fmt: str, dpi: int) -> Path:
    accuracies = fig_data.get("cv_accuracy_per_fold", [])
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))
    if accuracies:
        folds = list(range(1, len(accuracies) + 1))
        ax.bar(folds, accuracies, color="#4C72B0", edgecolor="#333333", linewidth=0.5)
        ax.set_xlabel("Fold")
        ax.set_ylabel("Accuracy")
        ax.set_title("CV Accuracy per Fold")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    else:
        ax.text(0.5, 0.5, "No per-fold data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("CV Accuracy per Fold")
    fig.tight_layout()
    return _save(fig, output_path, fmt, dpi)


def _render_cluster_sizes(fig_data: dict, output_path: Path, fmt: str, dpi: int) -> Path:
    sizes = fig_data["cluster_sizes"]
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))
    clusters = [f"C{i}" for i in range(len(sizes))]
    ax.bar(clusters, sizes, color="#4C72B0", edgecolor="#333333", linewidth=0.5)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Size")
    ax.set_title("Cluster Sizes")
    fig.tight_layout()
    return _save(fig, output_path, fmt, dpi)


_RENDERERS = {
    "confusion_matrix": _render_confusion_matrix,
    "feature_importances": _render_feature_importances,
    "explained_variance_ratio": _render_explained_variance,
    "predicted_vs_actual": _render_predicted_vs_actual,
    "cv_accuracy_per_fold": _render_cv_accuracy,
    "cluster_sizes": _render_cluster_sizes,
}


def render_figure(
    fig_data: dict,
    model_name: str,
    output_path: Path,
    format: str = "png",
    dpi: int = _FIG_DPI,
) -> Path:
    """Render *fig_data* to an image file and return the output path.

    Dispatches on the keys present in *fig_data* to select the appropriate
    renderer.  Raises :class:`RuntimeError` if matplotlib is unavailable.
    """
    _require_backend()

    output_path = Path(output_path)

    for key, renderer in _RENDERERS.items():
        if key in fig_data:
            return renderer(fig_data, output_path, format, dpi)

    log.warning("No recognised figure key in fig_data for %s", model_name)
    return output_path
