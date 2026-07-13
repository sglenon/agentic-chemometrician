"""Dataset loading and inspection for spectral data files.

Supported formats
-----------------
- Excel (.xlsx / .xls) with a two-sheet layout:
    - ``Spectra Metadata`` — one row per sample, columns are sample attributes.
    - ``Spectra <device>`` — first column is wavelength/wavenumber axis; remaining
      columns are samples identified by Spectrum ID, matching the metadata sheet.

The loader is deterministic: given the same file it always returns the same
``SpectralDataset`` and ``DatasetInspection`` with no random operations.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from chemometrics_contracts import (
    ArtifactReference,
    DatasetInspection,
    SpectralDataset,
    ValidationWarning,
)

_NIR_WAVELENGTH_RANGE = (700.0, 2500.0)
_FTIR_WAVENUMBER_RANGE = (400.0, 4000.0)

_CANDIDATE_LABEL_PATTERNS = re.compile(
    r"(label|class|category|type|material|product|description|grade|sample.name)",
    re.IGNORECASE,
)
_CANDIDATE_GROUP_PATTERNS = re.compile(
    r"(group|batch|replicate|rep|lot|sample.id|run|session|origin)",
    re.IGNORECASE,
)

_NUMERIC_DTYPE_KINDS = frozenset(("f", "i", "u"))


def _infer_modality(axis_values: np.ndarray) -> str | None:
    axis_min = float(axis_values.min())
    axis_max = float(axis_values.max())
    if _NIR_WAVELENGTH_RANGE[0] <= axis_min and axis_max <= _NIR_WAVELENGTH_RANGE[1]:
        return "NIR"
    if _FTIR_WAVENUMBER_RANGE[0] <= axis_min and axis_max <= _FTIR_WAVENUMBER_RANGE[1]:
        return "FTIR"
    return None


def _candidate_label_cols(metadata: pd.DataFrame) -> list[str]:
    return [
        col
        for col in metadata.columns
        if _CANDIDATE_LABEL_PATTERNS.search(str(col)) and metadata[col].nunique() > 1
    ]


def _candidate_group_cols(metadata: pd.DataFrame) -> list[str]:
    return [
        col
        for col in metadata.columns
        if _CANDIDATE_GROUP_PATTERNS.search(str(col)) and metadata[col].nunique() > 1
    ]


def _check_missing_values(X: np.ndarray) -> ValidationWarning | None:
    n_missing = int(np.isnan(X).sum())
    if n_missing > 0:
        return ValidationWarning(
            code="missing_values",
            message=f"Spectral matrix contains {n_missing} missing value(s).",
            category="data_quality",
            severity="error",
            affected_stage="inspection",
        )
    return None


def _check_non_numeric(X: np.ndarray) -> ValidationWarning | None:
    if X.dtype.kind not in _NUMERIC_DTYPE_KINDS:
        return ValidationWarning(
            code="non_numeric_spectra",
            message=f"Spectral matrix has non-numeric dtype: {X.dtype}.",
            category="data_quality",
            severity="error",
            affected_stage="inspection",
        )
    return None


def _check_small_sample(n_samples: int, threshold: int = 20) -> ValidationWarning | None:
    if n_samples < threshold:
        return ValidationWarning(
            code="small_sample",
            message=f"Only {n_samples} sample(s) found; results will be unreliable.",
            category="data_quality",
            severity="warning",
            affected_stage="inspection",
        )
    return None


def _check_duplicate_ids(sample_ids: Sequence[str]) -> ValidationWarning | None:
    seen: set[str] = set()
    dupes: list[str] = []
    for sid in sample_ids:
        if sid in seen:
            dupes.append(sid)
        seen.add(sid)
    if dupes:
        return ValidationWarning(
            code="duplicate_sample_ids",
            message=f"Duplicate sample IDs detected: {dupes[:5]}{'...' if len(dupes) > 5 else ''}",
            category="data_quality",
            severity="warning",
            affected_stage="inspection",
        )
    return None


def _check_shape_mismatch(n_meta: int, n_spectra: int) -> ValidationWarning | None:
    if n_meta != n_spectra:
        return ValidationWarning(
            code="shape_mismatch",
            message=(
                f"Metadata has {n_meta} rows but spectra sheet has {n_spectra} sample columns. "
                "Alignment may be incomplete."
            ),
            category="data_quality",
            severity="warning",
            affected_stage="inspection",
        )
    return None


def _check_ambiguous_labels(candidate_cols: list[str]) -> ValidationWarning | None:
    if len(candidate_cols) > 1:
        return ValidationWarning(
            code="ambiguous_label_columns",
            message=(
                f"Multiple candidate label columns found: {candidate_cols}. "
                "Specify label_column in the request to disambiguate."
            ),
            category="metadata",
            severity="info",
            affected_stage="inspection",
        )
    if len(candidate_cols) == 0:
        return ValidationWarning(
            code="no_label_columns",
            message="No candidate label columns found. Supervised tasks will not be available.",
            category="metadata",
            severity="warning",
            affected_stage="inspection",
        )
    return None


def load_excel_nir(
    source_path: str | Path,
    *,
    modality_override: str | None = None,
    label_column: str | None = None,
    sample_id_column: str | None = None,
) -> tuple[SpectralDataset, DatasetInspection]:
    """Load a two-sheet NIR Excel file and return dataset and inspection objects.

    Parameters
    ----------
    source_path:
        Path to the .xlsx file.
    modality_override:
        If supplied, overrides inferred modality (e.g. ``"NIR"``).
    label_column:
        Metadata column to use as the primary label. If ``None``, candidate
        columns are detected but no label is assigned to the dataset.
    sample_id_column:
        Metadata column to use as sample IDs. If ``None``, ``"Spectrum ID"``
        is used if present, otherwise integer position strings.
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if path.suffix.lower() not in {".xlsx", ".xls"}:
        raise ValueError(f"Unsupported file format: {path.suffix!r}. Expected .xlsx or .xls.")

    xl = pd.ExcelFile(path)
    sheet_names: list[str] = xl.sheet_names

    metadata_sheet = next(
        (s for s in sheet_names if "metadata" in s.lower()), None
    )
    spectra_sheet = next(
        (s for s in sheet_names if "spectra" in s.lower() and "metadata" not in s.lower()), None
    )

    if metadata_sheet is None or spectra_sheet is None:
        raise ValueError(
            f"Could not find required sheets. Expected 'Spectra Metadata' and a spectra sheet. "
            f"Found: {sheet_names}"
        )

    metadata_df: pd.DataFrame = xl.parse(metadata_sheet)
    spectra_df: pd.DataFrame = xl.parse(spectra_sheet)

    axis_col = spectra_df.columns[0]
    axis_values: np.ndarray = spectra_df[axis_col].to_numpy(dtype=float)

    spectrum_id_cols = spectra_df.columns[1:]
    spectra_matrix: np.ndarray = spectra_df[spectrum_id_cols].to_numpy(dtype=float).T

    if "Spectrum ID" in metadata_df.columns:
        meta_ids = metadata_df["Spectrum ID"].astype(str).tolist()
        spec_ids = [str(c) for c in spectrum_id_cols]
        ordered_meta = metadata_df.set_index("Spectrum ID").reindex(
            [int(c) if str(c).isdigit() else c for c in spectrum_id_cols]
        ).reset_index()
    else:
        ordered_meta = metadata_df
        meta_ids = [str(i) for i in range(len(metadata_df))]
        spec_ids = [str(i) for i in range(len(spectrum_id_cols))]

    n_samples = spectra_matrix.shape[0]
    n_features = spectra_matrix.shape[1]

    if sample_id_column and sample_id_column in ordered_meta.columns:
        sample_ids = ordered_meta[sample_id_column].astype(str).tolist()
    elif "Spectrum ID" in ordered_meta.columns:
        sample_ids = ordered_meta["Spectrum ID"].astype(str).tolist()
    else:
        sample_ids = [str(i) for i in range(n_samples)]

    metadata_records: list[dict[str, Any]] = ordered_meta.to_dict(orient="records")

    candidate_labels = _candidate_label_cols(ordered_meta)
    candidate_groups = _candidate_group_cols(ordered_meta)

    resolved_label_col = label_column if label_column else (
        candidate_labels[0] if len(candidate_labels) == 1 else None
    )
    labels: list[Any] | None = None
    if resolved_label_col and resolved_label_col in ordered_meta.columns:
        labels = ordered_meta[resolved_label_col].tolist()

    modality = modality_override or _infer_modality(axis_values)

    source_ref = ArtifactReference(
        kind="source_file",
        uri=path.as_posix(),
        label=path.name,
        mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    dataset = SpectralDataset(
        x=tuple(tuple(row) for row in spectra_matrix.tolist()),
        axis=tuple(axis_values.tolist()),
        metadata=tuple({str(k): v for k, v in rec.items()} for rec in metadata_records),
        labels=tuple(labels) if labels is not None else None,
        modality=modality,
        sample_ids=tuple(sample_ids),
        source_references=(source_ref,),
    )

    warnings: list[ValidationWarning] = []
    for check in [
        _check_non_numeric(spectra_matrix),
        _check_missing_values(spectra_matrix),
        _check_small_sample(n_samples),
        _check_duplicate_ids(sample_ids),
        _check_shape_mismatch(len(metadata_df), n_samples),
        _check_ambiguous_labels(candidate_labels),
    ]:
        if check is not None:
            warnings.append(check)

    inspection = DatasetInspection(
        sample_count=n_samples,
        feature_count=n_features,
        axis_min=float(axis_values.min()),
        axis_max=float(axis_values.max()),
        modality=modality,
        candidate_label_columns=tuple(candidate_labels),
        candidate_group_columns=tuple(candidate_groups),
        warnings=tuple(warnings),
    )

    return dataset, inspection


def load_composition_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Composition table not found: {path}")
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    table_lines = [l for l in lines if l.strip().startswith("|")]
    if len(table_lines) < 3:
        raise ValueError(f"No valid markdown table found in {path}")
    header_line = table_lines[0]
    headers = [h.strip() for h in header_line.strip("|").split("|")]
    data_rows = []
    for line in table_lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == len(headers):
            data_rows.append(cells)
    df = pd.DataFrame(data_rows, columns=headers)
    col_map = {}
    for col in df.columns:
        low = col.lower()
        if low == "label":
            col_map[col] = "label"
        elif "ami wt%" in low or "ami wt %" in low:
            col_map[col] = "ami_wt_pct"
        elif "fc wt%" in low or "fc wt %" in low:
            col_map[col] = "fc_wt_pct"
        elif "sb wt%" in low or "sb wt %" in low:
            col_map[col] = "sb_wt_pct"
        elif low == "prep":
            col_map[col] = "prep"
    df = df.rename(columns=col_map)
    required = {"label", "ami_wt_pct", "fc_wt_pct", "sb_wt_pct", "prep"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in composition table: {missing}")
    df["ami_wt_pct"] = pd.to_numeric(df["ami_wt_pct"], errors="coerce")
    df["fc_wt_pct"] = pd.to_numeric(df["fc_wt_pct"], errors="coerce")
    df["sb_wt_pct"] = pd.to_numeric(df["sb_wt_pct"], errors="coerce")
    return df[["label", "ami_wt_pct", "fc_wt_pct", "sb_wt_pct", "prep"]]


def _generate_synthetic_ftir(
    compositions: pd.DataFrame,
    wavenumber_range: tuple[float, float] = (400.0, 4000.0),
    n_points: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    axis = np.linspace(wavenumber_range[0], wavenumber_range[1], n_points)
    ref_centers = {
        "ami": [1620.0, 1580.0, 1250.0],
        "fc": [1700.0, 1640.0, 1350.0],
        "sb": [1590.0, 1510.0, 1280.0],
    }
    ref_widths = {
        "ami": [40.0, 35.0, 45.0],
        "fc": [38.0, 42.0, 50.0],
        "sb": [36.0, 40.0, 48.0],
    }

    def _reference(compound: str) -> np.ndarray:
        spec = np.zeros(n_points)
        for center, width in zip(ref_centers[compound], ref_widths[compound]):
            spec += np.exp(-0.5 * ((axis - center) / width) ** 2)
        return spec

    refs = {c: _reference(c) for c in ("ami", "fc", "sb")}
    n_samples = len(compositions)
    X = np.zeros((n_samples, n_points), dtype=float)
    for i, (_, row) in enumerate(compositions.iterrows()):
        X[i] = (
            row["ami_wt_pct"] / 100.0 * refs["ami"]
            + row["fc_wt_pct"] / 100.0 * refs["fc"]
            + row["sb_wt_pct"] / 100.0 * refs["sb"]
        )
    X += rng.normal(0, 0.005, size=X.shape)
    return X, axis


def load_ftir_composition(
    source_path: str | Path,
    *,
    spectra_path: str | Path | None = None,
    modality_override: str | None = None,
) -> tuple[SpectralDataset, DatasetInspection]:
    path = Path(source_path)
    compositions = load_composition_table(path)
    n_samples = len(compositions)

    if spectra_path is not None:
        sp = Path(spectra_path)
        xl = pd.ExcelFile(sp)
        metadata_sheet = next((s for s in xl.sheet_names if "metadata" in s.lower()), None)
        spectra_sheet = next(
            (s for s in xl.sheet_names if "spectra" in s.lower() and "metadata" not in s.lower()), None
        )
        if metadata_sheet is None or spectra_sheet is None:
            raise ValueError(f"Expected metadata and spectra sheets in {sp}")
        spectra_df = xl.parse(spectra_sheet)
        axis_col = spectra_df.columns[0]
        axis_values = spectra_df[axis_col].to_numpy(dtype=float)
        spectrum_cols = spectra_df.columns[1:]
        spectra_matrix = spectra_df[spectrum_cols].to_numpy(dtype=float).T
        is_synthetic = False
    else:
        spectra_matrix, axis_values = _generate_synthetic_ftir(compositions)
        is_synthetic = True

    n_features = spectra_matrix.shape[1]
    labels = compositions["label"].tolist()
    sample_ids = labels
    metadata_records = compositions.to_dict(orient="records")
    modality = modality_override or "FTIR"

    source_ref = ArtifactReference(
        kind="source_file",
        uri=path.as_posix(),
        label=path.name,
        mime_type="text/markdown",
    )

    dataset = SpectralDataset(
        x=tuple(tuple(row) for row in spectra_matrix.tolist()),
        axis=tuple(axis_values.tolist()),
        metadata=tuple({str(k): v for k, v in rec.items()} for rec in metadata_records),
        labels=tuple(labels),
        modality=modality,
        sample_ids=tuple(sample_ids),
        source_references=(source_ref,),
    )

    warnings: list[ValidationWarning] = []
    if is_synthetic:
        warnings.append(
            ValidationWarning(
                code="synthetic_data",
                message="Spectral data is synthetically generated. Replace with real measurements before drawing conclusions.",
                category="data_quality",
                severity="info",
                affected_stage="inspection",
            )
        )
    for check in [
        _check_non_numeric(spectra_matrix),
        _check_missing_values(spectra_matrix),
        _check_small_sample(n_samples),
    ]:
        if check is not None:
            warnings.append(check)

    inspection = DatasetInspection(
        sample_count=n_samples,
        feature_count=n_features,
        axis_min=float(axis_values.min()),
        axis_max=float(axis_values.max()),
        modality=modality,
        candidate_label_columns=("label",),
        candidate_group_columns=("prep",),
        warnings=tuple(warnings),
    )

    return dataset, inspection


def save_inspection_artifact(
    inspection: DatasetInspection,
    artifact_dir: Path,
    filename: str = "dataset_inspection.json",
) -> Path:
    """Write inspection to a JSON file inside *artifact_dir* and return the path."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out = artifact_dir / filename
    out.write_text(json.dumps(inspection.to_dict(), indent=2, default=str), encoding="utf-8")
    return out
