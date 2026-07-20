"""MCP tool: inspect_dataset

Loads a spectral data file, inspects its shape, axis, labels, groups, and
data quality, then saves a structured JSON artifact. Returns a ToolResponse
wrapping a DatasetInspection payload.

This tool never guesses ambiguous labels silently — it surfaces them as
``ambiguous_label_columns`` warnings so the agent can ask the user.
"""
from __future__ import annotations

import datetime
from pathlib import Path

from chemometrics_contracts import (
    DatasetInspection,
    InspectDatasetRequest,
    RunMetadata,
    ToolResponse,
)

from chemometrics_mcp.artifacts import ensure_run_dir, make_run_id, artifact_ref
from chemometrics_mcp.core.datasets import (
    load_excel_nir,
    load_ftir_composition,
    load_ftir_real,
    save_inspection_artifact,
)

_SUPPORTED_EXTENSIONS = frozenset({".xlsx", ".xls", ".txt"})


def _unsupported_format_response(source_uri: str, ext: str) -> ToolResponse[DatasetInspection]:
    return ToolResponse(
        tool_name="inspect_dataset",
        ok=False,
        error=f"Unsupported file format {ext!r}. Supported: {sorted(_SUPPORTED_EXTENSIONS)}.",
        message=f"Cannot inspect {source_uri!r}: unsupported format.",
    )


def _file_not_found_response(source_uri: str) -> ToolResponse[DatasetInspection]:
    return ToolResponse(
        tool_name="inspect_dataset",
        ok=False,
        error=f"File not found: {source_uri!r}.",
        message=f"The dataset file does not exist at the specified path.",
    )


def run(request: InspectDatasetRequest, *, runs_root: str | Path = "runs") -> ToolResponse[DatasetInspection]:
    """Execute the inspect_dataset tool.

    Parameters
    ----------
    request:
        Validated :class:`InspectDatasetRequest` instance.
    runs_root:
        Root directory under which run artifact directories are created.
        Tools may not write outside this directory.

    Returns
    -------
    :class:`ToolResponse` with a :class:`DatasetInspection` payload on success,
    or ``ok=False`` with an ``error`` message on failure.
    """
    source_path = Path(request.source_uri)
    ext = source_path.suffix.lower()

    # Directories are valid for FTIR real-data mode; skip extension check for them.
    is_dir_target = source_path.is_dir() or request.source_format == "ftir_dir"
    if not is_dir_target and ext not in _SUPPORTED_EXTENSIONS:
        return _unsupported_format_response(request.source_uri, ext)

    if not source_path.exists():
        return _file_not_found_response(request.source_uri)

    dataset_id = request.dataset_id or source_path.stem
    run_id = make_run_id(slug=f"inspect-{dataset_id}"[:32])
    artifact_dir = ensure_run_dir(run_id, runs_root)

    started_at = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    try:
        if source_path.is_dir() or request.source_format == "ftir_dir":
            # Directory of real FTIR .txt measurement files
            _dataset, inspection = load_ftir_real(
                source_path,
                modality_override=request.modality_override,
            )
        elif ext == ".txt" or request.source_format == "composition_md":
            # Single composition-table .txt file (markdown table format)
            _dataset, inspection = load_ftir_composition(
                source_path,
                modality_override=request.modality_override,
            )
        else:
            _dataset, inspection = load_excel_nir(
                source_path,
                modality_override=request.modality_override,
                label_column=request.label_column,
                sample_id_column=request.sample_id_column,
            )
    except Exception as exc:  # noqa: BLE001
        return ToolResponse(
            tool_name="inspect_dataset",
            ok=False,
            error=str(exc),
            message=f"Failed to load dataset from {request.source_uri!r}.",
        )

    inspection_path = save_inspection_artifact(inspection, artifact_dir)
    completed_at = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()

    run_meta = RunMetadata(
        run_id=run_id,
        tool_name="inspect_dataset",
        dataset_id=dataset_id,
        status="completed",
        created_at=started_at,
        parameters={
            "source_uri": request.source_uri,
            "modality_override": request.modality_override,
            "label_column": request.label_column,
        },
    )

    inspection_artifact = artifact_ref(
        run_id,
        inspection_path.name,
        kind="inspection_json",
        label="Dataset inspection summary",
        mime_type="application/json",
        runs_root=runs_root,
    )

    return ToolResponse(
        tool_name="inspect_dataset",
        ok=True,
        payload=inspection,
        warnings=inspection.warnings,
        artifacts=(inspection_artifact,),
        metadata=run_meta,
        message=(
            f"Inspected {inspection.sample_count} samples × {inspection.feature_count} features "
            f"({inspection.modality or 'unknown modality'}). "
            f"Artifact saved to {inspection_artifact.uri}."
        ),
    )
