"""Format-neutral, conservative ingestion of measurement files.

Parsing deliberately does not interpret an axis range as a modality.  It only
records explicit format/header hints and leaves scientific interpretation to a
later stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import re
import zipfile
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class IngestionIssue:
    code: str
    message: str
    source_path: Path | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InventoryEntry:
    source_path: Path
    relative_path: Path
    extension: str
    supported: bool
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ParsedMeasurement:
    source_path: Path
    measurement_name: str
    axis: tuple[float, ...]
    signal: tuple[float, ...]
    parser_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    modality: str | None = None
    axis_kind: str | None = None
    axis_unit: str | None = None
    signal_kind: str | None = None
    signal_unit: str | None = None


class ParserRegistry:
    """Registry-like facade for the intentionally small set of safe readers."""

    supported_extensions = frozenset({
        ".csv", ".tsv", ".txt", ".asc", ".dat", ".xy", ".xye",
        ".xlsx", ".xls", ".jdx", ".dx",
    })

    def __init__(
        self,
        *,
        max_file_bytes: int = 256 * 1024 * 1024,
        max_rows: int = 2_000_000,
        max_signal_columns: int = 10_000,
        max_excel_sheets: int = 100,
        max_excel_uncompressed_bytes: int = 1024 * 1024 * 1024,
    ):
        self.max_file_bytes = max_file_bytes
        self.max_rows = max_rows
        self.max_signal_columns = max_signal_columns
        self.max_excel_sheets = max_excel_sheets
        self.max_excel_uncompressed_bytes = max_excel_uncompressed_bytes

    def inventory_directory(self, source_root: str | Path) -> tuple[tuple[InventoryEntry, ...], tuple[IngestionIssue, ...]]:
        root = Path(source_root)
        entries: list[InventoryEntry] = []
        issues: list[IngestionIssue] = []
        if not root.is_dir():
            return (), (IngestionIssue("not_a_directory", f"Not a directory: {root}", root),)
        for path in sorted(root.rglob("*")):
            if path.is_symlink() or not path.is_file() or "chemometrics-output" in path.parts:
                continue
            extension = path.suffix.lower()
            entry = InventoryEntry(path, path.relative_to(root), extension, extension in self.supported_extensions, path.stat().st_size)
            entries.append(entry)
            if not entry.supported:
                issues.append(IngestionIssue("unsupported_file", f"Unsupported file extension: {extension or '(none)'}", path))
        return tuple(entries), tuple(issues)

    def parse(self, source_path: str | Path) -> tuple[tuple[ParsedMeasurement, ...], tuple[IngestionIssue, ...]]:
        path = Path(source_path)
        if not path.is_file():
            return (), (IngestionIssue("file_not_found", f"File not found: {path}", path),)
        suffix = path.suffix.lower()
        if suffix not in self.supported_extensions:
            return (), (IngestionIssue("unsupported_file", f"Unsupported file extension: {suffix or '(none)'}", path),)
        if path.stat().st_size > self.max_file_bytes:
            return (), (
                IngestionIssue(
                    "resource_limit_exceeded",
                    "Input file exceeds the configured size limit.",
                    path,
                    {
                        "size_bytes": path.stat().st_size,
                        "max_file_bytes": self.max_file_bytes,
                    },
                ),
            )
        try:
            if suffix in {".xlsx", ".xls"}:
                return self._parse_excel(path)
            if suffix in {".jdx", ".dx"}:
                return self._parse_jcamp(path)
            return self._parse_text(path, suffix)
        except (OSError, UnicodeDecodeError, ValueError, csv.Error, ImportError) as exc:
            return (), (IngestionIssue("malformed_input", f"Could not parse {path.name}: {exc}", path),)

    def ingest_directory(self, source_root: str | Path) -> tuple[tuple[ParsedMeasurement, ...], tuple[IngestionIssue, ...]]:
        entries, issues = self.inventory_directory(source_root)
        parsed: list[ParsedMeasurement] = []
        all_issues = list(issues)
        for entry in entries:
            if entry.supported:
                measurements, parse_issues = self.parse(entry.source_path)
                parsed.extend(measurements)
                all_issues.extend(parse_issues)
        return tuple(parsed), tuple(all_issues)

    def _parse_text(self, path: Path, suffix: str) -> tuple[tuple[ParsedMeasurement, ...], tuple[IngestionIssue, ...]]:
        headers: dict[str, str] = {}
        lines: list[str] = []
        for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("##") and "=" in line:
                key, value = line[2:].split("=", 1)
                headers[key.strip().upper()] = value.strip()
                continue
            if line.startswith("#"):
                continue
            lines.append(line)
        if not lines:
            return (), (IngestionIssue("malformed_input", "No numeric data rows found", path),)
        delimiter = "\t" if suffix == ".tsv" else ("," if suffix == ".csv" else None)
        rows = [self._tokens(line, delimiter) for line in lines]
        header: list[str] | None = None
        try:
            [float(token) for token in rows[0]]
        except ValueError:
            header = rows.pop(0)
        if not rows:
            return (), (IngestionIssue("malformed_input", "Expected an axis and at least one signal column", path),)
        if len(rows[0]) == 1:
            # Single-column Y-only file — axis must be reconstructed from JCAMP geometry headers.
            return self._parse_text_yonly(path, rows, headers)
        if any(len(row) < 2 for row in rows):
            return (), (IngestionIssue("malformed_input", "Expected an axis and at least one signal column", path),)
        if (
            len(rows) > self.max_rows
            or len(rows[0]) - 1 > self.max_signal_columns
        ):
            return (), (
                IngestionIssue(
                    "resource_limit_exceeded",
                    "Input table exceeds configured row/column limits.",
                    path,
                ),
            )
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            return (), (IngestionIssue("malformed_input", "Inconsistent number of columns", path),)
        try:
            numeric = [[float(token) for token in row] for row in rows]
        except ValueError:
            return (), (IngestionIssue("malformed_input", "Non-numeric value in data rows", path),)
        axis = tuple(row[0] for row in numeric)
        names = header[1:] if header and len(header) == width else [path.stem if width == 2 else f"{path.stem}_{i}" for i in range(1, width)]
        x_unit = headers.get("XUNITS")
        y_unit = headers.get("YUNITS")
        x_token = (x_unit or "").strip().lower().replace(" ", "")
        y_token = (y_unit or "").strip().lower().replace(" ", "")
        axis_kind = (
            "wavenumber" if x_token in {"cm-1", "cm^-1", "1/cm"}
            else "wavelength" if x_token in {"nm", "um", "µm"}
            else "mass_to_charge" if x_token in {"m/z", "mz"}
            else None
        )
        signal_kind = (
            "percent_transmittance" if y_token in {"%t", "percenttransmittance"}
            else "absorbance" if y_token in {"abs", "absorbance"}
            else "reflectance" if y_token in {"%r", "reflectance"}
            else "counts" if y_token in {"count", "counts"}
            else None
        )
        metadata: dict[str, Any] = {}
        if header:
            metadata["header"] = header
        if headers:
            metadata["headers"] = headers
        measurements = tuple(
            ParsedMeasurement(
                path,
                str(names[i - 1]),
                axis,
                tuple(row[i] for row in numeric),
                "delimited-text-v1",
                dict(metadata),
                axis_kind=axis_kind,
                axis_unit=x_unit,
                signal_kind=signal_kind,
                signal_unit=y_unit,
            )
            for i in range(1, width)
        )
        return measurements, ()

    def _parse_text_yonly(
        self,
        path: Path,
        rows: list[list[str]],
        headers: dict[str, str],
    ) -> tuple[tuple[ParsedMeasurement, ...], tuple[IngestionIssue, ...]]:
        """Parse single-column Y-only files, reconstructing the axis from JCAMP geometry headers.

        Requires FIRSTX and NPOINTS, plus either LASTX (preferred) or DELTAX.
        If XFACTOR/YFACTOR are present and numeric they are applied; non-numeric
        values (e.g. the placeholder ``Anything`` used by some exporters) are
        silently ignored.  Descending axes (FIRSTX > LASTX) are preserved as-is —
        do not flip or sort.
        """
        # --- Parse signal values ---
        try:
            signal_list = [float(row[0]) for row in rows]
        except ValueError:
            return (), (IngestionIssue("malformed_input", "Non-numeric value in Y-only data rows", path),)

        n_actual = len(signal_list)

        if n_actual > self.max_rows:
            return (), (
                IngestionIssue(
                    "resource_limit_exceeded",
                    "Input table exceeds configured row/column limits.",
                    path,
                ),
            )

        # --- Validate NPOINTS ---
        npoints_str = headers.get("NPOINTS")
        if npoints_str is None:
            return (), (
                IngestionIssue(
                    "jcamp_missing_geometry",
                    "Single-column Y-only file requires a NPOINTS header for axis reconstruction",
                    path,
                ),
            )
        try:
            npoints = int(float(npoints_str))
        except ValueError:
            return (), (
                IngestionIssue(
                    "jcamp_missing_geometry",
                    f"NPOINTS header is not numeric: {npoints_str!r}",
                    path,
                ),
            )

        if npoints != n_actual:
            return (), (
                IngestionIssue(
                    "jcamp_npoints_mismatch",
                    f"NPOINTS={npoints} declared in header but {n_actual} data rows found",
                    path,
                    {"npoints_header": npoints, "rows_found": n_actual},
                ),
            )

        # --- Require FIRSTX ---
        firstx_str = headers.get("FIRSTX")
        if firstx_str is None:
            return (), (
                IngestionIssue(
                    "jcamp_missing_geometry",
                    "Single-column Y-only file requires a FIRSTX header for axis reconstruction",
                    path,
                ),
            )
        try:
            firstx = float(firstx_str)
        except ValueError:
            return (), (
                IngestionIssue(
                    "jcamp_missing_geometry",
                    f"FIRSTX header is not numeric: {firstx_str!r}",
                    path,
                ),
            )

        # --- Reconstruct axis from LASTX (preferred) or DELTAX ---
        lastx_str = headers.get("LASTX")
        deltax_str = headers.get("DELTAX")

        if lastx_str is not None:
            try:
                lastx = float(lastx_str)
                axis_array = np.linspace(firstx, lastx, npoints)
            except ValueError:
                axis_array = None
        else:
            axis_array = None

        if axis_array is None and deltax_str is not None:
            try:
                deltax = float(deltax_str)
                axis_array = firstx + deltax * np.arange(npoints)
            except ValueError:
                axis_array = None

        if axis_array is None:
            return (), (
                IngestionIssue(
                    "jcamp_missing_geometry",
                    "Single-column Y-only file requires LASTX or DELTAX for axis reconstruction "
                    "(FIRSTX was found but neither LASTX nor DELTAX is present/numeric)",
                    path,
                ),
            )

        # --- Apply XFACTOR / YFACTOR if present and numeric (skip silently if non-numeric) ---
        xfactor_str = headers.get("XFACTOR")
        if xfactor_str is not None:
            try:
                axis_array = axis_array * float(xfactor_str)
            except ValueError:
                pass  # non-numeric placeholder — ignore

        signal_array = np.array(signal_list, dtype=float)
        yfactor_str = headers.get("YFACTOR")
        if yfactor_str is not None:
            try:
                signal_array = signal_array * float(yfactor_str)
            except ValueError:
                pass  # non-numeric placeholder — ignore

        # --- Unit / kind mapping (mirrors the 2-column branch exactly) ---
        x_unit = headers.get("XUNITS")
        y_unit = headers.get("YUNITS")
        x_token = (x_unit or "").strip().lower().replace(" ", "")
        y_token = (y_unit or "").strip().lower().replace(" ", "")
        axis_kind = (
            "wavenumber" if x_token in {"cm-1", "cm^-1", "1/cm"}
            else "wavelength" if x_token in {"nm", "um", "µm"}
            else "mass_to_charge" if x_token in {"m/z", "mz"}
            else None
        )
        signal_kind = (
            "percent_transmittance" if y_token in {"%t", "percenttransmittance"}
            else "absorbance" if y_token in {"abs", "absorbance"}
            else "reflectance" if y_token in {"%r", "reflectance"}
            else "counts" if y_token in {"count", "counts"}
            else None
        )

        metadata: dict[str, Any] = {"headers": headers} if headers else {}

        return (
            ParsedMeasurement(
                path,
                path.stem,
                tuple(float(x) for x in axis_array),
                tuple(float(y) for y in signal_array),
                "delimited-text-yonly-v1",
                metadata,
                axis_kind=axis_kind,
                axis_unit=x_unit,
                signal_kind=signal_kind,
                signal_unit=y_unit,
            ),
        ), ()

    @staticmethod
    def _tokens(line: str, delimiter: str | None) -> list[str]:
        return [
            value.strip()
            for value in (
                line.split(delimiter)
                if delimiter
                else re.split(r"[\s,;]+", line.strip())
            )
        ]

    def _parse_excel(self, path: Path) -> tuple[tuple[ParsedMeasurement, ...], tuple[IngestionIssue, ...]]:
        if path.suffix.lower() == ".xlsx" and zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                uncompressed = sum(
                    item.file_size for item in archive.infolist()
                )
                if uncompressed > self.max_excel_uncompressed_bytes:
                    return (), (
                        IngestionIssue(
                            "resource_limit_exceeded",
                            "Excel archive exceeds the configured uncompressed-size limit.",
                            path,
                        ),
                    )
        workbook = pd.ExcelFile(path)
        if len(workbook.sheet_names) > self.max_excel_sheets:
            return (), (
                IngestionIssue(
                    "resource_limit_exceeded",
                    "Workbook exceeds the configured sheet limit.",
                    path,
                ),
            )
        metadata_name = next(
            (name for name in workbook.sheet_names if "metadata" in name.lower()),
            None,
        )
        spectra_name = next(
            (
                name
                for name in workbook.sheet_names
                if "spectra" in name.lower() and "metadata" not in name.lower()
            ),
            None,
        )
        if metadata_name and spectra_name:
            metadata = workbook.parse(metadata_name)
            spectra = workbook.parse(spectra_name)
            return self._measurements_from_frame(path, spectra, "excel-spectra-metadata-v1", metadata)
        result: list[ParsedMeasurement] = []
        issues: list[IngestionIssue] = []
        for sheet_name in workbook.sheet_names:
            frame = workbook.parse(sheet_name)
            measurements, frame_issues = self._measurements_from_frame(path, frame, "excel-generic-v1", None, sheet_name)
            result.extend(measurements)
            issues.extend(frame_issues)
        return tuple(result), tuple(issues)

    def _measurements_from_frame(self, path: Path, frame: pd.DataFrame, parser_id: str, metadata_frame: pd.DataFrame | None, sheet_name: str | None = None) -> tuple[tuple[ParsedMeasurement, ...], tuple[IngestionIssue, ...]]:
        if frame.shape[1] < 2 or frame.empty:
            return (), (IngestionIssue("malformed_input", "Expected an axis and at least one signal column", path, {"sheet": sheet_name}),)
        if (
            frame.shape[0] > self.max_rows
            or frame.shape[1] - 1 > self.max_signal_columns
        ):
            return (), (
                IngestionIssue(
                    "resource_limit_exceeded",
                    "Worksheet exceeds configured row/column limits.",
                    path,
                    {"sheet": sheet_name},
                ),
            )
        try:
            axis = tuple(float(value) for value in frame.iloc[:, 0])
            signals = [tuple(float(value) for value in frame.iloc[:, index]) for index in range(1, frame.shape[1])]
        except (TypeError, ValueError):
            return (), (IngestionIssue("malformed_input", "Excel spectra must contain numeric columns", path, {"sheet": sheet_name}),)
        records = metadata_frame.to_dict(orient="records") if metadata_frame is not None else []
        measurements = []
        for position, signal in enumerate(signals):
            name = str(frame.columns[position + 1])
            meta: dict[str, Any] = {"sheet": sheet_name} if sheet_name else {}
            if position < len(records):
                meta["sample_metadata"] = records[position]
            measurements.append(ParsedMeasurement(path, name, axis, signal, parser_id, meta))
        return tuple(measurements), ()

    def _parse_jcamp(self, path: Path) -> tuple[tuple[ParsedMeasurement, ...], tuple[IngestionIssue, ...]]:
        headers: dict[str, str] = {}
        pairs: list[tuple[float, float]] = []
        data_started = False
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("##"):
                key, _, value = line[2:].partition("=")
                key = key.strip().upper()
                headers[key] = value.strip()
                if key in {"XYDATA", "DATATABLE"} and "++" in value:
                    return (), (IngestionIssue("unsupported_jcamp_encoding", "Compressed JCAMP encodings are not supported; provide explicit XY pairs", path),)
                data_started = key in {"PEAK TABLE", "XYDATA", "DATATABLE"} or data_started
                continue
            if data_started or not line.startswith("##"):
                tokens = re.split(r"[\s,;]+", line)
                if len(tokens) != 2:
                    return (), (IngestionIssue("unsupported_jcamp_encoding", "JCAMP reader supports only explicit two-value XY rows", path),)
                try:
                    pairs.append((float(tokens[0]), float(tokens[1])))
                except ValueError:
                    return (), (IngestionIssue("malformed_input", "Invalid numeric JCAMP XY row", path),)
        if not pairs:
            return (), (IngestionIssue("malformed_input", "No explicit JCAMP XY pairs found", path),)
        return (ParsedMeasurement(path, headers.get("TITLE", path.stem), tuple(x for x, _ in pairs), tuple(y for _, y in pairs), "jcamp-explicit-xy-v1", {"headers": headers}, axis_unit=headers.get("XUNITS"), signal_unit=headers.get("YUNITS")),), ()
