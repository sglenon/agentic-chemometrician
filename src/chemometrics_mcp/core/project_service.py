"""Folder-first project creation and reopening for parsed measurements."""

from __future__ import annotations

from io import BytesIO
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from chemometrics_contracts.project import ProjectManifest, ValidationIssue, WarningLevel
from chemometrics_mcp.core.ingestion import IngestionIssue, ParserRegistry, ParsedMeasurement
from chemometrics_mcp.core.manifests import apply_manifest_updates, build_draft_manifest
from chemometrics_mcp.core.project_store import ProjectStore, create_project_layout, fingerprint_source
from chemometrics_mcp.core.units import validate_axis, validate_signal


class ProjectService:
    """Persist manifests separately from the numeric arrays they describe."""

    def __init__(self, store: ProjectStore):
        self.store = store

    @classmethod
    def create(cls, source_root: str | Path, output_root: str | Path | None = None,
               project_id: str | None = None) -> "ProjectService":
        store = create_project_layout(source_root, output_root, project_id)
        service = cls(store)
        root = Path(source_root).resolve()
        metadata = store.read_json("project.json")
        registry = ParserRegistry()
        inventory_entries, ingestion_issues = registry.inventory_directory(root)
        # ``ParserRegistry`` deliberately knows only the conventional output
        # name.  A caller may select another output folder inside the source
        # tree, so enforce the project boundary here as well.
        inventory_entries = tuple(
            entry for entry in inventory_entries
            if not _is_within(entry.source_path, store.output_root)
        )
        ingestion_issues = tuple(
            issue for issue in ingestion_issues
            if issue.source_path is None or not _is_within(issue.source_path, store.output_root)
        )
        fingerprints = {record["relative_path"]: record for record in fingerprint_source(root, store.output_root)}
        inventory: list[dict[str, Any]] = []
        for entry in inventory_entries:
            relative = entry.relative_path.as_posix()
            fingerprint = fingerprints.get(relative)
            # Inventory and fingerprint use the same tree.  Missing fingerprints
            # are not fabricated: build_draft_manifest will retain a deterministic
            # fallback hash if a race removed an input file.
            inventory.append({
                "relative_path": relative,
                "sha256": fingerprint["sha256"] if fingerprint else "",
                "size_bytes": fingerprint["size_bytes"] if fingerprint else entry.size_bytes,
                "format": fingerprint["format"] if fingerprint else entry.extension.lstrip("."),
                "supported": entry.supported,
            })
        parsed: list[ParsedMeasurement] = []
        all_issues = list(ingestion_issues)
        for entry in inventory_entries:
            if entry.supported:
                measurements, issues = registry.parse(entry.source_path)
                parsed.extend(measurements)
                all_issues.extend(issues)
        manifest = build_draft_manifest(metadata["project_id"], str(root), inventory, parsed)
        converted_issues = tuple(_validation_issue(issue, root) for issue in all_issues)
        if converted_issues:
            manifest = manifest.model_copy(update={
                "unresolved_issues": manifest.unresolved_issues + converted_issues,
                "manifest_hash": None,
            })
            manifest = _with_hash(manifest)
        measurement_data, storage_metadata = service._write_measurements(
            root, manifest, parsed
        )
        if storage_metadata:
            measurements = tuple(
                row.model_copy(
                    update={
                        "metadata": {
                            **dict(row.metadata),
                            **storage_metadata.get(row.measurement_id, {}),
                        }
                    }
                )
                for row in manifest.measurements
            )
            manifest = _with_hash(
                manifest.model_copy(
                    update={"measurements": measurements, "manifest_hash": None}
                )
            )
        unit_issues = service._unit_validation_issues(manifest, measurement_data)
        if unit_issues:
            manifest = manifest.model_copy(
                update={
                    "unresolved_issues": manifest.unresolved_issues + unit_issues,
                    "manifest_hash": None,
                }
            )
            manifest = _with_hash(manifest)
        service._persist_manifest(manifest)
        return service

    @classmethod
    def open(cls, output_root: str | Path) -> "ProjectService":
        store = ProjectStore(output_root)
        service = cls(store)
        # Loading through the strict project model is deliberately part of opening a
        # project; malformed evidence must not enter downstream tools.
        service.get_manifest()
        return service

    def get_manifest(self) -> ProjectManifest:
        metadata = self.store.read_json("project.json")
        pointer = metadata.get("current_manifest", "manifests/current.json")
        manifest = ProjectManifest.model_validate(self.store.read_json(pointer))
        expected = _with_hash(
            manifest.model_copy(update={"manifest_hash": None})
        ).manifest_hash
        if not manifest.manifest_hash or manifest.manifest_hash != expected:
            raise ValueError("persisted manifest hash does not match its contents")
        return manifest

    def update_manifest(self, updates: Mapping[str, Any]) -> ProjectManifest:
        updated = apply_manifest_updates(self.get_manifest(), updates)
        retained = tuple(
            issue
            for issue in updated.unresolved_issues
            if issue.stage != "unit_validation"
        )
        measurement_data = {
            row.measurement_id: self.load_measurement(row.measurement_id)
            for row in updated.measurements
        }
        unit_issues = self._unit_validation_issues(updated, measurement_data)
        updated = updated.model_copy(
            update={
                "unresolved_issues": retained + unit_issues,
                "manifest_hash": None,
            }
        )
        updated = _with_hash(updated)
        self._persist_manifest(updated)
        return updated

    def get_summary(self) -> dict[str, Any]:
        manifest = self.get_manifest()
        return {
            "project_id": manifest.project_id,
            "manifest_version": manifest.version,
            "manifest_hash": manifest.manifest_hash,
            "source_root": manifest.source_root,
            "asset_count": len(manifest.assets),
            "sample_count": len(manifest.samples),
            "measurement_count": len(manifest.measurements),
            "measurement_ids": [row.measurement_id for row in manifest.measurements],
            "issues": [issue.model_dump(mode="json") for issue in manifest.unresolved_issues],
        }

    def load_measurement(self, measurement_id: str) -> dict[str, np.ndarray]:
        index = self.store.read_json("data/index.json")
        entry = index.get(measurement_id)
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"Measurement {measurement_id!r} lacks a hash-bound storage record"
            )
        relative = entry.get("path")
        if not relative:
            raise KeyError(f"Unknown measurement id: {measurement_id}")
        record = next(
            (
                item
                for item in self.get_manifest().measurements
                if item.measurement_id == measurement_id
            ),
            None,
        )
        if record is None:
            raise KeyError(f"Unknown measurement id: {measurement_id}")
        if (
            entry.get("sha256") != record.metadata.get("storage_sha256")
            or entry.get("content_hash") != record.metadata.get("content_hash")
        ):
            raise ValueError(
                f"Measurement storage index is not bound to the manifest: {measurement_id}"
            )
        path = self.store.path_for(relative)
        from chemometrics_mcp.core.project_store import sha256_file

        if sha256_file(path) != entry.get("sha256"):
            raise ValueError(
                f"Stored measurement payload hash mismatch: {measurement_id}"
            )
        with np.load(path, allow_pickle=False) as archive:
            result = {
                "axis": archive["axis"].copy(),
                "signal": archive["signal"].copy(),
            }
        if _array_content_hash(result["axis"], result["signal"]) != entry.get(
            "content_hash"
        ):
            raise ValueError(
                f"Stored measurement content hash mismatch: {measurement_id}"
            )
        return result

    def _write_measurements(self, source_root: Path, manifest: ProjectManifest,
                            parsed: list[ParsedMeasurement]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, str]]]:
        index: dict[str, dict[str, str]] = {}
        measurement_data: dict[str, dict[str, np.ndarray]] = {}
        storage_metadata: dict[str, dict[str, str]] = {}
        by_key: dict[tuple[str, str], list[ParsedMeasurement]] = {}
        for item in parsed:
            relative = item.source_path.resolve().relative_to(source_root).as_posix()
            by_key.setdefault((relative, item.measurement_name), []).append(item)
        # The draft manifest's deterministic ids include the per-asset parsed
        # position. Pop parsed rows in the same grouped order to handle duplicate
        # signal names without placing arrays in the manifest.
        positions: dict[tuple[str, str], int] = {}
        asset_path = {asset.asset_id: asset.relative_path for asset in manifest.assets}
        for measurement in manifest.measurements:
            relative = asset_path[measurement.asset_id]
            name = str(measurement.metadata.get("measurement_name", ""))
            key = (relative, name)
            position = positions.get(key, 0)
            rows = by_key.get(key, [])
            if position >= len(rows):
                continue
            positions[key] = position + 1
            item = rows[position]
            buffer = BytesIO()
            axis = np.asarray(item.axis)
            signal = np.asarray(item.signal)
            np.savez_compressed(buffer, axis=axis, signal=signal)
            payload = buffer.getvalue()
            relative_data = f"data/{measurement.measurement_id}.npz"
            self.store.write_bytes(relative_data, payload)
            storage_hash = hashlib.sha256(payload).hexdigest()
            content_hash = _array_content_hash(axis, signal)
            index[measurement.measurement_id] = {
                "path": relative_data,
                "sha256": storage_hash,
                "content_hash": content_hash,
            }
            storage_metadata[measurement.measurement_id] = {
                "storage_sha256": storage_hash,
                "content_hash": content_hash,
            }
            measurement_data[measurement.measurement_id] = {
                "axis": np.asarray(axis, dtype=float),
                "signal": np.asarray(signal, dtype=float),
            }
        self.store.write_json("data/index.json", index)
        return measurement_data, storage_metadata

    @staticmethod
    def _unit_validation_issues(
        manifest: ProjectManifest,
        measurement_data: Mapping[str, Mapping[str, np.ndarray]],
    ) -> tuple[ValidationIssue, ...]:
        unresolved_fields = {
            (str(issue.details.get("measurement_id")), str(issue.details.get("field")))
            for issue in manifest.unresolved_issues
            if issue.stage == "manifest"
        }
        issues: list[ValidationIssue] = []
        for measurement in manifest.measurements:
            values = measurement_data.get(measurement.measurement_id)
            if values is None:
                continue
            axis_kind = (
                None
                if (measurement.measurement_id, "axis_kind") in unresolved_fields
                else measurement.axis_kind.value
            )
            axis_unit = (
                None
                if (measurement.measurement_id, "axis_unit") in unresolved_fields
                else measurement.axis_unit
            )
            signal_kind = (
                None
                if (measurement.measurement_id, "signal_kind") in unresolved_fields
                else measurement.signal_kind.value
            )
            signal_unit = (
                None
                if (measurement.measurement_id, "signal_unit") in unresolved_fields
                else measurement.signal_unit
            )
            checks = (
                validate_axis(values["axis"], axis_kind, axis_unit),
                validate_signal(
                    values["signal"], signal_kind, signal_unit, quantitative=True
                ),
            )
            for check in checks:
                for issue in check.issues:
                    # Missing semantics are already represented once by the
                    # manifest firewall; keep physical/data issues here.
                    if issue.code.endswith("_unknown"):
                        continue
                    issues.append(
                        ValidationIssue(
                            code=issue.code,
                            message=issue.message,
                            level=(
                                WarningLevel.BLOCKER
                                if issue.severity == "blocker"
                                else WarningLevel.ADVISORY
                            ),
                            stage="unit_validation",
                            details={"measurement_id": measurement.measurement_id},
                        )
                    )
        return tuple(issues)

    def _persist_manifest(self, manifest: ProjectManifest) -> None:
        # Validate from serialized form before every write, preserving the strict
        # contract boundary even when callers construct an object themselves.
        strict = ProjectManifest.model_validate(manifest.model_dump(mode="json"))
        payload = strict.model_dump(mode="json")
        self.store.save_manifest(payload, "manifest", strict.version)
        self.store.write_json("manifests/current.json", payload)
        metadata = self.store.read_json("project.json")
        metadata.update({
            "current_manifest": "manifests/current.json",
            "current_manifest_hash": strict.manifest_hash,
            "current_manifest_version": strict.version,
        })
        self.store.write_json("project.json", metadata)


def _validation_issue(issue: IngestionIssue, root: Path) -> ValidationIssue:
    unsupported = issue.code == "unsupported_file"
    path = issue.source_path
    try:
        relative = path.resolve().relative_to(root).as_posix() if path else None
    except ValueError:
        relative = str(path) if path else None
    return ValidationIssue(
        code=issue.code,
        message=issue.message,
        level=WarningLevel.INFORMATION if unsupported else WarningLevel.BLOCKER,
        stage="parse",
        details={**issue.details, **({"relative_path": relative} if relative else {})},
    )


def _with_hash(manifest: ProjectManifest) -> ProjectManifest:
    from chemometrics_mcp.core.project_store import data_hash
    clean = manifest.model_copy(update={"manifest_hash": None})
    return clean.model_copy(update={"manifest_hash": data_hash(clean.canonical_dict())})


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _array_content_hash(axis: np.ndarray, signal: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name, value in (("axis", axis), ("signal", signal)):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()
