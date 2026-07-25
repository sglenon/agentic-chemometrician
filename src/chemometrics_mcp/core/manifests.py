"""Deterministic construction and conservative validation of project manifests."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from chemometrics_contracts.project import (
    AxisKind, MeasurementRecord, MeasurementRole, Modality, ProjectManifest,
    SampleRecord, SignalKind, SourceAsset, ValidationIssue, WarningLevel,
)
from chemometrics_mcp.core.project_store import data_hash
from chemometrics_mcp.core.manifest_hints import ManifestHints


def _get(value: Mapping[str, Any] | object | None, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default) if value is not None else default


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return result or "unnamed"


def _id(kind: str, value: str) -> str:
    return f"{kind}-{data_hash(value)[:16]}"


def _enum_or(value: Any, enum: type, fallback: Any) -> Any:
    try:
        return enum(value) if value is not None else fallback
    except ValueError:
        return fallback


def _manifest_hash(manifest: ProjectManifest) -> str:
    return data_hash(manifest.model_copy(update={"manifest_hash": None}).canonical_dict())


def _issue(code: str, message: str, measurement_id: str, field: str) -> ValidationIssue:
    return ValidationIssue(
        code=code, message=message, level=WarningLevel.BLOCKER, stage="manifest",
        details={"measurement_id": measurement_id, "field": field},
    )


def build_draft_manifest(
    project_id: str,
    source_root: str,
    inventory: Sequence[Mapping[str, Any] | object],
    parsed_measurements: Sequence[Mapping[str, Any] | object] = (),
    *,
    hints: ManifestHints | None = None,
    infer_roles_from_filenames: bool = False,
) -> ProjectManifest:
    """Build a draft without guessing scientific roles or group hierarchy.

    Parsed measurements are associated by relative path when available, then by
    input order.  Filename stems only supply clearly provisional sample ids.

    When *hints* is provided (a :class:`~chemometrics_mcp.core.manifest_hints.ManifestHints`
    loaded from a ``manifest_hints.json`` in the source root), its entries are applied
    directly to matching samples and measurements before the manifest hash is computed.
    Declared hints (``provenance="declared"``) are treated as authoritative user-supplied
    truth and applied silently.

    When *infer_roles_from_filenames* is ``True``, an additional filename-convention
    heuristic pass is run for any stem not already covered by *hints*.  Each inferred
    role/composition is tagged ``roles_inferred_from_filename=True`` in the sample
    metadata alongside a human-readable ``hints_note`` so the scientist knows to
    confirm.  This tier is opt-in and never applied automatically.

    When both parameters are at their defaults (``hints=None``,
    ``infer_roles_from_filenames=False``) the output is byte-identical to the
    pre-hints behaviour.
    """
    root = Path(source_root).resolve()
    parsed_by_path: dict[str, list[Mapping[str, Any] | object]] = {}
    unassigned_parsed: list[Mapping[str, Any] | object] = []
    for parsed in parsed_measurements:
        raw_relative = _get(parsed, "relative_path")
        if raw_relative:
            relative = str(raw_relative).replace("\\", "/")
        else:
            source_path = _get(parsed, "source_path")
            if source_path is None:
                unassigned_parsed.append(parsed)
                continue
            candidate = Path(source_path)
            try:
                relative = candidate.resolve().relative_to(root).as_posix()
            except (OSError, ValueError):
                relative = candidate.as_posix()
        parsed_by_path.setdefault(relative, []).append(parsed)
    assets: list[SourceAsset] = []
    samples: list[SampleRecord] = []
    measurements: list[MeasurementRecord] = []
    issues: list[ValidationIssue] = []
    used_samples: set[str] = set()
    sorted_inventory = sorted(
        inventory, key=lambda x: str(_get(x, "relative_path", ""))
    )
    if unassigned_parsed:
        if len(sorted_inventory) == 1:
            only_path = str(_get(sorted_inventory[0], "relative_path", "")).replace(
                "\\", "/"
            )
            parsed_by_path.setdefault(only_path, []).extend(unassigned_parsed)
        elif len(unassigned_parsed) == len(sorted_inventory):
            for item, parsed in zip(sorted_inventory, unassigned_parsed):
                path_key = str(_get(item, "relative_path", "")).replace("\\", "/")
                parsed_by_path.setdefault(path_key, []).append(parsed)
        else:
            raise ValueError(
                "parsed measurements without source paths cannot be associated "
                "unambiguously with inventory assets"
            )
    for item in sorted_inventory:
        relative_path = str(_get(item, "relative_path", "")).replace("\\", "/")
        if not relative_path:
            raise ValueError("inventory entries require relative_path")
        digest = str(_get(item, "sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            digest = data_hash({"relative_path": relative_path, "size_bytes": _get(item, "size_bytes", 0)})
        asset_id = _id("asset", relative_path)
        parsed_for_asset = parsed_by_path.get(relative_path, [])
        parser_id = (
            str(_get(parsed_for_asset[0], "parser_id"))
            if parsed_for_asset
            else str(_get(item, "parser_id", "unparsed"))
        )
        assets.append(SourceAsset(
            asset_id=asset_id, relative_path=relative_path, sha256=digest,
            size_bytes=int(_get(item, "size_bytes", 0) or 0),
            format=str(_get(item, "format", Path(relative_path).suffix.lstrip(".") or "unknown")),
            parser_id=parser_id, media_type=_get(item, "media_type"),
        ))
        if not parsed_for_asset:
            level = (
                WarningLevel.INFORMATION
                if _get(item, "supported", True) is False
                else WarningLevel.BLOCKER
            )
            issues.append(
                ValidationIssue(
                    code=(
                        "unsupported_asset"
                        if level == WarningLevel.INFORMATION
                        else "asset_not_parsed"
                    ),
                    message=(
                        "Asset is unsupported and was retained for inventory only."
                        if level == WarningLevel.INFORMATION
                        else "Supported asset did not yield a parsed measurement."
                    ),
                    level=level,
                    stage="manifest",
                    details={"asset_id": asset_id, "relative_path": relative_path},
                )
            )
            continue

        for parsed_index, parsed in enumerate(parsed_for_asset):
            measurement_name = str(
                _get(parsed, "measurement_name", Path(relative_path).stem)
            )
            stem = _slug(measurement_name)
            sample_id = f"sample-{stem}"
            if sample_id in used_samples:
                sample_id = f"{sample_id}-{data_hash(f'{relative_path}:{parsed_index}')[:8]}"
            used_samples.add(sample_id)
            sample_metadata = dict(_get(parsed, "metadata", {}) or {})
            sample_metadata.update(
                {
                    "provisional_sample_id": True,
                    "filename_stem": Path(relative_path).stem,
                    "measurement_name": measurement_name,
                }
            )
            samples.append(
                SampleRecord(
                    sample_id=sample_id,
                    role=MeasurementRole.SAMPLE,
                    metadata=sample_metadata,
                )
            )
            measurement_id = _id(
                "measurement", f"{relative_path}:{measurement_name}:{parsed_index}"
            )
            raw_modality = _get(parsed, "modality")
            raw_axis_kind = _get(parsed, "axis_kind")
            raw_axis_unit = _get(parsed, "axis_unit")
            raw_signal_kind = _get(parsed, "signal_kind")
            raw_signal_unit = _get(parsed, "signal_unit")
            measurement = MeasurementRecord(
                measurement_id=measurement_id,
                asset_id=asset_id,
                sample_id=sample_id,
                modality=_enum_or(raw_modality, Modality, Modality.OTHER),
                axis_kind=_enum_or(raw_axis_kind, AxisKind, AxisKind.INDEX),
                axis_unit=raw_axis_unit,
                signal_kind=_enum_or(
                    raw_signal_kind, SignalKind, SignalKind.INTENSITY
                ),
                signal_unit=raw_signal_unit,
                role=MeasurementRole.UNKNOWN,
                metadata={
                    "measurement_name": measurement_name,
                    "parser_id": parser_id,
                },
            )
            measurements.append(measurement)
            for field, raw, code in (
                ("modality", raw_modality, "missing_modality"),
                ("axis_kind", raw_axis_kind, "missing_axis_kind"),
                ("axis_unit", raw_axis_unit, "missing_axis_unit"),
                ("signal_kind", raw_signal_kind, "missing_signal_kind"),
                ("signal_unit", raw_signal_unit, "missing_signal_unit"),
            ):
                if raw is None or raw == "":
                    issues.append(
                        _issue(
                            code,
                            f"Measurement requires explicit {field}.",
                            measurement_id,
                            field,
                        )
                    )

    # --------------------------------------------------------------------------
    # Hints application — runs BEFORE the manifest hash is computed so the hash
    # reflects the actual (possibly hint-informed) state of the draft.
    # Default path (hints=None, infer_roles_from_filenames=False): no-op.
    # --------------------------------------------------------------------------
    if hints is not None or infer_roles_from_filenames:
        # Build a combined hints lookup: declared hints take priority; filename
        # heuristic fills in only stems not already covered by declared hints.
        effective_hints: ManifestHints | None = hints
        if infer_roles_from_filenames:
            from chemometrics_mcp.core.manifest_hints import infer_hints_from_filenames
            all_stems = [Path(s.metadata.get("filename_stem", s.sample_id)).stem for s in samples]
            inferred = infer_hints_from_filenames(all_stems)
            if hints is None:
                effective_hints = inferred
            else:
                # Merge: declared wins on any key that exists in both.
                merged_entries = {**inferred.entries, **hints.entries}
                from chemometrics_mcp.core.manifest_hints import ManifestHints as _MH
                effective_hints = _MH(entries=merged_entries, source=hints.source)

        if effective_hints is not None:
            updated_samples: list[SampleRecord] = []
            for sample in samples:
                stem = sample.metadata.get("filename_stem", "")
                hint = effective_hints.lookup(str(stem))
                if hint is None:
                    updated_samples.append(sample)
                    continue
                patch: dict[str, Any] = {}
                if hint.role is not None:
                    patch["role"] = _enum_or(hint.role, MeasurementRole, MeasurementRole.UNKNOWN)
                if hint.composition is not None:
                    patch["composition"] = hint.composition
                # Build merged metadata first so reference_name and advisory markers
                # end up in the same dict that goes into model_validate.
                meta_patch: dict[str, Any] = dict(sample.metadata)
                if hint.reference_name is not None:
                    meta_patch["reference_name"] = hint.reference_name
                if hint.provenance == "inferred":
                    meta_patch["roles_inferred_from_filename"] = True
                    if hint.note:
                        meta_patch["hints_note"] = hint.note
                elif hint.note:
                    meta_patch["hints_note"] = hint.note
                updated_samples.append(
                    sample.__class__.model_validate({
                        **sample.model_dump(mode="json"),
                        **patch,
                        "metadata": meta_patch,
                    })
                )
            samples = updated_samples

            # Apply measurement role from hint (measurement role mirrors sample role
            # for declared hints; advisory tag is on the sample record).
            sample_role_by_id = {s.sample_id: s.role for s in samples}
            updated_measurements: list[MeasurementRecord] = []
            for measurement in measurements:
                stem = ""
                # Derive stem from the asset's relative path via the asset lookup built above.
                for asset in assets:
                    if asset.asset_id == measurement.asset_id:
                        stem = Path(asset.relative_path).stem
                        break
                hint = effective_hints.lookup(stem)
                if hint is None or hint.role is None:
                    updated_measurements.append(measurement)
                    continue
                new_role = _enum_or(hint.role, MeasurementRole, MeasurementRole.UNKNOWN)
                meta_patch = dict(measurement.metadata)
                if hint.provenance == "inferred":
                    meta_patch["roles_inferred_from_filename"] = True
                    if hint.note:
                        meta_patch["hints_note"] = hint.note
                updated_measurements.append(
                    measurement.__class__.model_validate({
                        **measurement.model_dump(mode="json"),
                        "role": new_role.value,
                        "metadata": meta_patch,
                    })
                )
            measurements = updated_measurements

    manifest = ProjectManifest(project_id=project_id, source_root=source_root, assets=tuple(assets), samples=tuple(samples),
                               measurements=tuple(measurements), unresolved_issues=tuple(issues))
    return manifest.model_copy(update={"manifest_hash": _manifest_hash(manifest)})


_MEASUREMENT_FIELDS = frozenset({"asset_id", "sample_id", "modality", "representation", "axis_kind", "axis_unit", "signal_kind", "signal_unit", "role", "metadata"})
_SAMPLE_FIELDS = frozenset({"preparation_id", "technical_replicate_id", "batch_id", "instrument_id", "operator_id", "role", "physical_state", "composition", "metadata"})


def _update_rows(rows: tuple[Any, ...], updates: Mapping[str, Mapping[str, Any]], allowed: frozenset[str], label: str) -> tuple[Any, ...]:
    indexed = {row.measurement_id if label == "measurement" else row.sample_id: row for row in rows}
    unknown = set(updates) - set(indexed)
    if unknown:
        raise ValueError(f"Unknown {label} id(s): {sorted(unknown)}")
    result = []
    for identifier, row in indexed.items():
        patch = dict(updates.get(identifier, {}))
        invalid = set(patch) - allowed
        if invalid:
            raise ValueError(f"Unknown {label} field(s): {sorted(invalid)}")
        if "metadata" in patch:
            metadata_patch = patch["metadata"]
            if not isinstance(metadata_patch, Mapping):
                raise ValueError(f"{label} metadata update must be a mapping")
            if label == "measurement":
                reserved = {"storage_sha256", "content_hash"}
                attempted = reserved.intersection(metadata_patch)
                if attempted:
                    raise ValueError(
                        "Measurement storage metadata is managed internally and "
                        f"cannot be updated: {sorted(attempted)}"
                    )
            patch["metadata"] = {
                **dict(row.metadata),
                **dict(metadata_patch),
            }
        # ``model_copy(update=...)`` intentionally skips Pydantic validation;
        # manifest edits are an external boundary, so validate enum values and
        # all field types before admitting them into the next immutable model.
        result.append(row.__class__.model_validate({**row.model_dump(mode="json"), **patch}))
    return tuple(result)


def apply_manifest_updates(manifest: ProjectManifest, updates: Mapping[str, Any]) -> ProjectManifest:
    """Apply only explicit record patches and produce a new hashed version."""
    invalid = set(updates) - {"measurements", "samples"}
    if invalid:
        raise ValueError(f"Unknown manifest update field(s): {sorted(invalid)}")
    measurements = _update_rows(manifest.measurements, updates.get("measurements", {}), _MEASUREMENT_FIELDS, "measurement")
    samples = _update_rows(manifest.samples, updates.get("samples", {}), _SAMPLE_FIELDS, "sample")
    resolved = {(mid, field) for mid, patch in updates.get("measurements", {}).items() for field, value in patch.items() if value not in (None, "")}
    issues = tuple(issue for issue in manifest.unresolved_issues if (issue.details.get("measurement_id"), issue.details.get("field")) not in resolved)
    try:
        version = str(int(manifest.version) + 1)
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest version must be integer-like") from exc
    changed = manifest.model_copy(update={"measurements": measurements, "samples": samples, "unresolved_issues": issues, "version": version, "manifest_hash": None})
    return changed.model_copy(update={"manifest_hash": _manifest_hash(changed)})


def validate_manifest(manifest: ProjectManifest, *, quantitative_grouped: bool = False,
                      composition_total: float = 1.0, composition_tolerance: float = 1e-6) -> tuple[ValidationIssue, ...]:
    """Return deterministic validation issues without silently modifying records."""
    issues = list(manifest.unresolved_issues)
    asset_ids = [asset.asset_id for asset in manifest.assets]
    sample_ids = [sample.sample_id for sample in manifest.samples]
    if len(asset_ids) != len(set(asset_ids)) or len(sample_ids) != len(set(sample_ids)):
        issues.append(ValidationIssue(code="duplicate_ids", message="Asset and sample IDs must be unique.", level=WarningLevel.BLOCKER, stage="manifest"))
    measurement_ids = [row.measurement_id for row in manifest.measurements]
    if len(measurement_ids) != len(set(measurement_ids)):
        issues.append(ValidationIssue(code="duplicate_measurement_ids", message="Measurement IDs must be unique.", level=WarningLevel.BLOCKER, stage="manifest"))
    sample_by_id = {sample.sample_id: sample for sample in manifest.samples}
    for row in manifest.measurements:
        if row.asset_id not in set(asset_ids) or row.sample_id not in sample_by_id:
            issues.append(ValidationIssue(code="broken_measurement_reference", message="Measurement references an unknown asset or sample.", level=WarningLevel.BLOCKER, stage="manifest", details={"measurement_id": row.measurement_id}))
            continue
        sample = sample_by_id[row.sample_id]
        if quantitative_grouped and not sample.preparation_id:
            issues.append(ValidationIssue(code="missing_preparation_id", message="Quantitative grouped work requires preparation_id.", level=WarningLevel.BLOCKER, stage="manifest", details={"sample_id": sample.sample_id}))
        numeric = [value for value in sample.composition.values() if isinstance(value, (int, float)) and not isinstance(value, bool)]
        if sample.composition and (not numeric or abs(sum(numeric) - composition_total) > composition_tolerance):
            issues.append(ValidationIssue(code="composition_not_closed", message="Sample composition does not close to the configured total.", level=WarningLevel.BLOCKER, stage="manifest", details={"sample_id": sample.sample_id}))
        if row.role == MeasurementRole.REFERENCE and sample.physical_state and row.metadata.get("physical_state") and sample.physical_state != row.metadata["physical_state"]:
            issues.append(ValidationIssue(code="reference_physical_state_mismatch", message="Reference measurement and sample physical states differ.", level=WarningLevel.ADVISORY, stage="manifest", details={"measurement_id": row.measurement_id}))
    return tuple(issues)


def finalize_manifest(manifest: ProjectManifest, **validation_options: Any) -> ProjectManifest:
    """Return a hash-stable final manifest only when no blocker remains."""
    issues = validate_manifest(manifest, **validation_options)
    if any(issue.level == WarningLevel.BLOCKER for issue in issues):
        raise ValueError("Cannot finalize manifest while blocker issues remain")
    finalized = manifest.model_copy(update={"unresolved_issues": issues, "manifest_hash": None})
    return finalized.model_copy(update={"manifest_hash": _manifest_hash(finalized)})
