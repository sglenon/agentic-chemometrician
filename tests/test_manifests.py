import pytest

from chemometrics_contracts.project import MeasurementRole
from chemometrics_mcp.core.manifests import (
    apply_manifest_updates, build_draft_manifest, finalize_manifest, validate_manifest,
)


INVENTORY = [{"relative_path": "raw/a.csv", "sha256": "a" * 64, "size_bytes": 4, "format": "csv", "parser_id": "csv"}]
PARSED = [{"modality": "nir", "axis_kind": "wavelength", "axis_unit": "nm", "signal_kind": "absorbance", "signal_unit": "au"}]


def test_draft_ids_are_deterministic():
    one = build_draft_manifest("p", ".", INVENTORY, PARSED)
    two = build_draft_manifest("p", ".", INVENTORY, PARSED)
    assert one == two
    assert one.measurements[0].sample_id.startswith("sample-a")


def test_multiple_measurements_in_one_asset_are_preserved(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    asset = source / "multi.csv"
    asset.write_text("x,a,b\n1,2,3\n")
    parsed = [
        {
            "source_path": asset,
            "measurement_name": name,
            "modality": "nir",
            "axis_kind": "wavelength",
            "axis_unit": "nm",
            "signal_kind": "absorbance",
            "signal_unit": "au",
            "parser_id": "csv",
        }
        for name in ("a", "b")
    ]
    manifest = build_draft_manifest(
        "p",
        str(source),
        [{**INVENTORY[0], "relative_path": "multi.csv"}],
        parsed,
    )
    assert len(manifest.assets) == 1
    assert len(manifest.measurements) == 2
    assert len(manifest.samples) == 2


def test_missing_parse_fields_are_unresolved_blockers_and_labels_do_not_create_groups():
    draft = build_draft_manifest("p", ".", INVENTORY, [{"target_label": "batch-17"}])
    assert {issue.code for issue in draft.unresolved_issues} >= {"missing_modality", "missing_axis_kind", "missing_axis_unit", "missing_signal_kind", "missing_signal_unit"}
    assert draft.samples[0].preparation_id is None
    assert draft.measurements[0].role == MeasurementRole.UNKNOWN


def test_explicit_updates_resolve_draft_fields_and_change_version_hash():
    draft = build_draft_manifest("p", ".", INVENTORY, [{}])
    mid = draft.measurements[0].measurement_id
    resolved = apply_manifest_updates(draft, {"measurements": {mid: {"modality": "nir", "axis_kind": "wavelength", "axis_unit": "nm", "signal_kind": "absorbance", "signal_unit": "au"}}})
    assert resolved.version == "3"
    assert resolved.manifest_hash != draft.manifest_hash
    assert not resolved.unresolved_issues
    assert finalize_manifest(resolved).manifest_hash == resolved.manifest_hash


def test_unknown_updates_are_rejected():
    draft = build_draft_manifest("p", ".", INVENTORY, PARSED)
    with pytest.raises(ValueError):
        apply_manifest_updates(draft, {"measurements": {"missing": {"modality": "nir"}}})


def test_metadata_updates_merge_and_storage_integrity_fields_are_reserved():
    draft = build_draft_manifest("p", ".", INVENTORY, PARSED)
    mid = draft.measurements[0].measurement_id
    sid = draft.samples[0].sample_id
    changed = apply_manifest_updates(
        draft,
        {
            "measurements": {mid: {"metadata": {"physical_state": "solid"}}},
            "samples": {sid: {"metadata": {"note": "pellet"}}},
        },
    )
    assert changed.measurements[0].metadata["measurement_name"] == "a"
    assert changed.measurements[0].metadata["physical_state"] == "solid"
    assert changed.samples[0].metadata["filename_stem"] == "a"
    assert changed.samples[0].metadata["note"] == "pellet"

    with pytest.raises(ValueError, match="managed internally"):
        apply_manifest_updates(
            changed,
            {"measurements": {mid: {"metadata": {"storage_sha256": "0" * 64}}}},
        )


def test_composition_closure_and_reference_state_advisory():
    draft = build_draft_manifest("p", ".", INVENTORY, PARSED)
    mid, sid = draft.measurements[0].measurement_id, draft.samples[0].sample_id
    changed = apply_manifest_updates(draft, {"measurements": {mid: {"role": "reference", "metadata": {"physical_state": "liquid"}}}, "samples": {sid: {"physical_state": "solid", "composition": {"a": 0.7, "b": 0.2}}}})
    codes = {item.code for item in validate_manifest(changed)}
    assert "composition_not_closed" in codes
    assert "reference_physical_state_mismatch" in codes
