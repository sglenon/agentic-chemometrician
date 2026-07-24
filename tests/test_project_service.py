from pathlib import Path

import numpy as np
import pytest

from chemometrics_mcp.core.project_service import ProjectService
from chemometrics_mcp.core.project_store import ProjectStore


def test_folder_project_parses_multisignal_and_reopens(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    raw = source / "signals.csv"
    raw.write_text("axis,a,b\n1,10,20\n2,11,21\n", encoding="utf-8")
    service = ProjectService.create(source)
    manifest = service.get_manifest()
    assert len(manifest.measurements) == 2
    loaded = service.load_measurement(manifest.measurements[0].measurement_id)
    assert np.array_equal(loaded["axis"], [1, 2]) and np.array_equal(loaded["signal"], [10, 11])
    assert ProjectService.open(source / "chemometrics-output").get_manifest().manifest_hash == manifest.manifest_hash
    summary = service.get_summary()
    assert "measurements" not in summary
    assert all(not isinstance(value, np.ndarray) for value in summary.values())
    assert raw.read_text(encoding="utf-8") == "axis,a,b\n1,10,20\n2,11,21\n"


def test_updates_persist_new_version_and_hash(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.csv").write_text("x,y\n1,2\n2,3\n", encoding="utf-8")
    service = ProjectService.create(source)
    first = service.get_manifest()
    row = first.measurements[0]
    second = service.update_manifest({"measurements": {row.measurement_id: {"axis_unit": "nm"}}})
    assert second.version == "3" and second.manifest_hash != first.manifest_hash
    assert ProjectService.open(service.store.output_root).get_manifest().manifest_hash == second.manifest_hash


def test_invalid_percent_transmittance_is_stored_raw_and_malformed_is_issue(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "ftir.txt").write_text("##XUNITS=cm-1\n##YUNITS=%T\n1000 101\n900 0\n", encoding="utf-8")
    (source / "broken.csv").write_text("x,y\nnot,a-number\n", encoding="utf-8")
    service = ProjectService.create(source)
    manifest = service.get_manifest()
    measurement = next(row for row in manifest.measurements if row.metadata["measurement_name"] == "ftir")
    assert np.array_equal(service.load_measurement(measurement.measurement_id)["signal"], [101, 0])
    assert any(issue.code == "malformed_input" and issue.level.value == "blocker" for issue in manifest.unresolved_issues)
    assert any(
        issue.code == "percent_transmittance_out_of_range"
        and issue.level.value == "blocker"
        for issue in manifest.unresolved_issues
    )


def test_project_store_traversal_remains_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ProjectStore(tmp_path / "out").write_bytes("../escape.npz", b"no")


def test_refresh_excludes_custom_output_folder_inside_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    output = source / "derived-evidence"
    first = ProjectService.create(source, output)
    refreshed = ProjectService.create(source, output)
    assert len(first.get_manifest().assets) == len(refreshed.get_manifest().assets) == 1


def test_manifest_and_measurement_tampering_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.csv").write_text("x,y\n1,2\n2,3\n", encoding="utf-8")
    service = ProjectService.create(source)
    manifest = service.get_manifest()
    index = service.store.read_json("data/index.json")
    entry = index[manifest.measurements[0].measurement_id]
    service.store.write_bytes(entry["path"], b"replaced")
    with pytest.raises(ValueError, match="payload hash mismatch"):
        service.load_measurement(manifest.measurements[0].measurement_id)

    payload = service.store.read_json("manifests/current.json")
    payload["source_root"] = "/tampered"
    service.store.write_json("manifests/current.json", payload)
    with pytest.raises(ValueError, match="manifest hash"):
        service.get_manifest()
