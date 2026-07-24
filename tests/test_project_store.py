from pathlib import Path

import pytest

from chemometrics_mcp.core.project_store import (
    ProjectStore, canonical_json_bytes, create_project_layout, data_hash,
    fingerprint_source, sha256_file,
)


def test_hashes_are_deterministic(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("evidence", encoding="utf-8")
    assert sha256_file(sample) == sha256_file(sample)
    assert canonical_json_bytes({"b": 2, "a": 1}) == canonical_json_bytes({"a": 1, "b": 2})
    assert data_hash({"b": 2, "a": 1}) == data_hash({"a": 1, "b": 2})


def test_default_layout_does_not_change_inputs_and_has_safe_unique_ids(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text("x,y\n1,2\n", encoding="utf-8")
    before = raw.read_bytes()
    first = create_project_layout(tmp_path, project_id="My project / 2026!")
    assert first.output_root == tmp_path / "chemometrics-output"
    assert raw.read_bytes() == before
    assert (first.output_root / "project.json").is_file()
    assert all((first.output_root / name).is_dir() for name in ("manifests", "plans", "runs"))
    assert first.read_json("project.json")["project_id"] == "my-project-2026"
    # Default ids are slug-safe and incorporate random entropy for independent layouts.
    other_root = tmp_path / "other"
    other_root.mkdir()
    other = create_project_layout(other_root)
    assert other.read_json("project.json")["project_id"].replace("-", "").isalnum()
    assert first.read_json("project.json")["project_id"] != other.read_json("project.json")["project_id"]


def test_atomic_json_round_trip_and_artifacts(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "out")
    target = store.write_json("nested/result.json", {"z": 1, "a": [2]})
    assert store.read_json("nested/result.json") == {"a": [2], "z": 1}
    assert not list(target.parent.glob(".tmp-*"))
    assert store.save_manifest({"x": 1}).parent.name == "manifests"
    assert store.save_plan({"x": 1}).parent.name == "plans"
    run = store.save_run({"ok": True}, "run 1")
    assert store.list_runs() == [run]


@pytest.mark.parametrize("name", ["../escape.json", "/tmp/escape.json", "a/../../escape.json"])
def test_traversal_is_rejected(tmp_path: Path, name: str) -> None:
    store = ProjectStore(tmp_path / "out")
    with pytest.raises(ValueError):
        store.write_json(name, {"no": "escape"})


def test_fingerprint_excludes_output_directory(tmp_path: Path) -> None:
    (tmp_path / "input.txt").write_text("input", encoding="utf-8")
    store = create_project_layout(tmp_path)
    store.write_json("runs/evidence.json", {"derived": True})
    records = fingerprint_source(tmp_path, store.output_root)
    assert records == [{
        "relative_path": "input.txt",
        "sha256": sha256_file(tmp_path / "input.txt"),
        "size_bytes": 5,
        "format": "txt",
    }]


def test_optional_deployment_root_allowlist(monkeypatch, tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    monkeypatch.setenv("CHEMOMETRICS_ALLOWED_ROOTS", str(allowed))
    ProjectStore(allowed / "project")
    with pytest.raises(ValueError, match="outside configured"):
        ProjectStore(outside / "project")
