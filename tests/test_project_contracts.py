from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from chemometrics_contracts.project import (
    ProjectAnalysisPlan,
    Modality,
    ProjectManifest,
    SampleRecord,
    SourceAsset,
    ValidationIssue,
    WarningLevel,
)


def asset(**overrides: object) -> SourceAsset:
    data: dict[str, object] = {
        "asset_id": "asset-1",
        "relative_path": "raw/spectrum.csv",
        "sha256": "a" * 64,
        "size_bytes": 12,
        "format": "csv",
        "parser_id": "csv-v1",
    }
    data.update(overrides)
    return SourceAsset(**data)


def test_contracts_are_frozen_and_forbid_unknown_fields() -> None:
    issue = ValidationIssue(code="x", message="message")
    with pytest.raises(ValidationError):
        issue.code = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ValidationIssue(code="x", message="message", unknown=True)


@pytest.mark.parametrize("path", ["../raw.csv", "/raw.csv", "raw/../x.csv", "raw//x.csv"])
def test_asset_rejects_non_relative_or_traversing_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        asset(relative_path=path)


def test_asset_validates_sha256_and_normalizes_windows_separators() -> None:
    assert asset(relative_path="raw\\spectrum.csv").relative_path == "raw/spectrum.csv"
    with pytest.raises(ValidationError):
        asset(sha256="A" * 64)
    with pytest.raises(ValidationError):
        asset(sha256="abc")


def test_json_schema_contains_strict_contract_and_enum_definitions() -> None:
    schema = ProjectManifest.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Modality"]["enum"] == [member.value for member in Modality]


def test_lossless_json_round_trip_and_canonical_serialization() -> None:
    manifest = ProjectManifest(
        project_id="demo",
        assets=(asset(),),
        unresolved_issues=(ValidationIssue(code="note", message="ok", level=WarningLevel.INFORMATION),),
        manifest_hash="f" * 64,
    )
    restored = ProjectManifest.model_validate_json(manifest.model_dump_json())
    assert restored == manifest
    assert "manifest_hash" not in manifest.canonical_dict()
    assert json.loads(manifest.canonical_json())["project_id"] == "demo"


def test_defaults_construct_minimal_plans_and_manifests() -> None:
    assert ProjectManifest(project_id="empty").assets == ()
    plan = ProjectAnalysisPlan(plan_id="plan-1")
    assert plan.pipelines == ()
    assert plan.task is None


def test_contract_metadata_is_deeply_immutable_and_json_serializable() -> None:
    source = {"nested": {"items": [1, 2]}}
    sample = SampleRecord(sample_id="s", metadata=source)
    source["nested"]["items"].append(3)
    assert sample.metadata["nested"]["items"] == (1, 2)
    with pytest.raises(TypeError):
        sample.metadata["new"] = "value"
    with pytest.raises(TypeError):
        sample.metadata["nested"]["new"] = "value"
    assert sample.model_dump(mode="json")["metadata"]["nested"]["items"] == [
        1,
        2,
    ]
    copied = sample.model_copy(
        update={"metadata": {"nested": {"items": [3, 4]}}}
    )
    with pytest.raises(TypeError):
        copied.metadata["nested"]["items"] = ()
