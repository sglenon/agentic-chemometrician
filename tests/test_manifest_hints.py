"""Tests for manifest_hints module and its integration with build_draft_manifest.

Three scenarios required by the task spec:
  (a) No-hints default: build_draft_manifest output byte-identical to pre-hints.
  (b) Declared-mapping hints applied correctly.
  (c) Filename-heuristic hints applied only when opted in and marked advisory.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from chemometrics_mcp.core.manifest_hints import (
    ManifestHints,
    SampleHint,
    infer_hints_from_filenames,
    load_hints_file,
)
from chemometrics_mcp.core.manifests import build_draft_manifest


# ---------------------------------------------------------------------------
# Helpers — minimal parsed measurement fixture
# ---------------------------------------------------------------------------

def _inventory(name: str, sha: str | None = None):
    return {
        "relative_path": f"{name}.csv",
        "sha256": sha or ("a" * 64),
        "size_bytes": 100,
        "format": "csv",
        "supported": True,
    }


def _parsed(name: str):
    return {
        "source_path": Path(f"/fake/{name}.csv"),
        "relative_path": f"{name}.csv",
        "measurement_name": name,
        "parser_id": "csv",
        "modality": "ftir",
        "axis_kind": "wavenumber",
        "axis_unit": "cm-1",
        "signal_kind": "absorbance",
        "signal_unit": "absorbance",
        "axis": [1000, 2000],
        "signal": [0.1, 0.2],
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# (a) No-hints default: output byte-identical
# ---------------------------------------------------------------------------

class TestNoHintsDefaultUnchanged:
    """When hints=None and infer_roles_from_filenames=False, output is identical."""

    def test_sample_role_is_sample_by_default(self):
        """Pre-hints baseline: SampleRecord.role should be SAMPLE (not UNKNOWN)."""
        inv = [_inventory("sample_a")]
        parsed = [_parsed("sample_a")]
        manifest = build_draft_manifest("proj", "/fake", inv, parsed)
        assert len(manifest.samples) == 1
        assert manifest.samples[0].role.value == "sample"

    def test_measurement_role_is_unknown_by_default(self):
        """Pre-hints baseline: MeasurementRecord.role should be UNKNOWN."""
        inv = [_inventory("sample_a")]
        parsed = [_parsed("sample_a")]
        manifest = build_draft_manifest("proj", "/fake", inv, parsed)
        assert len(manifest.measurements) == 1
        assert manifest.measurements[0].role.value == "unknown"

    def test_no_hints_note_in_metadata(self):
        """No advisory markers in metadata when hints are off."""
        inv = [_inventory("pure_compound")]
        parsed = [_parsed("pure_compound")]
        manifest = build_draft_manifest("proj", "/fake", inv, parsed)
        sample = manifest.samples[0]
        assert "roles_inferred_from_filename" not in sample.metadata
        assert "hints_note" not in sample.metadata

    def test_hash_is_stable_without_hints(self):
        """Hash is deterministic across two identical calls."""
        inv = [_inventory("sample_a")]
        parsed = [_parsed("sample_a")]
        m1 = build_draft_manifest("proj", "/fake", inv, parsed)
        m2 = build_draft_manifest("proj", "/fake", inv, parsed)
        assert m1.manifest_hash == m2.manifest_hash

    def test_explicit_none_hints_identical_to_default(self):
        """Passing hints=None explicitly produces same result as omitting it."""
        inv = [_inventory("pure_ref")]
        parsed = [_parsed("pure_ref")]
        m_default = build_draft_manifest("proj", "/fake", inv, parsed)
        m_none = build_draft_manifest("proj", "/fake", inv, parsed, hints=None, infer_roles_from_filenames=False)
        assert m_default.manifest_hash == m_none.manifest_hash


# ---------------------------------------------------------------------------
# (b) Declared-mapping hints applied correctly
# ---------------------------------------------------------------------------

class TestDeclaredHintsApplied:
    """Declared hints (provenance=declared) are applied with full confidence."""

    def _hints(self, stem: str, **kwargs) -> ManifestHints:
        return ManifestHints(entries={stem: SampleHint(provenance="declared", **kwargs)})

    def test_declared_role_applied_to_sample(self):
        inv = [_inventory("compound_a")]
        parsed = [_parsed("compound_a")]
        hints = self._hints("compound_a", role="reference")
        manifest = build_draft_manifest("proj", "/fake", inv, parsed, hints=hints)
        assert manifest.samples[0].role.value == "reference"

    def test_declared_role_applied_to_measurement(self):
        inv = [_inventory("compound_a")]
        parsed = [_parsed("compound_a")]
        hints = self._hints("compound_a", role="reference")
        manifest = build_draft_manifest("proj", "/fake", inv, parsed, hints=hints)
        assert manifest.measurements[0].role.value == "reference"

    def test_declared_composition_applied(self):
        inv = [_inventory("mixture_x")]
        parsed = [_parsed("mixture_x")]
        hints = self._hints("mixture_x", role="sample", composition={"A": 0.3, "B": 0.7})
        manifest = build_draft_manifest("proj", "/fake", inv, parsed, hints=hints)
        sample = manifest.samples[0]
        assert sample.composition == {"A": 0.3, "B": 0.7}

    def test_declared_reference_name_in_metadata(self):
        inv = [_inventory("compound_a")]
        parsed = [_parsed("compound_a")]
        hints = self._hints("compound_a", role="reference", reference_name="API-Form-I")
        manifest = build_draft_manifest("proj", "/fake", inv, parsed, hints=hints)
        assert manifest.samples[0].metadata.get("reference_name") == "API-Form-I"

    def test_declared_hint_does_not_set_advisory_marker(self):
        """Declared hints must NOT set roles_inferred_from_filename."""
        inv = [_inventory("compound_a")]
        parsed = [_parsed("compound_a")]
        hints = self._hints("compound_a", role="reference")
        manifest = build_draft_manifest("proj", "/fake", inv, parsed, hints=hints)
        assert "roles_inferred_from_filename" not in manifest.samples[0].metadata

    def test_unmatched_samples_unaffected_by_hints(self):
        """Samples not in the hints dict keep default role/composition."""
        inv = [_inventory("compound_a"), _inventory("other_b", sha="b" * 64)]
        parsed = [_parsed("compound_a"), _parsed("other_b")]
        hints = self._hints("compound_a", role="reference")
        manifest = build_draft_manifest("proj", "/fake", inv, parsed, hints=hints)
        by_stem = {s.metadata["filename_stem"]: s for s in manifest.samples}
        assert by_stem["compound_a"].role.value == "reference"
        assert by_stem["other_b"].role.value == "sample"

    def test_hash_changes_when_hints_applied(self):
        """Manifest hash must differ when hints change the role."""
        inv = [_inventory("compound_a")]
        parsed = [_parsed("compound_a")]
        no_hints = build_draft_manifest("proj", "/fake", inv, parsed)
        hints = self._hints("compound_a", role="reference")
        with_hints = build_draft_manifest("proj", "/fake", inv, parsed, hints=hints)
        assert no_hints.manifest_hash != with_hints.manifest_hash

    def test_glob_pattern_hint_matches(self):
        """A glob pattern key like 'pure_*' should match 'pure_acetone'."""
        inv = [_inventory("pure_acetone")]
        parsed = [_parsed("pure_acetone")]
        hints = ManifestHints(entries={"pure_*": SampleHint(role="reference", provenance="declared")})
        manifest = build_draft_manifest("proj", "/fake", inv, parsed, hints=hints)
        assert manifest.samples[0].role.value == "reference"

    def test_hints_loaded_from_json_file(self, tmp_path):
        """load_hints_file reads manifest_hints.json and returns correct hints."""
        hints_data = {
            "pure_ref": {"role": "reference", "reference_name": "Compound-X"},
            "mix_sample": {"role": "sample", "composition": {"A": 0.4, "B": 0.6}},
        }
        (tmp_path / "manifest_hints.json").write_text(json.dumps(hints_data))
        loaded = load_hints_file(tmp_path)
        assert loaded is not None
        assert loaded.lookup("pure_ref").role == "reference"
        assert loaded.lookup("pure_ref").reference_name == "Compound-X"
        assert loaded.lookup("mix_sample").composition == {"A": 0.4, "B": 0.6}

    def test_no_hints_file_returns_none(self, tmp_path):
        """load_hints_file returns None when no manifest_hints.json exists."""
        assert load_hints_file(tmp_path) is None

    def test_malformed_hints_file_raises(self, tmp_path):
        (tmp_path / "manifest_hints.json").write_text("{not valid json")
        with pytest.raises(ValueError, match="Cannot parse"):
            load_hints_file(tmp_path)


# ---------------------------------------------------------------------------
# (c) Filename-heuristic hints: opt-in only, advisory markers present
# ---------------------------------------------------------------------------

class TestFilenameHeuristicOptIn:
    """Filename heuristic is only applied when infer_roles_from_filenames=True."""

    def test_pure_prefix_not_applied_by_default(self):
        """pure_compound stays role=UNKNOWN in measurement, no advisory marker."""
        inv = [_inventory("pure_compound")]
        parsed = [_parsed("pure_compound")]
        manifest = build_draft_manifest("proj", "/fake", inv, parsed)
        # measurement role stays UNKNOWN (sample role is SAMPLE — that's the baseline)
        assert manifest.measurements[0].role.value == "unknown"
        assert "roles_inferred_from_filename" not in manifest.samples[0].metadata

    def test_pure_prefix_applied_when_opted_in(self):
        """pure_compound gets role=reference when infer_roles_from_filenames=True."""
        inv = [_inventory("pure_compound")]
        parsed = [_parsed("pure_compound")]
        manifest = build_draft_manifest("proj", "/fake", inv, parsed,
                                        infer_roles_from_filenames=True)
        assert manifest.measurements[0].role.value == "reference"

    def test_advisory_marker_present_when_opted_in(self):
        """Inferred hints set roles_inferred_from_filename=True on sample metadata."""
        inv = [_inventory("pure_compound")]
        parsed = [_parsed("pure_compound")]
        manifest = build_draft_manifest("proj", "/fake", inv, parsed,
                                        infer_roles_from_filenames=True)
        assert manifest.samples[0].metadata.get("roles_inferred_from_filename") is True

    def test_hints_note_present_when_opted_in(self):
        """Inferred hints include a human-readable hints_note."""
        inv = [_inventory("pure_compound")]
        parsed = [_parsed("pure_compound")]
        manifest = build_draft_manifest("proj", "/fake", inv, parsed,
                                        infer_roles_from_filenames=True)
        assert "hints_note" in manifest.samples[0].metadata
        assert "pure_compound" in manifest.samples[0].metadata["hints_note"]

    def test_mix_positional_composition_inferred(self):
        """mix_30_70 stem yields advisory composition {"A": 0.3, "B": 0.7}."""
        inv = [_inventory("mix_30_70")]
        parsed = [_parsed("mix_30_70")]
        manifest = build_draft_manifest("proj", "/fake", inv, parsed,
                                        infer_roles_from_filenames=True)
        sample = manifest.samples[0]
        assert sample.composition is not None
        assert abs(sample.composition.get("A", 0) - 0.3) < 1e-5
        assert abs(sample.composition.get("B", 0) - 0.7) < 1e-5
        assert sample.metadata.get("roles_inferred_from_filename") is True

    def test_mix_named_composition_inferred(self):
        """mix_api_30_exc_70 stem yields advisory composition {"api": 0.3, "exc": 0.7}."""
        inv = [_inventory("mix_api_30_exc_70")]
        parsed = [_parsed("mix_api_30_exc_70")]
        manifest = build_draft_manifest("proj", "/fake", inv, parsed,
                                        infer_roles_from_filenames=True)
        sample = manifest.samples[0]
        assert sample.composition is not None
        assert abs(sample.composition.get("api", 0) - 0.3) < 1e-5
        assert abs(sample.composition.get("exc", 0) - 0.7) < 1e-5

    def test_unrecognized_stem_not_affected_by_heuristic(self):
        """Stems like 'C', 'I1', 'S31SOLID' produce no inferred hint."""
        hints_result = infer_hints_from_filenames(["C", "I1", "S31SOLID", "8ami", "3fc"])
        assert hints_result.entries == {}

    def test_declared_wins_over_inferred_for_same_stem(self):
        """When same stem appears in declared hints, declared role takes priority."""
        inv = [_inventory("pure_ref")]
        parsed = [_parsed("pure_ref")]
        # Declared: calibration; heuristic would infer reference.
        declared = ManifestHints(
            entries={"pure_ref": SampleHint(role="calibration", provenance="declared")}
        )
        manifest = build_draft_manifest("proj", "/fake", inv, parsed,
                                        hints=declared, infer_roles_from_filenames=True)
        assert manifest.measurements[0].role.value == "calibration"
        assert "roles_inferred_from_filename" not in manifest.samples[0].metadata

    def test_heuristic_hash_differs_from_no_hints(self):
        """Hash changes when heuristic is applied."""
        inv = [_inventory("pure_compound")]
        parsed = [_parsed("pure_compound")]
        no_infer = build_draft_manifest("proj", "/fake", inv, parsed)
        with_infer = build_draft_manifest("proj", "/fake", inv, parsed,
                                          infer_roles_from_filenames=True)
        assert no_infer.manifest_hash != with_infer.manifest_hash


# ---------------------------------------------------------------------------
# load_hints_file + infer_hints_from_filenames unit tests
# ---------------------------------------------------------------------------

class TestHintsModuleUnit:
    """Unit tests for manifest_hints.py functions directly."""

    def test_load_hints_file_full_schema(self, tmp_path):
        data = {
            "8ami": {"role": "sample", "composition": {"A": 0.8, "B": 0.2}},
            "3fc": {"role": "reference", "reference_name": "FC-form"},
        }
        (tmp_path / "manifest_hints.json").write_text(json.dumps(data))
        hints = load_hints_file(tmp_path)
        assert hints.lookup("8ami").composition == {"A": 0.8, "B": 0.2}
        assert hints.lookup("3fc").role == "reference"
        assert hints.lookup("3fc").reference_name == "FC-form"
        assert hints.lookup("missing") is None

    def test_infer_ref_patterns(self):
        for stem in ("pure_ethanol", "ref_api", "reference_compound", "ethanol_pure", "api_ref"):
            hint = infer_hints_from_filenames([stem]).lookup(stem)
            assert hint is not None, f"No hint for {stem!r}"
            assert hint.role == "reference", f"Wrong role for {stem!r}"
            assert hint.provenance == "inferred"

    def test_infer_mix_positional(self):
        hint = infer_hints_from_filenames(["mix_30_70"]).lookup("mix_30_70")
        assert hint is not None
        assert abs(hint.composition["A"] - 0.3) < 1e-5
        assert abs(hint.composition["B"] - 0.7) < 1e-5

    def test_infer_mix_named(self):
        hint = infer_hints_from_filenames(["mix_api_40_exc_60"]).lookup("mix_api_40_exc_60")
        assert hint is not None
        assert abs(hint.composition["api"] - 0.4) < 1e-5
        assert abs(hint.composition["exc"] - 0.6) < 1e-5

    def test_infer_no_match_returns_empty(self):
        hints = infer_hints_from_filenames(["C", "M1", "J2", "ETOH", "S31SOLID"])
        assert hints.entries == {}

    def test_lookup_glob_fallback(self):
        hints = ManifestHints(entries={"mix_*": SampleHint(role="sample", provenance="declared")})
        assert hints.lookup("mix_30_70") is not None
        assert hints.lookup("pure_a") is None

    def test_lookup_exact_beats_glob(self):
        hints = ManifestHints(entries={
            "pure_*": SampleHint(role="reference", provenance="declared"),
            "pure_special": SampleHint(role="calibration", provenance="declared"),
        })
        assert hints.lookup("pure_special").role == "calibration"
        assert hints.lookup("pure_other").role == "reference"
