"""Smoke tests for the MCP server and inspect_dataset tool.

These tests verify:
  - All 8 tools are registered and discoverable.
  - Invalid inputs to inspect_dataset return explicit error responses, not exceptions.
  - inspect_dataset on the real NIR Excel file returns a valid DatasetInspection payload.
  - Artifact path safety: run IDs with path separators are rejected.
  - Deferred tools return ok=False with a clear error, not silently faked responses.
"""
from __future__ import annotations

import asyncio
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repo root
NIR_FILE = ROOT / "2026-05-21_22-59_Method Dev_wavelength_range_1450-2450.xlsx"

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from chemometrics_contracts import InspectDatasetRequest
from chemometrics_mcp import server as mcp_server
from chemometrics_mcp.artifacts import make_run_id, run_artifacts_dir
from chemometrics_mcp.tools import inspect_dataset


class ServerToolRegistrationTests(unittest.TestCase):
    def test_list_tools_returns_all_eight(self) -> None:
        tools = asyncio.run(mcp_server.list_tools())
        names = {t.name for t in tools}
        expected = {
            "inspect_dataset",
            "propose_analysis_plan",
            "run_analysis",
            "validate_results",
            "select_best_model",
            "recommend_next_model",
            "interpret_results",
            "generate_report",
            "save_method_memory",
            "search_method_memory",
            "recommend_from_memory",
        }
        self.assertEqual(names, expected)

    def test_all_tools_have_descriptions(self) -> None:
        tools = asyncio.run(mcp_server.list_tools())
        for tool in tools:
            self.assertTrue(
                tool.description and len(tool.description) > 10,
                f"Tool {tool.name!r} has no meaningful description.",
            )

    def test_all_tools_have_input_schemas(self) -> None:
        tools = asyncio.run(mcp_server.list_tools())
        for tool in tools:
            self.assertIn("type", tool.inputSchema, f"Tool {tool.name!r} missing inputSchema type.")

    def test_no_tools_say_deferred_in_description(self) -> None:
        tools = asyncio.run(mcp_server.list_tools())
        deferred = {t.name for t in tools if "deferred" in (t.description or "").lower()}
        self.assertEqual(deferred, set(), "All tools are implemented; none should claim deferred status.")


class InspectDatasetInvalidInputTests(unittest.TestCase):
    def _tmp_runs(self) -> Path:
        return Path("/tmp/chemometrics_test_runs")

    def test_file_not_found_returns_error_response(self) -> None:
        req = InspectDatasetRequest(source_uri="/nonexistent/file.xlsx")
        resp = inspect_dataset.run(req, runs_root=self._tmp_runs())
        self.assertFalse(resp.ok)
        self.assertIsNotNone(resp.error)
        self.assertIn("not found", resp.error.lower())

    def test_unsupported_format_returns_error_response(self) -> None:
        req = InspectDatasetRequest(source_uri="/some/file.csv")
        resp = inspect_dataset.run(req, runs_root=self._tmp_runs())
        self.assertFalse(resp.ok)
        self.assertIsNotNone(resp.error)
        self.assertIn("unsupported", resp.error.lower())

    def test_unsupported_format_txt_returns_error_response(self) -> None:
        req = InspectDatasetRequest(source_uri="/some/file.txt")
        resp = inspect_dataset.run(req, runs_root=self._tmp_runs())
        self.assertFalse(resp.ok)


class InspectDatasetRealFileTests(unittest.TestCase):
    @unittest.skipUnless(NIR_FILE.exists(), "NIR Excel file not available at expected path.")
    def test_inspect_real_nir_file(self) -> None:
        req = InspectDatasetRequest(
            source_uri=str(NIR_FILE),
            dataset_id="nir-flooring-v1",
            modality_override="NIR",
            label_column="Measurement Description",
        )
        runs_root = Path("/tmp/chemometrics_test_runs")
        resp = inspect_dataset.run(req, runs_root=runs_root)

        self.assertTrue(resp.ok, f"Expected ok=True, got error: {resp.error}")
        self.assertIsNotNone(resp.payload)
        inspection = resp.payload

        self.assertEqual(inspection.sample_count, 146)
        self.assertEqual(inspection.feature_count, 249)
        self.assertEqual(inspection.modality, "NIR")
        self.assertAlmostEqual(inspection.axis_min, 1454.0, places=0)
        self.assertAlmostEqual(inspection.axis_max, 2446.0, places=0)
        self.assertIn("Measurement Description", inspection.candidate_label_columns)

    @unittest.skipUnless(NIR_FILE.exists(), "NIR Excel file not available at expected path.")
    def test_inspect_without_label_override_auto_selects_single_candidate(self) -> None:
        req = InspectDatasetRequest(source_uri=str(NIR_FILE))
        runs_root = Path("/tmp/chemometrics_test_runs")
        resp = inspect_dataset.run(req, runs_root=runs_root)

        self.assertTrue(resp.ok)
        self.assertIsNotNone(resp.payload)
        # With exactly one candidate label column the loader auto-selects it without warning.
        self.assertIn("Measurement Description", resp.payload.candidate_label_columns)
        # No ambiguous_label_columns warning expected for a single candidate.
        warning_codes = [w.code for w in resp.warnings]
        self.assertNotIn("ambiguous_label_columns", warning_codes)

    @unittest.skipUnless(NIR_FILE.exists(), "NIR Excel file not available at expected path.")
    def test_inspect_is_deterministic(self) -> None:
        req = InspectDatasetRequest(
            source_uri=str(NIR_FILE),
            modality_override="NIR",
            label_column="Measurement Description",
        )
        runs_root = Path("/tmp/chemometrics_test_runs")
        resp1 = inspect_dataset.run(req, runs_root=runs_root)
        resp2 = inspect_dataset.run(req, runs_root=runs_root)

        self.assertEqual(resp1.payload.sample_count, resp2.payload.sample_count)
        self.assertEqual(resp1.payload.feature_count, resp2.payload.feature_count)
        self.assertEqual(resp1.payload.axis_min, resp2.payload.axis_min)
        self.assertEqual(resp1.payload.axis_max, resp2.payload.axis_max)

    @unittest.skipUnless(NIR_FILE.exists(), "NIR Excel file not available at expected path.")
    def test_inspect_saves_artifact(self) -> None:
        req = InspectDatasetRequest(
            source_uri=str(NIR_FILE),
            dataset_id="nir-smoke-artifact",
            modality_override="NIR",
        )
        runs_root = Path("/tmp/chemometrics_test_runs")
        resp = inspect_dataset.run(req, runs_root=runs_root)

        self.assertTrue(resp.ok)
        self.assertTrue(len(resp.artifacts) > 0)
        artifact_path = Path(resp.artifacts[0].uri)
        self.assertTrue(artifact_path.exists(), f"Artifact not written: {artifact_path}")


class ArtifactPathSafetyTests(unittest.TestCase):
    def test_run_id_with_path_separator_raises(self) -> None:
        with self.assertRaises(ValueError):
            run_artifacts_dir("../escape/path")

    def test_run_id_with_slash_raises(self) -> None:
        with self.assertRaises(ValueError):
            run_artifacts_dir("run/subdir")

    def test_make_run_id_strips_special_chars(self) -> None:
        run_id = make_run_id(slug="my dataset/file.xlsx")
        self.assertNotIn("/", run_id)
        self.assertNotIn(".", run_id)

    def test_make_run_id_is_unique(self) -> None:
        id1 = make_run_id(slug="test")
        id2 = make_run_id(slug="test")
        self.assertNotEqual(id1, id2)


class DeferredToolsReturnErrorTests(unittest.TestCase):
    def test_propose_analysis_plan_implemented(self) -> None:
        """Phase 5: propose_analysis_plan is now implemented; verify basic success path."""
        import tempfile
        from chemometrics_contracts import DatasetInspection, ProposeAnalysisPlanRequest
        from chemometrics_mcp.tools import propose_analysis_plan

        req = ProposeAnalysisPlanRequest(
            dataset_inspection=DatasetInspection(sample_count=10, feature_count=50)
        )
        with tempfile.TemporaryDirectory() as tmp:
            resp = propose_analysis_plan.run(req, runs_root=tmp)
        # No label columns → unsupervised_exploration → ok=True
        self.assertTrue(resp.ok)
        self.assertIsNotNone(resp.payload)
        self.assertEqual(resp.payload.task_name, "unsupervised_exploration")

    def test_validate_results_no_results(self) -> None:
        """validate_results is now implemented (Phase 7). Empty request returns ok=False."""
        from chemometrics_contracts import ValidateResultsRequest
        from chemometrics_mcp.tools import validate_results

        resp = validate_results.run(ValidateResultsRequest())
        self.assertFalse(resp.ok)
        self.assertIn("No results", resp.error)

    def test_generate_report_implemented(self) -> None:
        """generate_report is now implemented (Phase 6b). Verify it returns ok=True."""
        import tempfile
        from pathlib import Path
        from chemometrics_contracts import AnalysisRun, GenerateReportRequest
        from chemometrics_mcp.tools import generate_report

        with tempfile.TemporaryDirectory() as tmp:
            resp = generate_report.run(
                GenerateReportRequest(analysis_run=AnalysisRun()),
                runs_root=Path(tmp),
            )
        self.assertTrue(resp.ok)
        self.assertIsNotNone(resp.payload)


if __name__ == "__main__":
    unittest.main()
