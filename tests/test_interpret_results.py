"""Tests for interpret_results tool and core/interpretation.interpret_results logic.

All tests use synthetic data only — no file I/O except artifact writes
which are directed to a temporary directory via tmp_path.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from chemometrics_contracts import (
    AnalysisResult,
    InterpretResultsRequest,
    InterpretationSummary,
    ValidationSummary,
    ValidationWarning,
)
from chemometrics_mcp.core.interpretation import interpret_results as core_interpret
from chemometrics_mcp.tools import interpret_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(
    model_name: str,
    accuracy: float = 0.80,
    selected_features: tuple = (),
    task_name: str = "binary_classification",
    warnings: tuple = (),
) -> AnalysisResult:
    return AnalysisResult(
        task_name=task_name,
        model_name=model_name,
        metrics={"accuracy": accuracy},
        selected_features=selected_features,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Core interpret_results tests
# ---------------------------------------------------------------------------


class TestCoreInterpretResults(unittest.TestCase):
    def test_empty_results_no_feature_importance_summary(self):
        # Empty results — no results at all, but core function still works
        # The tool layer handles the empty-results guard; core can receive empty list.
        summary = core_interpret([])
        self.assertIn("No feature importance", summary.summary)

    def test_result_no_features_says_no_feature_importance(self):
        r = _result("svm_rbf")
        summary = core_interpret([r])
        self.assertIn("No feature importance", summary.summary)

    def test_selected_features_appear_in_important_features(self):
        r = _result("random_forest", selected_features=(1450.0, 1600.0))
        summary = core_interpret([r])
        self.assertIn(1450.0, summary.important_features)
        self.assertIn(1600.0, summary.important_features)

    def test_two_results_with_features_intersection_in_summary(self):
        r1 = _result("random_forest", selected_features=(1450.0, 1600.0, 1750.0))
        r2 = _result("plsr", selected_features=(1450.0, 1600.0, 1900.0))
        summary = core_interpret([r1, r2])
        # 1450.0 and 1600.0 appear in both → should be mentioned in summary
        self.assertIn("1450.0", summary.summary)
        self.assertIn("1600.0", summary.summary)

    def test_model_comparisons_one_per_result(self):
        r1 = _result("svm_rbf")
        r2 = _result("random_forest")
        summary = core_interpret([r1, r2])
        self.assertEqual(len(summary.model_comparisons), 2)

    def test_model_comparison_contains_model_name(self):
        r = _result("svm_rbf", accuracy=0.91)
        summary = core_interpret([r])
        self.assertTrue(any("svm_rbf" in line for line in summary.model_comparisons))

    def test_model_comparison_with_features_mentions_features(self):
        r = _result("random_forest", accuracy=0.88, selected_features=(1450.0, 1600.0, 1750.0))
        summary = core_interpret([r])
        self.assertTrue(any("top features" in line for line in summary.model_comparisons))

    def test_model_comparison_without_features_says_no_importance(self):
        r = _result("svm_rbf", accuracy=0.91)
        summary = core_interpret([r])
        self.assertTrue(
            any("no feature importance available" in line for line in summary.model_comparisons)
        )

    def test_summary_ends_with_chemical_causality_disclaimer(self):
        r = _result("random_forest", selected_features=(1450.0,))
        summary = core_interpret([r])
        self.assertIn(
            "Do not interpret as chemical causality without domain expert review",
            summary.summary,
        )

    def test_no_features_disclaimer_in_summary(self):
        r = _result("random_forest", selected_features=(1450.0,))
        summary = core_interpret([r])
        self.assertIn("Wavelength importance reflects model evidence only", summary.summary)

    def test_warning_emitted_for_model_with_empty_features(self):
        r = _result("svm_rbf")  # no selected_features
        summary = core_interpret([r])
        codes = [w.code for w in summary.warnings]
        self.assertIn("no_feature_importance", codes)

    def test_no_warning_for_model_with_features(self):
        r = _result("random_forest", selected_features=(1450.0,))
        summary = core_interpret([r])
        codes = [w.code for w in summary.warnings]
        self.assertNotIn("no_feature_importance", codes)

    def test_validation_summary_warnings_included(self):
        r = _result("random_forest", selected_features=(1450.0,))
        w = ValidationWarning(code="class_imbalance", message="imbalanced", severity="warning")
        vs = ValidationSummary(passed=True, checks={}, warnings=(w,))
        summary = core_interpret([r], validation_summary=vs)
        codes = [w.code for w in summary.warnings]
        self.assertIn("class_imbalance", codes)

    def test_important_features_deduplicated_across_models(self):
        r1 = _result("random_forest", selected_features=(1450.0, 1600.0))
        r2 = _result("plsr", selected_features=(1450.0, 1900.0))
        summary = core_interpret([r1, r2])
        # 1450.0 appears in both but should only be listed once
        self.assertEqual(list(summary.important_features).count(1450.0), 1)

    def test_feature_importance_count_in_summary(self):
        r1 = _result("random_forest", selected_features=(1450.0,))
        r2 = _result("plsr", selected_features=(1600.0,))
        summary = core_interpret([r1, r2])
        self.assertIn("2 model(s)", summary.summary)


# ---------------------------------------------------------------------------
# Tool boundary tests
# ---------------------------------------------------------------------------


class TestInterpretResultsTool(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.mkdtemp()

    def _run(self, results, dataset=None, validation_summary=None):
        request = InterpretResultsRequest(
            results=tuple(results),
            dataset=dataset,
            validation_summary=validation_summary,
        )
        return interpret_results.run(request, runs_root=self._tmp)

    def test_ok_true_for_non_empty_results(self):
        r = _result("random_forest", selected_features=(1450.0,))
        resp = self._run([r])
        self.assertTrue(resp.ok)
        self.assertEqual(resp.tool_name, "interpret_results")

    def test_payload_is_interpretation_summary(self):
        r = _result("random_forest", selected_features=(1450.0,))
        resp = self._run([r])
        self.assertIsInstance(resp.payload, InterpretationSummary)

    def test_ok_false_for_empty_results(self):
        resp = self._run([])
        self.assertFalse(resp.ok)
        self.assertIsNotNone(resp.error)

    def test_artifact_is_saved(self):
        r = _result("random_forest", selected_features=(1450.0,))
        resp = self._run([r])
        self.assertTrue(resp.artifacts)
        artifact_path = Path(resp.artifacts[0].uri)
        self.assertTrue(artifact_path.exists())
        data = json.loads(artifact_path.read_text())
        self.assertIn("summary", data)
        self.assertIn("important_features", data)

    def test_artifact_filename_is_interpretation_summary(self):
        r = _result("random_forest", selected_features=(1450.0,))
        resp = self._run([r])
        artifact_path = Path(resp.artifacts[0].uri)
        self.assertEqual(artifact_path.name, "interpretation_summary.json")

    def test_no_feature_importance_result_ok_true(self):
        r = _result("svm_rbf")  # no selected_features
        resp = self._run([r])
        self.assertTrue(resp.ok)
        # warning should be in response
        codes = [w.code for w in resp.warnings]
        self.assertIn("no_feature_importance", codes)


if __name__ == "__main__":
    unittest.main()
