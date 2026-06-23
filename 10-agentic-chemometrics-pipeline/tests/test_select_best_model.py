"""Tests for select_best_model tool and core/interpretation.select_best_model logic.

All tests use synthetic data only — no file I/O except artifact writes
which are directed to a temporary directory via tmp_path.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from chemometrics_contracts import (
    AnalysisResult,
    ModelSelectionRecommendation,
    SelectBestModelRequest,
    ValidationSummary,
    ValidationWarning,
)
from chemometrics_mcp.core.interpretation import score_model
from chemometrics_mcp.core.interpretation import select_best_model as core_select
from chemometrics_mcp.tools import select_best_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cls_result(
    model_name: str,
    accuracy: float,
    balanced_accuracy: float | None = None,
    selected_features: tuple = (),
    warnings: tuple = (),
    task_name: str = "binary_classification",
) -> AnalysisResult:
    metrics: dict = {"accuracy": accuracy}
    if balanced_accuracy is not None:
        metrics["balanced_accuracy"] = balanced_accuracy
    return AnalysisResult(
        task_name=task_name,
        model_name=model_name,
        metrics=metrics,
        selected_features=selected_features,
        warnings=warnings,
    )


def _warning(code: str = "some_warning", severity: str = "warning") -> ValidationWarning:
    return ValidationWarning(code=code, message="test warning", severity=severity)


# ---------------------------------------------------------------------------
# score_model tests
# ---------------------------------------------------------------------------


class TestScoreModel(unittest.TestCase):
    def test_performance_uses_accuracy(self):
        result = _cls_result("svm_rbf", accuracy=0.85)
        scores = score_model(result)
        self.assertAlmostEqual(scores["performance"], 0.85)

    def test_performance_prefers_balanced_accuracy(self):
        result = _cls_result("svm_rbf", accuracy=0.90, balanced_accuracy=0.78)
        scores = score_model(result)
        self.assertAlmostEqual(scores["performance"], 0.78)

    def test_reliability_penalised_by_warning(self):
        w = _warning(severity="warning")
        result = _cls_result("svm_rbf", accuracy=0.80, warnings=(w,))
        scores = score_model(result)
        self.assertAlmostEqual(scores["reliability"], 0.7)

    def test_reliability_penalised_by_error(self):
        w = _warning(severity="error")
        result = _cls_result("svm_rbf", accuracy=0.80, warnings=(w,))
        scores = score_model(result)
        self.assertAlmostEqual(scores["reliability"], 0.7)

    def test_reliability_not_penalised_by_info(self):
        w = _warning(severity="info")
        result = _cls_result("svm_rbf", accuracy=0.80, warnings=(w,))
        scores = score_model(result)
        self.assertAlmostEqual(scores["reliability"], 1.0)

    def test_interpretability_full_when_features_present(self):
        result = _cls_result("random_forest", accuracy=0.80, selected_features=(1450.0, 1600.0))
        scores = score_model(result)
        self.assertAlmostEqual(scores["interpretability"], 1.0)

    def test_interpretability_half_when_no_features(self):
        result = _cls_result("svm_rbf", accuracy=0.80)
        scores = score_model(result)
        self.assertAlmostEqual(scores["interpretability"], 0.5)

    def test_complexity_pca(self):
        result = AnalysisResult(
            task_name="unsupervised_exploration",
            model_name="pca",
            metrics={"explained_variance_ratio_cumulative": 0.9},
        )
        scores = score_model(result)
        self.assertAlmostEqual(scores["complexity"], 0.9)

    def test_complexity_pca_lda(self):
        result = _cls_result("pca_lda", accuracy=0.80)
        scores = score_model(result)
        self.assertAlmostEqual(scores["complexity"], 0.9)

    def test_complexity_random_forest(self):
        result = _cls_result("random_forest", accuracy=0.80)
        scores = score_model(result)
        self.assertAlmostEqual(scores["complexity"], 0.6)

    def test_complexity_plsr(self):
        result = AnalysisResult(
            task_name="regression",
            model_name="plsr",
            metrics={"r2": 0.85},
        )
        scores = score_model(result)
        self.assertAlmostEqual(scores["complexity"], 0.8)

    def test_composite_between_zero_and_one(self):
        result = _cls_result("svm_rbf", accuracy=0.75, warnings=(_warning(),))
        scores = score_model(result)
        self.assertGreaterEqual(scores["composite"], 0.0)
        self.assertLessEqual(scores["composite"], 1.0)

    def test_composite_formula(self):
        result = _cls_result("random_forest", accuracy=0.80, selected_features=(1.0,))
        scores = score_model(result)
        expected = 0.4 * 0.80 + 0.3 * 1.0 + 0.2 * 1.0 + 0.1 * 0.6
        self.assertAlmostEqual(scores["composite"], expected, places=6)


# ---------------------------------------------------------------------------
# select_best_model (core) tests
# ---------------------------------------------------------------------------


class TestCoreSelectBestModel(unittest.TestCase):
    def test_single_result_selected(self):
        r = _cls_result("random_forest", accuracy=0.82, selected_features=(1450.0,))
        rec = core_select([r])
        self.assertEqual(rec.selected_model, "random_forest")

    def test_higher_accuracy_with_warning_may_lose(self):
        # svm_rbf has higher accuracy but a validation warning → lower composite
        w = _warning(severity="warning")
        r_svm = _cls_result("svm_rbf", accuracy=0.95, warnings=(w,))
        r_rf = _cls_result("random_forest", accuracy=0.82, selected_features=(1450.0,))
        rec = core_select([r_svm, r_rf])
        # svm: 0.4*0.95 + 0.3*0.7 + 0.2*0.5 + 0.1*0.5 = 0.38+0.21+0.10+0.05 = 0.74
        # rf:  0.4*0.82 + 0.3*1.0 + 0.2*1.0 + 0.1*0.6 = 0.328+0.30+0.20+0.06 = 0.888
        self.assertEqual(rec.selected_model, "random_forest")

    def test_empty_results_returns_none(self):
        rec = core_select([])
        self.assertIsNone(rec.selected_model)
        self.assertIn("No results", rec.rationale)

    def test_task_name_filter(self):
        r1 = _cls_result("svm_rbf", accuracy=0.90, task_name="binary_classification")
        r2 = _cls_result("random_forest", accuracy=0.70, task_name="multiclass_classification")
        rec = core_select([r1, r2], task_name="binary_classification")
        self.assertEqual(rec.selected_model, "svm_rbf")
        self.assertNotIn("random_forest", rec.candidate_models)

    def test_task_name_filter_no_match_returns_none(self):
        r = _cls_result("svm_rbf", accuracy=0.90, task_name="binary_classification")
        rec = core_select([r], task_name="regression")
        self.assertIsNone(rec.selected_model)

    def test_requires_human_approval_true_for_suspicious_warning(self):
        w = ValidationWarning(
            code="suspicious_high_metric",
            message="Suspiciously high accuracy",
            severity="warning",
        )
        r = _cls_result("svm_rbf", accuracy=0.99, warnings=(w,))
        rec = core_select([r])
        self.assertTrue(rec.requires_human_approval)

    def test_requires_human_approval_false_when_no_issues(self):
        r = _cls_result("random_forest", accuracy=0.82)
        rec = core_select([r])
        self.assertFalse(rec.requires_human_approval)

    def test_requires_human_approval_true_when_validation_failed(self):
        r = _cls_result("svm_rbf", accuracy=0.80)
        vs = ValidationSummary(passed=False, checks={}, warnings=())
        rec = core_select([r], validation_summary=vs)
        self.assertTrue(rec.requires_human_approval)

    def test_warnings_deduplicated_by_code(self):
        w1 = ValidationWarning(code="same_code", message="first", severity="warning")
        w2 = ValidationWarning(code="same_code", message="second", severity="warning")
        r1 = _cls_result("svm_rbf", accuracy=0.80, warnings=(w1,))
        r2 = _cls_result("random_forest", accuracy=0.70, warnings=(w2,))
        rec = core_select([r1, r2])
        codes = [w.code for w in rec.warnings]
        self.assertEqual(codes.count("same_code"), 1)

    def test_candidate_models_contains_all(self):
        r1 = _cls_result("svm_rbf", accuracy=0.80)
        r2 = _cls_result("random_forest", accuracy=0.75)
        rec = core_select([r1, r2])
        self.assertIn("svm_rbf", rec.candidate_models)
        self.assertIn("random_forest", rec.candidate_models)


# ---------------------------------------------------------------------------
# Tool boundary tests
# ---------------------------------------------------------------------------


class TestSelectBestModelTool(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.mkdtemp()

    def _run(self, results, task_name=None):
        request = SelectBestModelRequest(results=tuple(results), task_name=task_name)
        return select_best_model.run(request, runs_root=self._tmp)

    def test_ok_true_for_non_empty_results(self):
        r = _cls_result("random_forest", accuracy=0.82)
        resp = self._run([r])
        self.assertTrue(resp.ok)
        self.assertEqual(resp.tool_name, "select_best_model")

    def test_payload_is_model_selection_recommendation(self):
        r = _cls_result("random_forest", accuracy=0.82)
        resp = self._run([r])
        self.assertIsInstance(resp.payload, ModelSelectionRecommendation)

    def test_ok_false_for_empty_results(self):
        resp = self._run([])
        self.assertFalse(resp.ok)
        self.assertIsNotNone(resp.error)

    def test_artifact_is_saved(self):
        r = _cls_result("random_forest", accuracy=0.82)
        resp = self._run([r])
        self.assertTrue(resp.artifacts)
        artifact_path = Path(resp.artifacts[0].uri)
        self.assertTrue(artifact_path.exists())
        data = json.loads(artifact_path.read_text())
        self.assertIn("selected_model", data)

    def test_selected_model_in_response(self):
        r = _cls_result("random_forest", accuracy=0.82, selected_features=(1.0,))
        resp = self._run([r])
        self.assertEqual(resp.payload.selected_model, "random_forest")


if __name__ == "__main__":
    unittest.main()
