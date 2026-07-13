"""Tests for validate_results tool and core/validation.py checks.

All tests use synthetic data only — no file I/O except artifact writes
which are directed to a temporary directory via tmp_path.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from chemometrics_contracts import (
    AnalysisResult,
    DatasetInspection,
    RunMetadata,
    SpectralDataset,
    ValidateResultsRequest,
    ValidationSummary,
)
from chemometrics_mcp.core.validation import (
    check_class_imbalance,
    check_group_leakage,
    check_missing_metadata,
    check_regression_target_leakage,
    check_replicate_leakage,
    check_small_sample_per_class,
    check_split_instability,
    check_suspicious_metrics,
    run_all_checks,
)
from chemometrics_mcp.tools import validate_results


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _classification_result(
    model_name: str,
    accuracy: float,
    balanced_accuracy: float | None = None,
    *,
    run_metadata: RunMetadata | None = None,
) -> AnalysisResult:
    metrics: dict = {"accuracy": accuracy}
    if balanced_accuracy is not None:
        metrics["balanced_accuracy"] = balanced_accuracy
    return AnalysisResult(
        task_name="binary_classification",
        model_name=model_name,
        metrics=metrics,
        run_metadata=run_metadata,
    )


def _regression_result(
    model_name: str,
    r2: float,
    *,
    run_metadata: RunMetadata | None = None,
) -> AnalysisResult:
    return AnalysisResult(
        task_name="regression",
        model_name=model_name,
        metrics={"r2": r2},
        run_metadata=run_metadata,
    )


def _dataset_with_labels(labels: list) -> SpectralDataset:
    n = len(labels)
    # Minimal valid SpectralDataset — one feature column
    return SpectralDataset(
        x=tuple((float(i),) for i in range(n)),
        axis=(1.0,),
        labels=tuple(labels),
    )


def _dummy_run_metadata(run_id: str = "run-test-001") -> RunMetadata:
    return RunMetadata(run_id=run_id, tool_name="run_analysis")


# ---------------------------------------------------------------------------
# check_suspicious_metrics
# ---------------------------------------------------------------------------

class TestCheckSuspiciousMetrics(unittest.TestCase):

    def test_returns_none_for_safe_accuracy(self):
        result = _classification_result("LDA", accuracy=0.95)
        self.assertIsNone(check_suspicious_metrics(result))

    def test_returns_warning_for_high_accuracy(self):
        result = _classification_result("LDA", accuracy=0.999)
        w = check_suspicious_metrics(result)
        self.assertIsNotNone(w)
        self.assertEqual(w.code, "suspicious_high_metric")
        self.assertEqual(w.severity, "warning")
        self.assertEqual(w.category, "reliability")
        self.assertIn("LDA", w.message)

    def test_returns_none_for_regression_task(self):
        result = _regression_result("PLS", r2=0.999)
        self.assertIsNone(check_suspicious_metrics(result))

    def test_triggers_on_balanced_accuracy(self):
        result = _classification_result("SVM", accuracy=0.85, balanced_accuracy=0.995)
        w = check_suspicious_metrics(result)
        self.assertIsNotNone(w)
        self.assertIn("balanced_accuracy", w.message)

    def test_exact_threshold_triggers(self):
        # >= 0.99 should trigger
        result = _classification_result("LDA", accuracy=0.99)
        self.assertIsNotNone(check_suspicious_metrics(result))

    def test_just_below_threshold_passes(self):
        result = _classification_result("LDA", accuracy=0.989)
        self.assertIsNone(check_suspicious_metrics(result))


# ---------------------------------------------------------------------------
# check_small_sample_per_class
# ---------------------------------------------------------------------------

class TestCheckSmallSamplePerClass(unittest.TestCase):

    def test_returns_none_when_all_classes_have_enough_samples(self):
        dataset = _dataset_with_labels(["A"] * 5 + ["B"] * 5)
        result = _classification_result("LDA", accuracy=0.8)
        self.assertIsNone(check_small_sample_per_class(result, dataset))

    def test_returns_warning_for_class_with_3_samples(self):
        dataset = _dataset_with_labels(["A"] * 10 + ["B"] * 3)
        result = _classification_result("LDA", accuracy=0.8)
        w = check_small_sample_per_class(result, dataset)
        self.assertIsNotNone(w)
        self.assertEqual(w.code, "small_sample_per_class")
        self.assertIn("B", w.message)
        self.assertIn("3", w.message)

    def test_returns_none_when_dataset_is_none(self):
        result = _classification_result("LDA", accuracy=0.8)
        self.assertIsNone(check_small_sample_per_class(result, None))

    def test_returns_none_when_labels_are_none(self):
        dataset = SpectralDataset(x=((1.0,), (2.0,)), axis=(1.0,), labels=None)
        result = _classification_result("LDA", accuracy=0.8)
        self.assertIsNone(check_small_sample_per_class(result, dataset))

    def test_exactly_5_samples_passes(self):
        dataset = _dataset_with_labels(["A"] * 5 + ["B"] * 5)
        result = _classification_result("LDA", accuracy=0.8)
        self.assertIsNone(check_small_sample_per_class(result, dataset))

    def test_4_samples_triggers(self):
        dataset = _dataset_with_labels(["A"] * 4 + ["B"] * 10)
        result = _classification_result("LDA", accuracy=0.8)
        w = check_small_sample_per_class(result, dataset)
        self.assertIsNotNone(w)


# ---------------------------------------------------------------------------
# check_class_imbalance
# ---------------------------------------------------------------------------

class TestCheckClassImbalance(unittest.TestCase):

    def test_returns_none_for_balanced_classes(self):
        dataset = _dataset_with_labels(["A"] * 10 + ["B"] * 10)
        result = _classification_result("LDA", accuracy=0.8)
        self.assertIsNone(check_class_imbalance(result, dataset))

    def test_returns_warning_for_10_to_1_ratio(self):
        dataset = _dataset_with_labels(["A"] * 10 + ["B"] * 1)
        result = _classification_result("LDA", accuracy=0.8)
        w = check_class_imbalance(result, dataset)
        self.assertIsNotNone(w)
        self.assertEqual(w.code, "class_imbalance")
        self.assertIn("10.0x", w.message)

    def test_returns_none_when_dataset_is_none(self):
        result = _classification_result("LDA", accuracy=0.8)
        self.assertIsNone(check_class_imbalance(result, None))

    def test_returns_none_for_regression_task(self):
        dataset = _dataset_with_labels([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
        result = _regression_result("PLS", r2=0.8)
        self.assertIsNone(check_class_imbalance(result, dataset))

    def test_exactly_3_ratio_passes(self):
        dataset = _dataset_with_labels(["A"] * 3 + ["B"] * 1)
        result = _classification_result("LDA", accuracy=0.8)
        self.assertIsNone(check_class_imbalance(result, dataset))

    def test_just_above_3_triggers(self):
        dataset = _dataset_with_labels(["A"] * 4 + ["B"] * 1)
        result = _classification_result("LDA", accuracy=0.8)
        w = check_class_imbalance(result, dataset)
        self.assertIsNotNone(w)


# ---------------------------------------------------------------------------
# check_missing_metadata
# ---------------------------------------------------------------------------

class TestCheckMissingMetadata(unittest.TestCase):

    def test_returns_warning_when_run_metadata_is_none(self):
        result = _classification_result("LDA", accuracy=0.8, run_metadata=None)
        w = check_missing_metadata(result)
        self.assertIsNotNone(w)
        self.assertEqual(w.code, "missing_run_metadata")
        self.assertEqual(w.severity, "info")
        self.assertEqual(w.category, "provenance")
        self.assertIn("LDA", w.message)

    def test_returns_none_when_run_metadata_is_present(self):
        meta = _dummy_run_metadata()
        result = _classification_result("LDA", accuracy=0.8, run_metadata=meta)
        self.assertIsNone(check_missing_metadata(result))


# ---------------------------------------------------------------------------
# check_regression_target_leakage
# ---------------------------------------------------------------------------

class TestCheckRegressionTargetLeakage(unittest.TestCase):

    def test_returns_none_for_safe_r2(self):
        result = _regression_result("PLS", r2=0.85)
        self.assertIsNone(check_regression_target_leakage(result))

    def test_returns_warning_for_high_r2(self):
        result = _regression_result("PLS", r2=0.999)
        w = check_regression_target_leakage(result)
        self.assertIsNotNone(w)
        self.assertEqual(w.code, "suspicious_regression_r2")
        self.assertEqual(w.severity, "warning")
        self.assertEqual(w.category, "reliability")
        self.assertIn("PLS", w.message)

    def test_returns_none_for_classification_task(self):
        result = _classification_result("LDA", accuracy=0.999)
        self.assertIsNone(check_regression_target_leakage(result))

    def test_exactly_at_threshold_does_not_trigger(self):
        # strictly > 0.99
        result = _regression_result("PLS", r2=0.99)
        self.assertIsNone(check_regression_target_leakage(result))

    def test_just_above_threshold_triggers(self):
        result = _regression_result("PLS", r2=0.991)
        self.assertIsNotNone(check_regression_target_leakage(result))


# ---------------------------------------------------------------------------
# run_all_checks
# ---------------------------------------------------------------------------

class TestRunAllChecks(unittest.TestCase):

    def test_all_checks_pass(self):
        meta = _dummy_run_metadata()
        result = _classification_result("LDA", accuracy=0.85, run_metadata=meta)
        dataset = _dataset_with_labels(["A"] * 10 + ["B"] * 10)
        summary = run_all_checks([result], dataset)
        self.assertTrue(summary.passed)
        self.assertTrue(all(summary.checks.values()))
        self.assertEqual(len(summary.warnings), 0)

    def test_suspicious_metric_causes_failure(self):
        meta = _dummy_run_metadata()
        result = _classification_result("LDA", accuracy=0.999, run_metadata=meta)
        dataset = _dataset_with_labels(["A"] * 10 + ["B"] * 10)
        summary = run_all_checks([result], dataset)
        self.assertFalse(summary.passed)
        self.assertFalse(summary.checks["suspicious_metrics"])
        # Other checks should still pass
        self.assertTrue(summary.checks["class_imbalance"])
        self.assertTrue(summary.checks["metadata_present"])

    def test_warnings_accumulated_from_multiple_results(self):
        # Two results: one with suspicious accuracy, one with missing metadata
        r1 = _classification_result("LDA", accuracy=0.999, run_metadata=_dummy_run_metadata("r1"))
        r2 = _classification_result("SVM", accuracy=0.85, run_metadata=None)
        summary = run_all_checks([r1, r2])
        codes = {w.code for w in summary.warnings}
        self.assertIn("suspicious_high_metric", codes)
        self.assertIn("missing_run_metadata", codes)
        self.assertFalse(summary.passed)

    def test_empty_results_gives_passed_none(self):
        summary = run_all_checks([])
        self.assertIsNone(summary.passed)
        self.assertEqual(len(summary.warnings), 0)

    def test_all_checks_keys_present(self):
        meta = _dummy_run_metadata()
        result = _classification_result("LDA", accuracy=0.8, run_metadata=meta)
        summary = run_all_checks([result])
        expected_keys = {
            "suspicious_metrics",
            "small_sample_per_class",
            "class_imbalance",
            "metadata_present",
            "regression_leakage",
            "replicate_leakage",
            "group_leakage",
            "split_instability",
            "modality_consistency",
        }
        self.assertEqual(set(summary.checks.keys()), expected_keys)

    def test_multiple_small_sample_classes_all_warned(self):
        dataset = _dataset_with_labels(["A"] * 2 + ["B"] * 2 + ["C"] * 20)
        meta = _dummy_run_metadata()
        result = _classification_result("LDA", accuracy=0.8, run_metadata=meta)
        summary = run_all_checks([result], dataset)
        small_sample_warnings = [w for w in summary.warnings if w.code == "small_sample_per_class"]
        # Both A and B should be flagged
        self.assertEqual(len(small_sample_warnings), 2)


# ---------------------------------------------------------------------------
# Tool boundary tests (validate_results.run)
# ---------------------------------------------------------------------------

class TestValidateResultsTool(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.runs_root = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_returns_ok_true_for_non_empty_results(self):
        meta = _dummy_run_metadata()
        result = _classification_result("LDA", accuracy=0.8, run_metadata=meta)
        request = ValidateResultsRequest(results=(result,))
        response = validate_results.run(request, runs_root=self.runs_root)
        self.assertTrue(response.ok)
        self.assertEqual(response.tool_name, "validate_results")
        self.assertIsNotNone(response.payload)
        self.assertIsInstance(response.payload, ValidationSummary)

    def test_empty_results_no_analysis_run_returns_ok_false(self):
        request = ValidateResultsRequest()
        response = validate_results.run(request, runs_root=self.runs_root)
        self.assertFalse(response.ok)
        self.assertIn("No results", response.message or response.error or "")

    def test_validation_summary_json_artifact_is_created(self):
        meta = _dummy_run_metadata()
        result = _classification_result("SVM", accuracy=0.75, run_metadata=meta)
        request = ValidateResultsRequest(results=(result,))
        response = validate_results.run(request, runs_root=self.runs_root)
        self.assertTrue(response.ok)
        # There should be exactly one artifact pointing to validation_summary.json
        self.assertEqual(len(response.artifacts), 1)
        artifact_uri = response.artifacts[0].uri
        artifact_path = Path(artifact_uri)
        self.assertTrue(artifact_path.exists(), f"Artifact not found: {artifact_path}")
        data = json.loads(artifact_path.read_text())
        # The JSON should contain the ValidationSummary fields
        self.assertIn("passed", data)
        self.assertIn("checks", data)
        self.assertIn("warnings", data)

    def test_results_from_analysis_run_are_used_when_request_results_empty(self):
        from chemometrics_contracts import AnalysisRun
        meta = _dummy_run_metadata()
        result = _classification_result("LDA", accuracy=0.8, run_metadata=meta)
        analysis_run = AnalysisRun(results=(result,))
        request = ValidateResultsRequest(analysis_run=analysis_run)
        response = validate_results.run(request, runs_root=self.runs_root)
        self.assertTrue(response.ok)

    def test_warnings_from_summary_propagated_to_response(self):
        # No metadata → warning should propagate
        result = _classification_result("LDA", accuracy=0.8, run_metadata=None)
        request = ValidateResultsRequest(results=(result,))
        response = validate_results.run(request, runs_root=self.runs_root)
        self.assertTrue(response.ok)
        codes = {w.code for w in response.warnings}
        self.assertIn("missing_run_metadata", codes)


# ---------------------------------------------------------------------------
# Helper builders for leakage tests
# ---------------------------------------------------------------------------

def _classification_result_with_predictions(
    model_name: str,
    predictions: list,
    *,
    validation_strategy: str | None = None,
    run_metadata: RunMetadata | None = None,
) -> AnalysisResult:
    return AnalysisResult(
        task_name="binary_classification",
        model_name=model_name,
        metrics={"accuracy": 0.85},
        predictions=tuple(predictions),
        run_metadata=run_metadata,
        validation_strategy=validation_strategy,
    )


def _dataset_with_metadata_and_predictions(
    metadata: list[dict[str, str]],
) -> SpectralDataset:
    n = len(metadata)
    return SpectralDataset(
        x=tuple((float(i),) for i in range(n)),
        axis=(1.0,),
        metadata=tuple(metadata),
        sample_ids=tuple(f"sample_{i}" for i in range(n)),
    )


def _inspection_with_groups(
    candidate_group_columns: tuple[str, ...] = ("batch",),
) -> DatasetInspection:
    return DatasetInspection(
        sample_count=10,
        feature_count=100,
        modality="NIR",
        candidate_group_columns=candidate_group_columns,
    )


# ---------------------------------------------------------------------------
# check_replicate_leakage
# ---------------------------------------------------------------------------

class TestCheckReplicateLeakage(unittest.TestCase):

    def test_returns_empty_when_no_inspection(self):
        result = _classification_result_with_predictions("LDA", ["A", "A", "B", "B"])
        dataset = _dataset_with_metadata_and_predictions([
            {"batch": "B1"}, {"batch": "B1"}, {"batch": "B2"}, {"batch": "B2"},
        ])
        warnings = check_replicate_leakage(result, dataset, None)
        self.assertEqual(warnings, [])

    def test_returns_empty_when_no_predictions(self):
        result = _classification_result_with_predictions("LDA", [])
        dataset = _dataset_with_metadata_and_predictions([
            {"batch": "B1"}, {"batch": "B1"},
        ])
        inspection = _inspection_with_groups(("batch",))
        warnings = check_replicate_leakage(result, dataset, inspection)
        self.assertEqual(warnings, [])

    def test_returns_empty_when_no_candidate_groups(self):
        result = _classification_result_with_predictions("LDA", ["A", "A", "B", "B"])
        dataset = _dataset_with_metadata_and_predictions([
            {"batch": "B1"}, {"batch": "B1"}, {"batch": "B2"}, {"batch": "B2"},
        ])
        inspection = DatasetInspection(sample_count=4, feature_count=100, candidate_group_columns=())
        warnings = check_replicate_leakage(result, dataset, inspection)
        self.assertEqual(warnings, [])

    def test_returns_empty_when_metadata_length_mismatch(self):
        result = _classification_result_with_predictions("LDA", ["A", "A", "B", "B"])
        dataset = _dataset_with_metadata_and_predictions([
            {"batch": "B1"}, {"batch": "B1"},
        ])
        inspection = _inspection_with_groups(("batch",))
        warnings = check_replicate_leakage(result, dataset, inspection)
        self.assertEqual(warnings, [])

    def test_high_consistency_triggers_warning(self):
        predictions = ["A", "A", "A", "A", "B", "B", "B", "B"]
        result = _classification_result_with_predictions("LDA", predictions)
        metadata = [
            {"batch": "B1"}, {"batch": "B1"}, {"batch": "B1"}, {"batch": "B1"},
            {"batch": "B2"}, {"batch": "B2"}, {"batch": "B2"}, {"batch": "B2"},
        ]
        dataset = _dataset_with_metadata_and_predictions(metadata)
        inspection = _inspection_with_groups(("batch",))
        warnings = check_replicate_leakage(result, dataset, inspection)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].code, "replicate_leakage")
        self.assertIn("batch", warnings[0].message)

    def test_low_consistency_does_not_trigger(self):
        predictions = ["A", "B", "A", "B", "A", "B", "A", "B"]
        result = _classification_result_with_predictions("LDA", predictions)
        metadata = [
            {"batch": "B1"}, {"batch": "B1"}, {"batch": "B1"}, {"batch": "B1"},
            {"batch": "B2"}, {"batch": "B2"}, {"batch": "B2"}, {"batch": "B2"},
        ]
        dataset = _dataset_with_metadata_and_predictions(metadata)
        inspection = _inspection_with_groups(("batch",))
        warnings = check_replicate_leakage(result, dataset, inspection)
        self.assertEqual(warnings, [])

    def test_single_member_groups_no_warning(self):
        predictions = ["A", "A", "B", "B"]
        result = _classification_result_with_predictions("LDA", predictions)
        metadata = [
            {"batch": "B1"}, {"batch": "B2"}, {"batch": "B3"}, {"batch": "B4"},
        ]
        dataset = _dataset_with_metadata_and_predictions(metadata)
        inspection = _inspection_with_groups(("batch",))
        warnings = check_replicate_leakage(result, dataset, inspection)
        self.assertEqual(warnings, [])

    def test_deterministic_with_large_group(self):
        n = 25
        predictions = ["A"] * n
        result = _classification_result_with_predictions("LDA", predictions)
        metadata = [{"batch": "B1"} for _ in range(n)]
        dataset = _dataset_with_metadata_and_predictions(metadata)
        inspection = _inspection_with_groups(("batch",))
        w1 = check_replicate_leakage(result, dataset, inspection)
        w2 = check_replicate_leakage(result, dataset, inspection)
        self.assertEqual(len(w1), len(w2))
        for a, b in zip(w1, w2):
            self.assertEqual(a.code, b.code)
            self.assertEqual(a.message, b.message)
            self.assertEqual(a.details, b.details)

    def test_multiple_group_columns_each_checked(self):
        predictions = ["A", "A", "A", "A"]
        result = _classification_result_with_predictions("LDA", predictions)
        metadata = [
            {"batch": "B1", "replicate": "R1"},
            {"batch": "B1", "replicate": "R1"},
            {"batch": "B1", "replicate": "R2"},
            {"batch": "B1", "replicate": "R2"},
        ]
        dataset = _dataset_with_metadata_and_predictions(metadata)
        inspection = _inspection_with_groups(("batch", "replicate"))
        warnings = check_replicate_leakage(result, dataset, inspection)
        codes = {w.code for w in warnings}
        self.assertIn("replicate_leakage", codes)


# ---------------------------------------------------------------------------
# check_group_leakage
# ---------------------------------------------------------------------------

class TestCheckGroupLeakage(unittest.TestCase):

    def test_returns_empty_when_no_inspection(self):
        result = _classification_result_with_predictions(
            "LDA", ["A", "B"], validation_strategy="stratified_kfold_5"
        )
        warnings = check_group_leakage([result], None)
        self.assertEqual(warnings, [])

    def test_returns_empty_when_no_candidate_groups(self):
        result = _classification_result_with_predictions(
            "LDA", ["A", "B"], validation_strategy="stratified_kfold_5"
        )
        inspection = DatasetInspection(sample_count=2, feature_count=100, candidate_group_columns=())
        warnings = check_group_leakage([result], inspection)
        self.assertEqual(warnings, [])

    def test_grouped_strategy_does_not_trigger(self):
        result = _classification_result_with_predictions(
            "LDA", ["A", "B"], validation_strategy="grouped_kfold_5"
        )
        inspection = _inspection_with_groups(("batch",))
        warnings = check_group_leakage([result], inspection)
        self.assertEqual(warnings, [])

    def test_non_grouped_strategy_triggers_warning(self):
        result = _classification_result_with_predictions(
            "LDA", ["A", "B"], validation_strategy="stratified_kfold_5"
        )
        inspection = _inspection_with_groups(("batch",))
        warnings = check_group_leakage([result], inspection)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].code, "group_leakage_risk")
        self.assertIn("batch", warnings[0].message)
        self.assertIn("stratified_kfold_5", warnings[0].message)

    def test_missing_validation_strategy_triggers_warning(self):
        result = _classification_result_with_predictions("LDA", ["A", "B"])
        inspection = _inspection_with_groups(("batch",))
        warnings = check_group_leakage([result], inspection)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].code, "group_leakage_risk")
        self.assertIn("not specified", warnings[0].message)

    def test_multiple_results_each_warned(self):
        results = [
            _classification_result_with_predictions("LDA", ["A", "B"], validation_strategy="stratified_kfold_5"),
            _classification_result_with_predictions("SVM", ["A", "B"], validation_strategy="stratified_kfold_5"),
        ]
        inspection = _inspection_with_groups(("batch",))
        warnings = check_group_leakage(results, inspection)
        self.assertEqual(len(warnings), 2)

    def test_mixed_strategies_only_non_grouped_warned(self):
        results = [
            _classification_result_with_predictions("LDA", ["A", "B"], validation_strategy="grouped_kfold_5"),
            _classification_result_with_predictions("SVM", ["A", "B"], validation_strategy="stratified_kfold_5"),
        ]
        inspection = _inspection_with_groups(("batch",))
        warnings = check_group_leakage(results, inspection)
        self.assertEqual(len(warnings), 1)
        self.assertIn("SVM", warnings[0].message)


# ---------------------------------------------------------------------------
# run_all_checks with new leakage checks
# ---------------------------------------------------------------------------

class TestRunAllChecksLeakage(unittest.TestCase):

    def test_new_check_keys_present(self):
        meta = _dummy_run_metadata()
        result = _classification_result("LDA", accuracy=0.8, run_metadata=meta)
        summary = run_all_checks([result])
        self.assertIn("replicate_leakage", summary.checks)
        self.assertIn("group_leakage", summary.checks)

    def test_replicate_leakage_detected_in_run_all_checks(self):
        predictions = ["A", "A", "A", "A", "B", "B", "B", "B"]
        meta = _dummy_run_metadata()
        result = _classification_result_with_predictions("LDA", predictions, run_metadata=meta)
        metadata = [
            {"batch": "B1"}, {"batch": "B1"}, {"batch": "B1"}, {"batch": "B1"},
            {"batch": "B2"}, {"batch": "B2"}, {"batch": "B2"}, {"batch": "B2"},
        ]
        dataset = _dataset_with_metadata_and_predictions(metadata)
        inspection = _inspection_with_groups(("batch",))
        summary = run_all_checks([result], dataset, inspection)
        self.assertFalse(summary.checks["replicate_leakage"])
        self.assertFalse(summary.checks["group_leakage"])
        codes = {w.code for w in summary.warnings}
        self.assertIn("replicate_leakage", codes)
        self.assertIn("group_leakage_risk", codes)

    def test_grouped_strategy_passes_group_leakage(self):
        predictions = ["A", "A", "A", "A", "B", "B", "B", "B"]
        meta = _dummy_run_metadata()
        result = _classification_result_with_predictions(
            "LDA", predictions, validation_strategy="grouped_kfold_5", run_metadata=meta
        )
        metadata = [
            {"batch": "B1"}, {"batch": "B1"}, {"batch": "B1"}, {"batch": "B1"},
            {"batch": "B2"}, {"batch": "B2"}, {"batch": "B2"}, {"batch": "B2"},
        ]
        dataset = _dataset_with_metadata_and_predictions(metadata)
        inspection = _inspection_with_groups(("batch",))
        summary = run_all_checks([result], dataset, inspection)
        self.assertTrue(summary.checks["group_leakage"])
        self.assertFalse(summary.checks["replicate_leakage"])

    def test_no_inspection_means_leakage_checks_pass(self):
        meta = _dummy_run_metadata()
        result = _classification_result("LDA", accuracy=0.8, run_metadata=meta)
        summary = run_all_checks([result])
        self.assertTrue(summary.checks["replicate_leakage"])
        self.assertTrue(summary.checks["group_leakage"])


# ---------------------------------------------------------------------------
# check_split_instability
# ---------------------------------------------------------------------------

class TestCheckSplitInstability(unittest.TestCase):

    def test_single_result_per_model_no_warning(self):
        r1 = _classification_result("svm_rbf", accuracy=0.85)
        r2 = _regression_result("plsr", r2=0.80)
        warnings = check_split_instability([r1, r2])
        self.assertEqual(warnings, [])

    def test_low_variance_no_warning(self):
        results = [
            _classification_result("svm_rbf", accuracy=0.80),
            _classification_result("svm_rbf", accuracy=0.81),
            _classification_result("svm_rbf", accuracy=0.805),
        ]
        warnings = check_split_instability(results)
        self.assertEqual(warnings, [])

    def test_high_variance_returns_warning(self):
        results = [
            _classification_result("svm_rbf", accuracy=0.60),
            _classification_result("svm_rbf", accuracy=0.85),
            _classification_result("svm_rbf", accuracy=0.65),
        ]
        warnings = check_split_instability(results)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].code, "split_instability")
        self.assertEqual(warnings[0].severity, "warning")

    def test_very_high_variance_returns_error(self):
        results = [
            _regression_result("plsr", r2=0.10),
            _regression_result("plsr", r2=0.95),
            _regression_result("plsr", r2=0.15),
        ]
        warnings = check_split_instability(results)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].severity, "error")

    def test_mixed_classification_and_regression(self):
        results = [
            _classification_result("svm_rbf", accuracy=0.60),
            _classification_result("svm_rbf", accuracy=0.85),
            _classification_result("svm_rbf", accuracy=0.65),
            _regression_result("plsr", r2=0.80),
            _regression_result("plsr", r2=0.81),
            _regression_result("plsr", r2=0.805),
        ]
        warnings = check_split_instability(results)
        svm_warnings = [w for w in warnings if "svm_rbf" in w.message]
        plsr_warnings = [w for w in warnings if "plsr" in w.message]
        self.assertEqual(len(svm_warnings), 1)
        self.assertEqual(len(plsr_warnings), 0)

    def test_split_instability_in_run_all_checks(self):
        results = [
            _classification_result("svm_rbf", accuracy=0.60),
            _classification_result("svm_rbf", accuracy=0.85),
            _classification_result("svm_rbf", accuracy=0.65),
        ]
        summary = run_all_checks(results)
        self.assertIn("split_instability", summary.checks)
        self.assertFalse(summary.checks["split_instability"])
        codes = {w.code for w in summary.warnings}
        self.assertIn("split_instability", codes)

    def test_split_instability_disabled(self):
        results = [
            _classification_result("svm_rbf", accuracy=0.60),
            _classification_result("svm_rbf", accuracy=0.85),
            _classification_result("svm_rbf", accuracy=0.65),
        ]
        summary = run_all_checks(results, split_instability=False)
        self.assertTrue(summary.checks["split_instability"])
        codes = {w.code for w in summary.warnings}
        self.assertNotIn("split_instability", codes)


if __name__ == "__main__":
    unittest.main()
