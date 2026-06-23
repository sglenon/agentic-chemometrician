"""Tests for propose_analysis_plan: core planning logic and tool boundary."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from chemometrics_contracts import (
    DatasetInspection,
    ProposeAnalysisPlanRequest,
    ValidationWarning,
)

from chemometrics_mcp.core.planning import (
    build_plan,
    infer_task_name,
    recommend_model_families,
    recommend_preprocessing,
    recommend_validation_strategy,
)
from chemometrics_mcp.tools import propose_analysis_plan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inspection(**kwargs) -> DatasetInspection:
    defaults = dict(
        sample_count=146,
        feature_count=249,
        axis_min=1100.0,
        axis_max=2500.0,
        modality="NIR",
        candidate_label_columns=(),
        candidate_group_columns=(),
        warnings=(),
    )
    defaults.update(kwargs)
    return DatasetInspection(**defaults)


def _request(inspection: DatasetInspection, **kwargs) -> ProposeAnalysisPlanRequest:
    defaults = dict(
        dataset_inspection=inspection,
        user_intent=None,
        task_hint=None,
        allow_supervised_planning=True,
    )
    defaults.update(kwargs)
    return ProposeAnalysisPlanRequest(**defaults)


# ---------------------------------------------------------------------------
# Core planning: infer_task_name
# ---------------------------------------------------------------------------

class TestInferTaskName(unittest.TestCase):

    def test_one_label_column_defaults_to_multi_class(self):
        insp = _inspection(candidate_label_columns=("Measurement Description",))
        result = infer_task_name(insp, task_hint=None)
        self.assertEqual(result, "multi_class_classification")

    def test_no_label_columns_gives_unsupervised(self):
        insp = _inspection(candidate_label_columns=())
        result = infer_task_name(insp, task_hint=None)
        self.assertEqual(result, "unsupervised_exploration")

    def test_multiple_label_columns_gives_none(self):
        insp = _inspection(candidate_label_columns=("col_a", "col_b"))
        result = infer_task_name(insp, task_hint=None)
        self.assertIsNone(result)

    def test_task_hint_regression_overrides(self):
        insp = _inspection(candidate_label_columns=("Measurement Description",))
        result = infer_task_name(insp, task_hint="regression")
        self.assertEqual(result, "regression")

    def test_task_hint_binary_classification_overrides(self):
        insp = _inspection(candidate_label_columns=("Measurement Description",))
        result = infer_task_name(insp, task_hint="binary_classification")
        self.assertEqual(result, "binary_classification")

    def test_task_hint_clustering_overrides(self):
        insp = _inspection(candidate_label_columns=("Measurement Description",))
        result = infer_task_name(insp, task_hint="clustering")
        self.assertEqual(result, "clustering")

    def test_allow_supervised_planning_false_gives_unsupervised(self):
        insp = _inspection(candidate_label_columns=("Measurement Description",))
        result = infer_task_name(insp, task_hint=None, allow_supervised_planning=False)
        self.assertEqual(result, "unsupervised_exploration")


# ---------------------------------------------------------------------------
# Core planning: recommend_preprocessing
# ---------------------------------------------------------------------------

class TestRecommendPreprocessing(unittest.TestCase):

    def test_nir_modality_standard_candidates(self):
        insp = _inspection(modality="NIR")
        result = recommend_preprocessing(insp)
        self.assertEqual(result[:4], ["snv", "msc", "sg_1st_deriv", "sg_2nd_deriv"])
        self.assertNotIn("baseline_correction", result)

    def test_ftir_modality_adds_extra(self):
        insp = _inspection(modality="FTIR")
        result = recommend_preprocessing(insp)
        self.assertIn("baseline_correction", result)
        self.assertIn("area_normalization", result)

    def test_ftir_case_insensitive(self):
        insp = _inspection(modality="ftir")
        result = recommend_preprocessing(insp)
        self.assertIn("baseline_correction", result)

    def test_order_of_standard_candidates(self):
        insp = _inspection(modality="NIR")
        result = recommend_preprocessing(insp)
        self.assertEqual(result[0], "snv")
        self.assertEqual(result[1], "msc")
        self.assertEqual(result[2], "sg_1st_deriv")
        self.assertEqual(result[3], "sg_2nd_deriv")


# ---------------------------------------------------------------------------
# Core planning: recommend_validation_strategy
# ---------------------------------------------------------------------------

class TestRecommendValidationStrategy(unittest.TestCase):

    def test_group_columns_gives_grouped_kfold(self):
        insp = _inspection(candidate_group_columns=("batch",), sample_count=100)
        result = recommend_validation_strategy(insp)
        self.assertEqual(result, "grouped_kfold_5")

    def test_large_dataset_gives_stratified_kfold_5(self):
        insp = _inspection(sample_count=100)
        result = recommend_validation_strategy(insp)
        self.assertEqual(result, "stratified_kfold_5")

    def test_exactly_50_samples_gives_stratified_kfold_5(self):
        insp = _inspection(sample_count=50)
        result = recommend_validation_strategy(insp)
        self.assertEqual(result, "stratified_kfold_5")

    def test_none_sample_count_gives_stratified_kfold_5(self):
        insp = _inspection(sample_count=None)
        result = recommend_validation_strategy(insp)
        self.assertEqual(result, "stratified_kfold_5")

    def test_medium_dataset_gives_stratified_kfold_3(self):
        insp = _inspection(sample_count=30)
        result = recommend_validation_strategy(insp)
        self.assertEqual(result, "stratified_kfold_3")

    def test_small_dataset_gives_loocv(self):
        insp = _inspection(sample_count=10)
        result = recommend_validation_strategy(insp)
        self.assertEqual(result, "loocv")


# ---------------------------------------------------------------------------
# Core planning: recommend_model_families
# ---------------------------------------------------------------------------

class TestRecommendModelFamilies(unittest.TestCase):

    def test_multi_class_classification(self):
        insp = _inspection()
        result = recommend_model_families("multi_class_classification", insp)
        self.assertEqual(result, ["svm_rbf", "random_forest", "pca_lda"])

    def test_binary_classification(self):
        insp = _inspection()
        result = recommend_model_families("binary_classification", insp)
        self.assertEqual(result, ["svm_rbf", "random_forest", "pca_lda"])

    def test_regression(self):
        insp = _inspection()
        result = recommend_model_families("regression", insp)
        self.assertEqual(result, ["plsr", "svr", "random_forest"])

    def test_unsupervised_exploration(self):
        insp = _inspection()
        result = recommend_model_families("unsupervised_exploration", insp)
        self.assertEqual(result, ["pca", "kmeans"])

    def test_clustering(self):
        insp = _inspection()
        result = recommend_model_families("clustering", insp)
        self.assertEqual(result, ["pca", "kmeans"])

    def test_none_task_gives_exploratory_fallback(self):
        insp = _inspection()
        result = recommend_model_families(None, insp)
        self.assertEqual(result, ["pca"])


# ---------------------------------------------------------------------------
# Core planning: build_plan (integration-style)
# ---------------------------------------------------------------------------

class TestBuildPlan(unittest.TestCase):

    def test_nir_with_one_label_column(self):
        insp = _inspection(candidate_label_columns=("Measurement Description",))
        req = _request(insp)
        plan = build_plan(req)
        self.assertEqual(plan.task_name, "multi_class_classification")

    def test_no_label_columns_unsupervised(self):
        insp = _inspection(candidate_label_columns=())
        req = _request(insp)
        plan = build_plan(req)
        self.assertEqual(plan.task_name, "unsupervised_exploration")

    def test_multiple_label_columns_ambiguous(self):
        insp = _inspection(candidate_label_columns=("col_a", "col_b"))
        req = _request(insp)
        plan = build_plan(req)
        self.assertIsNone(plan.task_name)
        warning_codes = [w.code for w in plan.warnings]
        self.assertIn("ambiguous_label_columns", warning_codes)

    def test_task_hint_regression(self):
        insp = _inspection(candidate_label_columns=("Measurement Description",))
        req = _request(insp, task_hint="regression")
        plan = build_plan(req)
        self.assertEqual(plan.task_name, "regression")

    def test_allow_supervised_planning_false(self):
        insp = _inspection(candidate_label_columns=("Measurement Description",))
        req = _request(insp, allow_supervised_planning=False)
        plan = build_plan(req)
        self.assertEqual(plan.task_name, "unsupervised_exploration")

    def test_group_columns_gives_grouped_kfold(self):
        insp = _inspection(
            candidate_label_columns=("label",),
            candidate_group_columns=("batch",),
        )
        req = _request(insp)
        plan = build_plan(req)
        self.assertEqual(plan.validation_strategy, "grouped_kfold_5")

    def test_large_dataset_gives_stratified_kfold_5(self):
        insp = _inspection(sample_count=100, candidate_label_columns=("label",))
        req = _request(insp)
        plan = build_plan(req)
        self.assertEqual(plan.validation_strategy, "stratified_kfold_5")

    def test_small_sample_warning_emitted(self):
        insp = _inspection(sample_count=15, candidate_label_columns=("label",))
        req = _request(insp)
        plan = build_plan(req)
        warning_codes = [w.code for w in plan.warnings]
        self.assertIn("small_sample", warning_codes)

    def test_ftir_modality_preprocessing(self):
        insp = _inspection(modality="FTIR", candidate_label_columns=("label",))
        req = _request(insp)
        plan = build_plan(req)
        self.assertIn("baseline_correction", plan.preprocessing_candidates)

    def test_nir_modality_no_baseline_correction(self):
        insp = _inspection(modality="NIR", candidate_label_columns=("label",))
        req = _request(insp)
        plan = build_plan(req)
        self.assertNotIn("baseline_correction", plan.preprocessing_candidates)

    def test_inspection_warnings_propagated(self):
        existing_warning = ValidationWarning(
            code="existing_issue",
            message="something was off",
            category="data_quality",
        )
        insp = _inspection(
            candidate_label_columns=("label",),
            warnings=(existing_warning,),
        )
        req = _request(insp)
        plan = build_plan(req)
        warning_codes = [w.code for w in plan.warnings]
        self.assertIn("existing_issue", warning_codes)

    def test_human_readable_plan_ends_with_approval(self):
        insp = _inspection(candidate_label_columns=("Measurement Description",))
        req = _request(insp)
        plan = build_plan(req)
        self.assertTrue(
            plan.human_readable_plan.endswith("Approval required before running."),
            msg=f"Plan did not end with approval line: {plan.human_readable_plan!r}",
        )


# ---------------------------------------------------------------------------
# Tool boundary tests
# ---------------------------------------------------------------------------

class TestProposeAnalysisPlanTool(unittest.TestCase):

    def test_unambiguous_inspection_returns_ok_true(self):
        insp = _inspection(candidate_label_columns=("Measurement Description",))
        req = _request(insp)
        with tempfile.TemporaryDirectory() as tmp:
            response = propose_analysis_plan.run(req, runs_root=tmp)
        self.assertTrue(response.ok)
        self.assertEqual(response.tool_name, "propose_analysis_plan")
        self.assertIsNotNone(response.payload)
        self.assertEqual(response.payload.task_name, "multi_class_classification")

    def test_ambiguous_labels_returns_ok_false(self):
        insp = _inspection(candidate_label_columns=("col_a", "col_b"))
        req = _request(insp)
        with tempfile.TemporaryDirectory() as tmp:
            response = propose_analysis_plan.run(req, runs_root=tmp)
        self.assertFalse(response.ok)
        self.assertIn("ambiguous", response.error.lower())

    def test_artifact_saved_at_expected_path(self):
        insp = _inspection(candidate_label_columns=("Measurement Description",))
        req = _request(insp)
        with tempfile.TemporaryDirectory() as tmp:
            response = propose_analysis_plan.run(req, runs_root=tmp)
            self.assertTrue(response.ok)
            self.assertEqual(len(response.artifacts), 1)
            artifact_uri = response.artifacts[0].uri
            artifact_path = Path(artifact_uri)
            self.assertTrue(artifact_path.exists(), f"Artifact not found at {artifact_uri}")
            # Verify it is valid JSON containing the plan
            data = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertIn("task_name", data)
            self.assertEqual(data["task_name"], "multi_class_classification")

    def test_artifact_filename_is_plan_json(self):
        insp = _inspection(candidate_label_columns=("label",))
        req = _request(insp)
        with tempfile.TemporaryDirectory() as tmp:
            response = propose_analysis_plan.run(req, runs_root=tmp)
        self.assertTrue(response.ok)
        artifact_uri = response.artifacts[0].uri
        self.assertTrue(artifact_uri.endswith("plan.json"), f"Expected plan.json, got: {artifact_uri}")

    def test_response_includes_run_metadata(self):
        insp = _inspection(candidate_label_columns=("label",))
        req = _request(insp)
        with tempfile.TemporaryDirectory() as tmp:
            response = propose_analysis_plan.run(req, runs_root=tmp)
        self.assertTrue(response.ok)
        self.assertIsNotNone(response.metadata)
        self.assertEqual(response.metadata.tool_name, "propose_analysis_plan")


if __name__ == "__main__":
    unittest.main()
