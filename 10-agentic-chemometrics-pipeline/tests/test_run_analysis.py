"""Tests for run_analysis tool and its core preprocessing/modeling infrastructure.

Uses synthetic 30 × 20 spectral data — does NOT load the real NIR file.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemometrics_contracts import (  # noqa: E402
    AnalysisPlan,
    SpectralDataset,
)
from chemometrics_mcp.core import modeling, preprocessing  # noqa: E402
from chemometrics_mcp.tools.run_analysis import _spectral_dataset_from_dict, run  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------

N_SAMPLES = 30
N_FEATURES = 20
AXIS_START = 1000.0
AXIS_STOP = 2000.0

RNG = np.random.default_rng(0)


def _make_X() -> np.ndarray:
    """30 × 20 synthetic spectral matrix."""
    return RNG.random((N_SAMPLES, N_FEATURES)).astype(float) + 0.1


def _make_y_binary() -> np.ndarray:
    """15 + 15 binary class labels."""
    return np.array(["A"] * 15 + ["B"] * 15)


def _make_axis() -> np.ndarray:
    return np.linspace(AXIS_START, AXIS_STOP, N_FEATURES)


def _make_spectral_dataset(*, include_labels: bool = True) -> SpectralDataset:
    X = _make_X()
    axis = _make_axis()
    labels: tuple | None = tuple(_make_y_binary().tolist()) if include_labels else None
    return SpectralDataset(
        x=tuple(tuple(float(v) for v in row) for row in X.tolist()),
        axis=tuple(float(v) for v in axis.tolist()),
        labels=labels,
    )


def _make_cv():
    return modeling.make_cv_splitter("stratified_kfold_5")


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.X = _make_X()

    # 1. raw returns same shape
    def test_raw_same_shape(self):
        X_out, details = preprocessing.apply(self.X, "raw")
        self.assertEqual(X_out.shape, self.X.shape)
        self.assertEqual(details["method"], "raw")
        self.assertEqual(details["shape_in"], list(self.X.shape))

    # 2. snv returns same shape; each row ~zero mean, ~unit std
    def test_snv_shape_and_statistics(self):
        X_out, details = preprocessing.apply(self.X, "snv")
        self.assertEqual(X_out.shape, self.X.shape)
        row_means = X_out.mean(axis=1)
        row_stds = X_out.std(axis=1)
        np.testing.assert_allclose(row_means, 0.0, atol=1e-10)
        np.testing.assert_allclose(row_stds, 1.0, atol=1e-10)

    # 3. msc returns same shape
    def test_msc_same_shape(self):
        X_out, details = preprocessing.apply(self.X, "msc")
        self.assertEqual(X_out.shape, self.X.shape)
        self.assertIn("reference_mean_shape", details)

    # 4. sg_1st_deriv returns same shape (savgol_filter on axis=1 preserves shape)
    def test_sg_1st_deriv_shape(self):
        X_out, details = preprocessing.apply(self.X, "sg_1st_deriv")
        self.assertEqual(X_out.shape, self.X.shape)
        self.assertEqual(details["deriv"], 1)

    # 5. sg_2nd_deriv returns same shape
    def test_sg_2nd_deriv_shape(self):
        X_out, details = preprocessing.apply(self.X, "sg_2nd_deriv")
        self.assertEqual(X_out.shape, self.X.shape)
        self.assertEqual(details["deriv"], 2)

    # 6. Unknown method raises ValueError
    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            preprocessing.apply(self.X, "nonexistent_method")

    # 7. 1D input raises ValueError
    def test_1d_input_raises(self):
        with self.assertRaises(ValueError) as ctx:
            preprocessing.apply(self.X[0], "raw")
        self.assertIn("2D", str(ctx.exception))


# ---------------------------------------------------------------------------
# Modeling tests
# ---------------------------------------------------------------------------


class TestModeling(unittest.TestCase):

    def setUp(self):
        self.X = _make_X()
        self.y = _make_y_binary()
        self.axis = _make_axis()
        self.cv = _make_cv()

    # 8. svm_rbf returns AnalysisResult with accuracy in metrics
    def test_svm_rbf_classification(self):
        result, fig_data = modeling.run_cv_model(
            self.X, self.y, self.axis, "svm_rbf", self.cv,
            "binary_classification", "snv",
        )
        self.assertIn("accuracy", result.metrics)
        self.assertIn("balanced_accuracy", result.metrics)
        self.assertIn("f1_macro", result.metrics)
        self.assertIn("confusion_matrix", fig_data)
        self.assertIn("class_labels", fig_data)

    # 9. random_forest returns AnalysisResult with non-empty selected_features
    def test_random_forest_selected_features(self):
        result, fig_data = modeling.run_cv_model(
            self.X, self.y, self.axis, "random_forest", self.cv,
            "binary_classification", "raw",
        )
        self.assertTrue(len(result.selected_features) > 0)
        self.assertIn("feature_importances", fig_data)

    # 10. pca returns AnalysisResult with explained_variance_ratio_cumulative
    def test_pca_unsupervised(self):
        result, fig_data = modeling.run_cv_model(
            self.X, None, self.axis, "pca", self.cv,
            "unsupervised_exploration", "raw",
        )
        self.assertIn("explained_variance_ratio_cumulative", result.metrics)
        self.assertIn("explained_variance_ratio", fig_data)

    # 11. Unknown model name raises ValueError
    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            modeling.run_cv_model(
                self.X, self.y, self.axis, "bogus_model", self.cv,
                "binary_classification", "raw",
            )

    # 12. svm_rbf with y=None raises ValueError
    def test_svm_rbf_no_labels_raises(self):
        with self.assertRaises(ValueError):
            modeling.run_cv_model(
                self.X, None, self.axis, "svm_rbf", self.cv,
                "binary_classification", "raw",
            )

    # 13. make_cv_splitter returns StratifiedKFold for "stratified_kfold_5"
    def test_make_cv_splitter_stratified(self):
        from sklearn.model_selection import StratifiedKFold
        cv = modeling.make_cv_splitter("stratified_kfold_5")
        self.assertIsInstance(cv, StratifiedKFold)
        self.assertEqual(cv.n_splits, 5)

    # 14. make_cv_splitter returns LeaveOneOut for "loocv"
    def test_make_cv_splitter_loocv(self):
        from sklearn.model_selection import LeaveOneOut
        cv = modeling.make_cv_splitter("loocv")
        self.assertIsInstance(cv, LeaveOneOut)

    # 15. Unknown strategy returns KFold(n_splits=5)
    def test_make_cv_splitter_unknown_returns_kfold(self):
        from sklearn.model_selection import KFold
        cv = modeling.make_cv_splitter("totally_unknown_strategy")
        self.assertIsInstance(cv, KFold)
        self.assertEqual(cv.n_splits, 5)


# ---------------------------------------------------------------------------
# Tool boundary tests
# ---------------------------------------------------------------------------


class TestRunAnalysisTool(unittest.TestCase):

    def _make_classification_plan(self) -> AnalysisPlan:
        return AnalysisPlan(
            task_name="binary_classification",
            preprocessing_candidates=("snv",),
            validation_strategy="stratified_kfold_5",
            model_families=("svm_rbf",),
        )

    def _make_pca_plan(self) -> AnalysisPlan:
        return AnalysisPlan(
            task_name="unsupervised_exploration",
            preprocessing_candidates=("raw",),
            validation_strategy="stratified_kfold_5",
            model_families=("pca",),
        )

    # 16. Full run() with classification plan returns ok=True
    def test_classification_run_ok(self):
        from chemometrics_contracts import RunAnalysisRequest
        dataset = _make_spectral_dataset(include_labels=True)
        plan = self._make_classification_plan()
        request = RunAnalysisRequest(dataset=dataset, approved_plan=plan)
        with tempfile.TemporaryDirectory() as tmpdir:
            response = run(request, runs_root=tmpdir)
        self.assertTrue(response.ok, msg=f"Expected ok=True, got error: {response.error}")

    # 17. Full run() with PCA plan returns ok=True (no labels needed)
    def test_pca_run_ok(self):
        from chemometrics_contracts import RunAnalysisRequest
        dataset = _make_spectral_dataset(include_labels=False)
        plan = self._make_pca_plan()
        request = RunAnalysisRequest(dataset=dataset, approved_plan=plan)
        with tempfile.TemporaryDirectory() as tmpdir:
            response = run(request, runs_root=tmpdir)
        self.assertTrue(response.ok, msg=f"Expected ok=True, got error: {response.error}")

    # 18. Empty dataset returns ok=False
    def test_empty_dataset_returns_error(self):
        from chemometrics_contracts import RunAnalysisRequest
        empty_dataset = SpectralDataset(x=(), axis=())
        plan = self._make_classification_plan()
        request = RunAnalysisRequest(dataset=empty_dataset, approved_plan=plan)
        with tempfile.TemporaryDirectory() as tmpdir:
            response = run(request, runs_root=tmpdir)
        self.assertFalse(response.ok)
        self.assertIsNotNone(response.error)

    # 19. All-models-fail: svm_rbf with no labels → ok=False
    def test_all_models_fail_returns_error(self):
        from chemometrics_contracts import RunAnalysisRequest
        # No labels, svm_rbf requires them
        dataset = _make_spectral_dataset(include_labels=False)
        plan = AnalysisPlan(
            task_name="binary_classification",
            preprocessing_candidates=("raw",),
            validation_strategy="stratified_kfold_5",
            model_families=("svm_rbf",),
        )
        request = RunAnalysisRequest(dataset=dataset, approved_plan=plan)
        with tempfile.TemporaryDirectory() as tmpdir:
            response = run(request, runs_root=tmpdir)
        self.assertFalse(response.ok)

    # 20. Artifact dir is created and run_summary.json exists
    def test_artifact_directory_and_summary_created(self):
        from chemometrics_contracts import RunAnalysisRequest
        dataset = _make_spectral_dataset(include_labels=True)
        plan = self._make_classification_plan()
        request = RunAnalysisRequest(dataset=dataset, approved_plan=plan)
        with tempfile.TemporaryDirectory() as tmpdir:
            response = run(request, runs_root=tmpdir)
            self.assertTrue(response.ok, msg=f"Run failed: {response.error}")
            # Find the summary artifact
            summary_artifact = None
            for a in response.artifacts:
                if a.kind == "run_summary":
                    summary_artifact = a
                    break
            self.assertIsNotNone(summary_artifact, "No run_summary artifact found")
            summary_path = Path(summary_artifact.uri)
            self.assertTrue(summary_path.exists(), f"run_summary.json not found at {summary_path}")

    # 21. _spectral_dataset_from_dict round-trip
    def test_spectral_dataset_from_dict_roundtrip(self):
        original = _make_spectral_dataset(include_labels=True)
        d = {
            "x": [list(row) for row in original.x],
            "axis": list(original.axis),
            "metadata": list(original.metadata),
            "labels": list(original.labels) if original.labels is not None else None,
            "modality": original.modality,
            "sample_ids": list(original.sample_ids) if original.sample_ids is not None else None,
        }
        reconstructed = _spectral_dataset_from_dict(d)
        self.assertEqual(len(reconstructed.x), N_SAMPLES)
        self.assertEqual(len(reconstructed.x[0]), N_FEATURES)
        self.assertEqual(len(reconstructed.axis), N_FEATURES)
        self.assertEqual(reconstructed.labels, original.labels)


if __name__ == "__main__":
    unittest.main()
