"""Tests for harder NIR tasks: wear-layer regression and species multi-class.

Uses synthetic spectral data so no real Excel file is required.
Integration tests against the real flooring Excel are skipped when the file
is absent (CI-safe).
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from chemometrics_contracts import (  # noqa: E402
    AnalysisPlan,
    SpectralDataset,
)
from chemometrics_mcp.core import modeling, preprocessing  # noqa: E402
from chemometrics_mcp.core.datasets import _parse_flooring_description  # noqa: E402
from chemometrics_mcp.core.reporting import build_model_comparison_table  # noqa: E402
from chemometrics_mcp.core.validation import check_easy_task  # noqa: E402
from chemometrics_mcp.tools.run_analysis import run  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(7)
_AXIS_NIR = np.linspace(1450.0, 2450.0, 100)


def _make_wear_layer_dataset(n_per_class: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic wear-layer regression dataset.

    Classes: 6.0, 12.0, 22.0 mil.  Each class has a different spectral shape
    to make regression non-trivial but learnable.
    """
    rng = np.random.default_rng(11)
    wear_layers = [6.0, 12.0, 22.0]
    X_parts, y_parts = [], []
    for wl in wear_layers:
        # Create a spectral peak whose position scales with wear layer
        center = 1700.0 + wl * 10.0
        base = np.exp(-0.5 * ((_AXIS_NIR - center) / 80.0) ** 2)
        noise = rng.normal(0, 0.02, size=(n_per_class, len(_AXIS_NIR)))
        X_parts.append(np.tile(base, (n_per_class, 1)) + noise)
        y_parts.extend([wl] * n_per_class)
    X = np.vstack(X_parts)
    y = np.array(y_parts, dtype=float)
    return X, y, _AXIS_NIR


def _make_species_dataset(n_per_species: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic species multi-class dataset (6 species)."""
    rng = np.random.default_rng(13)
    species = ["fir", "mahogany", "oak", "pine", "poplar", "particle_board"]
    X_parts, y_parts = [], []
    for i, sp in enumerate(species):
        center = 1600.0 + i * 90.0
        base = np.exp(-0.5 * ((_AXIS_NIR - center) / 60.0) ** 2)
        noise = rng.normal(0, 0.03, size=(n_per_species, len(_AXIS_NIR)))
        X_parts.append(np.tile(base, (n_per_species, 1)) + noise)
        y_parts.extend([sp] * n_per_species)
    X = np.vstack(X_parts)
    y = np.array(y_parts)
    return X, y, _AXIS_NIR


def _make_spectral_dataset_from_arrays(
    X: np.ndarray,
    y,
    axis: np.ndarray,
) -> SpectralDataset:
    return SpectralDataset(
        x=tuple(tuple(float(v) for v in row) for row in X.tolist()),
        axis=tuple(float(v) for v in axis.tolist()),
        labels=tuple(y.tolist()),
    )


# ---------------------------------------------------------------------------
# 1. Flooring description parser
# ---------------------------------------------------------------------------


class TestParseFlooringDescription(unittest.TestCase):

    def test_lumber_fir(self):
        result = _parse_flooring_description("Lumber, fir, 11.45")
        self.assertEqual(result["material_type"], "lumber")
        self.assertEqual(result["species"], "fir")
        self.assertIsNone(result["wear_layer_mil"])

    def test_lumber_particle_board(self):
        result = _parse_flooring_description("Lumber, particle board")
        self.assertEqual(result["material_type"], "lumber")
        self.assertEqual(result["species"], "particle_board")

    def test_lumber_mahogany_trailing_space(self):
        result = _parse_flooring_description("Lumber, mahogany , 12")
        self.assertEqual(result["material_type"], "lumber")
        self.assertEqual(result["species"], "mahogany")

    def test_vinyl_wl6(self):
        result = _parse_flooring_description("Vinyl, traffic master, wl 6, birchwood, 1.99")
        self.assertEqual(result["material_type"], "vinyl")
        self.assertEqual(result["wear_layer_mil"], 6)
        self.assertIsNone(result["species"])

    def test_vinyl_wl12(self):
        result = _parse_flooring_description(
            "Vinyl, home decorators collection, wl 12, Wayne canyon oak, 2.49"
        )
        self.assertEqual(result["material_type"], "vinyl")
        self.assertEqual(result["wear_layer_mil"], 12)

    def test_vinyl_wl22(self):
        result = _parse_flooring_description("Vinyl, life proof, wl 22, brooks oak, 3.29")
        self.assertEqual(result["material_type"], "vinyl")
        self.assertEqual(result["wear_layer_mil"], 22)

    def test_unknown_description(self):
        result = _parse_flooring_description("Water bottle")
        self.assertIsNone(result["material_type"])
        self.assertIsNone(result["species"])
        self.assertIsNone(result["wear_layer_mil"])

    def test_empty_string(self):
        result = _parse_flooring_description("")
        self.assertIsNone(result["material_type"])


# ---------------------------------------------------------------------------
# 2. Wear-layer regression models
# ---------------------------------------------------------------------------


class TestWearLayerRegression(unittest.TestCase):

    def setUp(self):
        self.X, self.y, self.axis = _make_wear_layer_dataset(n_per_class=20)
        from sklearn.model_selection import KFold
        self.cv = KFold(n_splits=5, shuffle=True, random_state=42)

    def test_plsr_regression_metrics(self):
        result, fig_data = modeling.run_cv_model(
            self.X, self.y, self.axis, "plsr", self.cv, "regression", "snv"
        )
        self.assertIn("r2", result.metrics)
        self.assertIn("rmse", result.metrics)
        self.assertIn("mae", result.metrics)
        self.assertIn("predicted_vs_actual", fig_data)
        self.assertIn("residuals", fig_data)

    def test_svr_regression_metrics(self):
        result, fig_data = modeling.run_cv_model(
            self.X, self.y, self.axis, "svr", self.cv, "regression", "snv"
        )
        self.assertIn("r2", result.metrics)
        self.assertIn("residuals", fig_data)

    def test_ridge_regression_metrics(self):
        result, fig_data = modeling.run_cv_model(
            self.X, self.y, self.axis, "ridge", self.cv, "regression", "standard_scaler"
        )
        self.assertIn("r2", result.metrics)
        self.assertIn("rmse", result.metrics)
        self.assertIn("mae", result.metrics)
        self.assertIn("feature_importances", fig_data)
        self.assertIn("residuals", fig_data)
        self.assertTrue(len(result.selected_features) > 0)

    def test_random_forest_reg_metrics(self):
        result, fig_data = modeling.run_cv_model(
            self.X, self.y, self.axis, "random_forest_reg", self.cv, "regression", "snv"
        )
        self.assertIn("r2", result.metrics)
        self.assertIn("feature_importances", fig_data)
        self.assertIn("residuals", fig_data)
        self.assertTrue(len(result.selected_features) > 0)

    def test_ridge_no_y_raises(self):
        with self.assertRaises(ValueError):
            modeling.run_cv_model(
                self.X, None, self.axis, "ridge", self.cv, "regression", "raw"
            )

    def test_random_forest_reg_no_y_raises(self):
        with self.assertRaises(ValueError):
            modeling.run_cv_model(
                self.X, None, self.axis, "random_forest_reg", self.cv, "regression", "raw"
            )

    def test_run_analysis_wear_layer(self):
        """End-to-end run_analysis with regression plan and wear-layer data."""
        dataset = _make_spectral_dataset_from_arrays(self.X, self.y, self.axis)
        plan = AnalysisPlan(
            task_name="regression",
            preprocessing_candidates=("snv",),
            validation_strategy="grouped_kfold_5",
            model_families=("ridge", "svr"),
        )
        from chemometrics_contracts import RunAnalysisRequest
        request = RunAnalysisRequest(dataset=dataset, approved_plan=plan)
        with tempfile.TemporaryDirectory() as tmpdir:
            response = run(request, runs_root=tmpdir)
        self.assertTrue(response.ok, msg=f"run_analysis failed: {response.error}")
        self.assertEqual(len(response.payload.results), 2)
        for result in response.payload.results:
            self.assertIn("r2", result.metrics)


# ---------------------------------------------------------------------------
# 3. Species multi-class classification models
# ---------------------------------------------------------------------------


class TestSpeciesMultiClass(unittest.TestCase):

    def setUp(self):
        self.X, self.y, self.axis = _make_species_dataset(n_per_species=10)
        from sklearn.model_selection import StratifiedKFold
        self.cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def test_random_forest_multiclass(self):
        result, fig_data = modeling.run_cv_model(
            self.X, self.y, self.axis, "random_forest", self.cv,
            "multi_class_classification", "snv"
        )
        self.assertIn("accuracy", result.metrics)
        self.assertIn("f1_weighted", result.metrics)
        self.assertIn("f1_macro", result.metrics)
        self.assertIn("confusion_matrix", fig_data)
        self.assertIn("class_labels", fig_data)
        self.assertEqual(len(fig_data["class_labels"]), 6)

    def test_logistic_regression_multiclass(self):
        result, fig_data = modeling.run_cv_model(
            self.X, self.y, self.axis, "logistic_regression", self.cv,
            "multi_class_classification", "snv"
        )
        self.assertIn("accuracy", result.metrics)
        self.assertIn("f1_weighted", result.metrics)
        self.assertIn("confusion_matrix", fig_data)
        self.assertIn("feature_importances", fig_data)
        self.assertTrue(len(result.selected_features) > 0)

    def test_svm_rbf_multiclass(self):
        result, fig_data = modeling.run_cv_model(
            self.X, self.y, self.axis, "svm_rbf", self.cv,
            "multi_class_classification", "snv"
        )
        self.assertIn("accuracy", result.metrics)
        self.assertIn("confusion_matrix", fig_data)

    def test_logistic_regression_no_labels_raises(self):
        with self.assertRaises(ValueError):
            modeling.run_cv_model(
                self.X, None, self.axis, "logistic_regression", self.cv,
                "multi_class_classification", "raw"
            )

    def test_run_analysis_species(self):
        """End-to-end run_analysis with multi-class plan and species data."""
        dataset = _make_spectral_dataset_from_arrays(self.X, self.y, self.axis)
        plan = AnalysisPlan(
            task_name="multi_class_classification",
            preprocessing_candidates=("snv",),
            validation_strategy="stratified_kfold_3",
            model_families=("random_forest", "logistic_regression"),
        )
        from chemometrics_contracts import RunAnalysisRequest
        request = RunAnalysisRequest(dataset=dataset, approved_plan=plan)
        with tempfile.TemporaryDirectory() as tmpdir:
            response = run(request, runs_root=tmpdir)
        self.assertTrue(response.ok, msg=f"run_analysis failed: {response.error}")
        self.assertEqual(len(response.payload.results), 2)
        for result in response.payload.results:
            self.assertIn("accuracy", result.metrics)
            self.assertIn("f1_weighted", result.metrics)


# ---------------------------------------------------------------------------
# 4. Preprocessing: standard_scaler and robust_scaler
# ---------------------------------------------------------------------------


class TestNewPreprocessing(unittest.TestCase):

    def setUp(self):
        self.X, _, _ = _make_wear_layer_dataset(n_per_class=10)

    def test_standard_scaler_shape(self):
        X_out, details = preprocessing.apply(self.X, "standard_scaler")
        self.assertEqual(X_out.shape, self.X.shape)
        self.assertEqual(details["method"], "standard_scaler")

    def test_standard_scaler_column_zero_mean(self):
        X_out, _ = preprocessing.apply(self.X, "standard_scaler")
        col_means = X_out.mean(axis=0)
        np.testing.assert_allclose(col_means, 0.0, atol=1e-10)

    def test_robust_scaler_shape(self):
        X_out, details = preprocessing.apply(self.X, "robust_scaler")
        self.assertEqual(X_out.shape, self.X.shape)
        self.assertEqual(details["method"], "robust_scaler")


# ---------------------------------------------------------------------------
# 5. Model comparison table
# ---------------------------------------------------------------------------


class TestModelComparisonTable(unittest.TestCase):

    def _make_result(self, task: str, model: str, acc: float):
        from chemometrics_contracts import AnalysisResult
        return AnalysisResult(
            task_name=task,
            model_name=model,
            metrics={"accuracy": acc, "f1_weighted": acc * 0.98},
        )

    def _make_reg_result(self, model: str, r2: float):
        from chemometrics_contracts import AnalysisResult
        return AnalysisResult(
            task_name="regression",
            model_name=model,
            metrics={"r2": r2, "rmse": 1.0 - r2},
        )

    def test_empty_results(self):
        table = build_model_comparison_table([])
        self.assertIn("No results", table)

    def test_classification_table_contains_models(self):
        results = [
            self._make_result("multi_class_classification", "random_forest", 0.85),
            self._make_result("multi_class_classification", "logistic_regression", 0.78),
        ]
        table = build_model_comparison_table(results)
        self.assertIn("random_forest", table)
        self.assertIn("logistic_regression", table)
        self.assertIn("accuracy", table)

    def test_regression_table_shows_r2(self):
        results = [
            self._make_reg_result("ridge", 0.92),
            self._make_reg_result("svr", 0.88),
        ]
        table = build_model_comparison_table(results)
        self.assertIn("ridge", table)
        self.assertIn("svr", table)
        self.assertIn("r2", table)

    def test_complexity_column_present(self):
        results = [self._make_result("binary_classification", "random_forest", 0.9)]
        table = build_model_comparison_table(results)
        self.assertIn("100 trees", table)

    def test_stability_na_for_single_run(self):
        results = [self._make_result("binary_classification", "svm_rbf", 0.9)]
        table = build_model_comparison_table(results)
        self.assertIn("n/a", table)


# ---------------------------------------------------------------------------
# 6. Caveat logic: check_easy_task
# ---------------------------------------------------------------------------


class TestCheckEasyTask(unittest.TestCase):

    def test_highly_separated_binary_triggers_warning(self):
        """Two perfectly separated binary classes → easy task warning."""
        rng = np.random.default_rng(0)
        X0 = rng.normal(loc=0.0, scale=0.1, size=(20, 50))
        X1 = rng.normal(loc=10.0, scale=0.1, size=(20, 50))
        X = np.vstack([X0, X1])
        y = np.array(["A"] * 20 + ["B"] * 20)
        warning = check_easy_task(X, y, "binary_classification")
        self.assertIsNotNone(warning)
        self.assertEqual(warning.code, "easy_task_detected")
        self.assertEqual(warning.severity, "info")

    def test_overlapping_classes_no_warning(self):
        """Fully overlapping classes → no easy-task warning."""
        rng = np.random.default_rng(1)
        X = rng.normal(loc=0.0, scale=1.0, size=(40, 50))
        y = np.array(["A"] * 20 + ["B"] * 20)
        warning = check_easy_task(X, y, "binary_classification")
        self.assertIsNone(warning)

    def test_regression_task_returns_none(self):
        """Regression tasks are not checked for easy-task separation."""
        rng = np.random.default_rng(2)
        X = rng.random((40, 50))
        y = rng.random(40)
        warning = check_easy_task(X, y, "regression")
        self.assertIsNone(warning)

    def test_multiclass_separated_triggers_warning(self):
        """Multi-class with clear separation should also trigger."""
        rng = np.random.default_rng(3)
        centers = [0.0, 8.0, 16.0]
        X = np.vstack([rng.normal(c, 0.05, (15, 50)) for c in centers])
        y = np.array(["A"] * 15 + ["B"] * 15 + ["C"] * 15)
        warning = check_easy_task(X, y, "multi_class_classification")
        self.assertIsNotNone(warning)


# ---------------------------------------------------------------------------
# 7. Planning: regression validation strategy
# ---------------------------------------------------------------------------


class TestPlanningRegressionStrategy(unittest.TestCase):

    def test_regression_uses_kfold(self):
        from chemometrics_mcp.core.planning import recommend_validation_strategy
        from chemometrics_contracts import DatasetInspection
        inspection = DatasetInspection(sample_count=90, feature_count=100)
        strategy = recommend_validation_strategy(inspection, task_name="regression")
        self.assertEqual(strategy, "grouped_kfold_5")

    def test_classification_still_stratified(self):
        from chemometrics_mcp.core.planning import recommend_validation_strategy
        from chemometrics_contracts import DatasetInspection
        inspection = DatasetInspection(sample_count=90, feature_count=100)
        strategy = recommend_validation_strategy(inspection, task_name="binary_classification")
        self.assertEqual(strategy, "stratified_kfold_5")


# ---------------------------------------------------------------------------
# 8. Integration: load_flooring_nir (skipped if Excel not present)
# ---------------------------------------------------------------------------

_FLOORING_EXCEL = ROOT / "2026-05-21_22-59_Method Dev_wavelength_range_1450-2450.xlsx"


@unittest.skipUnless(_FLOORING_EXCEL.exists(), "Flooring Excel not present")
class TestLoadFlooringNIR(unittest.TestCase):

    def test_species_task_loads(self):
        from chemometrics_mcp.core.datasets import load_flooring_nir
        dataset, inspection = load_flooring_nir(_FLOORING_EXCEL, task="species")
        self.assertGreater(inspection.sample_count, 30)
        self.assertEqual(inspection.candidate_label_columns, ("species",))
        species_set = set(dataset.labels)
        expected = {"fir", "mahogany", "oak", "pine", "poplar", "particle_board"}
        self.assertLessEqual(species_set, expected)

    def test_wear_layer_task_loads(self):
        from chemometrics_mcp.core.datasets import load_flooring_nir
        dataset, inspection = load_flooring_nir(_FLOORING_EXCEL, task="wear_layer")
        self.assertGreater(inspection.sample_count, 60)
        self.assertEqual(inspection.candidate_label_columns, ("wear_layer_mil",))
        wear_layer_set = set(dataset.labels)
        self.assertLessEqual(wear_layer_set, {6.0, 12.0, 22.0})

    def test_material_type_task_loads(self):
        from chemometrics_mcp.core.datasets import load_flooring_nir
        dataset, inspection = load_flooring_nir(_FLOORING_EXCEL, task="material_type")
        self.assertGreater(inspection.sample_count, 100)
        self.assertIn("lumber", set(dataset.labels))
        self.assertIn("vinyl", set(dataset.labels))

    def test_unknown_task_raises(self):
        from chemometrics_mcp.core.datasets import load_flooring_nir
        with self.assertRaises(ValueError):
            load_flooring_nir(_FLOORING_EXCEL, task="bogus")


if __name__ == "__main__":
    unittest.main()
