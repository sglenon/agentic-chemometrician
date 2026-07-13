"""Tests for SHAP/LIME interpretability integration."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from chemometrics_contracts import AnalysisResult, InterpretationResult
from chemometrics_mcp.core.interpretation import (
    compute_lime_importance,
    compute_shap_importance,
    interpret_results,
)
from chemometrics_mcp.core.modeling import run_cv_model


@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=60, n_features=10, n_informative=5, random_state=42
    )
    return X, y


@pytest.fixture
def regression_data():
    X, y = make_regression(
        n_samples=60, n_features=10, n_informative=5, random_state=42
    )
    return X, y


@pytest.fixture
def fitted_rf(classification_data):
    X, y = classification_data
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    return clf, X, y


class TestComputeShapImportance:
    def test_returns_correct_structure_with_mock(self, fitted_rf):
        clf, X, _ = fitted_rf
        mock_shap_values = np.random.rand(10, X.shape[1])
        mock_shap = MagicMock()
        mock_shap.kmeans.return_value = X[:5]
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = mock_shap_values
        mock_shap.KernelExplainer.return_value = mock_explainer

        with patch.dict(sys.modules, {"shap": mock_shap}):
            result = compute_shap_importance(clf, X, max_samples=10, seed=42)

        assert "feature_scores" in result
        assert "top_features" in result
        assert "mean_abs_shap" in result
        assert isinstance(result["feature_scores"], list)
        assert isinstance(result["top_features"], list)
        assert len(result["feature_scores"]) == X.shape[1]

    def test_feature_scores_have_correct_keys_with_mock(self, fitted_rf):
        clf, X, _ = fitted_rf
        mock_shap_values = np.random.rand(10, X.shape[1])
        mock_shap = MagicMock()
        mock_shap.kmeans.return_value = X[:5]
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = mock_shap_values
        mock_shap.KernelExplainer.return_value = mock_explainer

        with patch.dict(sys.modules, {"shap": mock_shap}):
            result = compute_shap_importance(clf, X, max_samples=10, seed=42)

        for entry in result["feature_scores"]:
            assert "feature_index" in entry
            assert "score" in entry
            assert isinstance(entry["score"], float)

    def test_axis_provided_with_mock(self, fitted_rf):
        clf, X, _ = fitted_rf
        axis = np.arange(X.shape[1], dtype=float) * 100
        mock_shap_values = np.random.rand(10, X.shape[1])
        mock_shap = MagicMock()
        mock_shap.kmeans.return_value = X[:5]
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = mock_shap_values
        mock_shap.KernelExplainer.return_value = mock_explainer

        with patch.dict(sys.modules, {"shap": mock_shap}):
            result = compute_shap_importance(clf, X, axis=axis, max_samples=10, seed=42)

        for entry in result["feature_scores"]:
            assert "axis_value" in entry
            assert "feature_index" in entry

    def test_axis_none_with_mock(self, fitted_rf):
        clf, X, _ = fitted_rf
        mock_shap_values = np.random.rand(10, X.shape[1])
        mock_shap = MagicMock()
        mock_shap.kmeans.return_value = X[:5]
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = mock_shap_values
        mock_shap.KernelExplainer.return_value = mock_explainer

        with patch.dict(sys.modules, {"shap": mock_shap}):
            result = compute_shap_importance(clf, X, axis=None, max_samples=10, seed=42)

        for entry in result["feature_scores"]:
            assert "feature_index" in entry
            assert "axis_value" not in entry

    def test_determinism_with_mock(self, fitted_rf):
        clf, X, _ = fitted_rf
        mock_shap_values = np.random.rand(10, X.shape[1])
        mock_shap = MagicMock()
        mock_shap.kmeans.return_value = X[:5]
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = mock_shap_values
        mock_shap.KernelExplainer.return_value = mock_explainer

        with patch.dict(sys.modules, {"shap": mock_shap}):
            r1 = compute_shap_importance(clf, X, max_samples=10, seed=42)
            r2 = compute_shap_importance(clf, X, max_samples=10, seed=42)

        assert r1["top_features"] == r2["top_features"]
        np.testing.assert_array_almost_equal(r1["mean_abs_shap"], r2["mean_abs_shap"])

    def test_shap_not_installed(self, fitted_rf):
        clf, X, _ = fitted_rf
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def mock_import(name, *args, **kwargs):
            if name == "shap":
                raise ImportError("shap is not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="shap is not installed"):
                compute_shap_importance(clf, X, max_samples=10)


class TestComputeLimeImportance:
    def test_returns_correct_structure_with_mock(self, fitted_rf):
        clf, X, _ = fitted_rf
        mock_exp = MagicMock()
        mock_exp.local_exp = {0: [(0, 0.1), (1, 0.2)]}
        mock_explainer = MagicMock()
        mock_explainer.explain_instance.return_value = mock_exp

        mock_lime_tabular = MagicMock()
        mock_lime_tabular.LimeTabularExplainer.return_value = mock_explainer
        mock_lime = MagicMock()
        mock_lime.lime_tabular = mock_lime_tabular

        with patch.dict(sys.modules, {"lime": mock_lime, "lime.lime_tabular": mock_lime_tabular}):
            result = compute_lime_importance(clf, X, n_samples=5, seed=42)

        assert "feature_scores" in result
        assert "top_features" in result
        assert "mean_abs_lime" in result
        assert len(result["feature_scores"]) == X.shape[1]

    def test_axis_provided_with_mock(self, fitted_rf):
        clf, X, _ = fitted_rf
        axis = np.arange(X.shape[1], dtype=float) * 100
        mock_exp = MagicMock()
        mock_exp.local_exp = {0: [(0, 0.1), (1, 0.2)]}
        mock_explainer = MagicMock()
        mock_explainer.explain_instance.return_value = mock_exp

        mock_lime_tabular = MagicMock()
        mock_lime_tabular.LimeTabularExplainer.return_value = mock_explainer
        mock_lime = MagicMock()
        mock_lime.lime_tabular = mock_lime_tabular

        with patch.dict(sys.modules, {"lime": mock_lime, "lime.lime_tabular": mock_lime_tabular}):
            result = compute_lime_importance(clf, X, axis=axis, n_samples=5, seed=42)

        for entry in result["feature_scores"]:
            assert "axis_value" in entry

    def test_lime_not_installed(self, fitted_rf):
        clf, X, _ = fitted_rf
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def mock_import(name, *args, **kwargs):
            if name == "lime" or name.startswith("lime."):
                raise ImportError("lime is not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="lime is not installed"):
                compute_lime_importance(clf, X, n_samples=5)


class TestInterpretResultsWithShap:
    def test_methods_used_populated(self, fitted_rf):
        clf, X, _ = fitted_rf
        mock_shap_values = np.random.rand(10, X.shape[1])
        mock_shap = MagicMock()
        mock_shap.kmeans.return_value = X[:5]
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = mock_shap_values
        mock_shap.KernelExplainer.return_value = mock_explainer

        with patch.dict(sys.modules, {"shap": mock_shap}):
            shap_data = compute_shap_importance(clf, X, max_samples=10, seed=42)

        ir = InterpretationResult(
            model_name="random_forest",
            method="shap",
            feature_scores=tuple(shap_data["feature_scores"]),
            top_features=tuple(shap_data["top_features"]),
        )
        result = AnalysisResult(
            task_name="binary_classification",
            model_name="random_forest",
            metrics={"accuracy": 0.9},
            selected_features=(1, 2, 3),
            interpretation_results=(ir,),
        )
        summary = interpret_results([result])
        assert "shap" in summary.interpretation_methods_used

    def test_methods_used_empty_when_no_interpretation(self):
        result = AnalysisResult(
            task_name="binary_classification",
            model_name="random_forest",
            metrics={"accuracy": 0.9},
            selected_features=(1, 2, 3),
        )
        summary = interpret_results([result])
        assert summary.interpretation_methods_used == ()


class TestIntegrationWithModeling:
    def test_random_forest_with_shap_mock(self, classification_data):
        X, y = classification_data
        axis = np.arange(X.shape[1], dtype=float) * 100

        mock_shap_values = np.random.rand(10, X.shape[1])
        mock_shap = MagicMock()
        mock_shap.kmeans.return_value = X[:5]
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = mock_shap_values
        mock_shap.KernelExplainer.return_value = mock_explainer

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        with patch.dict(sys.modules, {"shap": mock_shap}):
            result, fig_data = run_cv_model(
                X, y, axis, "random_forest", cv,
                "binary_classification", "none",
                enable_shap=True,
            )

        assert len(result.interpretation_results) > 0
        ir = result.interpretation_results[0]
        assert ir.method == "shap"
        assert ir.model_name == "random_forest"
        assert len(ir.feature_scores) > 0
        assert len(ir.top_features) > 0

    def test_random_forest_without_shap(self, classification_data):
        X, y = classification_data
        axis = np.arange(X.shape[1], dtype=float) * 100
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        result, fig_data = run_cv_model(
            X, y, axis, "random_forest", cv,
            "binary_classification", "none",
            enable_shap=False,
        )
        assert len(result.interpretation_results) == 0

    def test_plsr_with_shap_mock(self, regression_data):
        X, y = regression_data
        axis = np.arange(X.shape[1], dtype=float) * 100

        mock_shap_values = np.random.rand(10, X.shape[1])
        mock_shap = MagicMock()
        mock_shap.kmeans.return_value = X[:5]
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = mock_shap_values
        mock_shap.KernelExplainer.return_value = mock_explainer

        from sklearn.model_selection import KFold
        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        with patch.dict(sys.modules, {"shap": mock_shap}):
            result, fig_data = run_cv_model(
                X, y, axis, "plsr", cv,
                "regression", "none",
                enable_shap=True,
            )

        assert len(result.interpretation_results) > 0
        ir = result.interpretation_results[0]
        assert ir.method == "shap"
