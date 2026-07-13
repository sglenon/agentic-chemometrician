"""Tests for the figure rendering module."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from chemometrics_mcp.core.figures import render_figure  # noqa: E402


class TestRenderConfusionMatrix(unittest.TestCase):

    def setUp(self):
        self.fig_data = {
            "confusion_matrix": [[10, 2], [3, 15]],
            "class_labels": ["A", "B"],
        }

    def test_produces_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "cm.png"
            result = render_figure(self.fig_data, "svm_rbf", out)
            self.assertTrue(result.exists())

    def test_file_nonempty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "cm.png"
            render_figure(self.fig_data, "svm_rbf", out)
            self.assertGreater(out.stat().st_size, 0)


class TestRenderFeatureImportances(unittest.TestCase):

    def setUp(self):
        self.fig_data = {
            "feature_importances": [0.3, 0.2, 0.15, 0.1, 0.05],
            "axis": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
        }

    def test_produces_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "fi.png"
            result = render_figure(self.fig_data, "random_forest", out)
            self.assertTrue(result.exists())

    def test_file_nonempty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "fi.png"
            render_figure(self.fig_data, "random_forest", out)
            self.assertGreater(out.stat().st_size, 0)


class TestRenderExplainedVariance(unittest.TestCase):

    def setUp(self):
        self.fig_data = {
            "explained_variance_ratio": [0.4, 0.2, 0.1, 0.05],
            "cumulative_variance": [0.4, 0.6, 0.7, 0.75],
            "n_components": 4,
        }

    def test_produces_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "evr.png"
            result = render_figure(self.fig_data, "pca", out)
            self.assertTrue(result.exists())

    def test_file_nonempty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "evr.png"
            render_figure(self.fig_data, "pca", out)
            self.assertGreater(out.stat().st_size, 0)


class TestRenderPredictedVsActual(unittest.TestCase):

    def setUp(self):
        self.fig_data = {
            "predicted_vs_actual": {
                "predicted": [1.0, 2.1, 3.0, 4.2, 5.0],
                "actual": [1.1, 2.0, 3.1, 4.0, 5.1],
            }
        }

    def test_produces_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "pva.png"
            result = render_figure(self.fig_data, "plsr", out)
            self.assertTrue(result.exists())

    def test_file_nonempty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "pva.png"
            render_figure(self.fig_data, "plsr", out)
            self.assertGreater(out.stat().st_size, 0)


class TestRenderCvAccuracy(unittest.TestCase):

    def setUp(self):
        self.fig_data = {
            "cv_accuracy_per_fold": [0.85, 0.90, 0.88, 0.92, 0.87],
            "n_lda_components": 1,
        }

    def test_produces_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "cv.png"
            result = render_figure(self.fig_data, "pca_lda", out)
            self.assertTrue(result.exists())

    def test_file_nonempty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "cv.png"
            render_figure(self.fig_data, "pca_lda", out)
            self.assertGreater(out.stat().st_size, 0)


class TestRenderClusterSizes(unittest.TestCase):

    def setUp(self):
        self.fig_data = {"cluster_sizes": [12, 10, 8]}

    def test_produces_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "cs.png"
            result = render_figure(self.fig_data, "kmeans", out)
            self.assertTrue(result.exists())

    def test_file_nonempty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "cs.png"
            render_figure(self.fig_data, "kmeans", out)
            self.assertGreater(out.stat().st_size, 0)


class TestRenderPdfFormat(unittest.TestCase):

    def test_pdf_output(self):
        fig_data = {"cluster_sizes": [5, 10]}
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "cs.pdf"
            result = render_figure(fig_data, "kmeans", out, format="pdf")
            self.assertTrue(result.exists())
            self.assertGreater(out.stat().st_size, 0)


class TestUnknownFigureKey(unittest.TestCase):

    def test_unknown_key_returns_path_without_error(self):
        fig_data = {"unknown_type": [1, 2, 3]}
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "unk.png"
            result = render_figure(fig_data, "test_model", out)
            self.assertEqual(result, out)


class TestGracefulDegradation(unittest.TestCase):

    def test_render_raises_when_matplotlib_unavailable(self):
        import chemometrics_mcp.core.figures as fig_mod
        original = fig_mod._HAS_MPL
        try:
            fig_mod._HAS_MPL = False
            with self.assertRaises(RuntimeError):
                fig_mod.render_figure({"cluster_sizes": [1]}, "m", Path("/tmp/x.png"))
        finally:
            fig_mod._HAS_MPL = original


class TestRunAnalysisIntegration(unittest.TestCase):

    def test_render_failure_does_not_crash_pipeline(self):
        import tempfile
        import numpy as np
        from chemometrics_contracts import AnalysisPlan, RunAnalysisRequest, SpectralDataset

        from chemometrics_mcp.tools.run_analysis import run

        X = np.random.default_rng(0).random((30, 20)) + 0.1
        y = ["A"] * 15 + ["B"] * 15
        dataset = SpectralDataset(
            x=tuple(tuple(float(v) for v in row) for row in X.tolist()),
            axis=tuple(np.linspace(1000, 2000, 20).tolist()),
            labels=tuple(y),
        )
        plan = AnalysisPlan(
            task_name="binary_classification",
            preprocessing_candidates=("snv",),
            validation_strategy="stratified_kfold_5",
            model_families=("svm_rbf",),
        )
        request = RunAnalysisRequest(dataset=dataset, approved_plan=plan)

        with tempfile.TemporaryDirectory() as tmpdir:
            import chemometrics_mcp.tools.run_analysis as ra_mod
            original_render = ra_mod._render_figure
            ra_mod._render_figure = None
            try:
                response = run(request, runs_root=tmpdir)
                self.assertTrue(response.ok, msg=response.error)
            finally:
                ra_mod._render_figure = original_render


if __name__ == "__main__":
    unittest.main()
