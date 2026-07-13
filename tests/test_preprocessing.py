"""Tests for preprocessing module: new methods, adaptive SG, backward compatibility."""
from __future__ import annotations

import unittest

import numpy as np

from chemometrics_mcp.core.preprocessing import apply


class TestBaselineCorrection(unittest.TestCase):

    def test_removes_known_baseline(self):
        rng = np.random.RandomState(42)
        n, p = 5, 200
        x = np.linspace(0, 10, p)
        signal = np.exp(-0.5 * ((x - 3) / 0.3) ** 2) + np.exp(-0.5 * ((x - 7) / 0.3) ** 2)
        baseline = 0.05 * x + 0.01 * x ** 2
        X = np.tile(signal, (n, 1)) + np.tile(baseline, (n, 1)) + rng.normal(0, 0.005, (n, p))
        X_out, details = apply(X, "baseline_correction")
        self.assertEqual(X_out.shape, X.shape)
        self.assertEqual(details["method"], "baseline_correction")
        residual = np.mean(np.abs(X_out - np.tile(signal, (n, 1))))
        self.assertLess(residual, 0.1)

    def test_deterministic(self):
        rng = np.random.RandomState(7)
        X = rng.randn(3, 80)
        out1, _ = apply(X, "baseline_correction")
        out2, _ = apply(X, "baseline_correction")
        np.testing.assert_array_equal(out1, out2)


class TestAreaNormalization(unittest.TestCase):

    def test_spectra_integrate_to_one(self):
        rng = np.random.RandomState(0)
        X = np.abs(rng.randn(10, 50)) + 0.1
        X_out, details = apply(X, "area_normalization")
        self.assertEqual(details["method"], "area_normalization")
        for i in range(X_out.shape[0]):
            area = np.trapezoid(X_out[i])
            self.assertAlmostEqual(area, 1.0, places=5)

    def test_with_axis(self):
        rng = np.random.RandomState(1)
        X = np.abs(rng.randn(4, 60)) + 0.1
        axis = np.linspace(400, 4000, 60)
        X_out, details = apply(X, "area_normalization", axis=axis)
        self.assertTrue(details["used_axis"])
        for i in range(X_out.shape[0]):
            area = np.trapezoid(X_out[i], x=axis)
            self.assertAlmostEqual(area, 1.0, places=5)


class TestAdaptiveSgWindow(unittest.TestCase):

    def test_narrow_input_no_crash(self):
        rng = np.random.RandomState(99)
        X = rng.randn(5, 7)
        X_out, details = apply(X, "sg_1st_deriv")
        self.assertEqual(X_out.shape, X.shape)
        self.assertLessEqual(details["window_length"], 7)
        self.assertGreaterEqual(details["window_length"], 5)
        self.assertEqual(details["window_length"] % 2, 1)

    def test_wide_input_uses_11(self):
        rng = np.random.RandomState(10)
        X = rng.randn(5, 50)
        _, details = apply(X, "sg_1st_deriv")
        self.assertEqual(details["window_length"], 11)


class TestBackwardCompatibility(unittest.TestCase):

    def test_snv_without_axis(self):
        rng = np.random.RandomState(5)
        X = rng.randn(8, 30)
        X_out, details = apply(X, "snv")
        self.assertEqual(X_out.shape, X.shape)
        self.assertEqual(details["method"], "snv")

    def test_raw_without_axis(self):
        X = np.ones((3, 10))
        X_out, details = apply(X, "raw")
        np.testing.assert_array_equal(X_out, X)

    def test_msc_without_axis(self):
        rng = np.random.RandomState(3)
        X = rng.randn(6, 20)
        X_out, details = apply(X, "msc")
        self.assertEqual(X_out.shape, X.shape)


class TestOutputShapeProperty(unittest.TestCase):

    def test_all_methods_preserve_shape(self):
        rng = np.random.RandomState(123)
        X = np.abs(rng.randn(10, 50)) + 0.1
        methods = ["raw", "snv", "msc", "sg_1st_deriv", "sg_2nd_deriv",
                   "baseline_correction", "area_normalization"]
        for method in methods:
            X_out, _ = apply(X, method)
            self.assertEqual(X_out.shape, X.shape, f"Shape mismatch for method={method}")


if __name__ == "__main__":
    unittest.main()
