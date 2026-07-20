"""Tests for load_ftir_real() against the fwdftirjune262026 dataset.

Run with:
    pytest tests/test_ftir_real_data.py -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from chemometrics_mcp.core.datasets import (  # noqa: E402
    _FTIR_REAL_PURITY_MANIFEST,
    load_ftir_real,
)

FTIR_DIR = ROOT / "ftir-purity-dataset" / "fwdftirjune262026"

# Expected number of .txt files in the directory (as of June 2026 dataset)
EXPECTED_SAMPLE_COUNT = 20
# Common grid: 400–4000 cm⁻¹ @ 4 cm⁻¹ step = 901 points
EXPECTED_FEATURE_COUNT = 901


@unittest.skipUnless(FTIR_DIR.exists(), f"FTIR directory not found: {FTIR_DIR}")
class TestLoadFtirReal(unittest.TestCase):
    """Integration tests against real FTIR measurement files."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset, cls.inspection = load_ftir_real(FTIR_DIR)

    # --- shape / structure -----------------------------------------------

    def test_sample_count(self) -> None:
        self.assertEqual(self.dataset.x.__len__(), EXPECTED_SAMPLE_COUNT,
                         msg=f"Expected {EXPECTED_SAMPLE_COUNT} samples")

    def test_feature_count(self) -> None:
        first_spectrum = self.dataset.x[0]
        self.assertEqual(len(first_spectrum), EXPECTED_FEATURE_COUNT,
                         msg=f"Expected {EXPECTED_FEATURE_COUNT} features (wavenumber grid points)")

    def test_axis_length(self) -> None:
        self.assertEqual(len(self.dataset.axis), EXPECTED_FEATURE_COUNT)

    def test_axis_range(self) -> None:
        axis = np.array(self.dataset.axis)
        self.assertAlmostEqual(float(axis.min()), 400.0, places=3)
        self.assertAlmostEqual(float(axis.max()), 4000.0, places=3)

    # --- metadata --------------------------------------------------------

    def test_modality_is_ftir(self) -> None:
        self.assertEqual(self.dataset.modality, "FTIR")

    def test_labels_assigned(self) -> None:
        self.assertIsNotNone(self.dataset.labels)
        self.assertEqual(len(self.dataset.labels), EXPECTED_SAMPLE_COUNT)

    def test_known_labels_present(self) -> None:
        label_set = set(self.dataset.labels)
        for expected_group in ("C", "I", "J", "M", "N", "L", "s31"):
            self.assertIn(expected_group, label_set,
                          msg=f"Purity group '{expected_group}' missing from labels")

    def test_sample_ids_match_filenames(self) -> None:
        # Every sample_id should correspond to an existing .txt stem
        txt_stems = {p.stem for p in FTIR_DIR.glob("*.txt")}
        for sid in self.dataset.sample_ids:
            self.assertIn(sid, txt_stems, msg=f"sample_id '{sid}' not found in directory")

    def test_inspection_sample_count(self) -> None:
        self.assertEqual(self.inspection.sample_count, EXPECTED_SAMPLE_COUNT)

    def test_inspection_feature_count(self) -> None:
        self.assertEqual(self.inspection.feature_count, EXPECTED_FEATURE_COUNT)

    def test_inspection_axis_range(self) -> None:
        self.assertAlmostEqual(self.inspection.axis_min, 400.0, places=3)
        self.assertAlmostEqual(self.inspection.axis_max, 4000.0, places=3)

    # --- data quality ----------------------------------------------------

    def test_no_nans(self) -> None:
        X = np.array(self.dataset.x)
        self.assertEqual(int(np.isnan(X).sum()), 0, msg="NaN values found in spectral matrix")

    def test_no_infs(self) -> None:
        X = np.array(self.dataset.x)
        self.assertFalse(np.any(np.isinf(X)), msg="Inf values found in spectral matrix")

    def test_missing_base_run_warning(self) -> None:
        """L.txt is absent; expect a warning about the missing base run."""
        codes = [w.code for w in self.inspection.warnings]
        self.assertIn("missing_base_run", codes,
                      msg="Expected 'missing_base_run' warning for absent L.txt")

    # --- PCA smoke test --------------------------------------------------

    def test_pca_runs_without_error(self) -> None:
        """Quick PCA sanity check: no exception, explained variance sums ≤ 1."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        X = np.array(self.dataset.x)
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=min(5, X.shape[0] - 1))
        pca.fit(X_scaled)
        self.assertLessEqual(float(pca.explained_variance_ratio_.sum()), 1.0 + 1e-9)

    # --- reporting -------------------------------------------------------

    def test_report_sample_and_wavenumber(self) -> None:
        """Print a summary to stdout for human review."""
        X = np.array(self.dataset.x)
        axis = np.array(self.dataset.axis)
        label_counts: dict[str, int] = {}
        for lbl in (self.dataset.labels or []):
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        print(
            f"\n[FTIR real data summary]\n"
            f"  samples       : {X.shape[0]}\n"
            f"  wavenumber pts: {X.shape[1]}\n"
            f"  axis range    : {axis.min():.1f} – {axis.max():.1f} cm⁻¹\n"
            f"  purity groups : {label_counts}\n"
            f"  warnings      : {[w.code for w in self.inspection.warnings]}"
        )


class TestFtirRealManifest(unittest.TestCase):
    """Unit tests for the built-in purity manifest."""

    def test_all_expected_stems_present(self) -> None:
        expected = {
            "C", "C1", "C2",
            "I", "I1", "I2",
            "J", "J1", "J2",
            "L1", "L2",
            "M", "M1", "M2",
            "N", "N1", "N2",
            "s31acn2", "s31meoh", "S31SOLID",
        }
        self.assertEqual(set(_FTIR_REAL_PURITY_MANIFEST.keys()), expected)

    def test_group_c_maps_correctly(self) -> None:
        for stem in ("C", "C1", "C2"):
            self.assertEqual(_FTIR_REAL_PURITY_MANIFEST[stem], "C")

    def test_s31_variants_same_group(self) -> None:
        for stem in ("s31acn2", "s31meoh", "S31SOLID"):
            self.assertEqual(_FTIR_REAL_PURITY_MANIFEST[stem], "s31")

    def test_no_l_base(self) -> None:
        """L.txt is absent; manifest must not contain a 'L' base entry."""
        self.assertNotIn("L", _FTIR_REAL_PURITY_MANIFEST)


@unittest.skipUnless(FTIR_DIR.exists(), f"FTIR directory not found: {FTIR_DIR}")
class TestLoadFtirRealOptions(unittest.TestCase):
    """Test optional parameters of load_ftir_real."""

    def test_modality_override(self) -> None:
        ds, insp = load_ftir_real(FTIR_DIR, modality_override="FTIR-ATR")
        self.assertEqual(ds.modality, "FTIR-ATR")
        self.assertEqual(insp.modality, "FTIR-ATR")

    def test_custom_grid(self) -> None:
        ds, insp = load_ftir_real(
            FTIR_DIR,
            wavenumber_start=500.0,
            wavenumber_end=3500.0,
            wavenumber_step=10.0,
        )
        expected_pts = int(round((3500.0 - 500.0) / 10.0)) + 1  # 301
        self.assertEqual(len(ds.axis), expected_pts)
        self.assertEqual(insp.feature_count, expected_pts)

    def test_custom_manifest_unknown_label(self) -> None:
        """Passing a manifest that omits most stems triggers unknown-label warning."""
        ds, insp = load_ftir_real(FTIR_DIR, purity_manifest={"C": "compound_c"})
        label_set = set(ds.labels or [])
        self.assertIn("unknown", label_set)
        codes = [w.code for w in insp.warnings]
        self.assertIn("unknown_sample_stems", codes)


if __name__ == "__main__":
    unittest.main()
