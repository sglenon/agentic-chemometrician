from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from chemometrics_mcp.core.datasets import (  # noqa: E402
    _generate_synthetic_ftir,
    load_composition_table,
    load_ftir_composition,
)

COMPOSITION_TABLE = ROOT / "ftir-purity-dataset" / "ftir_purity_composition_table.md"


class LoadCompositionTableTests(unittest.TestCase):
    def test_parses_15_rows(self) -> None:
        df = load_composition_table(COMPOSITION_TABLE)
        self.assertEqual(len(df), 15)

    def test_correct_columns(self) -> None:
        df = load_composition_table(COMPOSITION_TABLE)
        expected = {"label", "ami_wt_pct", "fc_wt_pct", "sb_wt_pct", "prep"}
        self.assertEqual(set(df.columns), expected)

    def test_numeric_columns_are_numeric(self) -> None:
        df = load_composition_table(COMPOSITION_TABLE)
        for col in ("ami_wt_pct", "fc_wt_pct", "sb_wt_pct"):
            self.assertTrue(pd.api.types.is_numeric_dtype(df[col]), f"{col} not numeric")

    def test_known_labels_present(self) -> None:
        df = load_composition_table(COMPOSITION_TABLE)
        self.assertIn("AMI-100", df["label"].tolist())
        self.assertIn("SB-AMI-FC-10-10", df["label"].tolist())


class GenerateSyntheticFtirTests(unittest.TestCase):
    def setUp(self) -> None:
        self.compositions = load_composition_table(COMPOSITION_TABLE)

    def test_output_shape(self) -> None:
        X, axis = _generate_synthetic_ftir(self.compositions, n_points=100)
        self.assertEqual(X.shape, (15, 100))
        self.assertEqual(axis.shape, (100,))

    def test_axis_range(self) -> None:
        _, axis = _generate_synthetic_ftir(self.compositions)
        self.assertAlmostEqual(float(axis.min()), 400.0, places=1)
        self.assertAlmostEqual(float(axis.max()), 4000.0, places=1)

    def test_deterministic(self) -> None:
        X1, axis1 = _generate_synthetic_ftir(self.compositions, seed=42)
        X2, axis2 = _generate_synthetic_ftir(self.compositions, seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(axis1, axis2)

    def test_different_seeds_differ(self) -> None:
        X1, _ = _generate_synthetic_ftir(self.compositions, seed=1)
        X2, _ = _generate_synthetic_ftir(self.compositions, seed=2)
        self.assertFalse(np.array_equal(X1, X2))


class LoadFtirCompositionTests(unittest.TestCase):
    def test_returns_valid_dataset(self) -> None:
        dataset, inspection = load_ftir_composition(COMPOSITION_TABLE)
        self.assertEqual(len(dataset.x), 15)
        self.assertEqual(len(dataset.axis), 100)
        self.assertEqual(dataset.modality, "FTIR")
        self.assertIsNotNone(dataset.labels)
        self.assertEqual(len(dataset.labels), 15)

    def test_axis_within_ftir_range(self) -> None:
        dataset, _ = load_ftir_composition(COMPOSITION_TABLE)
        axis = np.array(dataset.axis)
        self.assertGreaterEqual(float(axis.min()), 400.0)
        self.assertLessEqual(float(axis.max()), 4000.0)

    def test_synthetic_data_warning_emitted(self) -> None:
        _, inspection = load_ftir_composition(COMPOSITION_TABLE)
        codes = [w.code for w in inspection.warnings]
        self.assertIn("synthetic_data", codes)

    def test_source_references_point_to_composition_table(self) -> None:
        dataset, _ = load_ftir_composition(COMPOSITION_TABLE)
        self.assertTrue(len(dataset.source_references) > 0)
        self.assertIn("ftir_purity_composition_table.md", dataset.source_references[0].uri)

    def test_modality_override(self) -> None:
        dataset, inspection = load_ftir_composition(
            COMPOSITION_TABLE, modality_override="CUSTOM"
        )
        self.assertEqual(dataset.modality, "CUSTOM")
        self.assertEqual(inspection.modality, "CUSTOM")

    def test_inspection_sample_count(self) -> None:
        _, inspection = load_ftir_composition(COMPOSITION_TABLE)
        self.assertEqual(inspection.sample_count, 15)
        self.assertEqual(inspection.feature_count, 100)


if __name__ == "__main__":
    unittest.main()
