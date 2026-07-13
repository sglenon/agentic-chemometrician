"""Tests for multi-modality validation checks."""
from __future__ import annotations

import unittest

from chemometrics_contracts import (
    AnalysisResult,
    RunMetadata,
    SpectralDataset,
)
from chemometrics_mcp.core.validation import (
    check_cross_modality_comparability,
    check_modality_consistency,
    run_all_checks,
)


def _result(
    model_name: str = "LDA",
    preprocessing: tuple[str, ...] = ("snv",),
    accuracy: float = 0.85,
) -> AnalysisResult:
    return AnalysisResult(
        task_name="binary_classification",
        model_name=model_name,
        preprocessing=preprocessing,
        metrics={"accuracy": accuracy},
        run_metadata=RunMetadata(run_id="r1", tool_name="run_analysis"),
    )


def _nir_dataset() -> SpectralDataset:
    return SpectralDataset(
        x=((1.0, 2.0), (3.0, 4.0)),
        axis=(1.0, 2.0),
        modality="NIR",
    )


def _ftir_dataset() -> SpectralDataset:
    return SpectralDataset(
        x=((1.0, 2.0), (3.0, 4.0)),
        axis=(1.0, 2.0),
        modality="FTIR",
    )


class TestCheckModalityConsistency(unittest.TestCase):

    def test_no_warning_when_no_dataset(self):
        results = [_result(preprocessing=("baseline_correction",))]
        warnings = check_modality_consistency(results, None)
        self.assertEqual(warnings, [])

    def test_no_warning_when_no_modality(self):
        dataset = SpectralDataset(x=((1.0,),), axis=(1.0,))
        results = [_result(preprocessing=("baseline_correction",))]
        warnings = check_modality_consistency(results, dataset)
        self.assertEqual(warnings, [])

    def test_baseline_correction_on_nir_triggers_warning(self):
        results = [_result(preprocessing=("baseline_correction",))]
        warnings = check_modality_consistency(results, _nir_dataset())
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].code, "modality_preprocessing_mismatch")
        self.assertEqual(warnings[0].severity, "warning")
        self.assertIn("baseline_correction", warnings[0].message)
        self.assertIn("NIR", warnings[0].message)

    def test_area_normalization_on_nir_triggers_warning(self):
        results = [_result(preprocessing=("area_normalization",))]
        warnings = check_modality_consistency(results, _nir_dataset())
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].code, "modality_preprocessing_mismatch")
        self.assertIn("area_normalization", warnings[0].message)

    def test_no_warning_for_snv_on_nir(self):
        results = [_result(preprocessing=("snv",))]
        warnings = check_modality_consistency(results, _nir_dataset())
        self.assertEqual(warnings, [])

    def test_no_warning_for_baseline_correction_on_ftir(self):
        results = [_result(preprocessing=("baseline_correction",))]
        warnings = check_modality_consistency(results, _ftir_dataset())
        self.assertEqual(warnings, [])

    def test_no_warning_for_area_normalization_on_ftir(self):
        results = [_result(preprocessing=("area_normalization",))]
        warnings = check_modality_consistency(results, _ftir_dataset())
        self.assertEqual(warnings, [])

    def test_multiple_mismatches_produce_multiple_warnings(self):
        results = [
            _result(model_name="LDA", preprocessing=("baseline_correction",)),
            _result(model_name="SVM", preprocessing=("area_normalization",)),
        ]
        warnings = check_modality_consistency(results, _nir_dataset())
        self.assertEqual(len(warnings), 2)

    def test_empty_results_no_warnings(self):
        warnings = check_modality_consistency([], _nir_dataset())
        self.assertEqual(warnings, [])


class TestCheckCrossModalityComparability(unittest.TestCase):

    def test_no_warning_for_single_modality(self):
        results_by_modality = {"NIR": [_result(), _result(model_name="SVM")]}
        warnings = check_cross_modality_comparability(results_by_modality)
        self.assertEqual(warnings, [])

    def test_warning_for_two_modalities(self):
        results_by_modality = {
            "NIR": [_result(preprocessing=("snv",))],
            "FTIR": [_result(preprocessing=("baseline_correction",))],
        }
        warnings = check_cross_modality_comparability(results_by_modality)
        self.assertTrue(len(warnings) >= 1)
        codes = {w.code for w in warnings}
        self.assertIn("cross_modality_comparison", codes)

    def test_preprocessing_differs_warning(self):
        results_by_modality = {
            "NIR": [_result(preprocessing=("snv",))],
            "FTIR": [_result(preprocessing=("baseline_correction",))],
        }
        warnings = check_cross_modality_comparability(results_by_modality)
        codes = {w.code for w in warnings}
        self.assertIn("cross_modality_preprocessing_differs", codes)

    def test_shared_models_warning(self):
        results_by_modality = {
            "NIR": [_result(model_name="LDA")],
            "FTIR": [_result(model_name="LDA")],
        }
        warnings = check_cross_modality_comparability(results_by_modality)
        codes = {w.code for w in warnings}
        self.assertIn("cross_modality_hyperparameters", codes)

    def test_empty_mapping_no_warnings(self):
        warnings = check_cross_modality_comparability({})
        self.assertEqual(warnings, [])


class TestRunAllChecksModalityConsistency(unittest.TestCase):

    def test_modality_consistency_key_present(self):
        result = _result(preprocessing=("snv",))
        summary = run_all_checks([result], _nir_dataset())
        self.assertIn("modality_consistency", summary.checks)

    def test_passes_when_no_mismatch(self):
        result = _result(preprocessing=("snv",))
        summary = run_all_checks([result], _nir_dataset())
        self.assertTrue(summary.checks["modality_consistency"])

    def test_fails_when_mismatch(self):
        result = _result(preprocessing=("baseline_correction",))
        summary = run_all_checks([result], _nir_dataset())
        self.assertFalse(summary.checks["modality_consistency"])
        codes = {w.code for w in summary.warnings}
        self.assertIn("modality_preprocessing_mismatch", codes)

    def test_no_dataset_passes(self):
        result = _result(preprocessing=("baseline_correction",))
        summary = run_all_checks([result])
        self.assertTrue(summary.checks["modality_consistency"])


if __name__ == "__main__":
    unittest.main()
