"""Tests for Layer 1 evaluation harness.

Tests
-----
- Fixture loading (leakage_positive / leakage_clean / hallucination / fallback)
- compute_* functions on toy / mock data
- run_scenario_full S1–S4 with seed=42 → deterministic (run twice, outputs identical)
- plan_quality auto-score on synthetic plan dicts (pass/fail cases)
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is on path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from chemometrics_contracts import (
    AnalysisPlan,
    AnalysisResult,
    SpectralDataset,
)
from chemometrics_mcp.core.evaluation import (
    ScenarioResult,
    _make_toy_spectral_dataset,
    compute_effort_metrics,
    compute_fallback_correctness,
    compute_leakage_detection,
    compute_plan_quality,
    run_scenario_full,
)

_FIXTURE_DIR = _PROJECT_ROOT / "tests" / "fixtures" / "eval"
_FALLBACK_FIXTURE = _FIXTURE_DIR / "fallback_cases.json"
_HALLUCINATION_FIXTURE = _FIXTURE_DIR / "hallucination_probes.json"
_PLAN_RUBRIC = _FIXTURE_DIR / "plan_quality_rubric.txt"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fixture_exists(sub: str) -> bool:
    return (_FIXTURE_DIR / sub).exists()


# ---------------------------------------------------------------------------
# Fixture loading tests
# ---------------------------------------------------------------------------

class TestFixtureLoading:
    def test_leakage_positive_pkl_count(self):
        pos_dir = _FIXTURE_DIR / "leakage_positive"
        assert pos_dir.exists(), f"leakage_positive dir missing: {pos_dir}"
        pkls = list(pos_dir.glob("*.pkl"))
        assert len(pkls) == 3, f"Expected 3 positive fixtures, got {len(pkls)}"

    def test_leakage_clean_pkl_count(self):
        clean_dir = _FIXTURE_DIR / "leakage_clean"
        assert clean_dir.exists(), f"leakage_clean dir missing: {clean_dir}"
        pkls = list(clean_dir.glob("*.pkl"))
        assert len(pkls) == 3, f"Expected 3 clean fixtures, got {len(pkls)}"

    def test_leakage_positive_loads(self):
        pos_dir = _FIXTURE_DIR / "leakage_positive"
        for pkl_path in sorted(pos_dir.glob("*.pkl")):
            with open(pkl_path, "rb") as fh:
                obj = pickle.load(fh)
            # Each fixture is either a SpectralDataset or a dict with dataset key
            if isinstance(obj, dict):
                assert "dataset" in obj
                assert isinstance(obj["dataset"], SpectralDataset)
            else:
                assert isinstance(obj, SpectralDataset)

    def test_leakage_clean_loads(self):
        clean_dir = _FIXTURE_DIR / "leakage_clean"
        for pkl_path in sorted(clean_dir.glob("*.pkl")):
            with open(pkl_path, "rb") as fh:
                obj = pickle.load(fh)
            if isinstance(obj, dict):
                assert "dataset" in obj
                assert isinstance(obj["dataset"], SpectralDataset)
            else:
                assert isinstance(obj, SpectralDataset)

    def test_hallucination_probes_json(self):
        assert _HALLUCINATION_FIXTURE.exists()
        probes = json.loads(_HALLUCINATION_FIXTURE.read_text(encoding="utf-8"))
        assert isinstance(probes, list)
        assert len(probes) >= 3
        for p in probes:
            assert "probe_type" in p
            assert "claim_text" in p
            assert "detection_rule" in p
            assert p["probe_type"] in (
                "metric_fabricated", "causal_claim", "zero_importance_critical"
            )

    def test_fallback_cases_json(self):
        assert _FALLBACK_FIXTURE.exists()
        cases = json.loads(_FALLBACK_FIXTURE.read_text(encoding="utf-8"))
        assert isinstance(cases, list)
        assert len(cases) >= 5
        for c in cases:
            assert "failed_model" in c
            assert "failure_reason" in c
            assert "acceptable_fallback" in c
            assert isinstance(c["acceptable_fallback"], list)
            assert len(c["acceptable_fallback"]) >= 1

    def test_plan_quality_rubric_txt(self):
        assert _PLAN_RUBRIC.exists()
        text = _PLAN_RUBRIC.read_text(encoding="utf-8")
        assert "1" in text and "5" in text
        # Should contain Likert scale markers
        assert "Nonsensical" in text or "nonsensical" in text.lower()
        assert "Expert" in text or "expert" in text.lower()


# ---------------------------------------------------------------------------
# compute_plan_quality tests
# ---------------------------------------------------------------------------

class TestComputePlanQuality:
    def test_all_checks_pass(self):
        plan = {
            "task_name": "binary_classification",
            "preprocessing_candidates": ["snv", "msc"],
            "validation_strategy": "stratified_kfold_5",
            "model_families": ["random_forest", "svm_rbf"],
        }
        score, passed = compute_plan_quality(plan)
        assert score == 1.0
        assert passed is True

    def test_missing_task_name(self):
        plan = {
            "task_name": "",
            "preprocessing_candidates": ["snv"],
            "validation_strategy": "stratified_kfold_5",
            "model_families": ["random_forest"],
        }
        score, passed = compute_plan_quality(plan)
        assert score == pytest.approx(0.75)
        assert passed is True  # 3/4 checks

    def test_no_modality_preprocessing(self):
        plan = {
            "task_name": "regression",
            "preprocessing_candidates": ["standard_scaler", "pca"],
            "validation_strategy": "kfold_5",
            "model_families": ["ridge"],
        }
        score, passed = compute_plan_quality(plan)
        assert score == pytest.approx(0.75)
        # check2 fails (no modality preprocessing)

    def test_empty_model_families(self):
        plan = {
            "task_name": "regression",
            "preprocessing_candidates": ["snv"],
            "validation_strategy": "loocv",
            "model_families": [],
        }
        score, passed = compute_plan_quality(plan)
        assert score == pytest.approx(0.75)
        assert passed is True

    def test_all_checks_fail(self):
        plan = {
            "task_name": "",
            "preprocessing_candidates": [],
            "validation_strategy": "",
            "model_families": [],
        }
        score, passed = compute_plan_quality(plan)
        assert score == 0.0
        assert passed is False

    def test_only_two_checks_pass(self):
        plan = {
            "task_name": "classification",
            "preprocessing_candidates": ["snv"],
            "validation_strategy": "",
            "model_families": [],
        }
        score, passed = compute_plan_quality(plan)
        assert score == pytest.approx(0.5)
        assert passed is False

    def test_analysis_plan_dataclass(self):
        plan_obj = AnalysisPlan(
            task_name="multi_class_classification",
            preprocessing_candidates=("snv", "sg_1st_deriv"),
            validation_strategy="stratified_kfold_5",
            model_families=("random_forest", "svm_rbf", "pca_lda"),
        )
        score, passed = compute_plan_quality(plan_obj)
        assert score == 1.0
        assert passed is True

    def test_ftir_preprocessing_accepted(self):
        plan = {
            "task_name": "binary_classification",
            "preprocessing_candidates": ["baseline_correction", "area_normalization"],
            "validation_strategy": "grouped_kfold_5",
            "model_families": ["logistic_regression"],
        }
        score, passed = compute_plan_quality(plan)
        assert score == 1.0
        assert passed is True


# ---------------------------------------------------------------------------
# compute_effort_metrics tests
# ---------------------------------------------------------------------------

class TestComputeEffortMetrics:
    def test_basic(self):
        tools = [
            "inspect_dataset",
            "propose_analysis_plan",
            "run_analysis",
            "validate_results",
            "select_best_model",
            "interpret_results",
            "generate_report",
        ]
        metrics = compute_effort_metrics(tools, 12.3)
        assert metrics["tool_calls"] == 7
        assert metrics["wall_clock_s"] == pytest.approx(12.3, abs=0.01)
        assert metrics["decision_points"] == 2  # validate + select
        assert "validate_results" in metrics["unique_tools"]

    def test_empty_tools(self):
        metrics = compute_effort_metrics([], 0.0)
        assert metrics["tool_calls"] == 0
        assert metrics["decision_points"] == 0

    def test_decision_tools_counted(self):
        tools = ["recommend_next_model", "recommend_next_model"]
        metrics = compute_effort_metrics(tools, 1.0)
        assert metrics["decision_points"] == 2


# ---------------------------------------------------------------------------
# compute_fallback_correctness tests
# ---------------------------------------------------------------------------

class TestComputeFallbackCorrectness:
    def test_with_real_fixture(self):
        if not _FALLBACK_FIXTURE.exists():
            pytest.skip("fallback_cases.json not found")
        rate = compute_fallback_correctness(_FALLBACK_FIXTURE)
        assert 0.0 <= rate <= 1.0

    def test_correctness_reasonable(self):
        """Fallback correctness should be at least 50% with real fixture."""
        if not _FALLBACK_FIXTURE.exists():
            pytest.skip("fallback_cases.json not found")
        rate = compute_fallback_correctness(_FALLBACK_FIXTURE)
        assert rate >= 0.5, f"Expected ≥50% fallback correctness, got {rate:.2%}"


# ---------------------------------------------------------------------------
# compute_leakage_detection tests
# ---------------------------------------------------------------------------

class TestComputeLeakageDetection:
    def test_runs_without_error(self):
        if not (_FIXTURE_DIR / "leakage_positive").exists():
            pytest.skip("leakage fixtures not found")
        tpr, fpr, precision = compute_leakage_detection("full", _FIXTURE_DIR)
        assert 0.0 <= tpr <= 1.0
        assert 0.0 <= fpr <= 1.0
        assert 0.0 <= precision <= 1.0

    def test_tpr_positive(self):
        """At least 1 leakage fixture should be detected."""
        if not (_FIXTURE_DIR / "leakage_positive").exists():
            pytest.skip("leakage fixtures not found")
        tpr, fpr, precision = compute_leakage_detection("full", _FIXTURE_DIR)
        assert tpr > 0.0, f"Expected TPR > 0, got {tpr}"


# ---------------------------------------------------------------------------
# make_toy_spectral_dataset helper
# ---------------------------------------------------------------------------

class TestMakeToyDataset:
    def test_basic_shape(self):
        ds = _make_toy_spectral_dataset(n_samples=20, n_features=30, n_classes=2, seed=42)
        X = np.array(ds.x)
        assert X.shape == (20, 30)
        assert len(ds.labels) == 20
        assert len(set(ds.labels)) == 2

    def test_target_in_features_adds_column(self):
        ds = _make_toy_spectral_dataset(
            n_samples=20, n_features=30, n_classes=2, seed=42, target_in_features=True
        )
        X = np.array(ds.x)
        assert X.shape[1] == 31  # extra column

    def test_group_col_in_metadata(self):
        ds = _make_toy_spectral_dataset(
            n_samples=20, n_features=30, seed=42, group_col=True
        )
        if ds.metadata:
            assert "sample_group" in ds.metadata[0]


# ---------------------------------------------------------------------------
# run_scenario_full reproducibility tests
# ---------------------------------------------------------------------------

_WORKBOOK_PATH = _PROJECT_ROOT / "2026-05-21_22-59_Method Dev_wavelength_range_1450-2450.xlsx"
_FTIR_DIR = _PROJECT_ROOT / "ftir-purity-dataset" / "fwdftirjune262026"


def _has_workbook() -> bool:
    return _WORKBOOK_PATH.exists()


def _has_ftir() -> bool:
    return _FTIR_DIR.exists() and bool(list(_FTIR_DIR.glob("*.txt")))


@pytest.mark.skipif(not _has_workbook(), reason="NIR workbook not found")
class TestRunScenarioS1Reproducibility:
    def test_s1_runs_ok(self, tmp_path):
        result = run_scenario_full(
            "s1", seed=42,
            runs_root=tmp_path / "run1",
            workbook_path=_WORKBOOK_PATH,
        )
        assert result.ok, f"S1 failed: {result.error}"
        assert result.best_model is not None
        assert result.tool_calls >= 3

    def test_s1_byte_reproducible(self, tmp_path):
        r1 = run_scenario_full(
            "s1", seed=42,
            runs_root=tmp_path / "r1",
            workbook_path=_WORKBOOK_PATH,
        )
        r2 = run_scenario_full(
            "s1", seed=42,
            runs_root=tmp_path / "r2",
            workbook_path=_WORKBOOK_PATH,
        )
        assert r1.best_model == r2.best_model
        assert r1.best_metrics == r2.best_metrics
        assert r1.plan == r2.plan
        assert r1.plan_auto_score == r2.plan_auto_score


@pytest.mark.skipif(not _has_workbook(), reason="NIR workbook not found")
class TestRunScenarioS2Reproducibility:
    def test_s2_runs_ok(self, tmp_path):
        result = run_scenario_full(
            "s2", seed=42,
            runs_root=tmp_path / "run1",
            workbook_path=_WORKBOOK_PATH,
        )
        assert result.ok, f"S2 failed: {result.error}"

    def test_s2_byte_reproducible(self, tmp_path):
        r1 = run_scenario_full(
            "s2", seed=42,
            runs_root=tmp_path / "r1",
            workbook_path=_WORKBOOK_PATH,
        )
        r2 = run_scenario_full(
            "s2", seed=42,
            runs_root=tmp_path / "r2",
            workbook_path=_WORKBOOK_PATH,
        )
        assert r1.best_metrics == r2.best_metrics
        assert r1.plan == r2.plan


@pytest.mark.skipif(not _has_workbook(), reason="NIR workbook not found")
class TestRunScenarioS3Reproducibility:
    def test_s3_runs_ok(self, tmp_path):
        result = run_scenario_full(
            "s3", seed=42,
            runs_root=tmp_path / "run1",
            workbook_path=_WORKBOOK_PATH,
        )
        assert result.ok, f"S3 failed: {result.error}"

    def test_s3_byte_reproducible(self, tmp_path):
        r1 = run_scenario_full(
            "s3", seed=42,
            runs_root=tmp_path / "r1",
            workbook_path=_WORKBOOK_PATH,
        )
        r2 = run_scenario_full(
            "s3", seed=42,
            runs_root=tmp_path / "r2",
            workbook_path=_WORKBOOK_PATH,
        )
        assert r1.best_metrics == r2.best_metrics
        assert r1.plan == r2.plan


@pytest.mark.skipif(not _has_ftir(), reason="FTIR data not found")
class TestRunScenarioS4Reproducibility:
    def test_s4_runs_ok(self, tmp_path):
        result = run_scenario_full(
            "s4", seed=42,
            runs_root=tmp_path / "run1",
            ftir_dir=_FTIR_DIR,
        )
        assert result.ok, f"S4 failed: {result.error}"

    def test_s4_byte_reproducible(self, tmp_path):
        r1 = run_scenario_full(
            "s4", seed=42,
            runs_root=tmp_path / "r1",
            ftir_dir=_FTIR_DIR,
        )
        r2 = run_scenario_full(
            "s4", seed=42,
            runs_root=tmp_path / "r2",
            ftir_dir=_FTIR_DIR,
        )
        assert r1.best_metrics == r2.best_metrics
        assert r1.plan == r2.plan
