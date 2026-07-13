"""Tests for recommend_next_model tool and core/fallback logic.

All tests use synthetic data only — no file I/O except artifact writes
which are directed to a temporary directory via tmp_path.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from chemometrics_contracts import (
    NextModelRecommendation,
    RecommendNextModelRequest,
    ValidationWarning,
)
from chemometrics_mcp.core.fallback import (
    build_recommendation,
    classify_failure,
    recommend_fallback,
)
from chemometrics_mcp.tools import recommend_next_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    failed_model: str = "svm_rbf",
    failure_reason: str = "max_iter exceeded",
    candidate_models: tuple[str, ...] = (),
) -> RecommendNextModelRequest:
    return RecommendNextModelRequest(
        failed_model=failed_model,
        failure_reason=failure_reason,
        candidate_models=candidate_models,
    )


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------


class TestClassifyFailure(unittest.TestCase):
    def test_convergence_keyword(self):
        assert classify_failure("max_iter exceeded") == "convergence"

    def test_data_keyword(self):
        assert classify_failure("missing values in dataset") == "data"

    def test_dependency_keyword(self):
        assert classify_failure("scikit-learn not installed") == "dependency"

    def test_sample_size_keyword(self):
        assert classify_failure("n_samples too few for cross-validation") == "sample_size"

    def test_unsupported_task_keyword(self):
        assert classify_failure("unsupported task type") == "unsupported_task"

    def test_unknown_fallback(self):
        assert classify_failure("something completely unrelated") == "unknown"

    def test_case_insensitive(self):
        assert classify_failure("MAX_ITER EXCEEDED") == "convergence"

    def test_preprocessing_keyword(self):
        assert classify_failure("SNV preprocessing failed") == "preprocessing"

    def test_data_nan_keyword(self):
        assert classify_failure("NaN values detected in input") == "data"


# ---------------------------------------------------------------------------
# Fallback recommendation tests
# ---------------------------------------------------------------------------


class TestRecommendFallback(unittest.TestCase):
    def test_svm_rbf_convergence_gives_random_forest(self):
        result = recommend_fallback("svm_rbf", "convergence")
        assert result == "random_forest"

    def test_svm_rbf_data_gives_none(self):
        # data issue: _CLASSIFICATION_FALLBACKS["data"] = None
        # model-specific: _MODEL_SPECIFIC_FALLBACKS["svm_rbf"] = "random_forest"
        # Since model-specific has a value, it should return random_forest
        # But spec says data -> None; let's re-read the spec carefully:
        # Algorithm step 2: use MODEL_SPECIFIC if candidates empty
        # Algorithm step 3: use CLASSIFICATION if still None
        # So svm_rbf + data -> random_forest (model-specific wins over classification None)
        result = recommend_fallback("svm_rbf", "data")
        assert result == "random_forest"

    def test_pca_lda_unknown_gives_svm_rbf(self):
        # model-specific fallback for pca_lda is svm_rbf
        result = recommend_fallback("pca_lda", "unknown")
        assert result == "svm_rbf"

    def test_candidate_models_contains_preferred_fallback(self):
        # svm_rbf + convergence -> model-specific is random_forest
        # candidates contain random_forest -> use it
        result = recommend_fallback("svm_rbf", "convergence", ("random_forest", "plsr"))
        assert result == "random_forest"

    def test_candidate_models_missing_preferred_use_classification(self):
        # svm_rbf + convergence -> model-specific is random_forest
        # candidates don't have random_forest but have plsr (not classification fallback either)
        # classification fallback for convergence = random_forest (not in candidates)
        # Neither preferred option in candidates -> fall through to unconstrained -> random_forest
        result = recommend_fallback("svm_rbf", "convergence", ("plsr",))
        assert result == "random_forest"

    def test_candidate_models_no_preferred_uses_classification_fallback_if_present(self):
        # unknown model + convergence -> no model-specific -> classification fallback = random_forest
        # candidate contains random_forest -> found via classification path
        result = recommend_fallback("unknown_model", "convergence", ("random_forest", "pca"))
        assert result == "random_forest"

    def test_no_fallback_for_dependency_classification(self):
        # unknown model + dependency -> no model-specific -> classification None
        result = recommend_fallback("unknown_model", "dependency")
        assert result is None

    def test_plsr_fallback_is_svr(self):
        result = recommend_fallback("plsr", "unknown")
        assert result == "svr"


# ---------------------------------------------------------------------------
# build_recommendation tests
# ---------------------------------------------------------------------------


class TestBuildRecommendation(unittest.TestCase):
    def test_requires_human_approval_always_true(self):
        req = _make_request(failure_reason="max_iter exceeded")
        rec = build_recommendation(req)
        assert rec.requires_human_approval is True

    def test_requires_human_approval_true_even_for_unknown(self):
        req = _make_request(failure_reason="something completely random")
        rec = build_recommendation(req)
        assert rec.requires_human_approval is True

    def test_rationale_non_empty(self):
        req = _make_request(failure_reason="max_iter exceeded")
        rec = build_recommendation(req)
        assert rec.rationale and len(rec.rationale) > 0

    def test_fallback_required_warning_always_present(self):
        req = _make_request(failure_reason="max_iter exceeded")
        rec = build_recommendation(req)
        codes = [w.code for w in rec.warnings]
        assert "fallback_required" in codes

    def test_no_automatic_fallback_warning_when_fallback_is_none(self):
        # dependency failure on unknown model -> None fallback
        req = _make_request(
            failed_model="unknown_model",
            failure_reason="package not found: scipy",
        )
        rec = build_recommendation(req)
        codes = [w.code for w in rec.warnings]
        assert "no_automatic_fallback" in codes

    def test_data_failure_warning_present_for_data_classification(self):
        req = _make_request(failure_reason="missing values in dataset")
        rec = build_recommendation(req)
        codes = [w.code for w in rec.warnings]
        assert "data_failure" in codes

    def test_no_data_failure_warning_for_non_data_classification(self):
        req = _make_request(failure_reason="max_iter exceeded")
        rec = build_recommendation(req)
        codes = [w.code for w in rec.warnings]
        assert "data_failure" not in codes

    def test_failed_model_preserved_in_recommendation(self):
        req = _make_request(failed_model="plsr", failure_reason="max_iter exceeded")
        rec = build_recommendation(req)
        assert rec.failed_model == "plsr"

    def test_fallback_model_for_convergence(self):
        req = _make_request(failed_model="svm_rbf", failure_reason="max_iter exceeded")
        rec = build_recommendation(req)
        # svm_rbf model-specific fallback = random_forest
        assert rec.fallback_model == "random_forest"


# ---------------------------------------------------------------------------
# Tool boundary tests
# ---------------------------------------------------------------------------


class TestRecommendNextModelTool(unittest.TestCase):
    def test_run_returns_ok_true(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            req = _make_request(failure_reason="max_iter exceeded")
            response = recommend_next_model.run(req, runs_root=td)
            assert response.ok is True

    def test_artifact_fallback_recommendation_json_created(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            req = _make_request(failure_reason="max_iter exceeded")
            response = recommend_next_model.run(req, runs_root=td)
            # Verify at least one artifact exists and is the right kind
            assert len(response.artifacts) == 1
            artifact = response.artifacts[0]
            assert artifact.kind == "fallback_recommendation"
            # Verify the file was actually written
            artifact_path = Path(artifact.uri)
            assert artifact_path.exists(), f"Artifact not found: {artifact_path}"
            # Verify it's valid JSON
            data = json.loads(artifact_path.read_text())
            assert "failed_model" in data

    def test_payload_failed_model_matches_request(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            req = _make_request(failed_model="plsr", failure_reason="max_iter exceeded")
            response = recommend_next_model.run(req, runs_root=td)
            assert response.payload is not None
            assert response.payload.failed_model == "plsr"

    def test_run_returns_ok_true_even_when_no_fallback(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            req = _make_request(
                failed_model="unknown_model",
                failure_reason="package not found: scipy",
            )
            response = recommend_next_model.run(req, runs_root=td)
            # ok=True even when fallback_model is None
            assert response.ok is True
            assert response.payload is not None
            assert response.payload.fallback_model is None

    def test_warnings_propagated_to_tool_response(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            req = _make_request(failure_reason="max_iter exceeded")
            response = recommend_next_model.run(req, runs_root=td)
            assert len(response.warnings) > 0
            codes = [w.code for w in response.warnings]
            assert "fallback_required" in codes


if __name__ == "__main__":
    unittest.main()
