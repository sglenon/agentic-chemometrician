from pathlib import Path
import sys
import unittest
from typing import get_args, get_origin

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemometrics_contracts import (  # noqa: E402
    AnalysisPlan,
    AnalysisResult,
    AnalysisRun,
    ArtifactReference,
    DatasetInspection,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_RUNS_DIR,
    GenerateReportRequest,
    InspectDatasetResponse,
    InspectDatasetRequest,
    InterpretationSummary,
    InterpretResultsRequest,
    ModelSelectionRecommendation,
    NextModelRecommendation,
    ProposeAnalysisPlanRequest,
    RecommendNextModelRequest,
    ReportSummary,
    RunAnalysisRequest,
    RunAnalysisResponse,
    RunMetadata,
    RUN_ID_TEMPLATE,
    SelectBestModelRequest,
    SpectralDataset,
    ToolResponse,
    ValidationSummary,
    ValidationWarning,
    ValidateResultsRequest,
    run_artifact_directory,
)


class ContractImportTests(unittest.TestCase):
    def test_contracts_import_and_serialize_cleanly(self) -> None:
        warning = ValidationWarning(
            code="small_sample",
            message="Sample size is small.",
            category="data_quality",
            severity="warning",
            affected_stage="inspection",
        )
        artifact = ArtifactReference(kind="figure", uri="artifacts/roc.png")
        dataset = SpectralDataset(
            x=((1.0, 2.0), (3.0, 4.0)),
            axis=(1000.0, 1100.0),
            metadata=({"sample_id": "a"}, {"sample_id": "b"}),
            labels=("A", "B"),
            modality="NIR",
            source_references=(artifact,),
        )
        inspection = DatasetInspection(
            sample_count=2,
            feature_count=2,
            modality="NIR",
            candidate_label_columns=("label",),
            candidate_group_columns=("batch",),
            warnings=(warning,),
        )
        result = AnalysisResult(
            task_name="classification",
            model_name="svm",
            preprocessing=("snv",),
            metrics={"accuracy": 0.95},
            predictions=("A", "B"),
            selected_features=(1000.0, 1100.0),
            figures=(artifact,),
            warnings=(warning,),
            interpretation="Exploratory summary",
        )
        response = ToolResponse(
            tool_name="inspect_dataset",
            ok=True,
            payload=DatasetInspection(sample_count=2, feature_count=2, modality="NIR"),
            warnings=(warning,),
        )

        self.assertEqual(dataset.modality, "NIR")
        self.assertEqual(result.to_dict()["model_name"], "svm")
        self.assertEqual(response.to_dict()["payload"]["sample_count"], 2)
        self.assertIs(get_origin(InspectDatasetResponse), ToolResponse)
        self.assertEqual(get_args(InspectDatasetResponse)[0], DatasetInspection)


class ContractImmutabilityTests(unittest.TestCase):
    def test_validation_warning_is_frozen(self) -> None:
        w = ValidationWarning(code="x", message="y")
        with self.assertRaises((AttributeError, TypeError)):
            w.code = "changed"  # type: ignore[misc]

    def test_spectral_dataset_is_frozen(self) -> None:
        ds = SpectralDataset(x=((1.0,),), axis=(1000.0,))
        with self.assertRaises((AttributeError, TypeError)):
            ds.modality = "FTIR"  # type: ignore[misc]

    def test_analysis_result_is_frozen(self) -> None:
        r = AnalysisResult(task_name="cls", model_name="svm")
        with self.assertRaises((AttributeError, TypeError)):
            r.task_name = "regression"  # type: ignore[misc]


class ContractRequiredFieldTests(unittest.TestCase):
    def test_validation_warning_requires_code_and_message(self) -> None:
        with self.assertRaises(TypeError):
            ValidationWarning()  # type: ignore[call-arg]

    def test_analysis_result_requires_task_and_model(self) -> None:
        with self.assertRaises(TypeError):
            AnalysisResult()  # type: ignore[call-arg]

    def test_run_metadata_requires_run_id_and_tool_name(self) -> None:
        with self.assertRaises(TypeError):
            RunMetadata()  # type: ignore[call-arg]

    def test_artifact_reference_requires_kind_and_uri(self) -> None:
        with self.assertRaises(TypeError):
            ArtifactReference()  # type: ignore[call-arg]


class ContractDefaultFieldTests(unittest.TestCase):
    def test_spectral_dataset_defaults(self) -> None:
        ds = SpectralDataset(x=((1.0,),), axis=(1000.0,))
        self.assertIsNone(ds.labels)
        self.assertIsNone(ds.modality)
        self.assertIsNone(ds.sample_ids)
        self.assertEqual(list(ds.source_references), [])

    def test_analysis_result_defaults(self) -> None:
        r = AnalysisResult(task_name="regression", model_name="plsr")
        self.assertEqual(list(r.preprocessing), [])
        self.assertEqual(dict(r.metrics), {})
        self.assertIsNone(r.interpretation)
        self.assertIsNone(r.run_metadata)

    def test_analysis_plan_defaults(self) -> None:
        plan = AnalysisPlan()
        self.assertEqual(list(plan.preprocessing_candidates), [])
        self.assertEqual(list(plan.model_families), [])
        self.assertIsNone(plan.validation_strategy)

    def test_validation_summary_defaults(self) -> None:
        vs = ValidationSummary()
        self.assertIsNone(vs.passed)
        self.assertEqual(list(vs.warnings), [])

    def test_dataset_inspection_defaults(self) -> None:
        di = DatasetInspection()
        self.assertIsNone(di.sample_count)
        self.assertIsNone(di.feature_count)
        self.assertIsNone(di.modality)
        self.assertEqual(list(di.candidate_label_columns), [])
        self.assertEqual(list(di.candidate_group_columns), [])


class ContractSerializationTests(unittest.TestCase):
    def test_validation_warning_round_trip(self) -> None:
        w = ValidationWarning(
            code="leakage_risk",
            message="Replicate groups may overlap splits.",
            category="reliability",
            severity="error",
            affected_stage="validation",
        )
        d = w.to_dict()
        self.assertEqual(d["code"], "leakage_risk")
        self.assertEqual(d["severity"], "error")
        self.assertEqual(d["affected_stage"], "validation")

    def test_analysis_result_nested_round_trip(self) -> None:
        artifact = ArtifactReference(kind="figure", uri="runs/run-001/artifacts/roc.png")
        warning = ValidationWarning(code="small_n", message="N < 20 per class.")
        result = AnalysisResult(
            task_name="binary_classification",
            model_name="svm_rbf",
            preprocessing=("snv", "detrend"),
            metrics={"accuracy": 0.88, "roc_auc": 0.92},
            figures=(artifact,),
            warnings=(warning,),
        )
        d = result.to_dict()
        self.assertEqual(d["task_name"], "binary_classification")
        self.assertEqual(d["metrics"]["roc_auc"], 0.92)
        self.assertEqual(d["figures"][0]["kind"], "figure")
        self.assertEqual(d["warnings"][0]["code"], "small_n")

    def test_tool_response_ok_and_error_fields(self) -> None:
        ok_resp = ToolResponse(tool_name="inspect_dataset", ok=True, message="Done.")
        self.assertTrue(ok_resp.ok)
        self.assertIsNone(ok_resp.error)

        err_resp = ToolResponse(tool_name="run_analysis", ok=False, error="File not found.")
        self.assertFalse(err_resp.ok)
        self.assertEqual(err_resp.error, "File not found.")

    def test_nested_tool_response_payload_serializes(self) -> None:
        inspection = DatasetInspection(
            sample_count=146,
            feature_count=249,
            axis_min=1454.0,
            axis_max=2446.0,
            modality="NIR",
            candidate_label_columns=("Measurement Description",),
        )
        resp = ToolResponse(tool_name="inspect_dataset", ok=True, payload=inspection)
        d = resp.to_dict()
        self.assertEqual(d["payload"]["sample_count"], 146)
        self.assertEqual(d["payload"]["modality"], "NIR")


class ContractArtifactPathTests(unittest.TestCase):
    def test_run_artifact_directory_format(self) -> None:
        path = run_artifact_directory("run-20260624-120000-nir-demo")
        self.assertEqual(path, "runs/run-20260624-120000-nir-demo/artifacts")

    def test_run_id_template_placeholder(self) -> None:
        self.assertIn("{slug}", RUN_ID_TEMPLATE)
        self.assertIn("YYYYMMDD", RUN_ID_TEMPLATE)

    def test_default_dirs_constants(self) -> None:
        self.assertEqual(DEFAULT_RUNS_DIR, "runs")
        self.assertEqual(DEFAULT_ARTIFACTS_DIR, "artifacts")


class ContractInputSchemaTests(unittest.TestCase):
    def test_inspect_dataset_request_minimal(self) -> None:
        req = InspectDatasetRequest(source_uri="data/nir.xlsx")
        self.assertEqual(req.source_uri, "data/nir.xlsx")
        self.assertIsNone(req.dataset_id)
        self.assertIsNone(req.modality_override)
        self.assertIsNone(req.label_column)

    def test_inspect_dataset_request_full(self) -> None:
        req = InspectDatasetRequest(
            source_uri="data/nir.xlsx",
            dataset_id="nir-flooring-v1",
            modality_override="NIR",
            label_column="Measurement Description",
        )
        self.assertEqual(req.modality_override, "NIR")
        self.assertEqual(req.label_column, "Measurement Description")

    def test_recommend_next_model_request_requires_failed_model(self) -> None:
        with self.assertRaises(TypeError):
            RecommendNextModelRequest()  # type: ignore[call-arg]

    def test_select_best_model_request_defaults(self) -> None:
        req = SelectBestModelRequest()
        self.assertEqual(list(req.results), [])
        self.assertIsNone(req.task_name)


if __name__ == "__main__":
    unittest.main()
