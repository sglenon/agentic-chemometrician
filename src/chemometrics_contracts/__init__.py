from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Generic, Mapping, Sequence, TypeVar

T = TypeVar("T")

DEFAULT_RUNS_DIR = "runs"
DEFAULT_ARTIFACTS_DIR = "artifacts"
RUN_ID_TEMPLATE = "run-YYYYMMDD-HHMMSS-{slug}"


class SerializableContract:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ValidationWarning(SerializableContract):
    code: str
    message: str
    category: str = "validation"
    severity: str = "warning"
    affected_stage: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ArtifactReference(SerializableContract):
    kind: str
    uri: str
    label: str | None = None
    mime_type: str | None = None
    description: str | None = None


@dataclass(frozen=True, slots=True)
class RunMetadata(SerializableContract):
    run_id: str
    tool_name: str
    dataset_id: str | None = None
    status: str | None = None
    created_at: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SpectralDataset(SerializableContract):
    x: Sequence[Sequence[Any]]
    axis: Sequence[Any]
    metadata: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    labels: Sequence[Any] | None = None
    modality: str | None = None
    sample_ids: Sequence[str] | None = None
    source_references: Sequence[ArtifactReference] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class InterpretationResult(SerializableContract):
    model_name: str
    method: str
    feature_scores: Sequence[Mapping[str, Any]]
    top_features: Sequence[Any]
    summary: str | None = None


@dataclass(frozen=True, slots=True)
class AnalysisResult(SerializableContract):
    task_name: str
    model_name: str
    preprocessing: Sequence[str] = field(default_factory=tuple)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    predictions: Sequence[Any] = field(default_factory=tuple)
    selected_features: Sequence[Any] = field(default_factory=tuple)
    figures: Sequence[ArtifactReference] = field(default_factory=tuple)
    warnings: Sequence[ValidationWarning] = field(default_factory=tuple)
    interpretation: str | None = None
    run_metadata: RunMetadata | None = None
    validation_strategy: str | None = None
    interpretation_results: Sequence[InterpretationResult] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ToolResponse(Generic[T], SerializableContract):
    tool_name: str
    ok: bool
    payload: T | None = None
    warnings: Sequence[ValidationWarning] = field(default_factory=tuple)
    artifacts: Sequence[ArtifactReference] = field(default_factory=tuple)
    metadata: RunMetadata | None = None
    message: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetInspection(SerializableContract):
    sample_count: int | None = None
    feature_count: int | None = None
    axis_min: float | None = None
    axis_max: float | None = None
    modality: str | None = None
    candidate_label_columns: Sequence[str] = field(default_factory=tuple)
    candidate_group_columns: Sequence[str] = field(default_factory=tuple)
    warnings: Sequence[ValidationWarning] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class InspectDatasetRequest(SerializableContract):
    source_uri: str
    dataset_id: str | None = None
    modality_override: str | None = None
    sample_id_column: str | None = None
    label_column: str | None = None
    source_format: str | None = None


@dataclass(frozen=True, slots=True)
class ProposeAnalysisPlanRequest(SerializableContract):
    dataset_inspection: DatasetInspection
    user_intent: str | None = None
    task_hint: str | None = None
    allow_supervised_planning: bool = True


@dataclass(frozen=True, slots=True)
class RunAnalysisRequest(SerializableContract):
    dataset: SpectralDataset
    approved_plan: AnalysisPlan
    run_id: str | None = None


@dataclass(frozen=True, slots=True)
class ValidateResultsRequest(SerializableContract):
    analysis_run: AnalysisRun | None = None
    results: Sequence[AnalysisResult] = field(default_factory=tuple)
    dataset: SpectralDataset | None = None
    dataset_inspection: DatasetInspection | None = None


@dataclass(frozen=True, slots=True)
class SelectBestModelRequest(SerializableContract):
    results: Sequence[AnalysisResult] = field(default_factory=tuple)
    validation_summary: ValidationSummary | None = None
    task_name: str | None = None


@dataclass(frozen=True, slots=True)
class RecommendNextModelRequest(SerializableContract):
    failed_model: str
    failure_reason: str
    candidate_models: Sequence[str] = field(default_factory=tuple)
    validation_summary: ValidationSummary | None = None


@dataclass(frozen=True, slots=True)
class InterpretResultsRequest(SerializableContract):
    results: Sequence[AnalysisResult] = field(default_factory=tuple)
    dataset: SpectralDataset | None = None
    validation_summary: ValidationSummary | None = None


@dataclass(frozen=True, slots=True)
class GenerateReportRequest(SerializableContract):
    analysis_run: AnalysisRun
    validation_summary: ValidationSummary | None = None
    interpretation: InterpretationSummary | None = None


@dataclass(frozen=True, slots=True)
class DatasetProfile(SerializableContract):
    modality: str
    n_samples: int
    n_features: int
    n_classes: int | None = None
    axis_min: float | None = None
    axis_max: float | None = None
    label_column: str | None = None


@dataclass(frozen=True, slots=True)
class MethodMemory(SerializableContract):
    memory_id: str
    created_at: str
    modality: str
    task_name: str
    dataset_profile: DatasetProfile
    preprocessing: str
    model_name: str
    validation_strategy: str
    key_metrics: Mapping[str, Any]
    caveats: Sequence[str]
    reviewer_notes: str | None = None
    source_run_id: str = ""
    approval_status: str = "approved"


@dataclass(frozen=True, slots=True)
class MethodMemoryEntry(SerializableContract):
    memory_id: str
    modality: str
    task_name: str
    model_name: str
    preprocessing: str
    key_metric_value: float | None = None
    approval_status: str = "approved"
    created_at: str = ""


@dataclass(frozen=True, slots=True)
class MethodMemoryIndex(SerializableContract):
    entries: Sequence[MethodMemoryEntry] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class SaveMethodMemoryRequest(SerializableContract):
    memory: MethodMemory


@dataclass(frozen=True, slots=True)
class SearchMethodMemoryRequest(SerializableContract):
    modality: str | None = None
    task_name: str | None = None
    model_name: str | None = None
    min_metric: float | None = None
    approval_status: str = "approved"


@dataclass(frozen=True, slots=True)
class RecommendFromMemoryRequest(SerializableContract):
    dataset_profile: DatasetProfile
    top_k: int = 3


def run_artifact_directory(run_id: str) -> str:
    return f"{DEFAULT_RUNS_DIR}/{run_id}/{DEFAULT_ARTIFACTS_DIR}"


@dataclass(frozen=True, slots=True)
class AnalysisPlan(SerializableContract):
    task_name: str | None = None
    preprocessing_candidates: Sequence[str] = field(default_factory=tuple)
    validation_strategy: str | None = None
    model_families: Sequence[str] = field(default_factory=tuple)
    human_readable_plan: str | None = None
    warnings: Sequence[ValidationWarning] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class AnalysisRun(SerializableContract):
    run_metadata: RunMetadata | None = None
    results: Sequence[AnalysisResult] = field(default_factory=tuple)
    failed_models: Sequence[str] = field(default_factory=tuple)
    warnings: Sequence[ValidationWarning] = field(default_factory=tuple)
    artifacts: Sequence[ArtifactReference] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ValidationSummary(SerializableContract):
    passed: bool | None = None
    checks: Mapping[str, bool] = field(default_factory=dict)
    warnings: Sequence[ValidationWarning] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ModelSelectionRecommendation(SerializableContract):
    selected_model: str | None = None
    candidate_models: Sequence[str] = field(default_factory=tuple)
    rationale: str | None = None
    requires_human_approval: bool = False
    warnings: Sequence[ValidationWarning] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class NextModelRecommendation(SerializableContract):
    failed_model: str | None = None
    failure_classification: str | None = None
    fallback_model: str | None = None
    rationale: str | None = None
    requires_human_approval: bool = True
    warnings: Sequence[ValidationWarning] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class InterpretationSummary(SerializableContract):
    summary: str | None = None
    important_features: Sequence[Any] = field(default_factory=tuple)
    model_comparisons: Sequence[str] = field(default_factory=tuple)
    warnings: Sequence[ValidationWarning] = field(default_factory=tuple)
    interpretation_methods_used: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ReportSummary(SerializableContract):
    report_title: str | None = None
    human_readable_summary: str | None = None
    agent_readable_summary: str | None = None
    primary_report: ArtifactReference | None = None
    artifacts: Sequence[ArtifactReference] = field(default_factory=tuple)
    warnings: Sequence[ValidationWarning] = field(default_factory=tuple)


InspectDatasetResponse = ToolResponse[DatasetInspection]
ProposeAnalysisPlanResponse = ToolResponse[AnalysisPlan]
RunAnalysisResponse = ToolResponse[AnalysisRun]
ValidateResultsResponse = ToolResponse[ValidationSummary]
SelectBestModelResponse = ToolResponse[ModelSelectionRecommendation]
RecommendNextModelResponse = ToolResponse[NextModelRecommendation]
InterpretResultsResponse = ToolResponse[InterpretationSummary]
GenerateReportResponse = ToolResponse[ReportSummary]


__all__ = [
    "AnalysisPlan",
    "AnalysisResult",
    "AnalysisRun",
    "ArtifactReference",
    "DEFAULT_ARTIFACTS_DIR",
    "DEFAULT_RUNS_DIR",
    "DatasetInspection",
    "DatasetProfile",
    "GenerateReportRequest",
    "GenerateReportResponse",
    "InspectDatasetRequest",
    "InspectDatasetResponse",
    "InterpretationResult",
    "InterpretResultsRequest",
    "InterpretResultsResponse",
    "InterpretationSummary",
    "MethodMemory",
    "MethodMemoryEntry",
    "MethodMemoryIndex",
    "ModelSelectionRecommendation",
    "ProposeAnalysisPlanRequest",
    "NextModelRecommendation",
    "RecommendFromMemoryRequest",
    "RecommendNextModelRequest",
    "ProposeAnalysisPlanResponse",
    "RecommendNextModelResponse",
    "ReportSummary",
    "RunAnalysisRequest",
    "RunAnalysisResponse",
    "run_artifact_directory",
    "RunMetadata",
    "RUN_ID_TEMPLATE",
    "SaveMethodMemoryRequest",
    "SearchMethodMemoryRequest",
    "SelectBestModelRequest",
    "SelectBestModelResponse",
    "SerializableContract",
    "SpectralDataset",
    "ValidateResultsRequest",
    "ToolResponse",
    "ValidationSummary",
    "ValidationWarning",
    "ValidateResultsResponse",
]
