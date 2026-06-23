from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence


class SerializableDataclass:
    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SpectralDataset(SerializableDataclass):
    X: Sequence[Sequence[float]]
    axis: Sequence[float]
    metadata: Sequence[Mapping[str, object]]
    labels: Sequence[object] | None = None
    modality: str = ""
    sample_ids: Sequence[str] | None = None
    feature_names: Sequence[str] | None = None


@dataclass(slots=True)
class ValidationWarning(SerializableDataclass):
    code: str
    message: str
    severity: str = "warning"
    details: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class RunMetadata(SerializableDataclass):
    run_id: str
    tool_name: str = ""
    started_at: str = ""
    completed_at: str = ""
    parameters: dict[str, object] = field(default_factory=dict)
    source_dataset_id: str = ""


@dataclass(slots=True)
class ArtifactReference(SerializableDataclass):
    path: str
    kind: str = ""
    label: str = ""
    mime_type: str = ""


@dataclass(slots=True)
class AnalysisResult(SerializableDataclass):
    task_name: str
    model_name: str
    preprocessing: Sequence[str] = field(default_factory=list)
    metrics: dict[str, float | int | str] = field(default_factory=dict)
    predictions: Sequence[object] = field(default_factory=list)
    selected_features: Sequence[str] = field(default_factory=list)
    figures: Sequence[ArtifactReference] = field(default_factory=list)
    warnings: Sequence[ValidationWarning] = field(default_factory=list)
    interpretation: str = ""
    run_metadata: RunMetadata | None = None


@dataclass(slots=True)
class ToolResponse(SerializableDataclass):
    tool_name: str
    status: str = "ok"
    message: str = ""
    warnings: Sequence[ValidationWarning] = field(default_factory=list)
    artifacts: Sequence[ArtifactReference] = field(default_factory=list)
    run_metadata: RunMetadata | None = None


@dataclass(slots=True)
class InspectDatasetResponse(ToolResponse):
    tool_name: str = "inspect_dataset"
    dataset: SpectralDataset | None = None
    summary: dict[str, object] = field(default_factory=dict)
    candidate_labels: Sequence[str] = field(default_factory=list)
    candidate_groups: Sequence[str] = field(default_factory=list)
    modality: str = ""


@dataclass(slots=True)
class ProposeAnalysisPlanResponse(ToolResponse):
    tool_name: str = "propose_analysis_plan"
    plan: str = ""
    recommended_tasks: Sequence[str] = field(default_factory=list)
    recommended_preprocessing: Sequence[str] = field(default_factory=list)
    validation_strategy: str = ""
    approval_required: bool = True


@dataclass(slots=True)
class RunAnalysisResponse(ToolResponse):
    tool_name: str = "run_analysis"
    results: Sequence[AnalysisResult] = field(default_factory=list)
    failed_models: Sequence[str] = field(default_factory=list)


@dataclass(slots=True)
class ValidateResultsResponse(ToolResponse):
    tool_name: str = "validate_results"
    is_valid: bool = False
    validation_summary: str = ""


@dataclass(slots=True)
class SelectBestModelResponse(ToolResponse):
    tool_name: str = "select_best_model"
    selected_model_name: str = ""
    rationale: str = ""
    compared_models: Sequence[str] = field(default_factory=list)


@dataclass(slots=True)
class RecommendNextModelResponse(ToolResponse):
    tool_name: str = "recommend_next_model"
    next_model_name: str = ""
    failed_model_name: str = ""
    failure_reason: str = ""
    approval_required: bool = True
    rationale: str = ""


@dataclass(slots=True)
class InterpretResultsResponse(ToolResponse):
    tool_name: str = "interpret_results"
    interpretation: str = ""
    caveats: Sequence[str] = field(default_factory=list)


@dataclass(slots=True)
class GenerateReportResponse(ToolResponse):
    tool_name: str = "generate_report"
    report: str = ""
    report_format: str = ""
