from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence

WarningSeverity = Literal["info", "warning", "error"]


class DictSerializable:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ValidationWarning(DictSerializable):
    code: str
    message: str
    severity: WarningSeverity = "warning"
    source: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ArtifactReference(DictSerializable):
    uri: str
    kind: str
    description: str | None = None
    mime_type: str | None = None
    checksum: str | None = None


@dataclass(slots=True)
class RunMetadata(DictSerializable):
    run_id: str
    tool_name: str
    dataset_id: str | None = None
    model_name: str | None = None
    prompt_version: str | None = None
    status: str = "planned"
    started_at: str | None = None
    completed_at: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SpectralDataset(DictSerializable):
    X: Any
    axis: Any
    metadata: Any
    modality: str
    labels: Sequence[Any] | None = None
    sample_ids: Sequence[str] | None = None
    feature_names: Sequence[str] | None = None


@dataclass(slots=True)
class AnalysisResult(DictSerializable):
    task_name: str
    model_name: str
    preprocessing: Sequence[str] = field(default_factory=list)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    predictions: Any = None
    selected_features: Sequence[str] = field(default_factory=list)
    figures: Sequence[ArtifactReference] = field(default_factory=list)
    warnings: Sequence[ValidationWarning] = field(default_factory=list)
    interpretation: str | None = None
    run_metadata: RunMetadata | None = None
    artifacts: Sequence[ArtifactReference] = field(default_factory=list)


@dataclass(slots=True)
class DatasetInspection(DictSerializable):
    sample_count: int
    feature_count: int
    axis_min: float | None = None
    axis_max: float | None = None
    modality: str | None = None
    candidate_label_columns: Sequence[str] = field(default_factory=list)
    candidate_group_columns: Sequence[str] = field(default_factory=list)
    warnings: Sequence[ValidationWarning] = field(default_factory=list)


@dataclass(slots=True)
class InspectDatasetResponse(DictSerializable):
    inspection: DatasetInspection
    run_metadata: RunMetadata | None = None
    artifacts: Sequence[ArtifactReference] = field(default_factory=list)


@dataclass(slots=True)
class AnalysisPlan(DictSerializable):
    task_types: Sequence[str] = field(default_factory=list)
    preprocessing_candidates: Sequence[str] = field(default_factory=list)
    validation_strategy: str | None = None
    model_families: Sequence[str] = field(default_factory=list)
    requires_human_approval: bool = True
    rationale: str | None = None


@dataclass(slots=True)
class ProposeAnalysisPlanResponse(DictSerializable):
    plan: AnalysisPlan
    run_metadata: RunMetadata | None = None
    warnings: Sequence[ValidationWarning] = field(default_factory=list)


@dataclass(slots=True)
class RunAnalysisResponse(DictSerializable):
    run_id: str
    results: Sequence[AnalysisResult] = field(default_factory=list)
    artifacts: Sequence[ArtifactReference] = field(default_factory=list)
    failures: Sequence[ValidationWarning] = field(default_factory=list)
    run_metadata: RunMetadata | None = None


@dataclass(slots=True)
class ValidationSummary(DictSerializable):
    status: str
    warnings: Sequence[ValidationWarning] = field(default_factory=list)
    checks: Mapping[str, Any] = field(default_factory=dict)
    review_required: bool = False


@dataclass(slots=True)
class ValidateResultsResponse(DictSerializable):
    summary: ValidationSummary
    run_metadata: RunMetadata | None = None
    artifacts: Sequence[ArtifactReference] = field(default_factory=list)


@dataclass(slots=True)
class ModelCandidate(DictSerializable):
    model_name: str
    score: float | None = None
    notes: str | None = None


@dataclass(slots=True)
class BestModelRecommendation(DictSerializable):
    selected_model_name: str
    reason: str
    candidates: Sequence[ModelCandidate] = field(default_factory=list)
    warnings: Sequence[ValidationWarning] = field(default_factory=list)


@dataclass(slots=True)
class RecommendNextModelResponse(DictSerializable):
    failed_model_name: str
    failure_class: str
    recommended_model_name: str
    reason: str
    approval_required: bool = False
    warnings: Sequence[ValidationWarning] = field(default_factory=list)


@dataclass(slots=True)
class InterpretationSummary(DictSerializable):
    summary: str
    important_features: Sequence[str] = field(default_factory=list)
    feature_notes: Mapping[str, Any] = field(default_factory=dict)
    warnings: Sequence[ValidationWarning] = field(default_factory=list)


@dataclass(slots=True)
class InterpretResultsResponse(DictSerializable):
    interpretation: InterpretationSummary
    run_metadata: RunMetadata | None = None
    artifacts: Sequence[ArtifactReference] = field(default_factory=list)


@dataclass(slots=True)
class GenerateReportResponse(DictSerializable):
    report_title: str
    report_artifacts: Sequence[ArtifactReference] = field(default_factory=list)
    warnings: Sequence[ValidationWarning] = field(default_factory=list)
    run_metadata: RunMetadata | None = None
