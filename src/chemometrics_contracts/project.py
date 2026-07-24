"""Immutable boundary contracts for the scientist-facing project workflow."""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, ClassVar, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class FrozenDict(dict):
    """A JSON-serializable dict whose contents cannot change after creation."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        raise TypeError("contract mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenDict(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    return value


class _StringEnum(str, Enum):
    """Base class which serializes enum values as ordinary strings."""


class Modality(_StringEnum):
    NIR = "nir"
    FTIR = "ftir"
    RAMAN = "raman"
    UV_VIS = "uv_vis"
    PXRD = "pxrd"
    MASS_SPECTROMETRY = "mass_spectrometry"
    NMR = "nmr"
    HYPERSPECTRAL = "hyperspectral"
    OTHER = "other"


class Representation(_StringEnum):
    SPECTRUM = "spectrum"
    CONTINUOUS_SPECTRUM = "continuous_spectrum"
    PEAK_LIST = "peak_list"
    DIFFRACTION_PATTERN = "diffraction_pattern"
    CHROMATOGRAM = "chromatogram"
    IMAGE = "image"
    TABLE = "table"
    FEATURE_MATRIX = "feature_matrix"
    OTHER = "other"


class AxisKind(_StringEnum):
    WAVENUMBER = "wavenumber"
    WAVELENGTH = "wavelength"
    RAMAN_SHIFT = "raman_shift"
    MASS_TO_CHARGE = "mass_to_charge"
    TWO_THETA = "two_theta"
    ENERGY = "energy"
    FREQUENCY = "frequency"
    RETENTION_TIME = "retention_time"
    PIXEL = "pixel"
    INDEX = "index"
    OTHER = "other"


class SignalKind(_StringEnum):
    ABSORBANCE = "absorbance"
    TRANSMITTANCE = "transmittance"
    PERCENT_TRANSMITTANCE = "percent_transmittance"
    REFLECTANCE = "reflectance"
    INTENSITY = "intensity"
    SINGLE_BEAM_INTENSITY = "single_beam_intensity"
    RELATIVE_ABUNDANCE = "relative_abundance"
    DIFFRACTION_INTENSITY = "diffraction_intensity"
    COUNTS = "counts"
    OTHER = "other"


class MeasurementRole(_StringEnum):
    SAMPLE = "sample"
    PRECURSOR = "precursor"
    PRODUCT = "product"
    REFERENCE = "reference"
    SIMULATED_REFERENCE = "simulated_reference"
    SIDE_PRODUCT_CANDIDATE = "side_product_candidate"
    BLANK = "blank"
    CALIBRATION = "calibration"
    QUALITY_CONTROL = "quality_control"
    UNKNOWN = "unknown"


class WarningLevel(_StringEnum):
    BLOCKER = "blocker"
    ADVISORY = "advisory"
    INFORMATION = "information"


class ClaimLevel(_StringEnum):
    DESCRIPTIVE = "descriptive"
    EXPLORATORY = "exploratory"
    SCREENING = "screening"
    QUANTITATIVE_METHOD_CANDIDATE = "quantitative_method_candidate"
    VALIDATED_METHOD = "validated_method"


class TransformScope(_StringEnum):
    GLOBAL = "global"
    SAMPLE_LOCAL = "sample_local"
    TRAINING_FOLD = "training_fold"
    VALIDATION_FOLD = "validation_fold"
    TEST_SET = "test_set"
    INFERENCE = "inference"
    DISPLAY_ONLY = "display_only"


class RunStatus(_StringEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class ContractModel(BaseModel):
    """Strict immutable base with deterministic serialization helpers."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    _canonical_exclude: ClassVar[set[str]] = set()

    @model_validator(mode="after")
    def deeply_immutable(self) -> "ContractModel":
        # ``frozen=True`` is shallow; freeze arbitrary metadata too so a
        # retained dict/list reference cannot invalidate an approved hash.
        for name in type(self).model_fields:
            value = getattr(self, name)
            frozen = _deep_freeze(value)
            if frozen is not value:
                object.__setattr__(self, name, frozen)
        return self

    def canonical_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude=self._canonical_exclude)

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> "ContractModel":
        """Return a fully revalidated copy, preserving deep immutability."""
        del deep
        payload = self.model_dump(mode="python")
        payload.update(dict(update or {}))
        return type(self).model_validate(payload)

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class ValidationIssue(ContractModel):
    code: str
    message: str
    level: WarningLevel = WarningLevel.ADVISORY
    stage: str = "validation"
    details: Mapping[str, Any] = Field(default_factory=dict)


class SourceAsset(ContractModel):
    asset_id: str
    relative_path: str
    sha256: str
    size_bytes: int = Field(ge=0)
    format: str
    parser_id: str
    media_type: str | None = None

    @field_validator("relative_path")
    @classmethod
    def relative_non_traversing_path(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        if not normalized or normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized):
            raise ValueError("relative_path must be a non-empty relative path")
        if any(part in {"", ".", ".."} for part in normalized.split("/")):
            raise ValueError("relative_path must not contain '.', '..', or empty components")
        return normalized

    @field_validator("sha256")
    @classmethod
    def lowercase_sha256(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        return value


class SampleRecord(ContractModel):
    sample_id: str
    preparation_id: str | None = None
    technical_replicate_id: str | None = None
    batch_id: str | None = None
    instrument_id: str | None = None
    operator_id: str | None = None
    role: MeasurementRole = MeasurementRole.SAMPLE
    physical_state: str | None = None
    composition: Mapping[str, Any] = Field(default_factory=dict)
    metadata: Mapping[str, Any] = Field(default_factory=dict)


class MeasurementRecord(ContractModel):
    measurement_id: str
    asset_id: str
    sample_id: str
    modality: Modality
    representation: Representation = Representation.SPECTRUM
    axis_kind: AxisKind = AxisKind.INDEX
    axis_unit: str | None = None
    signal_kind: SignalKind = SignalKind.INTENSITY
    signal_unit: str | None = None
    role: MeasurementRole = MeasurementRole.UNKNOWN
    metadata: Mapping[str, Any] = Field(default_factory=dict)


class TransformationRecord(ContractModel):
    transformation_id: str
    name: str
    scope: TransformScope = TransformScope.GLOBAL
    parameters: Mapping[str, Any] = Field(default_factory=dict)
    input_representation: Representation | None = None
    output_representation: Representation | None = None
    input_ids: tuple[str, ...] = ()
    input_hashes: tuple[str, ...] = ()
    output_hash: str | None = None
    implementation_id: str | None = None
    applied_at: str | None = None


class ProjectManifest(ContractModel):
    project_id: str
    version: str = "2"
    source_root: str = "."
    assets: tuple[SourceAsset, ...] = ()
    samples: tuple[SampleRecord, ...] = ()
    measurements: tuple[MeasurementRecord, ...] = ()
    unresolved_issues: tuple[ValidationIssue, ...] = ()
    created_at: str | None = None
    manifest_hash: str | None = None
    _canonical_exclude: ClassVar[set[str]] = {"manifest_hash"}


class ScientificIntent(ContractModel):
    objective: str = ""
    task_kind: str | None = None
    target: str | None = None
    claim_level: ClaimLevel = ClaimLevel.EXPLORATORY
    intended_use: str | None = None
    constraints: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = Field(default_factory=dict)


class PipelineSpec(ContractModel):
    pipeline_id: str
    transformations: tuple[TransformationRecord, ...] = ()
    model_family: str | None = None
    hyperparameters: Mapping[str, Any] = Field(default_factory=dict)
    description: str | None = None


class SplitManifest(ContractModel):
    split_id: str
    strategy: str = "unspecified"
    seed: int | None = None
    train_ids: tuple[str, ...] = ()
    validation_ids: tuple[str, ...] = ()
    test_ids: tuple[str, ...] = ()
    group_key: str | None = None
    metadata: Mapping[str, Any] = Field(default_factory=dict)


class AnalysisTaskSpec(ContractModel):
    task_id: str
    task_type: str = "exploratory"
    target: str | None = None
    metric_names: tuple[str, ...] = ()
    scientific_intent: ScientificIntent | None = None
    metadata: Mapping[str, Any] = Field(default_factory=dict)


class ProjectAnalysisPlan(ContractModel):
    plan_id: str
    project_id: str | None = None
    manifest_hash: str | None = None
    task: AnalysisTaskSpec | None = None
    tasks: tuple[AnalysisTaskSpec, ...] = ()
    pipelines: tuple[PipelineSpec, ...] = ()
    split_manifest: SplitManifest | None = None
    issues: tuple[ValidationIssue, ...] = ()
    approval_required: bool = True
    compute_budget: Mapping[str, Any] = Field(default_factory=dict)
    output_formats: tuple[str, ...] = ("json", "markdown")
    created_at: str | None = None
    plan_hash: str | None = None
    _canonical_exclude: ClassVar[set[str]] = {"plan_hash"}


class PlanApproval(ContractModel):
    approval_id: str | None = None
    project_id: str | None = None
    plan_id: str
    plan_hash: str | None = None
    approved: bool = False
    approver_id: str | None = None
    approved_at: str | None = None
    notes: str | None = None


class ClaimEligibility(ContractModel):
    claim_level: ClaimLevel
    eligible: bool = False
    reasons: tuple[str, ...] = ()
    blocking_issues: tuple[ValidationIssue, ...] = ()


class EvidenceReference(ContractModel):
    evidence_id: str
    kind: str
    uri: str
    sha256: str | None = None
    description: str | None = None
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("sha256")
    @classmethod
    def optional_lowercase_sha256(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        return value


class EvidenceBundle(ContractModel):
    bundle_id: str
    project_id: str | None = None
    run_id: str | None = None
    manifest_hash: str | None = None
    plan_hash: str | None = None
    pipeline_ids: tuple[str, ...] = ()
    code_version: str | None = None
    environment_hash: str | None = None
    references: tuple[EvidenceReference, ...] = ()
    issues: tuple[ValidationIssue, ...] = ()
    claim_eligibility: ClaimEligibility | None = None
    created_at: str | None = None


class ProjectSummary(ContractModel):
    project_id: str
    manifest_hash: str | None = None
    asset_count: int = Field(default=0, ge=0)
    sample_count: int = Field(default=0, ge=0)
    measurement_count: int = Field(default=0, ge=0)
    unresolved_issue_count: int = Field(default=0, ge=0)
    status: RunStatus = RunStatus.PENDING


class ProjectRunSummary(ContractModel):
    run_id: str
    status: RunStatus = RunStatus.PENDING
    project_id: str | None = None
    plan_id: str | None = None
    pipeline_id: str | None = None
    evidence_bundle_id: str | None = None
    claim_eligibility: ClaimEligibility | None = None
    issues: tuple[ValidationIssue, ...] = ()
    started_at: str | None = None
    completed_at: str | None = None
    metadata: Mapping[str, Any] = Field(default_factory=dict)
