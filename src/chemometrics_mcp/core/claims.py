"""Small, dependency-free gates for scientific claims.

These gates deliberately assess evidence and study design; they do not derive
analytical figures of merit from model-fit metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence


CLAIM_LEVELS = (
    "descriptive",
    "exploratory",
    "screening",
    "quantitative_method_candidate",
    "validated_method",
)
_LEVEL_INDEX = {level: index for index, level in enumerate(CLAIM_LEVELS)}
_ISSUE_LEVELS = frozenset({"blocker", "advisory", "information"})


@dataclass(frozen=True)
class GateIssue:
    """An immutable issue which can limit the scientific claim being made."""

    code: str
    message: str
    level: str
    stage: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.level not in _ISSUE_LEVELS:
            raise ValueError(f"Unknown gate issue level: {self.level!r}")
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


@dataclass(frozen=True)
class GateSummary:
    """The allowed claim level and the evidence issues used to determine it."""

    can_execute: bool
    can_select: bool
    claim_level: str
    issues: tuple[GateIssue, ...]
    blockers: tuple[GateIssue, ...]
    advisories: tuple[GateIssue, ...]


@dataclass(frozen=True)
class DetectionLimitEligibility:
    """Whether the supplied design supports estimating a detection limit."""

    estimable: bool
    status: str
    missing_requirements: tuple[str, ...] = ()

    def __getitem__(self, key: str) -> Any:
        """Permit light dict-style consumption by tool layers."""
        return getattr(self, key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "estimable": self.estimable,
            "status": self.status,
            "missing_requirements": list(self.missing_requirements),
        }


def _value(source: Mapping[str, Any] | object | None, name: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _scope(issue: GateIssue) -> Any:
    """Return the optional affected scope used to keep distinct issues distinct."""
    for key in ("affected_scope", "scope", "affected"):
        value = issue.details.get(key)
        if value is not None:
            if isinstance(value, (list, tuple, set, frozenset)):
                return tuple(value)
            if isinstance(value, Mapping):
                return tuple(sorted(value.items()))
            return value
    return None


def deduplicate_issues(issues: Sequence[GateIssue]) -> tuple[GateIssue, ...]:
    """Merge duplicate code/stage/scope issues while preserving first order.

    Details from later instances supplement rather than overwrite the first
    useful value.  This makes independently emitted diagnostics composable.
    """
    merged: dict[tuple[Any, ...], GateIssue] = {}
    for issue in issues:
        key = (issue.code, issue.stage, _scope(issue))
        previous = merged.get(key)
        if previous is None:
            merged[key] = issue
            continue
        details = dict(previous.details)
        for detail_key, detail_value in issue.details.items():
            if detail_key not in details or details[detail_key] in (None, "", (), [], {}):
                details[detail_key] = detail_value
        # A duplicate should never silently reduce severity.
        level = min((previous.level, issue.level), key=lambda item: ("blocker", "advisory", "information").index(item))
        merged[key] = GateIssue(previous.code, previous.message, level, previous.stage, details)
    return tuple(merged.values())


def _requirement_issue(level: str, missing: list[str]) -> GateIssue:
    return GateIssue(
        code="claim_evidence_incomplete",
        message="Claim level requires additional independent validation evidence.",
        level="advisory",
        stage="claim_eligibility",
        details={"requested_level": level, "missing_requirements": tuple(missing)},
    )


def evaluate_claim_eligibility(
    requested_level: str,
    issues: Sequence[GateIssue] = (),
    design_metadata: Mapping[str, Any] | object | None = None,
) -> GateSummary:
    """Return the highest defensible level for a requested scientific claim."""
    if requested_level not in _LEVEL_INDEX:
        raise ValueError(f"Unknown claim level: {requested_level!r}")

    all_issues = list(deduplicate_issues(issues))
    blockers = [issue for issue in all_issues if issue.level == "blocker"]
    advisories = [issue for issue in all_issues if issue.level == "advisory"]
    allowed_index = _LEVEL_INDEX[requested_level]
    if blockers:
        allowed_index = min(allowed_index, _LEVEL_INDEX["descriptive"])
    elif advisories:
        allowed_index = min(allowed_index, _LEVEL_INDEX["screening"])

    if allowed_index >= _LEVEL_INDEX["quantitative_method_candidate"]:
        missing = []
        if (_value(design_metadata, "independent_preparations", 0) or 0) < 5:
            missing.append("independent_preparations>=5")
        for requirement in (
            "group_safe_validation",
            "fold_safe_pipeline",
            "calibration_coverage",
            "independent_levels",
            "endpoint_coverage",
            "batch_representation",
            "bias_acceptable",
            "applicability_domain",
        ):
            if not _value(design_metadata, requirement, False):
                missing.append(requirement)
        if missing:
            all_issues.append(_requirement_issue(requested_level, missing))
            allowed_index = _LEVEL_INDEX["screening"]

    if allowed_index >= _LEVEL_INDEX["validated_method"]:
        missing = [
            requirement for requirement in ("external_test", "user_validation_criteria")
            if not _value(design_metadata, requirement, False)
        ]
        if missing:
            all_issues.append(_requirement_issue(requested_level, missing))
            allowed_index = _LEVEL_INDEX["quantitative_method_candidate"]

    all_issues = list(deduplicate_issues(all_issues))
    blockers = tuple(issue for issue in all_issues if issue.level == "blocker")
    advisories = tuple(issue for issue in all_issues if issue.level == "advisory")
    return GateSummary(
        can_execute=not blockers,
        can_select=not blockers,
        claim_level=CLAIM_LEVELS[allowed_index],
        issues=tuple(all_issues),
        blockers=blockers,
        advisories=advisories,
    )


def evaluate_detection_limit_eligibility(
    design: Mapping[str, Any] | object | None,
) -> DetectionLimitEligibility:
    """Gate LOD estimation on experimental evidence, never prediction RMSE."""
    requirements = ("blanks", "low_level_standards", "independent_replicates", "declared_method")
    missing = tuple(name for name in requirements if not _value(design, name, False))
    if missing:
        return DetectionLimitEligibility(False, "not_estimable", missing)
    return DetectionLimitEligibility(True, "estimable")


def is_classification_task(task_name: str | None) -> bool:
    """Return whether the task explicitly represents classification."""
    return "classification" in (task_name or "").lower()


def is_regression_task(task_name: str | None) -> bool:
    """Return whether the task explicitly represents regression."""
    return "regression" in (task_name or "").lower()


def targets_are_classes(task_name: str | None) -> bool:
    """Compatibility spelling for older callers."""
    return is_classification_task(task_name)


def regression_targets_are_not_classes(task_name: str | None) -> bool:
    """Compatibility assertion for callers handling regression outputs."""
    return is_regression_task(task_name)
