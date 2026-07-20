"""Layer 1 evaluation harness — deterministic, seeded, no LLM.

All functions are reproducible: given the same seed, inputs, and fixture files,
they return byte-identical outputs.

Public API
----------
- ``ScenarioResult`` — result dataclass for a single scenario run
- ``run_scenario_full(scenario_id, seed)`` — orchestrate inspect→…→report
- ``compute_leakage_detection(config_name, fixture_dir)`` — (tpr, fpr, precision)
- ``compute_fallback_correctness(fixture_path)`` — correctness_rate
- ``compute_plan_quality(plan_obj)`` — (auto_score, auto_pass)
- ``compute_effort_metrics(tool_runs, wall_clock_s)`` — effort dict
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from chemometrics_contracts import (
    AnalysisPlan,
    AnalysisResult,
    AnalysisRun,
    DatasetInspection,
    InterpretResultsRequest,
    ProposeAnalysisPlanRequest,
    RecommendNextModelRequest,
    RunAnalysisRequest,
    SelectBestModelRequest,
    SpectralDataset,
    ValidateResultsRequest,
    ValidationSummary,
)

# ---------------------------------------------------------------------------
# Project-root detection
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
# src/chemometrics_mcp/core/evaluation.py  →  parents[3] = project root
# parents[0]=core, [1]=chemometrics_mcp, [2]=src, [3]=main (project root)
_PROJECT_ROOT = _THIS_FILE.parents[3]


def _default_workbook_path() -> Path:
    return _PROJECT_ROOT / "2026-05-21_22-59_Method Dev_wavelength_range_1450-2450.xlsx"


def _default_ftir_dir() -> Path:
    return _PROJECT_ROOT / "ftir-purity-dataset" / "fwdftirjune262026"


# ---------------------------------------------------------------------------
# ScenarioResult
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    """Collected outputs from a single Layer-1 scenario run."""

    scenario_id: str
    seed: int
    wall_clock_s: float = 0.0

    # Tool responses (serialised as dicts for JSON-serializability)
    inspection: dict[str, Any] = field(default_factory=dict)
    plan: dict[str, Any] = field(default_factory=dict)
    analysis_run: dict[str, Any] = field(default_factory=dict)
    validation_summary: dict[str, Any] = field(default_factory=dict)
    selection: dict[str, Any] = field(default_factory=dict)
    interpretation: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)

    # Aggregated metrics
    best_model: str | None = None
    best_metrics: dict[str, float] = field(default_factory=dict)

    # Plan quality
    plan_auto_score: float = 0.0
    plan_auto_pass: bool = False

    # Effort
    tool_calls: int = 0
    decision_points: int = 0

    # Status
    ok: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Scenario-specific dataset loaders
# ---------------------------------------------------------------------------

def _load_dataset_for_scenario(
    scenario_id: str,
    workbook_path: Path | None = None,
    ftir_dir: Path | None = None,
) -> tuple[SpectralDataset, DatasetInspection]:
    """Load dataset + inspection for the given scenario."""
    from chemometrics_mcp.core.datasets import (
        load_flooring_nir,
        load_ftir_real,
    )
    from chemometrics_mcp.tools import inspect_dataset as _inspect_tool
    from chemometrics_contracts import InspectDatasetRequest

    sid = scenario_id.lower()

    if sid == "s1":
        wb = workbook_path or _default_workbook_path()
        dataset, inspection = load_flooring_nir(wb, task="wear_layer")
    elif sid == "s2":
        wb = workbook_path or _default_workbook_path()
        dataset, inspection = load_flooring_nir(wb, task="species")
    elif sid == "s3":
        wb = workbook_path or _default_workbook_path()
        dataset, inspection = load_flooring_nir(wb, task="material_type")
    elif sid == "s4":
        fdir = ftir_dir or _default_ftir_dir()
        dataset, inspection = load_ftir_real(fdir)
    else:
        raise ValueError(f"Unknown scenario_id: {scenario_id!r}. Expected s1–s4.")

    return dataset, inspection


# ---------------------------------------------------------------------------
# Scenario-specific plan overrides (ensure reproducible plans)
# ---------------------------------------------------------------------------

_SCENARIO_TASK_HINTS: dict[str, str] = {
    "s1": "regression",
    "s2": "multi_class_classification",
    "s3": "binary_classification",
    "s4": "multi_class_classification",
}

_SCENARIO_USER_INTENTS: dict[str, str] = {
    "s1": "Predict vinyl flooring wear-layer thickness (mil) from NIR spectra.",
    "s2": "Classify lumber species from NIR spectra.",
    "s3": "Distinguish vinyl from wood/lumber using NIR spectra (binary).",
    "s4": "Classify compound purity groups from FTIR transmittance spectra.",
}


# ---------------------------------------------------------------------------
# run_scenario_full
# ---------------------------------------------------------------------------

def run_scenario_full(
    scenario_id: str,
    seed: int = 42,
    *,
    runs_root: str | Path | None = None,
    workbook_path: Path | None = None,
    ftir_dir: Path | None = None,
) -> ScenarioResult:
    """Orchestrate inspect→plan→run→validate→select→interpret→report.

    Parameters
    ----------
    scenario_id:
        One of ``"s1"``, ``"s2"``, ``"s3"``, ``"s4"``.
    seed:
        Random seed for all stochastic operations. Default 42.
    runs_root:
        Root directory for tool artifacts. Defaults to ``runs/eval/<timestamp>/``
        under the project root.
    workbook_path:
        Override path for the NIR Excel workbook (S1/S2/S3).
    ftir_dir:
        Override path for the FTIR .txt directory (S4).

    Returns
    -------
    ScenarioResult with all tool outputs and aggregated metrics.
    """
    import datetime

    t0 = time.perf_counter()

    if runs_root is None:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        runs_root = _PROJECT_ROOT / "runs" / "eval" / ts

    runs_root = Path(runs_root)
    sid = scenario_id.lower()

    result = ScenarioResult(scenario_id=sid, seed=seed)
    tool_calls = 0

    try:
        # ------------------------------------------------------------------
        # 1. Load dataset
        # ------------------------------------------------------------------
        dataset, inspection = _load_dataset_for_scenario(
            sid,
            workbook_path=workbook_path,
            ftir_dir=ftir_dir,
        )
        result.inspection = inspection.to_dict()
        tool_calls += 1

        # ------------------------------------------------------------------
        # 2. Propose analysis plan
        # ------------------------------------------------------------------
        from chemometrics_mcp.tools import propose_analysis_plan as _plan_tool

        plan_request = ProposeAnalysisPlanRequest(
            dataset_inspection=inspection,
            user_intent=_SCENARIO_USER_INTENTS.get(sid),
            task_hint=_SCENARIO_TASK_HINTS.get(sid),
            allow_supervised_planning=True,
        )
        plan_response = _plan_tool.run(plan_request, runs_root=runs_root)
        tool_calls += 1

        if not plan_response.ok or plan_response.payload is None:
            result.error = f"propose_analysis_plan failed: {plan_response.error}"
            result.tool_calls = tool_calls
            result.wall_clock_s = time.perf_counter() - t0
            return result

        plan = plan_response.payload
        result.plan = plan.to_dict()

        # Plan quality
        auto_score, auto_pass = compute_plan_quality(plan)
        result.plan_auto_score = auto_score
        result.plan_auto_pass = auto_pass

        # ------------------------------------------------------------------
        # 3. Run analysis
        # ------------------------------------------------------------------
        from chemometrics_mcp.tools import run_analysis as _run_tool

        run_request = RunAnalysisRequest(dataset=dataset, approved_plan=plan)
        run_response = _run_tool.run(run_request, runs_root=runs_root)
        tool_calls += 1

        if not run_response.ok or run_response.payload is None:
            result.error = f"run_analysis failed: {run_response.error}"
            result.tool_calls = tool_calls
            result.wall_clock_s = time.perf_counter() - t0
            return result

        analysis_run = run_response.payload
        result.analysis_run = analysis_run.to_dict()

        # ------------------------------------------------------------------
        # 4. Validate results
        # ------------------------------------------------------------------
        from chemometrics_mcp.tools import validate_results as _validate_tool

        validate_request = ValidateResultsRequest(
            analysis_run=analysis_run,
            dataset=dataset,
            dataset_inspection=inspection,
        )
        validate_response = _validate_tool.run(validate_request, runs_root=runs_root)
        tool_calls += 1

        if validate_response.ok and validate_response.payload is not None:
            result.validation_summary = validate_response.payload.to_dict()

        # ------------------------------------------------------------------
        # 5. Select best model
        # ------------------------------------------------------------------
        from chemometrics_mcp.tools import select_best_model as _select_tool

        select_request = SelectBestModelRequest(
            results=analysis_run.results,
            task_name=plan.task_name,
        )
        select_response = _select_tool.run(select_request, runs_root=runs_root)
        tool_calls += 1

        if select_response.ok and select_response.payload is not None:
            sel = select_response.payload
            result.selection = sel.to_dict()
            result.best_model = sel.selected_model

        # ------------------------------------------------------------------
        # 6. Interpret results
        # ------------------------------------------------------------------
        from chemometrics_mcp.tools import interpret_results as _interp_tool

        interp_request = InterpretResultsRequest(
            results=analysis_run.results,
            dataset=dataset,
        )
        interp_response = _interp_tool.run(interp_request, runs_root=runs_root)
        tool_calls += 1

        if interp_response.ok and interp_response.payload is not None:
            result.interpretation = interp_response.payload.to_dict()

        # ------------------------------------------------------------------
        # 7. Generate report
        # ------------------------------------------------------------------
        from chemometrics_mcp.tools import generate_report as _report_tool
        from chemometrics_contracts import GenerateReportRequest

        report_request = GenerateReportRequest(analysis_run=analysis_run)
        report_response = _report_tool.run(report_request, runs_root=runs_root)
        tool_calls += 1

        if report_response.ok and report_response.payload is not None:
            result.report = report_response.payload.to_dict()

        # ------------------------------------------------------------------
        # Aggregate best metrics
        # ------------------------------------------------------------------
        if analysis_run.results:
            # Find result for best_model or just use first result
            best_result: AnalysisResult | None = None
            if result.best_model:
                best_result = next(
                    (r for r in analysis_run.results if r.model_name == result.best_model),
                    None,
                )
            if best_result is None and analysis_run.results:
                best_result = analysis_run.results[0]
            if best_result is not None:
                result.best_metrics = dict(best_result.metrics)

        result.ok = True
        result.tool_calls = tool_calls
        result.decision_points = _count_decision_points(result)

    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
        result.tool_calls = tool_calls
        result.ok = False

    result.wall_clock_s = time.perf_counter() - t0
    return result


def _count_decision_points(result: ScenarioResult) -> int:
    """Count HITL decision points from validation summary and model selection."""
    n = 0
    # model selection always requires human approval
    if result.selection.get("requires_human_approval"):
        n += 1
    # any validation warnings requiring attention
    v_warnings = result.validation_summary.get("warnings", [])
    error_warnings = [w for w in v_warnings if isinstance(w, dict) and w.get("severity") in ("error", "warning")]
    if error_warnings:
        n += 1
    return n


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _load_pickle_fixture(path: Path) -> SpectralDataset:
    """Load a SpectralDataset from a pickle fixture file."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _make_toy_spectral_dataset(
    n_samples: int = 30,
    n_features: int = 50,
    n_classes: int = 2,
    seed: int = 42,
    group_col: bool = False,
    target_in_features: bool = False,
) -> SpectralDataset:
    """Build a small synthetic SpectralDataset for unit tests and fixtures."""
    rng = np.random.default_rng(seed)
    axis = np.linspace(1000.0, 2000.0, n_features).tolist()
    labels_arr = np.tile(np.arange(n_classes), n_samples // n_classes + 1)[:n_samples]
    rng.shuffle(labels_arr)
    labels = [str(int(lbl)) for lbl in labels_arr]

    X = rng.standard_normal((n_samples, n_features)).astype(float)
    # Add class signal
    for i, lbl in enumerate(labels_arr):
        X[i] += float(lbl) * 0.5

    if target_in_features:
        # Inject target as a perfect predictor column (leakage)
        target_col = labels_arr.astype(float).reshape(-1, 1)
        X = np.hstack([X, target_col])
        axis = axis + [9999.0]  # sentinel wavelength

    metadata = []
    for i, lbl in enumerate(labels_arr):
        row: dict[str, Any] = {"sample_idx": str(i), "label": str(int(lbl))}
        if group_col:
            row["sample_group"] = f"group_{i % 5}"
        metadata.append(row)

    return SpectralDataset(
        x=tuple(tuple(float(v) for v in row) for row in X.tolist()),
        axis=tuple(float(v) for v in axis),
        metadata=tuple(metadata),
        labels=tuple(labels),
        modality="NIR",
        sample_ids=tuple(str(i) for i in range(n_samples)),
    )


def _make_near_duplicate_dataset(
    n_samples: int = 30,
    n_features: int = 50,
    seed: int = 42,
) -> SpectralDataset:
    """Dataset with near-duplicate rows across classes (inter-fold leakage risk)."""
    rng = np.random.default_rng(seed)
    axis = np.linspace(1000.0, 2000.0, n_features).tolist()
    base_row = rng.standard_normal(n_features)
    # Half samples are near-duplicates of base_row (tiny noise)
    X = np.zeros((n_samples, n_features))
    labels = []
    for i in range(n_samples):
        if i < n_samples // 2:
            X[i] = base_row + rng.normal(0, 0.001, n_features)
            labels.append("0")
        else:
            X[i] = rng.standard_normal(n_features)
            labels.append("1")

    metadata = [{"sample_idx": str(i), "label": lbl} for i, lbl in enumerate(labels)]
    return SpectralDataset(
        x=tuple(tuple(float(v) for v in row) for row in X.tolist()),
        axis=tuple(float(v) for v in axis),
        metadata=tuple(metadata),
        labels=tuple(labels),
        modality="NIR",
        sample_ids=tuple(str(i) for i in range(n_samples)),
    )


# ---------------------------------------------------------------------------
# Fixture generation (called once to create .pkl files)
# ---------------------------------------------------------------------------

def generate_leakage_fixtures(fixture_dir: Path) -> None:
    """Generate leakage_positive and leakage_clean fixture .pkl files."""
    pos_dir = fixture_dir / "leakage_positive"
    clean_dir = fixture_dir / "leakage_clean"
    pos_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    # --- Positive fixtures (leakage present) ---

    # (a) target-in-features
    ds_a_pos = _make_toy_spectral_dataset(
        n_samples=40, n_features=50, n_classes=2, seed=42, target_in_features=True
    )
    with open(pos_dir / "leakage_a_target_in_features.pkl", "wb") as f:
        pickle.dump(ds_a_pos, f)

    # (b) grouped data + non-grouped CV config metadata
    # We embed the cv_config mismatch as a metadata flag
    ds_b_pos = _make_toy_spectral_dataset(
        n_samples=40, n_features=50, n_classes=2, seed=42, group_col=True
    )
    # Store as dict with cv_config to signal intended mismatch
    with open(pos_dir / "leakage_b_group_no_cv.pkl", "wb") as f:
        pickle.dump({"dataset": ds_b_pos, "cv_config": "stratified_kfold_5",
                     "leakage_type": "group_leakage"}, f)

    # (c) near-duplicate rows across splits
    ds_c_pos = _make_near_duplicate_dataset(n_samples=40, n_features=50, seed=42)
    with open(pos_dir / "leakage_c_near_duplicates.pkl", "wb") as f:
        pickle.dump(ds_c_pos, f)

    # --- Clean fixtures (no leakage) ---

    # (a) clean: no target column in features
    ds_a_clean = _make_toy_spectral_dataset(
        n_samples=40, n_features=50, n_classes=2, seed=42, target_in_features=False
    )
    with open(clean_dir / "clean_a_normal_features.pkl", "wb") as f:
        pickle.dump(ds_a_clean, f)

    # (b) clean: grouped data with group-aware CV config
    ds_b_clean = _make_toy_spectral_dataset(
        n_samples=40, n_features=50, n_classes=2, seed=42, group_col=True
    )
    with open(clean_dir / "clean_b_group_with_cv.pkl", "wb") as f:
        pickle.dump({"dataset": ds_b_clean, "cv_config": "grouped_kfold_5",
                     "leakage_type": "none"}, f)

    # (c) clean: truly distinct rows
    rng = np.random.default_rng(99)
    axis = np.linspace(1000.0, 2000.0, 50).tolist()
    X_clean = rng.standard_normal((40, 50))
    labels_clean = [str(int(i % 2)) for i in range(40)]
    ds_c_clean = SpectralDataset(
        x=tuple(tuple(float(v) for v in row) for row in X_clean.tolist()),
        axis=tuple(float(v) for v in axis),
        metadata=tuple({"sample_idx": str(i)} for i in range(40)),
        labels=tuple(labels_clean),
        modality="NIR",
        sample_ids=tuple(str(i) for i in range(40)),
    )
    with open(clean_dir / "clean_c_distinct_rows.pkl", "wb") as f:
        pickle.dump(ds_c_clean, f)


# ---------------------------------------------------------------------------
# compute_leakage_detection
# ---------------------------------------------------------------------------

def compute_leakage_detection(
    config_name: str,
    fixture_dir: Path | str,
) -> tuple[float, float, float]:
    """Compute leakage detection (TPR, FPR, precision) on fixture data.

    Runs validation checks on leakage_positive + leakage_clean fixture sets.
    Returns (tpr, fpr, precision).

    Parameters
    ----------
    config_name:
        Evaluation configuration label (informational only in Layer 1).
    fixture_dir:
        Path to tests/fixtures/eval/ directory.

    Returns
    -------
    (tpr, fpr, precision)
        tpr = TP / (TP + FN), fpr = FP / (FP + TN), precision = TP / (TP + FP)
    """
    from chemometrics_mcp.core import validation as _val
    from chemometrics_contracts import AnalysisResult

    fixture_dir = Path(fixture_dir)
    pos_dir = fixture_dir / "leakage_positive"
    clean_dir = fixture_dir / "leakage_clean"

    _LEAKAGE_WARNING_CODES = {
        "suspicious_high_metric",
        "suspicious_regression_r2",
        "replicate_leakage",
        "group_leakage_risk",
        "easy_task_detected",
    }

    def _run_check_on_fixture(pkl_path: Path) -> bool:
        """Return True if a leakage warning fired for this fixture."""
        with open(pkl_path, "rb") as fh:
            obj = pickle.load(fh)

        if isinstance(obj, dict):
            ds = obj["dataset"]
            cv_config = obj.get("cv_config", "stratified_kfold_5")
        else:
            ds = obj
            cv_config = "stratified_kfold_5"

        X = np.array(ds.x, dtype=float)
        y_raw = list(ds.labels) if ds.labels else ["0"] * X.shape[0]
        y = np.array(y_raw)

        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        clf = LogisticRegression(max_iter=500, random_state=42, solver="lbfgs")
        cv = StratifiedKFold(n_splits=min(3, len(set(y_enc))), shuffle=True, random_state=42)
        try:
            y_pred_enc = cross_val_predict(clf, X, y_enc, cv=cv)
            y_pred = le.inverse_transform(y_pred_enc)
        except Exception:
            return False

        from sklearn.metrics import accuracy_score as _acc
        acc = float(_acc(y, y_pred))

        # Build a minimal AnalysisResult
        result = AnalysisResult(
            task_name="binary_classification",
            model_name="logistic_regression_eval",
            metrics={"accuracy": acc, "balanced_accuracy": acc},
            predictions=tuple(str(p) for p in y_pred),
            validation_strategy=cv_config,
        )

        # Build inspection with group columns if present
        from chemometrics_contracts import DatasetInspection
        group_cols: list[str] = []
        if ds.metadata:
            for row in ds.metadata:
                group_cols = [k for k in row.keys() if "group" in k.lower()]
                break

        inspection = DatasetInspection(
            sample_count=X.shape[0],
            feature_count=X.shape[1],
            modality="NIR",
            candidate_label_columns=("label",),
            candidate_group_columns=tuple(group_cols),
        )

        summary = _val.run_all_checks([result], ds, inspection)
        fired_codes = {w.code for w in summary.warnings}
        return bool(fired_codes & _LEAKAGE_WARNING_CODES)

    # Positive fixtures
    pos_files = list(pos_dir.glob("*.pkl"))
    clean_files = list(clean_dir.glob("*.pkl"))

    tp = sum(1 for p in pos_files if _run_check_on_fixture(p))
    fn = len(pos_files) - tp
    fp = sum(1 for p in clean_files if _run_check_on_fixture(p))
    tn = len(clean_files) - fp

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return tpr, fpr, precision


# ---------------------------------------------------------------------------
# compute_fallback_correctness
# ---------------------------------------------------------------------------

def compute_fallback_correctness(fixture_path: Path | str) -> float:
    """For each fallback case, run recommend_next_model, check result in acceptable set.

    Parameters
    ----------
    fixture_path:
        Path to fallback_cases.json fixture file.

    Returns
    -------
    correctness_rate : float
        Fraction of cases where the recommended fallback is in acceptable_fallback.
    """
    from chemometrics_mcp.tools import recommend_next_model as _rnm_tool
    from chemometrics_contracts import RecommendNextModelRequest

    fixture_path = Path(fixture_path)
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))

    if not cases:
        return 0.0

    n_correct = 0
    for case in cases:
        request = RecommendNextModelRequest(
            failed_model=case["failed_model"],
            failure_reason=case["failure_reason"],
        )
        # Use a temp runs_root in scratchpad to avoid cluttering project
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            response = _rnm_tool.run(request, runs_root=tmp)

        if response.ok and response.payload is not None:
            recommended = response.payload.fallback_model
            acceptable = set(case.get("acceptable_fallback", []))
            if recommended is not None and recommended in acceptable:
                n_correct += 1

    return n_correct / len(cases)


# ---------------------------------------------------------------------------
# compute_plan_quality
# ---------------------------------------------------------------------------

def compute_plan_quality(plan_obj: AnalysisPlan | dict) -> tuple[float, bool]:
    """Score a plan with 4 boolean checks → auto_score in [0,1].

    Checks:
    1. task_name is present and non-empty
    2. preprocessing candidates include a modality-relevant method (snv, msc, sg, baseline)
    3. validation_strategy is present and non-empty
    4. model_families is non-empty

    Parameters
    ----------
    plan_obj:
        An AnalysisPlan dataclass or equivalent dict.

    Returns
    -------
    (auto_score, auto_pass)
        auto_score = mean of 4 bool checks (0.0–1.0).
        auto_pass = auto_score >= 0.75 (i.e., 3/4 checks pass).
    """
    if isinstance(plan_obj, dict):
        task_name = plan_obj.get("task_name") or ""
        preprocessing = plan_obj.get("preprocessing_candidates") or []
        validation_strategy = plan_obj.get("validation_strategy") or ""
        model_families = plan_obj.get("model_families") or []
    else:
        task_name = plan_obj.task_name or ""
        preprocessing = list(plan_obj.preprocessing_candidates)
        validation_strategy = plan_obj.validation_strategy or ""
        model_families = list(plan_obj.model_families)

    _MODALITY_PREPROCESSING = {"snv", "msc", "sg_1st_deriv", "sg_2nd_deriv", "baseline_correction", "area_normalization"}

    check1 = bool(task_name.strip())
    check2 = bool(set(preprocessing) & _MODALITY_PREPROCESSING)
    check3 = bool(validation_strategy.strip())
    check4 = bool(model_families)

    auto_score = float(sum([check1, check2, check3, check4])) / 4.0
    auto_pass = auto_score >= 0.75

    return auto_score, auto_pass


# ---------------------------------------------------------------------------
# compute_effort_metrics
# ---------------------------------------------------------------------------

def compute_effort_metrics(
    tool_runs: Sequence[str],
    wall_clock_s: float,
) -> dict[str, Any]:
    """Summarize effort from a list of tool call names and elapsed time.

    Parameters
    ----------
    tool_runs:
        Ordered list of tool names called during the run.
    wall_clock_s:
        Total elapsed wall-clock time in seconds.

    Returns
    -------
    dict with keys: tool_calls, wall_clock_s, decision_points, unique_tools.
    """
    _DECISION_TOOLS = {
        "select_best_model",
        "recommend_next_model",
        "validate_results",
    }
    decision_points = sum(1 for t in tool_runs if t in _DECISION_TOOLS)
    return {
        "tool_calls": len(tool_runs),
        "wall_clock_s": round(wall_clock_s, 3),
        "decision_points": decision_points,
        "unique_tools": sorted(set(tool_runs)),
    }
