import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from xml.etree import ElementTree

import pytest

from chemometrics_mcp.core.dashboard import _task_tables_and_figures
from chemometrics_mcp.core.project_service import ProjectService
from chemometrics_mcp.core.project_store import ProjectStore
from chemometrics_mcp.core.project_reporting import generate_report_for_run
from chemometrics_mcp.core.run_service import run_project_analysis
from chemometrics_mcp.tools import project_workflow


def _ready_ftir_project(tmp_path: Path) -> tuple[str, dict, dict]:
    source = tmp_path / "spectra"
    source.mkdir()
    (source / "product.csv").write_text(
        "1000,0.10\n1001,0.30\n1002,0.20\n", encoding="utf-8"
    )
    (source / "precursor.csv").write_text(
        "1000,0.20\n1001,0.10\n1002,0.20\n", encoding="utf-8"
    )
    created = project_workflow.create_project(
        str(source), project_id="dashboard"
    )
    output = created["output_root"]
    service = ProjectService.open(output)
    manifest = service.get_manifest()
    project_workflow.update_project_manifest(
        output,
        {
            "measurements": {
                row.measurement_id: {
                    "modality": "ftir",
                    "axis_kind": "wavenumber",
                    "axis_unit": "cm^-1",
                    "signal_kind": "absorbance",
                    "signal_unit": "absorbance",
                    "role": (
                        "product"
                        if "product" in str(row.metadata["measurement_name"])
                        else "precursor"
                    ),
                }
                for row in manifest.measurements
            },
            "samples": {
                sample.sample_id: {
                    "preparation_id": f"prep-{index}",
                    "role": (
                        "product"
                        if "product"
                        in str(sample.metadata["measurement_name"])
                        else "precursor"
                    ),
                }
                for index, sample in enumerate(manifest.samples)
            },
        },
    )
    plan = project_workflow.plan_project_analysis(
        output,
        "Compare product and precursor FTIR spectra",
        task_kind="spectral_comparison",
    )
    approval = project_workflow.approve_project_plan(
        output, plan["plan_id"], approved_by="scientist"
    )
    return output, plan, approval


def test_successful_run_creates_offline_dashboard_figures_and_tables(
    tmp_path: Path,
) -> None:
    output, plan, approval = _ready_ftir_project(tmp_path)
    result = run_project_analysis(
        output,
        plan["plan_id"],
        approval["approval_id"],
        "dashboard-run",
    )
    assert result["status"] == "succeeded"
    store = ProjectService.open(output).store
    record = store.read_json("runs/dashboard-run.json")
    kinds = [item["kind"] for item in record["artifacts"]]
    assert "analysis_evidence" in kinds
    assert "scientist_dashboard" in kinds
    assert "scientist_figure" in kinds
    assert "scientist_table" in kinds

    dashboard_artifact = next(
        item
        for item in record["artifacts"]
        if item["kind"] == "scientist_dashboard"
    )
    dashboard_path = store.path_for(dashboard_artifact["path"])
    dashboard = dashboard_path.read_text(encoding="utf-8")
    HTMLParser().feed(dashboard)
    assert "<svg" in dashboard
    assert "Source measurement overlay" in dashboard
    assert "Possible explanations" in dashboard
    assert "Unsupported claims" in dashboard
    assert "<script" not in dashboard
    assert "https://" not in dashboard
    assert "&#x27;-" not in dashboard
    assert ">RMSE<" in dashboard
    assert hashlib.sha256(dashboard_path.read_bytes()).hexdigest() == (
        dashboard_artifact["sha256"]
    )
    assert store.path_for(
        "runs/dashboard-run/figures/source-measurement-overlay.svg"
    ).is_file()
    ElementTree.parse(
        store.path_for(
            "runs/dashboard-run/figures/source-measurement-overlay.svg"
        )
    )
    assert store.path_for(
        "runs/dashboard-run/tables/sample-assignments.csv"
    ).is_file()
    assert store.path_for(
        "runs/dashboard-run/tables/pairwise-comparisons.csv"
    ).is_file()


def test_blocked_run_still_creates_inventory_dashboard(tmp_path: Path) -> None:
    output, plan, _ = _ready_ftir_project(tmp_path)
    result = run_project_analysis(
        output, plan["plan_id"], run_id="blocked-dashboard"
    )
    assert result["status"] == "blocked"
    store = ProjectService.open(output).store
    record = store.read_json("runs/blocked-dashboard.json")
    dashboard = next(
        item
        for item in record["artifacts"]
        if item["kind"] == "scientist_dashboard"
    )
    content = store.path_for(dashboard["path"]).read_text(encoding="utf-8")
    assert "approval" in content.lower()
    assert "Samples and measurement semantics" in content
    assert store.path_for(
        "runs/blocked-dashboard/tables/issues.csv"
    ).is_file()


def test_report_can_add_hash_bound_notebook_without_duplication(
    tmp_path: Path,
) -> None:
    output, plan, approval = _ready_ftir_project(tmp_path)
    run_project_analysis(
        output,
        plan["plan_id"],
        approval["approval_id"],
        "notebook-run",
    )
    report = generate_report_for_run(
        output, "notebook-run", include_notebook=True
    )
    assert report["scientist_report_path"] == "runs/notebook-run/report.md"
    assert report["dashboard_path"] == "runs/notebook-run/dashboard.html"
    assert report["notebook_path"] == (
        "runs/notebook-run/analysis-notebook.ipynb"
    )
    assert report["figure_paths"]
    assert report["table_paths"]
    store = ProjectService.open(output).store
    notebook = store.read_json(report["notebook_path"])
    assert notebook["nbformat"] == 4
    assert "does not refit models" in "".join(
        notebook["cells"][0]["source"]
    )
    assert store.path_for(report["scientist_report_path"]).is_file()
    again = generate_report_for_run(
        output, "notebook-run", include_notebook=True
    )
    assert again["notebook_path"] == report["notebook_path"]
    record = store.read_json("runs/notebook-run.json")
    assert (
        sum(
            item["kind"] == "reproducibility_notebook"
            for item in record["artifacts"]
        )
        == 1
    )


def test_tampered_dashboard_is_rejected_before_report(tmp_path: Path) -> None:
    output, plan, approval = _ready_ftir_project(tmp_path)
    run_project_analysis(
        output,
        plan["plan_id"],
        approval["approval_id"],
        "tamper-dashboard",
    )
    store = ProjectService.open(output).store
    store.path_for("runs/tamper-dashboard/dashboard.html").write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        generate_report_for_run(output, "tamper-dashboard")


def test_notebook_is_valid_json_and_uses_only_run_relative_artifacts(
    tmp_path: Path,
) -> None:
    output, plan, approval = _ready_ftir_project(tmp_path)
    run_project_analysis(
        output,
        plan["plan_id"],
        approval["approval_id"],
        "portable-notebook",
    )
    report = generate_report_for_run(
        output, "portable-notebook", include_notebook=True
    )
    raw = ProjectService.open(output).store.path_for(
        report["notebook_path"]
    ).read_text(encoding="utf-8")
    parsed = json.loads(raw)
    verification_source = "".join(parsed["cells"][1]["source"])
    assert str(Path(output).resolve()) not in verification_source
    assert "dashboard.html" in verification_source


@pytest.mark.parametrize(
    ("task_result", "expected_paths"),
    [
        (
            {
                "pca": {
                    "scores": [[-1.0, 0.2], [0.1, -0.3], [0.9, 0.1]]
                }
            },
            {"tables/pca-scores.csv", "figures/pca-scores.svg"},
        ),
        (
            {
                "evaluation": {
                    "predictions": [
                        {
                            "sample_id": "a",
                            "group": "g1",
                            "fold": 0,
                            "y_true": 1.0,
                            "y_pred": 1.1,
                        },
                        {
                            "sample_id": "b",
                            "group": "g2",
                            "fold": 1,
                            "y_true": 2.0,
                            "y_pred": 1.9,
                        },
                    ],
                    "selected_configs": [],
                }
            },
            {
                "tables/predictions.csv",
                "figures/predicted-vs-observed.svg",
            },
        ),
        (
            {
                "evidence": {
                    "job_plot_points": [
                        {
                            "mole_fraction": 0.0,
                            "response": 0.0,
                            "replicate_count": 1,
                        },
                        {
                            "mole_fraction": 0.5,
                            "response": 1.0,
                            "replicate_count": 2,
                        },
                        {
                            "mole_fraction": 1.0,
                            "response": 0.0,
                            "replicate_count": 1,
                        },
                    ]
                }
            },
            {"tables/jobs-plot.csv", "figures/jobs-plot.svg"},
        ),
        (
            {
                "ranked_results": [
                    {
                        "candidate_id": "reference-a",
                        "candidate_role": "reference",
                        "matched_peak_fraction": 0.75,
                        "mean_position_error_degrees": 0.03,
                        "whole_pattern_metrics": {
                            "cosine_similarity": 0.9,
                            "correlation": 0.8,
                        },
                        "overlap_fraction": 1.0,
                        "matched_peaks": [
                            {
                                "experimental_two_theta": 10.0,
                                "reference_two_theta": 10.02,
                                "position_error_degrees": 0.02,
                            }
                        ],
                    }
                ]
            },
            {
                "tables/pxrd-reference-ranking.csv",
                "tables/pxrd-matched-peaks.csv",
                "figures/pxrd-reference-ranking.svg",
            },
        ),
        (
            {
                "task_pack": "mass_spec",
                "evidence_rows": [
                    {
                        "candidate_id": "candidate-a",
                        "role": "reference",
                        "match_fraction": 0.5,
                        "intensity_cosine": 0.8,
                        "matches": [
                            {
                                "experimental_mz": 100.0,
                                "reference_mz": 100.0002,
                                "signed_error_da": -0.0002,
                                "absolute_error_da": 0.0002,
                                "signed_error_ppm": -2.0,
                                "absolute_error_ppm": 2.0,
                            }
                        ],
                    }
                ],
            },
            {
                "tables/ms-reference-ranking.csv",
                "tables/ms-matched-peaks.csv",
                "figures/ms-reference-ranking.svg",
            },
        ),
        (
            {
                "mixture_screening": {
                    "coefficients": [[0.7, 0.3]],
                    "mixture_measurement_ids": ["mixture-a"],
                    "provenance": {"reference_names": ["product", "precursor"]},
                }
            },
            {
                "tables/mixture-screening-coefficients.csv",
                "figures/mixture-screening-coefficients.svg",
            },
        ),
    ],
)
def test_modality_specific_dashboard_exports(
    tmp_path: Path,
    task_result: dict,
    expected_paths: set[str],
) -> None:
    store = ProjectStore(tmp_path)
    run = {
        "project_id": "project",
        "run_id": "task-run",
        "manifest_hash": "a" * 64,
        "plan_hash": "b" * 64,
        "artifacts": [],
    }
    artifacts, _, _ = _task_tables_and_figures(
        store,
        run,
        task_result,
        run_prefix="runs/task-run",
    )
    actual = {
        str(Path(item["path"]).relative_to("runs/task-run"))
        for item in artifacts
    }
    assert expected_paths.issubset(actual)
    for item in artifacts:
        path = store.path_for(item["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        if path.suffix == ".svg":
            ElementTree.parse(path)
