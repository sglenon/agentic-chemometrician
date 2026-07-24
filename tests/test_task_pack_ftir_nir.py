from __future__ import annotations

from chemometrics_mcp.core.task_packs.ftir_nir import run_ftir_nir_task


def measurement(identifier, signal, **extra):
    return {"measurement_id": identifier, "sample_id": identifier, "preparation_id": f"prep-{identifier}", "modality": "ftir", "role": "sample", "axis": (1000., 1001., 1002., 1003.), "signal": tuple(signal), "axis_kind": "wavenumber", "axis_unit": "cm-1", "signal_kind": "transmittance", "signal_unit": "%", **extra}


def test_transmittance_conversion_comparison_and_raw_immutability() -> None:
    raw = (100., 50., 25., 50.)
    result = run_ftir_nir_task([measurement("a", raw), measurement("b", raw)])
    assert raw == (100., 50., 25., 50.)
    assert result["evidence_rows"][0]["comparison"]["n_overlap_points"] == 4
    assert result["measurement_provenance"][0]["conversions"][0]["action"] == "percent_transmittance_to_absorbance"


def test_invalid_units_are_rejected_and_pca_counts_are_explicit() -> None:
    invalid = measurement("bad", [1, 2, 3, 4], signal_unit="bananas")
    assert any(item["level"] == "blocker" for item in run_ftir_nir_task([invalid])["issues"])
    result = run_ftir_nir_task([measurement("a", [90, 80, 70, 60]), measurement("b", [80, 70, 60, 50]), measurement("c", [70, 60, 50, 40])], "pca")
    assert result["pca"]["scan_count"] == 3 and "chemical identity" in result["pca"]["label"]


def test_mixture_requires_named_references_and_stays_screening() -> None:
    no_refs = run_ftir_nir_task([measurement("a", [90, 80, 70, 60])], "mixture")
    assert any(item["code"] == "explicit_references_required" for item in no_refs["issues"])
    result = run_ftir_nir_task([measurement("r", [90, 80, 70, 60], role="reference", reference_name="reference"), measurement("m", [80, 70, 60, 50])], "mixture")
    assert result["claim_ceiling"] == "screening" and "mixture_screening" in result
    assert result["mixture_screening"]["provenance"]["sum_to_one"] is True
    assert "raw_coefficients" in result["mixture_screening"]


def test_modality_and_signal_semantics_are_explicit() -> None:
    bad = measurement("bad", [1, 2, 3, 4], modality="uv_vis")
    assert any(
        issue["code"] == "ftir_nir_modality_required"
        for issue in run_ftir_nir_task([bad])["issues"]
    )
    reflectance = measurement(
        "reflectance",
        [0.2, 0.3, 0.4, 0.3],
        signal_kind="reflectance",
        signal_unit="fraction",
    )
    result = run_ftir_nir_task(
        [measurement("abs", [90, 80, 70, 60]), reflectance]
    )
    assert result["evidence_rows"] == []
    assert any(
        issue["code"] == "incompatible_signal_semantics"
        for issue in result["issues"]
    )
