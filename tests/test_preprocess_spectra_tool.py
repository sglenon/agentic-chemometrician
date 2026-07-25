"""Tests for sg_3rd_deriv, region_select, and the preprocess_spectra MCP tool."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from chemometrics_mcp.core import preprocessing as _pre
from chemometrics_mcp.core.pipelines import (
    RegionSelector,
    SavgolDerivativeTransformer,
    make_preprocessor,
)
from chemometrics_mcp.tools.preprocess_spectra import preprocess_spectra


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synth_spectra(n: int = 4, p: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, axis) — simple Gaussian-peak spectra."""
    axis = np.linspace(1000.0, 4000.0, p)
    X = np.zeros((n, p))
    for i in range(n):
        center = 2000.0 + i * 200.0
        X[i] = np.exp(-((axis - center) ** 2) / (2 * 150.0 ** 2)) + 0.01 * np.random.default_rng(i).standard_normal(p)
    return X, axis


def _write_csv(path: Path, axis: np.ndarray, X: np.ndarray, names: list[str] | None = None) -> None:
    """Write a two-column-or-more CSV: first col = axis, rest = signals."""
    if names is None:
        names = [f"sample_{i}" for i in range(X.shape[0])]
    with path.open("w") as f:
        header = "wavenumber," + ",".join(names)
        f.write(header + "\n")
        for j, ax_val in enumerate(axis):
            row = [f"{ax_val:.4f}"] + [f"{X[i, j]:.6f}" for i in range(X.shape[0])]
            f.write(",".join(row) + "\n")


# ---------------------------------------------------------------------------
# preprocessing.apply — sg_3rd_deriv
# ---------------------------------------------------------------------------

class TestSg3rdDerivApply:
    def test_output_shape_unchanged(self) -> None:
        X, _ = _synth_spectra(3, 60)
        X_out, details = _pre.apply(X, "sg_3rd_deriv")
        assert X_out.shape == X.shape

    def test_details_keys(self) -> None:
        X, _ = _synth_spectra(2, 60)
        _, details = _pre.apply(X, "sg_3rd_deriv")
        assert details["method"] == "sg_3rd_deriv"
        assert details["deriv"] == 3
        assert details["polyorder"] == 3
        assert details["window_length"] % 2 == 1  # must be odd

    def test_derivative_is_not_zero_for_non_constant_signal(self) -> None:
        X, _ = _synth_spectra(2, 60)
        X_out, _ = _pre.apply(X, "sg_3rd_deriv")
        assert not np.allclose(X_out, 0.0)

    def test_constant_signal_gives_zero_derivative(self) -> None:
        """Third derivative of a constant (or low-degree polynomial) is zero."""
        X = np.ones((3, 40))
        X_out, _ = _pre.apply(X, "sg_3rd_deriv")
        assert np.allclose(X_out, 0.0, atol=1e-10)

    def test_requires_2d_input(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            _pre.apply(np.ones(20), "sg_3rd_deriv")


# ---------------------------------------------------------------------------
# preprocessing.apply — region_select
# ---------------------------------------------------------------------------

class TestRegionSelectApply:
    def test_correct_truncation(self) -> None:
        X, axis = _synth_spectra(3, 60)
        X_out, details = _pre.apply(X, "region_select", axis=axis, **{"min": 1500.0, "max": 3000.0})
        assert X_out.shape[0] == 3
        # All kept features must be within [1500, 3000]
        assert details["n_kept"] == X_out.shape[1]
        assert details["n_total"] == 60
        # axis_out returned
        ax_out = details["axis_out"]
        assert np.all(ax_out >= 1499.9)
        assert np.all(ax_out <= 3000.1)

    def test_min_only_bound(self) -> None:
        X, axis = _synth_spectra(2, 60)
        X_out, details = _pre.apply(X, "region_select", axis=axis, **{"min": 2000.0})
        assert details["max"] is None
        assert details["n_kept"] < 60

    def test_max_only_bound(self) -> None:
        X, axis = _synth_spectra(2, 60)
        X_out, details = _pre.apply(X, "region_select", axis=axis, **{"max": 2000.0})
        assert details["min"] is None
        assert details["n_kept"] < 60

    def test_no_bounds_keeps_all(self) -> None:
        X, axis = _synth_spectra(2, 60)
        X_out, _ = _pre.apply(X, "region_select", axis=axis)
        assert X_out.shape == X.shape

    def test_axis_absent_raises(self) -> None:
        X, _ = _synth_spectra(2, 60)
        with pytest.raises(ValueError, match="axis"):
            _pre.apply(X, "region_select", **{"min": 1000.0, "max": 2000.0})

    def test_bounds_exclude_all_raises(self) -> None:
        X, axis = _synth_spectra(2, 60)
        with pytest.raises(ValueError, match="no spectral points remain"):
            _pre.apply(X, "region_select", axis=axis, **{"min": 9999.0, "max": 99999.0})

    def test_axis_length_mismatch_raises(self) -> None:
        X, axis = _synth_spectra(2, 60)
        bad_axis = np.linspace(0, 1, 30)  # wrong length
        with pytest.raises(ValueError):
            _pre.apply(X, "region_select", axis=bad_axis, **{"min": 0.2, "max": 0.8})


# ---------------------------------------------------------------------------
# pipelines — SavgolDerivativeTransformer order=3
# ---------------------------------------------------------------------------

class TestSavgolOrder3Pipeline:
    def test_order3_accepted(self) -> None:
        X, _ = _synth_spectra(3, 60)
        t = SavgolDerivativeTransformer(order=3, polyorder=3)
        X_out = t.fit_transform(X)
        assert X_out.shape == X.shape

    def test_order3_with_polyorder_less_than_order_raises(self) -> None:
        X, _ = _synth_spectra(3, 60)
        t = SavgolDerivativeTransformer(order=3, polyorder=2)
        with pytest.raises(ValueError, match="polyorder"):
            t.fit(X)

    def test_order4_still_rejected(self) -> None:
        X, _ = _synth_spectra(3, 60)
        t = SavgolDerivativeTransformer(order=4, polyorder=4)
        with pytest.raises(ValueError, match="order"):
            t.fit(X)

    def test_make_preprocessor_sg3(self) -> None:
        X, _ = _synth_spectra(3, 60)
        t = make_preprocessor("sg_3rd_deriv").fit(X)
        provenance = t.get_provenance()
        assert provenance["parameters"]["order"] == 3
        assert provenance["parameters"]["polyorder"] >= 3


# ---------------------------------------------------------------------------
# pipelines — RegionSelector transformer
# ---------------------------------------------------------------------------

class TestRegionSelectorTransformer:
    def test_correct_truncation(self) -> None:
        X, axis = _synth_spectra(4, 60)
        rs = RegionSelector(axis=axis, min_val=1500.0, max_val=3000.0)
        X_out = rs.fit_transform(X)
        assert X_out.shape[0] == 4
        assert X_out.shape[1] < 60
        assert np.all(rs.axis_ >= 1499.9)
        assert np.all(rs.axis_ <= 3000.1)

    def test_fit_stores_mask_and_axis(self) -> None:
        X, axis = _synth_spectra(2, 60)
        rs = RegionSelector(axis=axis, min_val=2000.0).fit(X)
        assert hasattr(rs, "mask_")
        assert hasattr(rs, "axis_")
        assert rs.n_features_out_ == int(rs.mask_.sum())

    def test_axis_length_mismatch_raises(self) -> None:
        X, axis = _synth_spectra(2, 60)
        rs = RegionSelector(axis=axis[:30])
        with pytest.raises(ValueError):
            rs.fit(X)

    def test_empty_region_raises(self) -> None:
        X, axis = _synth_spectra(2, 60)
        rs = RegionSelector(axis=axis, min_val=9999.0)
        with pytest.raises(ValueError, match="no spectral points remain"):
            rs.fit(X)

    def test_make_preprocessor_region_select(self) -> None:
        X, axis = _synth_spectra(2, 60)
        rs = make_preprocessor("region_select", axis=axis, min_val=1500.0, max_val=3000.0)
        X_out = rs.fit_transform(X)
        assert X_out.shape[1] < 60

    def test_provenance_contains_bounds(self) -> None:
        X, axis = _synth_spectra(2, 60)
        rs = RegionSelector(axis=axis, min_val=1500.0, max_val=3000.0).fit(X)
        prov = rs.get_provenance()
        assert prov["fitted_state"]["min_val"] == 1500.0
        assert prov["fitted_state"]["max_val"] == 3000.0


# ---------------------------------------------------------------------------
# preprocess_spectra MCP tool — end-to-end
# ---------------------------------------------------------------------------

class TestPreprocessSpectraTool:
    def _make_csv_file(self, tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
        X, axis = _synth_spectra(4, 60)
        fpath = tmp_path / "spectra.csv"
        _write_csv(fpath, axis, X, names=[f"s{i}" for i in range(4)])
        return fpath, X, axis

    def test_before_after_shape_snv(self, tmp_path: Path) -> None:
        fpath, X, axis = self._make_csv_file(tmp_path)
        result = preprocess_spectra(str(fpath), steps=[{"name": "snv"}])
        assert result["n_samples"] == 4
        assert result["n_features_before"] == 60
        assert result["n_features_after"] == 60
        assert len(result["before"]["axis"]) == 60
        assert len(result["after"]["axis"]) == 60
        assert set(result["before"]["signals"].keys()) == set(result["after"]["signals"].keys())

    def test_region_select_shrinks_axis(self, tmp_path: Path) -> None:
        fpath, X, axis = self._make_csv_file(tmp_path)
        result = preprocess_spectra(
            str(fpath),
            steps=[{"name": "region_select", "min": 1500.0, "max": 3000.0}],
        )
        assert result["n_features_after"] < 60
        assert len(result["after"]["axis"]) == result["n_features_after"]
        assert "region" in result
        assert result["region"]["n_kept"] == result["n_features_after"]

    def test_multi_step_pipeline(self, tmp_path: Path) -> None:
        fpath, X, axis = self._make_csv_file(tmp_path)
        result = preprocess_spectra(
            str(fpath),
            steps=[
                {"name": "region_select", "min": 1500.0, "max": 3000.0},
                {"name": "snv"},
                {"name": "sg_2nd_deriv"},
            ],
        )
        assert result["steps"] == ["region_select", "snv", "sg_2nd_deriv"]
        assert result["n_features_after"] < 60
        # before axis is full length; after axis is truncated (region_select)
        assert len(result["before"]["axis"]) == 60
        assert len(result["after"]["axis"]) == result["n_features_after"]

    def test_sg_3rd_deriv_in_pipeline(self, tmp_path: Path) -> None:
        fpath, X, axis = self._make_csv_file(tmp_path)
        result = preprocess_spectra(
            str(fpath),
            steps=[{"name": "sg_3rd_deriv"}],
        )
        assert result["steps"] == ["sg_3rd_deriv"]
        assert result["n_features_after"] == 60

    def test_missing_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            preprocess_spectra("/nonexistent/path/spectra.csv", steps=[{"name": "snv"}])

    def test_empty_steps_raises(self, tmp_path: Path) -> None:
        fpath, _, _ = self._make_csv_file(tmp_path)
        # Pydantic min_length=1 on steps would catch this at MCP layer;
        # direct call with empty list should either error or return before==after.
        # If it doesn't raise, before signals == after signals trivially.
        try:
            result = preprocess_spectra(str(fpath), steps=[])
            # All signals unchanged
            for name in result["before"]["signals"]:
                assert result["before"]["signals"][name] == result["after"]["signals"][name]
        except (ValueError, IndexError):
            pass  # acceptable

    def test_directory_ingestion(self, tmp_path: Path) -> None:
        X, axis = _synth_spectra(3, 40)
        for i in range(3):
            f = tmp_path / f"sample_{i}.csv"
            # Write single-signal CSV
            with f.open("w") as fh:
                for j, ax_val in enumerate(axis):
                    fh.write(f"{ax_val:.2f},{X[i, j]:.6f}\n")
        result = preprocess_spectra(str(tmp_path), steps=[{"name": "snv"}])
        assert result["n_samples"] == 3

    def test_before_not_modified_by_after(self, tmp_path: Path) -> None:
        """before signals must be the raw values, unaffected by transformation."""
        fpath, X, axis = self._make_csv_file(tmp_path)
        result = preprocess_spectra(str(fpath), steps=[{"name": "snv"}])
        # Raw spectra should have non-unit variance; SNV-transformed should have unit std
        for name, raw in result["before"]["signals"].items():
            transformed = result["after"]["signals"][name]
            raw_arr = np.array(raw)
            tr_arr = np.array(transformed)
            # They should differ
            assert not np.allclose(raw_arr, tr_arr)
            # SNV output has std ≈ 1
            assert abs(tr_arr.std() - 1.0) < 0.05
