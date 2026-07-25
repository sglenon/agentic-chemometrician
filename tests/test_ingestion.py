from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from chemometrics_mcp.core.ingestion import ParserRegistry  # noqa: E402

DATASET_DIR = ROOT / "ftir-purity-dataset" / "fwdftirjune262026"


def test_two_column_delimited_and_whitespace_layouts(tmp_path) -> None:
    registry = ParserRegistry()
    for extension, contents in [("csv", "x,y\n1,2\n3,4\n"), ("xy", "1 2\n3 4\n"), ("tsv", "x\ty\n1\t2\n3\t4\n")]:
        path = tmp_path / f"one.{extension}"
        path.write_text(contents)
        parsed, issues = registry.parse(path)
        assert not issues and parsed[0].axis == (1.0, 3.0) and parsed[0].signal == (2.0, 4.0)


def test_text_header_units_are_preserved_without_clipping(tmp_path) -> None:
    path = tmp_path / "ftir.asc"
    path.write_text("##XUNITS=CM-1\n##YUNITS=%T\n1,120\n2,80\n")
    parsed, issues = ParserRegistry().parse(path)
    assert not issues
    assert parsed[0].axis_kind == "wavenumber"
    assert parsed[0].signal_kind == "percent_transmittance"
    assert parsed[0].signal == (120.0, 80.0)


def test_axis_plus_multiple_signals_and_comments(tmp_path) -> None:
    path = tmp_path / "multi.txt"
    path.write_text("# comment\naxis A B\n1 2 3\n4 5 6\n")
    parsed, issues = ParserRegistry().parse(path)
    assert not issues
    assert [item.measurement_name for item in parsed] == ["A", "B"]
    assert parsed[1].signal == (3.0, 6.0)


def test_generic_and_spectra_metadata_excel(tmp_path) -> None:
    generic = tmp_path / "generic.xlsx"
    pd.DataFrame({"axis": [1, 2], "a": [3, 4]}).to_excel(generic, index=False)
    parsed, issues = ParserRegistry().parse(generic)
    assert not issues and parsed[0].measurement_name == "a"
    special = tmp_path / "special.xlsx"
    with pd.ExcelWriter(special) as writer:
        pd.DataFrame({"Spectrum ID": ["a", "b"], "batch": [1, 2]}).to_excel(writer, sheet_name="Spectra Metadata", index=False)
        pd.DataFrame({"axis": [1, 2], "a": [3, 4], "b": [5, 6]}).to_excel(writer, sheet_name="Spectra Device 1", index=False)
    parsed, issues = ParserRegistry().parse(special)
    assert not issues and parsed[0].metadata["sample_metadata"]["batch"] == 1


def test_jcamp_explicit_pairs_and_compressed_encoding_issue(tmp_path) -> None:
    valid = tmp_path / "sample.jdx"
    valid.write_text("##TITLE=Example\n##XUNITS=CM-1\n##YUNITS=ABS\n##PEAK TABLE=(XY..XY)\n1,2\n3,4\n##END=\n")
    parsed, issues = ParserRegistry().parse(valid)
    assert not issues and parsed[0].axis_unit == "CM-1" and parsed[0].signal == (2.0, 4.0)
    compressed = tmp_path / "bad.dx"
    compressed.write_text("##XYDATA=(X++(Y..Y))\n1 2\n")
    _, issues = ParserRegistry().parse(compressed)
    assert issues[0].code == "unsupported_jcamp_encoding"


def test_inventory_ignores_output_and_symlinks_and_reports_unsupported(tmp_path) -> None:
    (tmp_path / "good.csv").write_text("1,2\n")
    (tmp_path / "bad.pdf").write_bytes(b"x")
    output = tmp_path / "chemometrics-output"
    output.mkdir()
    (output / "hidden.csv").write_text("1,2\n")
    (tmp_path / "link.csv").symlink_to(tmp_path / "good.csv")
    entries, issues = ParserRegistry().inventory_directory(tmp_path)
    assert [entry.relative_path.as_posix() for entry in entries] == ["bad.pdf", "good.csv"]
    assert issues[0].code == "unsupported_file"


def test_malformed_input_returns_issue_not_exception(tmp_path) -> None:
    path = tmp_path / "broken.dat"
    path.write_text("axis signal\n1 nope\n")
    parsed, issues = ParserRegistry().parse(path)
    assert not parsed and issues[0].code == "malformed_input"


def test_configured_resource_limit_fails_closed(tmp_path) -> None:
    path = tmp_path / "large.csv"
    path.write_text("x,y\n1,2\n2,3\n")
    parsed, issues = ParserRegistry(max_file_bytes=4).parse(path)
    assert parsed == ()
    assert issues[0].code == "resource_limit_exceeded"


# ---------------------------------------------------------------------------
# Single-column Y-only / JCAMP-axis-reconstruction tests
# ---------------------------------------------------------------------------

def _make_yonly(tmp_path, extra_headers: str = "", data_lines: list[str] | None = None) -> Path:
    """Helper: write a minimal single-column .asc file."""
    path = tmp_path / "spec.asc"
    n = 5
    lines = data_lines if data_lines is not None else ["1.0", "2.0", "3.0", "4.0", "5.0"]
    header = (
        "##XUNITS=1/CM\n"
        "##YUNITS=%T\n"
        f"##FIRSTX=400.0\n"
        f"##LASTX=500.0\n"
        f"##NPOINTS={n}.0\n"
        f"{extra_headers}"
    )
    path.write_text(header + "\n".join(lines) + "\n")
    return path


def test_yonly_basic_linspace_axis(tmp_path) -> None:
    """FIRSTX/LASTX/NPOINTS reconstructs correct axis via linspace."""
    import numpy as np
    path = _make_yonly(tmp_path)
    parsed, issues = ParserRegistry().parse(path)
    assert not issues, issues
    assert len(parsed) == 1
    m = parsed[0]
    assert m.parser_id == "delimited-text-yonly-v1"
    assert len(m.axis) == 5
    assert len(m.signal) == 5
    expected_axis = np.linspace(400.0, 500.0, 5)
    for got, exp in zip(m.axis, expected_axis):
        assert abs(got - exp) < 1e-9, f"{got} != {exp}"
    assert m.signal == (1.0, 2.0, 3.0, 4.0, 5.0)


def test_yonly_deltax_axis(tmp_path) -> None:
    """FIRSTX/DELTAX/NPOINTS reconstructs correct axis via arange."""
    path = tmp_path / "spec.asc"
    path.write_text(
        "##XUNITS=1/CM\n##YUNITS=ABS\n"
        "##FIRSTX=100.0\n##DELTAX=10.0\n##NPOINTS=4\n"
        "1.0\n2.0\n3.0\n4.0\n"
    )
    parsed, issues = ParserRegistry().parse(path)
    assert not issues, issues
    m = parsed[0]
    assert m.parser_id == "delimited-text-yonly-v1"
    assert len(m.axis) == 4
    for i, expected in enumerate([100.0, 110.0, 120.0, 130.0]):
        assert abs(m.axis[i] - expected) < 1e-9, f"axis[{i}]={m.axis[i]} != {expected}"


def test_yonly_descending_axis_preserved(tmp_path) -> None:
    """Descending axis (FIRSTX > LASTX) must NOT be flipped or sorted."""
    path = tmp_path / "desc.asc"
    path.write_text(
        "##XUNITS=1/CM\n##YUNITS=%T\n"
        "##FIRSTX=4000.0\n##LASTX=400.0\n##NPOINTS=3\n"
        "10.0\n20.0\n30.0\n"
    )
    parsed, issues = ParserRegistry().parse(path)
    assert not issues, issues
    m = parsed[0]
    # linspace(4000, 400, 3) = [4000, 2200, 400]
    assert m.axis[0] > m.axis[-1], "Descending axis was incorrectly flipped"
    assert abs(m.axis[0] - 4000.0) < 1e-9
    assert abs(m.axis[-1] - 400.0) < 1e-9


def test_yonly_xfactor_yfactor_numeric(tmp_path) -> None:
    """Numeric XFACTOR and YFACTOR are applied to axis and signal."""
    path = tmp_path / "factors.asc"
    path.write_text(
        "##XUNITS=1/CM\n##YUNITS=%T\n"
        "##FIRSTX=100.0\n##LASTX=200.0\n##NPOINTS=3\n"
        "##XFACTOR=2.0\n##YFACTOR=0.5\n"
        "10.0\n20.0\n30.0\n"
    )
    parsed, issues = ParserRegistry().parse(path)
    assert not issues, issues
    m = parsed[0]
    # axis * XFACTOR: linspace(100,200,3)=[100,150,200] * 2 = [200,300,400]
    assert abs(m.axis[0] - 200.0) < 1e-9
    assert abs(m.axis[1] - 300.0) < 1e-9
    assert abs(m.axis[2] - 400.0) < 1e-9
    # signal * YFACTOR: [10,20,30] * 0.5 = [5,10,15]
    assert abs(m.signal[0] - 5.0) < 1e-9
    assert abs(m.signal[1] - 10.0) < 1e-9
    assert abs(m.signal[2] - 15.0) < 1e-9


def test_yonly_nonnumeric_yfactor_ignored(tmp_path) -> None:
    """Non-numeric YFACTOR (e.g. 'Anything') is silently ignored — no error."""
    path = tmp_path / "placeholder.asc"
    path.write_text(
        "##XUNITS=1/CM\n##YUNITS=%T\n"
        "##FIRSTX=400.0\n##LASTX=500.0\n##NPOINTS=3\n"
        "##YFACTOR=Anything\n"
        "10.0\n20.0\n30.0\n"
    )
    parsed, issues = ParserRegistry().parse(path)
    assert not issues, issues
    m = parsed[0]
    # signal unchanged (factor skipped)
    assert m.signal == (10.0, 20.0, 30.0)


def test_yonly_axis_kind_signal_kind_mapped(tmp_path) -> None:
    """XUNITS/YUNITS are mapped to axis_kind/signal_kind correctly."""
    path = _make_yonly(tmp_path)
    parsed, _ = ParserRegistry().parse(path)
    m = parsed[0]
    assert m.axis_kind == "wavenumber"
    assert m.signal_kind == "percent_transmittance"
    assert m.axis_unit == "1/CM"
    assert m.signal_unit == "%T"


def test_yonly_npoints_mismatch_rejected(tmp_path) -> None:
    """NPOINTS != len(rows) produces jcamp_npoints_mismatch issue."""
    path = tmp_path / "mismatch.asc"
    # Claim 10 points but only provide 3
    path.write_text(
        "##FIRSTX=400.0\n##LASTX=500.0\n##NPOINTS=10\n"
        "1.0\n2.0\n3.0\n"
    )
    parsed, issues = ParserRegistry().parse(path)
    assert not parsed
    assert issues[0].code == "jcamp_npoints_mismatch"
    assert issues[0].details["npoints_header"] == 10
    assert issues[0].details["rows_found"] == 3


def test_yonly_missing_geometry_rejected(tmp_path) -> None:
    """Single-column file with no geometry headers is rejected cleanly."""
    path = tmp_path / "nogeom.asc"
    # No FIRSTX/LASTX/NPOINTS — must reject, not silently create index axis
    path.write_text("1.0\n2.0\n3.0\n")
    parsed, issues = ParserRegistry().parse(path)
    assert not parsed
    assert issues[0].code == "jcamp_missing_geometry"


def test_yonly_missing_lastx_and_deltax_rejected(tmp_path) -> None:
    """FIRSTX + NPOINTS but neither LASTX nor DELTAX → jcamp_missing_geometry."""
    path = tmp_path / "partial.asc"
    path.write_text(
        "##FIRSTX=400.0\n##NPOINTS=3\n"
        "1.0\n2.0\n3.0\n"
    )
    parsed, issues = ParserRegistry().parse(path)
    assert not parsed
    assert issues[0].code == "jcamp_missing_geometry"


def test_yonly_real_file_3fc1(tmp_path) -> None:
    """Real dataset file 3fc1.asc parses without issues."""
    asc = DATASET_DIR / "3fc1.asc"
    if not asc.exists():
        import pytest
        pytest.skip(f"Real dataset file not present: {asc}")
    parsed, issues = ParserRegistry().parse(asc)
    assert not issues, [i.message for i in issues]
    assert len(parsed) == 1
    m = parsed[0]
    assert m.parser_id == "delimited-text-yonly-v1"
    assert len(m.axis) == 1868
    assert len(m.signal) == 1868
    assert m.axis_kind == "wavenumber"
    assert m.signal_kind == "percent_transmittance"
    # Axis should span ~399 to ~4000 cm-1
    assert min(m.axis) > 390.0
    assert max(m.axis) < 4010.0


def test_yonly_real_file_8ami1(tmp_path) -> None:
    """Real dataset file 8ami1.asc parses without issues."""
    asc = DATASET_DIR / "8ami1.asc"
    if not asc.exists():
        import pytest
        pytest.skip(f"Real dataset file not present: {asc}")
    parsed, issues = ParserRegistry().parse(asc)
    assert not issues, [i.message for i in issues]
    assert len(parsed) == 1
    m = parsed[0]
    assert m.parser_id == "delimited-text-yonly-v1"
    assert len(m.axis) == 1868
    assert len(m.signal) == 1868
