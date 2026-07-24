from __future__ import annotations

import pandas as pd

from chemometrics_mcp.core.ingestion import ParserRegistry


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
