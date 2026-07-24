from __future__ import annotations

import json

from chemometrics_mcp.cli import main


def test_doctor_emits_json(capsys) -> None:
    assert main(["doctor"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] and result["temp_writable"]


def test_argument_errors_are_nonzero(capsys) -> None:
    assert main(["show", "/does/not/exist"]) == 2
    assert capsys.readouterr().err


def test_parser_requires_approval_for_runs() -> None:
    # argparse raises SystemExit for an incomplete command before any service import.
    import pytest
    with pytest.raises(SystemExit): main(["run", "out", "plan"])


def test_plan_options_must_be_a_json_object(capsys) -> None:
    result = main(
        [
            "plan",
            "out",
            "--objective",
            "x",
            "--options-json",
            "[]",
        ]
    )
    assert result == 2
    assert "JSON object" in capsys.readouterr().err
