import asyncio
import json

from chemometrics_mcp import server


def _call(name, args):
    content = asyncio.run(server.call_tool(name, args))
    return json.loads(content[0].text)


def _has_raw_array_keys(value):
    if isinstance(value, dict):
        return any(key in {"axis", "signal", "x", "y"} or _has_raw_array_keys(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_has_raw_array_keys(item) for item in value)
    return False


def test_schemas_are_generated_and_strict():
    tools = {item["name"]: item for item in __import__("chemometrics_mcp.mcp", fromlist=["tool_definitions"]).tool_definitions()}
    assert tools["create_project"]["inputSchema"]["additionalProperties"] is False
    assert "source_root" in tools["create_project"]["inputSchema"]["required"]


def test_project_workflow_through_server_is_compact(tmp_path):
    source = tmp_path / "source"; source.mkdir()
    (source / "sample.csv").write_text("1,2\n2,3\n")
    output = tmp_path / "output"
    created = _call("create_project", {"source_root": str(source), "output_root": str(output), "project_id": "demo"})
    assert created["ok"] and not _has_raw_array_keys(created)
    got = _call("get_project", {"output_root": str(output)})
    assert got["ok"] and not _has_raw_array_keys(got)
    measurement = got["payload"]["measurement_ids"][0]
    updated = _call("update_project_manifest", {"output_root": str(output), "updates": {"measurements": {measurement: {"modality": "nir", "axis_kind": "wavelength", "axis_unit": "nm", "signal_kind": "absorbance", "signal_unit": "au"}}}})
    assert updated["ok"]
    planned = _call("plan_project_analysis", {"output_root": str(output), "objective": "explore spectra"})
    assert planned["ok"]
    approval = _call("approve_project_plan", {"output_root": str(output), "plan": planned["payload"]["plan_id"], "approved_by": "tester"})
    assert approval["ok"]


def test_invalid_arguments_return_envelope_error():
    response = _call("get_project", {"unexpected": True})
    assert response == {"tool_name": "get_project", "ok": False, "payload": None, "error": response["error"]}
    assert response["error"]


def test_run_schemas_are_strict_without_exposing_arrays():
    from chemometrics_mcp import mcp
    schemas = {item["name"]: item["inputSchema"] for item in mcp.tool_definitions()}
    assert schemas["run_project_analysis"]["additionalProperties"] is False
    report_properties = schemas["generate_project_report"]["properties"]
    assert report_properties["include_notebook"]["type"] == "boolean"
    assert report_properties["include_notebook"]["default"] is False
    response = mcp.dispatch("get_project_run", {"output_root": "x", "run_id": "bad/id"})
    assert not response["ok"]
