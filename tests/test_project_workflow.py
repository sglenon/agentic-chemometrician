from types import SimpleNamespace

from chemometrics_mcp.tools import project_workflow as workflow


class FakeService:
    def __init__(self, output_root): self.output_root = output_root; self.manifest = None
    def create(self, source_root, output_root=None, project_id=None):
        self.manifest = SimpleNamespace(project_id=project_id or "demo", manifest_hash="h", assets=(1,), samples=(1,), measurements=(1,), unresolved_issues=())
        return self
    def get_manifest(self): return self.manifest
    def get_summary(self): return {"project_id": self.manifest.project_id, "axis": [1, 2]}
    def update_manifest(self, updates): return self.manifest


def test_create_response_is_compact(monkeypatch, tmp_path):
    service = FakeService(str(tmp_path / "out"))
    monkeypatch.setattr(workflow, "_load_service", lambda: type("S", (), {"create": classmethod(lambda cls, *args, **kwargs: service.create(*args, **kwargs))}))
    response = workflow.create_project(str(tmp_path), str(tmp_path / "out"), "demo")
    assert response["project_id"] == "demo"
    assert "axis" not in str(response)


def test_get_response_removes_raw_arrays(monkeypatch, tmp_path):
    service = FakeService(str(tmp_path)); service.create(str(tmp_path), project_id="demo")
    monkeypatch.setattr(workflow, "_service", lambda root: service)
    assert workflow.get_project(str(tmp_path)) == {"project_id": "demo"}


def test_real_folder_workflow_can_resolve_plan_and_approve(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "sample.csv").write_text("axis,signal\n1,2\n2,3\n")
    created = workflow.create_project(str(source), project_id="workflow")
    output = created["output_root"]
    service = workflow._service(output)
    manifest = service.get_manifest()
    measurement = manifest.measurements[0]
    workflow.update_project_manifest(
        output,
        {
            "measurements": {
                measurement.measurement_id: {
                    "modality": "nir",
                    "axis_kind": "wavelength",
                    "axis_unit": "nm",
                    "signal_kind": "absorbance",
                    "signal_unit": "absorbance",
                }
            }
        },
    )
    planned = workflow.plan_project_analysis(
        output, "Compare this spectrum with the supplied references",
        task_kind="spectral_comparison",
    )
    assert planned["plan_hash"]
    assert planned["pipelines"]
    assert "axis" not in str(planned) and "signal" not in str(planned)
    approved = workflow.approve_project_plan(
        output, planned["plan_id"], approved_by="scientist"
    )
    assert approved["approved"] and approved["verified"]


def test_run_status_wrapper_is_compact(tmp_path):
    store = workflow.ProjectStore(tmp_path)
    store.write_json("runs/demo-run.json", {"run_id": "demo-run", "status": "succeeded", "axis": [1], "signal": [2], "issues": []})
    response = workflow.get_project_run(str(tmp_path), "demo-run")
    assert response == {"run_id": "demo-run", "status": "succeeded", "issues": []}
