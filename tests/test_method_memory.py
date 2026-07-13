"""Tests for method memory: save, load, search, recommend, rebuild."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from chemometrics_contracts import (
    DatasetProfile,
    MethodMemory,
    MethodMemoryEntry,
    MethodMemoryIndex,
    SaveMethodMemoryRequest,
    SearchMethodMemoryRequest,
    RecommendFromMemoryRequest,
)
from chemometrics_mcp.core.method_memory import (
    load_method,
    rebuild_index,
    recommend_from_memory,
    save_method,
    search_methods,
)
from chemometrics_mcp.tools import save_method_memory, search_method_memory, recommend_from_memory as rec_tool


def _dp(**overrides) -> DatasetProfile:
    defaults = dict(modality="NIR", n_samples=100, n_features=200)
    defaults.update(overrides)
    return DatasetProfile(**defaults)


def _mem(memory_id: str = "mem-001", **overrides) -> MethodMemory:
    defaults = dict(
        memory_id=memory_id,
        created_at="2026-07-12T00:00:00Z",
        modality="NIR",
        task_name="classification",
        dataset_profile=_dp(),
        preprocessing="snv",
        model_name="svm_rbf",
        validation_strategy="stratified_kfold_5",
        key_metrics={"accuracy": 0.92},
        caveats=("small sample",),
        approval_status="approved",
    )
    defaults.update(overrides)
    return MethodMemory(**defaults)


class TestSaveMethod(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def test_save_writes_json(self):
        mem = _mem()
        path = save_method(mem, self._tmp)
        self.assertTrue(path.exists())
        data = json.loads(path.read_text())
        self.assertEqual(data["memory_id"], "mem-001")
        self.assertEqual(data["modality"], "NIR")
        self.assertEqual(data["dataset_profile"]["n_samples"], 100)

    def test_save_returns_valid_path(self):
        mem = _mem()
        path = save_method(mem, self._tmp)
        self.assertIsInstance(path, Path)
        self.assertTrue(path.is_file())


class TestLoadMethod(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def test_round_trip(self):
        mem = _mem(reviewer_notes="Looks good", source_run_id="run-abc")
        save_method(mem, self._tmp)
        loaded = load_method("mem-001", self._tmp)
        self.assertEqual(loaded.memory_id, "mem-001")
        self.assertEqual(loaded.modality, "NIR")
        self.assertEqual(loaded.model_name, "svm_rbf")
        self.assertEqual(loaded.reviewer_notes, "Looks good")
        self.assertEqual(loaded.source_run_id, "run-abc")
        self.assertEqual(loaded.dataset_profile.n_samples, 100)
        self.assertEqual(loaded.dataset_profile.n_features, 200)
        self.assertEqual(list(loaded.caveats), ["small sample"])


class TestSearchMethods(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        entries = (
            MethodMemoryEntry(memory_id="m1", modality="NIR", task_name="classification",
                              model_name="svm_rbf", preprocessing="snv", key_metric_value=0.92,
                              approval_status="approved"),
            MethodMemoryEntry(memory_id="m2", modality="NIR", task_name="regression",
                              model_name="plsr", preprocessing="snv", key_metric_value=0.85,
                              approval_status="approved"),
            MethodMemoryEntry(memory_id="m3", modality="FTIR", task_name="classification",
                              model_name="random_forest", preprocessing="msc", key_metric_value=0.88,
                              approval_status="approved"),
            MethodMemoryEntry(memory_id="m4", modality="NIR", task_name="classification",
                              model_name="pca_lda", preprocessing="detrend", key_metric_value=0.78,
                              approval_status="rejected"),
        )
        self.index = MethodMemoryIndex(entries=entries)

    def test_filter_by_modality(self):
        results = search_methods(self.index, modality="NIR")
        ids = [e.memory_id for e in results]
        self.assertIn("m1", ids)
        self.assertIn("m2", ids)
        self.assertNotIn("m3", ids)

    def test_filter_by_task_name(self):
        results = search_methods(self.index, task_name="classification")
        ids = [e.memory_id for e in results]
        self.assertIn("m1", ids)
        self.assertNotIn("m2", ids)

    def test_filter_by_model_name(self):
        results = search_methods(self.index, model_name="plsr")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].memory_id, "m2")

    def test_filter_by_min_metric(self):
        results = search_methods(self.index, min_metric=0.90)
        ids = [e.memory_id for e in results]
        self.assertIn("m1", ids)
        self.assertNotIn("m2", ids)

    def test_unapproved_excluded_by_default(self):
        results = search_methods(self.index, modality="NIR", task_name="classification")
        ids = [e.memory_id for e in results]
        self.assertIn("m1", ids)
        self.assertNotIn("m4", ids)

    def test_empty_results_no_crash(self):
        empty = MethodMemoryIndex(entries=())
        results = search_methods(empty, modality="NIR")
        self.assertEqual(results, [])


class TestRecommendFromMemory(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        for mem_id, modality, metric in [("r1", "NIR", 0.90), ("r2", "NIR", 0.80), ("r3", "FTIR", 0.95)]:
            m = _mem(memory_id=mem_id, modality=modality, key_metrics={"accuracy": metric})
            save_method(m, self._tmp)
        self.index = rebuild_index(self._tmp)

    def test_ranks_by_metric_descending(self):
        dp = _dp(modality="NIR")
        results = recommend_from_memory(self.index, dp, top_k=3, memory_dir=self._tmp)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].memory_id, "r1")
        self.assertEqual(results[1].memory_id, "r2")

    def test_top_k_limits_results(self):
        dp = _dp(modality="NIR")
        results = recommend_from_memory(self.index, dp, top_k=1, memory_dir=self._tmp)
        self.assertEqual(len(results), 1)

    def test_modality_filter(self):
        dp = _dp(modality="FTIR")
        results = recommend_from_memory(self.index, dp, top_k=3, memory_dir=self._tmp)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].memory_id, "r3")

    def test_empty_dir_no_crash(self):
        empty_dir = tempfile.mkdtemp()
        empty_index = rebuild_index(empty_dir)
        dp = _dp()
        results = recommend_from_memory(empty_index, dp, memory_dir=empty_dir)
        self.assertEqual(results, [])


class TestRebuildIndex(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def test_discover_all_files(self):
        for i in range(3):
            save_method(_mem(memory_id=f"idx-{i}"), self._tmp)
        index = rebuild_index(self._tmp)
        ids = {e.memory_id for e in index.entries}
        self.assertEqual(ids, {"idx-0", "idx-1", "idx-2"})

    def test_index_file_written(self):
        save_method(_mem(), self._tmp)
        rebuild_index(self._tmp)
        index_path = Path(self._tmp) / "memory_index.json"
        self.assertTrue(index_path.exists())
        data = json.loads(index_path.read_text())
        self.assertIn("entries", data)

    def test_empty_directory(self):
        index = rebuild_index(self._tmp)
        self.assertEqual(list(index.entries), [])

    def test_nonexistent_directory(self):
        index = rebuild_index(Path(self._tmp) / "nonexistent")
        self.assertEqual(list(index.entries), [])


class TestToolWrappers(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def test_save_tool(self):
        mem = _mem()
        req = SaveMethodMemoryRequest(memory=mem)
        resp = save_method_memory.run(req, memory_dir=self._tmp)
        self.assertTrue(resp.ok)
        self.assertEqual(resp.payload["memory_id"], "mem-001")

    def test_search_tool(self):
        mem = _mem()
        save_method(mem, self._tmp)
        rebuild_index(self._tmp)
        req = SearchMethodMemoryRequest(modality="NIR")
        resp = search_method_memory.run(req, memory_dir=self._tmp)
        self.assertTrue(resp.ok)
        self.assertEqual(resp.payload["count"], 1)

    def test_recommend_tool(self):
        mem = _mem()
        save_method(mem, self._tmp)
        rebuild_index(self._tmp)
        dp = _dp()
        req = RecommendFromMemoryRequest(dataset_profile=dp)
        resp = rec_tool.run(req, memory_dir=self._tmp)
        self.assertTrue(resp.ok)
        self.assertGreaterEqual(resp.payload["count"], 1)


if __name__ == "__main__":
    unittest.main()
