"""Method memory: persistent storage and retrieval of reviewed analysis recipes."""
from __future__ import annotations

import json
from pathlib import Path

from chemometrics_contracts import (
    DatasetProfile,
    MethodMemory,
    MethodMemoryEntry,
    MethodMemoryIndex,
)

_DEFAULT_MEMORY_DIR = "agent-memory/methods"
_INDEX_FILENAME = "memory_index.json"


def _memory_dir_path(memory_dir: str | Path) -> Path:
    p = Path(memory_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _memory_file(memory_dir: str | Path, memory_id: str) -> Path:
    return Path(memory_dir) / f"{memory_id}.json"


def _index_path(memory_dir: str | Path) -> Path:
    return Path(memory_dir) / _INDEX_FILENAME


def _entry_from_memory(mem: MethodMemory) -> MethodMemoryEntry:
    key_metric_value = None
    for v in mem.key_metrics.values():
        if isinstance(v, (int, float)):
            key_metric_value = float(v)
            break
    return MethodMemoryEntry(
        memory_id=mem.memory_id,
        modality=mem.modality,
        task_name=mem.task_name,
        model_name=mem.model_name,
        preprocessing=mem.preprocessing,
        key_metric_value=key_metric_value,
        approval_status=mem.approval_status,
        created_at=mem.created_at,
    )


def save_method(
    memory: MethodMemory,
    memory_dir: str | Path = _DEFAULT_MEMORY_DIR,
) -> Path:
    dir_path = _memory_dir_path(memory_dir)
    file_path = _memory_file(dir_path, memory.memory_id)
    file_path.write_text(
        json.dumps(memory.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    return file_path


def load_method(
    memory_id: str,
    memory_dir: str | Path = _DEFAULT_MEMORY_DIR,
) -> MethodMemory:
    file_path = _memory_file(memory_dir, memory_id)
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    dp_raw = raw["dataset_profile"]
    dp = DatasetProfile(
        modality=dp_raw["modality"],
        n_samples=dp_raw["n_samples"],
        n_features=dp_raw["n_features"],
        n_classes=dp_raw.get("n_classes"),
        axis_min=dp_raw.get("axis_min"),
        axis_max=dp_raw.get("axis_max"),
        label_column=dp_raw.get("label_column"),
    )
    return MethodMemory(
        memory_id=raw["memory_id"],
        created_at=raw["created_at"],
        modality=raw["modality"],
        task_name=raw["task_name"],
        dataset_profile=dp,
        preprocessing=raw["preprocessing"],
        model_name=raw["model_name"],
        validation_strategy=raw["validation_strategy"],
        key_metrics=dict(raw.get("key_metrics", {})),
        caveats=tuple(raw.get("caveats", [])),
        reviewer_notes=raw.get("reviewer_notes"),
        source_run_id=raw.get("source_run_id", ""),
        approval_status=raw.get("approval_status", "approved"),
    )


def search_methods(
    index: MethodMemoryIndex,
    *,
    modality: str | None = None,
    task_name: str | None = None,
    model_name: str | None = None,
    min_metric: float | None = None,
    approval_status: str = "approved",
) -> list[MethodMemoryEntry]:
    results: list[MethodMemoryEntry] = []
    for entry in index.entries:
        if entry.approval_status != approval_status:
            continue
        if modality is not None and entry.modality != modality:
            continue
        if task_name is not None and entry.task_name != task_name:
            continue
        if model_name is not None and entry.model_name != model_name:
            continue
        if min_metric is not None and entry.key_metric_value is not None:
            if entry.key_metric_value < min_metric:
                continue
        results.append(entry)
    return results


def recommend_from_memory(
    index: MethodMemoryIndex,
    dataset_profile: DatasetProfile,
    *,
    top_k: int = 3,
    memory_dir: str | Path = _DEFAULT_MEMORY_DIR,
) -> list[MethodMemoryEntry]:
    approved = [
        e for e in index.entries if e.approval_status == "approved"
    ]
    matched = [
        e for e in approved
        if e.modality == dataset_profile.modality
    ]

    full_entries = _load_full_entries_for_ids(
        {e.memory_id for e in matched},
        memory_dir,
    )

    def _sort_key(entry: MethodMemoryEntry) -> tuple:
        full = full_entries.get(entry.memory_id)
        if full is not None:
            dp = full.dataset_profile
            samples_diff = abs(dp.n_samples - dataset_profile.n_samples)
            features_diff = abs(dp.n_features - dataset_profile.n_features)
        else:
            samples_diff = 0
            features_diff = 0
        metric_val = entry.key_metric_value if entry.key_metric_value is not None else -1e9
        return (samples_diff + features_diff, -metric_val)

    matched.sort(key=_sort_key)
    return matched[:top_k]


def _load_full_entries_for_ids(
    memory_ids: set[str],
    memory_dir: str | Path,
) -> dict[str, MethodMemory]:
    result: dict[str, MethodMemory] = {}
    for mid in memory_ids:
        try:
            result[mid] = load_method(mid, memory_dir)
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
            continue
    return result


def rebuild_index(
    memory_dir: str | Path = _DEFAULT_MEMORY_DIR,
) -> MethodMemoryIndex:
    dir_path = Path(memory_dir)
    entries: list[MethodMemoryEntry] = []
    if not dir_path.exists():
        return MethodMemoryIndex(entries=tuple(entries))
    for f in sorted(dir_path.glob("*.json")):
        if f.name == _INDEX_FILENAME:
            continue
        try:
            mem = load_method(f.stem, memory_dir)
            entries.append(_entry_from_memory(mem))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    index_path = _index_path(dir_path)
    index_obj = MethodMemoryIndex(entries=tuple(entries))
    index_path.write_text(
        json.dumps(index_obj.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    return index_obj
