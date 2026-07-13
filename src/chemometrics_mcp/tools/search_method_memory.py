"""MCP tool: search_method_memory"""
from __future__ import annotations

import json
from pathlib import Path

from chemometrics_contracts import (
    MethodMemoryIndex,
    SearchMethodMemoryRequest,
    ToolResponse,
)

from chemometrics_mcp.core.method_memory import _index_path, search_methods


def run(
    request: SearchMethodMemoryRequest,
    *,
    memory_dir: str | Path = "agent-memory/methods",
) -> ToolResponse[dict]:
    index_path = _index_path(memory_dir)
    if index_path.exists():
        raw = json.loads(index_path.read_text(encoding="utf-8"))
        from chemometrics_contracts import MethodMemoryEntry
        entries = tuple(
            MethodMemoryEntry(**e) for e in raw.get("entries", [])
        )
        index = MethodMemoryIndex(entries=entries)
    else:
        index = MethodMemoryIndex(entries=())

    results = search_methods(
        index,
        modality=request.modality,
        task_name=request.task_name,
        model_name=request.model_name,
        min_metric=request.min_metric,
        approval_status=request.approval_status,
    )
    return ToolResponse(
        tool_name="search_method_memory",
        ok=True,
        payload={"results": [e.to_dict() for e in results], "count": len(results)},
        message=f"Found {len(results)} method memory entries.",
    )
