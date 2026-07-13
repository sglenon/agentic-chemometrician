"""MCP tool: recommend_from_memory"""
from __future__ import annotations

import json
from pathlib import Path

from chemometrics_contracts import (
    DatasetProfile,
    MethodMemoryEntry,
    MethodMemoryIndex,
    RecommendFromMemoryRequest,
    ToolResponse,
)

from chemometrics_mcp.core.method_memory import _index_path, recommend_from_memory


def run(
    request: RecommendFromMemoryRequest,
    *,
    memory_dir: str | Path = "agent-memory/methods",
) -> ToolResponse[dict]:
    index_path = _index_path(memory_dir)
    if index_path.exists():
        raw = json.loads(index_path.read_text(encoding="utf-8"))
        entries = tuple(
            MethodMemoryEntry(**e) for e in raw.get("entries", [])
        )
        index = MethodMemoryIndex(entries=entries)
    else:
        index = MethodMemoryIndex(entries=())

    results = recommend_from_memory(
        index,
        request.dataset_profile,
        top_k=request.top_k,
        memory_dir=memory_dir,
    )
    return ToolResponse(
        tool_name="recommend_from_memory",
        ok=True,
        payload={"recommendations": [e.to_dict() for e in results], "count": len(results)},
        message=f"Recommended {len(results)} methods from memory.",
    )
