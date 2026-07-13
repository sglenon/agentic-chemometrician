"""MCP tool: save_method_memory"""
from __future__ import annotations

from pathlib import Path

from chemometrics_contracts import (
    SaveMethodMemoryRequest,
    ToolResponse,
)

from chemometrics_mcp.core.method_memory import rebuild_index, save_method


def run(
    request: SaveMethodMemoryRequest,
    *,
    memory_dir: str | Path = "agent-memory/methods",
) -> ToolResponse[dict]:
    file_path = save_method(request.memory, memory_dir)
    rebuild_index(memory_dir)
    return ToolResponse(
        tool_name="save_method_memory",
        ok=True,
        payload={"memory_id": request.memory.memory_id, "path": str(file_path)},
        message=f"Method memory {request.memory.memory_id!r} saved and index rebuilt.",
    )
