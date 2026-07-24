"""MCP server for the scientist-facing project workflow.

Run with:
    python -m chemometrics_mcp.server

The server exposes only the current folder-to-report workflow. Tools execute
bounded application logic and never expose arbitrary Python or shell execution.
Artifact writes are contained by the project store and may be restricted with
``CHEMOMETRICS_ALLOWED_ROOTS``.
"""
from __future__ import annotations

import importlib.metadata
import json
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server
from mcp.server.lowlevel.server import NotificationOptions
from mcp.server.models import InitializationOptions

from chemometrics_mcp import mcp

server = Server("chemometrics-mcp")


def _server_version() -> str:
    try:
        return importlib.metadata.version("agentic-chemometrician")
    except importlib.metadata.PackageNotFoundError:
        return "0.2.0"


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Return the current, schema-generated chemometrics tools."""
    return [
        types.Tool(
            name=item["name"],
            description=item["description"],
            inputSchema=item["inputSchema"],
        )
        for item in mcp.tool_definitions()
    ]


@server.call_tool()
async def call_tool(
    name: str, arguments: dict[str, Any]
) -> list[types.TextContent]:
    """Validate and dispatch one current workflow tool."""
    response = mcp.dispatch(name, arguments)
    return [
        types.TextContent(
            type="text",
            text=json.dumps(response, indent=2, default=str),
        )
    ]


async def main() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="chemometrics-mcp",
                server_version=_server_version(),
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
