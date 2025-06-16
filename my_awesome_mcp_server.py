#!/usr/bin/env python3
"""
my_awesome_mcp_server.py

A minimal MCP stdio server powered by the MCP Python SDK (FastMCP).
It implements a single tool, "KnowledgeTool", that returns:
- A real definition when you ask "what is mcp"
- Otherwise an echo of your query
"""

from mcp.server.fastmcp import FastMCP

# 1) Create your server host
mcp = FastMCP(server_name="DemoMCP")

# 2) Register your tool
@mcp.tool(name="KnowledgeTool")
def knowledge_tool(query: str) -> list[str]:
    """
    If the user asks 'what is mcp', return a hard-coded definition.
    Otherwise, echo back the original query.
    """
    q = query.strip().lower()
    if "what is mcp" in q:
        return [
            "MCP (Model Context Protocol) is Anthropic’s open standard "
            "for invoking external tools via JSON-RPC over stdin/stdout or HTTP. "
            "It defines how an LLM can call out to “tools” (e.g. knowledge stores) "
            "and receive structured responses."
        ]
    # Fallback for any other query
    return ["Error: Unsupported query. Only 'what is mcp' is currently handled."]

# 3) Run the server loop (handles JSON-RPC, TaskGroups, etc.)
if __name__ == "__main__":
    mcp.run()
