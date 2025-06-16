# my_awesome_mcp_server.py
from mcp.server.fastmcp import FastMCP

# Name your server anything you like
mcp = FastMCP("DemoServer")

# Expose a “KnowledgeTool” that just echoes back the query
@mcp.tool(name="KnowledgeTool")
def knowledge_tool(query: str) -> list[str]:
    """Return a single-item list echoing the query."""
    return [f"Echo from MCP: {query}"]

if __name__ == "__main__":
    # This will listen on stdin/stdout
    mcp.run()
