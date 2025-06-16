import os
os.environ["STREAMLIT_SERVER_FILEWATCHER_TYPE"] = "none"

import nest_asyncio
nest_asyncio.apply()

"""
NoEncode RAG with MCP — A new FedRAG core abstraction

This Streamlit demo app shows how to configure an MCP stdio server, perform natural language queries
against a NoEncode knowledge store, includes fixes for asyncio event loop handling and file watcher,
provides a loading spinner and logging, and now handles tuple‐wrapped nodes when needed.
"""

import streamlit as st
import asyncio
import logging

from fed_rag.knowledge_stores.no_encode import MCPKnowledgeStore, MCPStdioKnowledgeSource
from fed_rag.data_structures import KnowledgeNode
from mcp import StdioServerParameters

# Configure root logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    st.title("NoEncode RAG with MCP Demo")
    st.markdown(
        "Enter your MCP stdio server parameters below, then type a natural language query "
        "and click **Retrieve** to fetch knowledge nodes."
    )

    # Sidebar: MCP server inputs
    st.sidebar.header("MCP Server Configuration")
    command = st.sidebar.text_input("Command", value="python3")
    args_str = st.sidebar.text_input("Args (comma-separated)", value="my_awesome_mcp_server.py")
    tool_name = st.sidebar.text_input("Tool Name", value="KnowledgeTool")
    query_param = st.sidebar.text_input("Query Param Name", value="query")

    # Sidebar: Logs area
    log_container = st.sidebar.empty()
    log_lines: list[str] = []

    def log(msg: str):
        """Append a message to the sidebar log."""
        log_lines.append(msg)
        log_container.text("\n".join(log_lines))
        logger.info(msg)

    # Main: user query
    query_text = st.text_input("Natural Language Query", value="What is MCP?")

    if st.button("Retrieve"):
        args = [arg.strip() for arg in args_str.split(",") if arg.strip()]

        async def retrieve_nodes():
            log("📦 Setting up StdioServerParameters")
            server_params = StdioServerParameters(command=command, args=args)
            log(f"⚙️ Command: {command} {' '.join(args)}")
            log("🔧 Creating MCPStdioKnowledgeSource")
            mcp_source = MCPStdioKnowledgeSource(
                name="mcp-source",
                server_params=server_params,
                tool_name=tool_name,
                query_param_name=query_param,
            )
            log("📚 Building MCPKnowledgeStore")
            knowledge_store = MCPKnowledgeStore().add_source(mcp_source)
            log("⏳ Retrieving nodes from MCP…")
            return await knowledge_store.retrieve(query_text)

        try:
            with st.spinner("Retrieving knowledge nodes…"):
                loop = asyncio.get_event_loop()
                raw_nodes = loop.run_until_complete(retrieve_nodes())

            count = len(raw_nodes)
            log(f"✅ Retrieved {count} node(s)")

            if count:
                for i, item in enumerate(raw_nodes, start=1):
                    # handle tuple-wrapped nodes (score, KnowledgeNode)
                    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], KnowledgeNode):
                        score, node = item
                    else:
                        node = item  # assume it's a KnowledgeNode
                        score = getattr(node, "score", "N/A")

                    st.subheader(f"Node {i}")
                    st.write(f"**Score:** {score}")
                    st.write(f"**Source:** {node.metadata.get('name', 'unknown')}")
                    st.write(node.text_content)
            else:
                log("ℹ️ No nodes returned for that query.")
                st.info("No nodes were retrieved for that query.")

        except Exception as e:
            log(f"❌ Error: {e}")
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
