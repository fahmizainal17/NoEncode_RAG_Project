import os

# ─── Disable Streamlit’s file watcher to avoid scanning torch._classes ───
os.environ["STREAMLIT_SERVER_FILEWATCHER_TYPE"] = "none"

# ─── Allow nested asyncio loops inside Streamlit ───
import nest_asyncio
nest_asyncio.apply()

"""
NoEncode RAG with MCP — A new FedRAG core abstraction

This Streamlit demo app shows how to configure an MCP stdio server, perform natural language queries
against a NoEncode knowledge store, and includes fixes for asyncio event loop handling and file watcher.
"""

import streamlit as st
import asyncio
from fed_rag.knowledge_stores.no_encode import MCPKnowledgeStore, MCPStdioKnowledgeSource
from fed_rag.data_structures import KnowledgeNode
from mcp import StdioServerParameters


def main():
    st.title("NoEncode RAG with MCP Demo")
    st.markdown(
        "Enter your MCP stdio server parameters below, then type a natural language query "
        "and click **Retrieve** to fetch knowledge nodes."
    )

    # Sidebar inputs for configuring the MCP server and tool
    st.sidebar.header("MCP Server Configuration")
    command = st.sidebar.text_input("Command", value="python")
    args_str = st.sidebar.text_input("Args (comma-separated)", value="my_awesome_mcp_server.py")
    tool_name = st.sidebar.text_input("Tool Name", value="KnowledgeTool")
    query_param = st.sidebar.text_input("Query Param Name", value="query")

    # Main input for the query
    query_text = st.text_input("Natural Language Query", value="What is MCP?")

    if st.button("Retrieve"):
        args = [arg.strip() for arg in args_str.split(",")]

        async def retrieve_nodes():
            server_params = StdioServerParameters(command=command, args=args)
            mcp_source = MCPStdioKnowledgeSource(
                name="mcp-source",
                server_params=server_params,
                tool_name=tool_name,
                query_param_name=query_param,
            )
            knowledge_store = MCPKnowledgeStore().add_source(mcp_source)
            return await knowledge_store.retrieve(query_text)

        try:
            # use the existing event loop rather than asyncio.run()
            loop = asyncio.get_event_loop()
            nodes = loop.run_until_complete(retrieve_nodes())

            if nodes:
                for i, node in enumerate(nodes, start=1):
                    st.subheader(f"Node {i}")
                    st.write(f"**Score:** {getattr(node, 'score', 'N/A')}")
                    st.write(f"**Source:** {node.metadata.get('name', 'unknown')}")
                    st.write(node.text_content)
            else:
                st.info("No nodes were retrieved for that query.")
        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
