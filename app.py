import os
# Disable Streamlit’s file watcher to avoid torch errors
os.environ["STREAMLIT_SERVER_FILEWATCHER_TYPE"] = "none"

import nest_asyncio
nest_asyncio.apply()

import streamlit as st
# Load Gemini API key from Streamlit secrets
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

"""
NoEncode RAG with MCP — A new FedRAG core abstraction

This Streamlit app configures an MCP stdio server, retrieves relevant knowledge
via a NoEncodeKnowledgeStore, and then uses a generator model to produce
 a final answer—demonstrating the full NoEncode RAG workflow with Google Gemini 2.0 Flash.

Requirements:
- streamlit
- mcp[cli]
- fed-rag
- sentence-transformers
- torch
- transformers
"""

import asyncio
import logging

from fed_rag.knowledge_stores.no_encode import MCPKnowledgeStore, MCPStdioKnowledgeSource
from fed_rag import AsyncNoEncodeRAGSystem, RAGConfig
from fed_rag.generators import HFPretrainedModelGenerator
from fed_rag.data_structures import KnowledgeNode
from mcp import StdioServerParameters

# Set up logging to sidebar
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    st.title("NoEncode RAG with MCP + Gemini 2.0 Flash Demo")
    st.markdown(
        "Configure your MCP stdio server below, enter a question, and click **Retrieve** "
        "to get both retrieved contexts and a generated answer using Gemini 2.0 Flash."
    )

    # Sidebar: MCP server inputs
    st.sidebar.header("MCP Server Configuration")
    command    = st.sidebar.text_input("Command",                value="python3")
    args_str   = st.sidebar.text_input("Args (comma-separated)", value="my_awesome_mcp_server.py")
    tool_name  = st.sidebar.text_input("Tool Name",              value="KnowledgeTool")
    query_param= st.sidebar.text_input("Query Param Name",       value="query")

    # Sidebar: Logging area
    log_container = st.sidebar.empty()
    log_lines: list[str] = []
    def log(msg: str):
        log_lines.append(msg)
        log_container.text("\n".join(log_lines))
        logger.info(msg)

    # Main: user enters question
    query_text = st.text_input("Natural Language Question", value="What is MCP?")

    if st.button("Retrieve"):
        args = [a.strip() for a in args_str.split(",") if a.strip()]

        async def run_noencode_rag():
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

            log("📚 Building NoEncodeKnowledgeStore")
            ks = MCPKnowledgeStore().add_source(mcp_source)

            log("⏳ Retrieving knowledge nodes…")
            nodes = await ks.retrieve(query_text)

            log("🚀 Running generator (Gemini 2.0 Flash) on retrieved nodes…")
            generator = HFPretrainedModelGenerator(
                model_name="google/gemini-2.0-flash"
            )
            rag_system = AsyncNoEncodeRAGSystem(
                knowledge_store=ks,
                generator=generator,
                rag_config=RAGConfig(top_k=2),
            )
            response = await rag_system.query(query_text)
            return nodes, response

        try:
            with st.spinner("Running NoEncode RAG pipeline…"):
                loop = asyncio.get_event_loop()
                nodes, response = loop.run_until_complete(run_noencode_rag())

            log(f"✅ Retrieved {len(nodes)} node(s) and generated answer")

            st.subheader("Generated Answer")
            answer = getattr(response, "result", None) or getattr(response, "answer", None) or response
            st.write(answer)

            if nodes:
                st.subheader("Retrieved Contexts")
                for i, item in enumerate(nodes, start=1):
                    if isinstance(item, tuple) and len(item)==2 and isinstance(item[1], KnowledgeNode):
                        score, node = item
                    else:
                        node  = item
                        score = getattr(node, "score", "N/A")
                    st.markdown(f"**Node {i}** (Score: {score})")
                    st.write(node.text_content)
            else:
                st.info("No contexts were retrieved from MCP.")

        except Exception as e:
            log(f"❌ Error: {e}")
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
