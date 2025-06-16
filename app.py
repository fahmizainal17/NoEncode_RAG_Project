"""
NoEncode RAG with MCP + Gemini 2.0 Flash Demo

This Streamlit app configures an MCP stdio server, performs NoEncode retrieval,
then uses Google Gemini 2.0 Flash to generate the final answer.

Requirements:
- streamlit
- nest_asyncio
- mcp[cli]
- fed-rag[huggingface]
- google-generativeai
- torch
"""

import os
# Disable Streamlit’s file watcher to avoid torch errors
os.environ["STREAMLIT_SERVER_FILEWATCHER_TYPE"] = "none"

import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import asyncio
import logging

# Attempt to import Google Generative AI SDK
try:
    import google.generativeai as genai
except ImportError:
    st.error("`google.generativeai` module not found. Run `pip install google-generativeai`")
    raise

# Load Gemini API key from Streamlit secrets
gemini_key = st.secrets.get("GEMINI_API_KEY", "")
if not gemini_key:
    st.error("GEMINI_API_KEY missing in secrets.toml.")
    st.stop()

# Configure Gemini
os.environ["GEMINI_API_KEY"] = gemini_key
genai.configure(api_key=gemini_key)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

from fed_rag.knowledge_stores.no_encode import MCPKnowledgeStore, MCPStdioKnowledgeSource
from fed_rag.data_structures import KnowledgeNode
from mcp import StdioServerParameters

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    st.title("NoEncode RAG + Gemini 2.0 Flash Demo")

    # Sidebar description and requirements
    st.sidebar.title("NoEncode RAG + Gemini 2.0 Flash Demo")
    st.sidebar.markdown(
        "This Streamlit app configures an MCP stdio server, performs NoEncode retrieval,"
        " then uses Google Gemini 2.0 Flash to generate the final answer.\n\n"
        "**Requirements:**\n"
        "- streamlit\n"
        "- nest_asyncio\n"
        "- mcp[cli]\n"
        "- fed-rag[huggingface]\n"
        "- google-generativeai\n"
        "- torch"
    )

    # Sidebar log area
    log_container = st.sidebar.empty()
    log_msgs: list[str] = []
    def log(msg: str):
        log_msgs.append(msg)
        log_container.text("\n".join(log_msgs))
        logger.info(msg)

    # Main input for user question
    query_text = st.text_input("Enter your question:", value="What is MCP?")

    if st.button("Retrieve"):
        # Fixed MCP server parameters
        command = "python3"
        args = ["my_awesome_mcp_server.py"]
        tool_name = "KnowledgeTool"
        query_param = "query"

        async def pipeline():
            log("📦 Setting up MCP parameters...")
            params = StdioServerParameters(command=command, args=args)
            log(f"⚙️ Command: {command} {' '.join(args)}")

            log("🔧 Creating MCPStdioKnowledgeSource...")
            source = MCPStdioKnowledgeSource(
                name="mcp", server_params=params,
                tool_name=tool_name, query_param_name=query_param
            )

            log("📚 Building knowledge store...")
            store = MCPKnowledgeStore().add_source(source)

            log("⏳ Retrieving contexts...")
            nodes = await store.retrieve(query_text)

            # Combine context texts
            contexts = "\n---\n".join(
                (item[1].text_content if isinstance(item, tuple) else item.text_content)
                for item in nodes
            )

            log("🚀 Generating answer with Gemini Flash...")
            prompt_text = f"Context:\n{contexts}\n\nQuestion: {query_text}"
            gen = gemini_model.generate_content(prompt_text)
            return nodes, gen.text

        try:
            with st.spinner("Running pipeline…"):
                loop = asyncio.get_event_loop()
                nodes, answer = loop.run_until_complete(pipeline())

            log(f"✅ Retrieved {len(nodes)} contexts and generated answer.")

            st.subheader("Generated Answer")
            st.write(answer)

            if nodes:
                st.subheader("Retrieved Contexts")
                for i, item in enumerate(nodes, start=1):
                    node = item[1] if isinstance(item, tuple) else item
                    score = item[0] if isinstance(item, tuple) else getattr(item, 'score', 'N/A')
                    st.markdown(f"**Context {i}** (Score: {score})")
                    st.write(node.text_content)
            else:
                st.info("No contexts retrieved.")

        except Exception as e:
            log(f"❌ Pipeline error: {e}")
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
