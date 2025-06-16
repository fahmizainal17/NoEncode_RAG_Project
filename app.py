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

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    st.title("NoEncode RAG + Gemini 2.0 Flash Demo")

    # Sidebar with app description and requirements
    st.sidebar.title("NoEncode RAG + Gemini 2.0 Flash Demo")
    st.sidebar.markdown(
        "This Streamlit app demonstrates a NoEncode RAG workflow:\n"
        "1. Retrieve context using MCP (Model Context Protocol)\n"
        "2. Generate a final answer with Google Gemini 2.0 Flash.\n\n"
        "**Requirements:**\n"
        "- streamlit\n"
        "- nest_asyncio\n"
        "- mcp[cli]\n"
        "- fed-rag\n"
        "- google-generativeai\n"
        "- torch"
    )

    # Two tabs: Demo and Explanation
    demo_tab, explain_tab = st.tabs(["🚀 Demo", "📖 How it works"])

    with demo_tab:
        st.header("Run NoEncode RAG Pipeline")
        query_text = st.text_input("Enter your question:", value="What is MCP?")
        log_container = st.empty()
        log_msgs = []
        def log(msg: str):
            log_msgs.append(msg)
            log_container.text("\n".join(log_msgs))
            logger.info(msg)

        if st.button("Retrieve", key="run_demo"):
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

    with explain_tab:
        st.header("How NoEncode RAG Works")
        st.markdown(
            "### Workflow Overview\n"
            "1. **MCP Stdio Server**: A FastMCP server listens on stdin/stdout and registers tools you define.\n"
            "2. **MCPKnowledgeSource**: FedRAG creates a client to send natural-language queries to your MCP server and receives plain-text results.\n"
            "3. **NoEncodeKnowledgeStore**: Abstracts one or more MCP sources so you don’t need embeddings or retrievers; retrieval works on natural queries.\n"
            "4. **Generator LLM**: Passes retrieved contexts plus the query to an LLM (Gemini 2.0 Flash) for final answer generation.\n\n"
            "**Demo Limitation:** The provided `my_awesome_mcp_server.py` only handles the `what is mcp` query and returns a fixed definition. You can only ask this specific question unless you extend the server logic.

**Dummy Server Example (only supports `what is mcp`):**
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP('DemoServer')

@mcp.tool(name='KnowledgeTool')
def knowledge_tool(query: str) -> list[str]:
    q = query.strip().lower()
    if "what is mcp" in q:
        return [
            "MCP (Model Context Protocol) is Anthropic’s open standard "
            "for invoking external tools via JSON-RPC over stdin/stdout or HTTP. "
            "It defines how an LLM can call out to \"tools\" and receive structured responses."
        ]
    # For any other query, return an empty or error message
    return ["Error: Unsupported query. Only 'what is mcp' is currently handled."]
```

### Server Example\n"
            "FedRAG’s NoEncode source can be any tool or API you wrap as an MCP tool. Examples include:\n"
            "- **SQL/NoSQL Database**: Query records directly (e.g., PostgreSQL, MongoDB).\n"
            "- **REST/GraphQL API**: Fetch live data (e.g., weather, finance, internal services).\n"
            "- **Document Store**: Search S3 or Google Drive documents.\n"
            "- **Knowledge Graph**: Run SPARQL against RDF or Cypher on Neo4j.\n"
            "- **Search Engine**: Wrap Elasticsearch or Algolia.\n"
            "- **Custom Python Function**: Integrate any SDK (Salesforce, HubSpot, etc.).\n\n"
            "```python\n"
            "# Example: Database-backed MCP server\n"
            "from mcp.server.fastmcp import FastMCP\n"
            "import psycopg2\n\n"
            "mcp = FastMCP('DatabaseServer')\n\n"
            "@mcp.tool(name='CustomerLookup')\n"
            "def customer_lookup(query: str) -> list[str]:\n"
            "    conn = psycopg2.connect(\"dbname=crm user=admin\")\n"
            "    cur = conn.cursor()\n"
            "    cur.execute(\"SELECT notes FROM customers WHERE id = %s\", (query,))\n"
            "    return [row[0] for row in cur.fetchall()]\n\n"
            "if __name__ == '__main__':\n"
            "    mcp.run()\n"
            "```\n"
            "Use this pattern to connect **any** external system as a NoEncode knowledge source."
        )

if __name__ == '__main__':
    main()
