# NOENCODE\_RAG\_PROJECT

**Empowering Rapid Knowledge Access Through Seamless AI Integration**

---

<p align="center">
  <img src="https://img.shields.io/badge/last%20commit-today-blue" alt="last commit">
  <img src="https://img.shields.io/badge/python-100%25-brightgreen" alt="python coverage">
  <img src="https://img.shields.io/badge/languages-1-lightgrey" alt="languages">
</p>

## Built With

* **Markdown** • **Streamlit** • **FastAPI** • **Python** • **Google Gemini**
* **Federated RAG (FedRAG)** • **Model Context Protocol (MCP)** • **nest\_asyncio**

---

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Architecture](#architecture)
* [Installation](#installation)
* [Usage](#usage)
* [Server Examples](#server-examples)
* [Requirements](#requirements)
* [Limitations & Extensions](#limitations--extensions)
* [License](#license)

---

## Overview

This project demonstrates **NoEncode Retrieval-Augmented Generation (RAG)** powered by Anthropic’s Model Context Protocol (MCP) and Google Gemini 2.0 Flash. Unlike traditional RAG systems, NoEncode RAG retrieves natural-language context directly—no embedding or retriever model required—then generates answers using an LLM.

![Project Architecture](./assets/architecture.png)

---

## Features

* **NoEncode Retrieval:** Query any tool or API using plain text via MCP.
* **Seamless AI Generation:** Generate concise answers with Google Gemini 2.0 Flash.
* **Interactive UI:** Developed with Streamlit for quick prototyping.
* **Extensible Server:** Plug in any external system (DB, API, document store) as an MCP tool.

---

## Architecture

1. **MCP Stdio Server**: Implements JSON-RPC tools over stdin/stdout (e.g., via FastMCP).
2. **MCPKnowledgeSource**: FedRAG client that sends plain-text queries to the server.
3. **NoEncodeKnowledgeStore**: Aggregates one or more MCP sources—no retriever step.
4. **Generator LLM**: Passes contexts + query to Gemini for final answer.

---

## Installation

1. **Clone this repository**

   ```bash
   git clone https://github.com/yourusername/NoEncode_RAG_Project.git
   cd NoEncode_RAG_Project
   ```

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   .venv\Scripts\Activate.ps1  # Windows PowerShell
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure secrets**
   Create `.streamlit/secrets.toml` with:

   ```toml
   GEMINI_API_KEY = "<your_gemini_api_key>"
   ```

---

## Usage

1. **Run the MCP server** (dummy or your own):

   ```bash
   python3 my_awesome_mcp_server.py
   ```

2. **Start the Streamlit UI**

   ```bash
   streamlit run app.py --server.fileWatcherType none
   ```

3. **Interact**

   * Enter your question in the **Demo** tab.
   * View retrieved contexts and generated answers.
   * Explore how it works in the **How it works** tab.

---

## Server Examples

### Dummy Server (FAQ Only)

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP('DemoServer')

@mcp.tool(name='KnowledgeTool')
def knowledge_tool(query: str) -> list[str]:
    q = query.strip().lower()
    if "what is mcp" in q:
        return [
            "MCP (Model Context Protocol) is Anthropic’s open standard "
            "for invoking external tools via JSON-RPC..."
        ]
    return ["Error: Unsupported query. Only 'what is mcp' is handled."]

if __name__ == '__main__':
    mcp.run()
```

### Production-Ready Example (Database)

```python
from mcp.server.fastmcp import FastMCP
import psycopg2

mcp = FastMCP('DatabaseServer')

@mcp.tool(name='CustomerLookup')
def customer_lookup(query: str) -> list[str]:
    conn = psycopg2.connect("dbname=crm user=admin")
    cur = conn.cursor()
    cur.execute("SELECT notes FROM customers WHERE id = %s", (query,))
    return [row[0] for row in cur.fetchall()]

if __name__ == '__main__':
    mcp.run()
```

---

## Requirements

```text
streamlit
nest_asyncio
mcp[cli]
fed-rag
google-generativeai
torch
```

---

## Limitations & Extensions

* **Demo Limitation:** Only the `what is mcp` query is supported in the dummy server.
* **Extendability:** Add more tools, integrate REST APIs, document stores, or knowledge graphs.
* **Fine-Tuning:** Plug in FedRAG’s generator trainers to customize to your domain.

---

## License

MIT © 2025 Fahmi Zainal
