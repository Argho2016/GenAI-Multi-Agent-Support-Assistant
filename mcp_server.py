import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import glob
from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path=".langchain_llm_cache.db"))

from agents.policy_rag import ingest_pdfs, answer_policy
from agents.sql_agent import answer_customer
from agents.graph import build_graph

load_dotenv()

mcp = FastMCP("genai-multiagent-support")

graph = build_graph()


@mcp.tool()
def ingest_policy_pdfs(directory: str = "policy_uploads") -> dict:
    """
    Ingest all PDFs in a directory into the vector database.
    """
    os.makedirs(directory, exist_ok=True)
    pdfs = glob.glob(os.path.join(directory, "*.pdf"))
    if not pdfs:
        return {"ok": False, "message": f"No PDFs found in {directory}"}
    stats = ingest_pdfs(pdfs)
    return {"ok": True, "stats": stats}


@mcp.tool()
def ask_policy(question: str) -> dict:
    """
    Answer a policy question using RAG over ingested PDFs.
    """
    ans = answer_policy(question)
    return {"answer": ans.answer, "sources": ans.sources}


@mcp.tool()
def ask_customer(question: str) -> dict:
    """
    Answer a customer/ticket question using the SQL agent.
    """
    ans = answer_customer(question)
    return {"answer": ans.answer, "sql": ans.sql, "rows": ans.rows}


@mcp.tool()
def ask_router(question: str) -> dict:
    """
    Main multi-agent entry: routes to POLICY / SQL / BOTH and returns final answer.
    """
    state = {"user_question": question}
    out = graph.invoke(state)
    return {"final_answer": out["final_answer"], "route": out["route"]}


if __name__ == "__main__":
    # Easiest demo transport: stdio (works with MCP inspector / desktop hosts)
    mcp.run(transport="stdio")
