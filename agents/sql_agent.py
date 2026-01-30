from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from .prompts import SQL_SYSTEM


DB_PATH = "data/customers.db"


@dataclass
class SQLAnswer:
    answer: str
    sql: str
    rows: List[Dict[str, Any]]


def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
    )


def _get_schema(conn: sqlite3.Connection) -> str:
    cur = conn.cursor()
    tables = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
    ).fetchall()
    schema_lines = []
    for (t,) in tables:
        cols = cur.execute(f"PRAGMA table_info({t});").fetchall()
        col_str = ", ".join([f"{c[1]} {c[2]}" for c in cols])
        schema_lines.append(f"{t}({col_str})")
    return "\n".join(schema_lines)


def _safe_sql(sql: str) -> str:
    s = sql.strip().lower()
    if not s.startswith("select"):
        raise ValueError("Only SELECT statements are allowed.")
    banned = ["insert", "update", "delete", "drop", "alter", "create", "pragma", "attach", "detach"]
    if any(b in s for b in banned):
        raise ValueError("Unsafe SQL detected.")
    return sql


def _run_query(conn: sqlite3.Connection, sql: str) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return [dict(zip(cols, r)) for r in rows]


def answer_customer(question: str) -> SQLAnswer:
    conn = sqlite3.connect(DB_PATH)
    schema = _get_schema(conn)
    llm = _get_llm()

    messages = [
        SystemMessage(content=SQL_SYSTEM),
        HumanMessage(content=f"User question: {question}\n\nDB schema:\n{schema}"),
    ]

    raw = llm.invoke(messages).content
    try:
        parsed = json.loads(raw)
        sql = parsed["sql"]
    except Exception:
        raise ValueError(f"Model did not return valid JSON. Got: {raw}")

    sql = _safe_sql(sql)

    # Force output limits if user didn't specify
    if "limit" not in sql.lower():
        sql = sql.rstrip(";") + " LIMIT 50;"

    rows = _run_query(conn, sql)

    # Summarize rows
    summary_messages = [
        SystemMessage(content="You summarize SQL results for a customer support executive. Be concise, factual."),
        HumanMessage(content=f"Question: {question}\nSQL: {sql}\nRows (JSON): {rows}"),
    ]
    answer = llm.invoke(summary_messages).content

    conn.close()
    return SQLAnswer(answer=answer, sql=sql, rows=rows)
