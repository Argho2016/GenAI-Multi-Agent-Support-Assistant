from __future__ import annotations

import json
from typing import TypedDict, Optional, Literal, Any, Dict

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from .prompts import ROUTER_SYSTEM
from .policy_rag import answer_policy
from .sql_agent import answer_customer


class GraphState(TypedDict, total=False):
    user_question: str
    route: Literal["POLICY", "SQL", "BOTH"]
    policy_question: str
    sql_question: str
    policy_answer: str
    sql_answer: str
    final_answer: str


def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
    )


def router_node(state: GraphState) -> GraphState:
    llm = _get_llm()
    q = state["user_question"]

    out = llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=q),
    ]).content

    parsed = json.loads(out)
    state["route"] = parsed["route"]
    if parsed["route"] in ("POLICY", "BOTH"):
        state["policy_question"] = parsed["policy_question"]
    if parsed["route"] in ("SQL", "BOTH"):
        state["sql_question"] = parsed["sql_question"]
    return state


def policy_node(state: GraphState) -> GraphState:
    q = state.get("policy_question", state["user_question"])
    ans = answer_policy(q)
    # Include sources in a compact form at the end
    citations = "\n".join([f"- (source: {s['file']}, page {s['page']})" for s in ans.sources[:5]])
    state["policy_answer"] = f"{ans.answer}\n\nCitations:\n{citations}" if citations else ans.answer
    return state


def sql_node(state: GraphState) -> GraphState:
    q = state.get("sql_question", state["user_question"])
    ans = answer_customer(q)
    state["sql_answer"] = f"{ans.answer}\n\n(Generated SQL: {ans.sql})"
    return state


def combine_node(state: GraphState) -> GraphState:
    route = state["route"]
    if route == "POLICY":
        state["final_answer"] = state["policy_answer"]
    elif route == "SQL":
        state["final_answer"] = state["sql_answer"]
    else:
        state["final_answer"] = (
            "Policy info:\n"
            f"{state.get('policy_answer','')}\n\n"
            "Customer / ticket info:\n"
            f"{state.get('sql_answer','')}"
        )
    return state


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("router", router_node)
    g.add_node("policy", policy_node)
    g.add_node("sql", sql_node)
    g.add_node("combine", combine_node)

    g.set_entry_point("router")

    def route_decider(state: GraphState):
        r = state["route"]
        if r == "POLICY":
            return "policy"
        if r == "SQL":
            return "sql"
        return "policy"  # for BOTH, do policy then sql

    g.add_conditional_edges("router", route_decider, {
        "policy": "policy",
        "sql": "sql",
    })

    # If BOTH, we run policy -> sql -> combine.
    # If POLICY only: policy -> combine
    # If SQL only: sql -> combine
    def after_policy(state: GraphState):
        return "sql" if state["route"] == "BOTH" else "combine"

    g.add_conditional_edges("policy", after_policy, {"sql": "sql", "combine": "combine"})
    g.add_edge("sql", "combine")
    g.add_edge("combine", END)

    return g.compile()
    