import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path=".langchain_llm_cache.db"))

import streamlit as st

from agents.policy_rag import ingest_pdfs
from agents.graph import build_graph

load_dotenv()
graph = build_graph()

st.set_page_config(page_title="Multi-Agent Support Assistant", layout="wide")
st.title("Multi-Agent Support Assistant")

st.sidebar.header("1) Upload policy PDFs")
os.makedirs("policy_uploads", exist_ok=True)
uploaded = st.sidebar.file_uploader(
    "Upload one or more policy PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

if st.sidebar.button("Ingest PDFs"):
    if not uploaded:
        st.sidebar.warning("Upload at least one PDF first.")
    else:
        paths = []
        for f in uploaded:
            path = os.path.join("policy_uploads", f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            paths.append(path)
        stats = ingest_pdfs(paths)
        st.sidebar.success(f"Ingested {stats['pdfs']} PDFs, {stats['chunks']} chunks.")

st.sidebar.header("2) Seed demo customer DB")
st.sidebar.code("python data/seed_customers.py", language="bash")

st.markdown("### Ask a question")
st.caption("Examples: “What is the current refund policy?” or “Give me an overview of customer Ema’s profile and past tickets.”")

if "chat" not in st.session_state:
    st.session_state.chat = []

q = st.text_input("Your question", placeholder="Type here…")

col1, col2 = st.columns([1, 3])
with col1:
    ask = st.button("Ask")

if ask and q.strip():
    st.session_state.chat.append(("You", q))
    out = graph.invoke({"user_question": q})
    st.session_state.chat.append(("Assistant", out["final_answer"]))

st.divider()
for speaker, msg in st.session_state.chat:
    if speaker == "You":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Assistant:**\n\n{msg}")
