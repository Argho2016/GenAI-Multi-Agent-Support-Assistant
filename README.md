# GenAI Multi-Agent Support Assistant
## Gemini 2.5 Flash · LangGraph · RAG · SQL · MCP

---

## 1. Project Overview

This project implements a production-style multi-agent Generative AI system designed for customer support workflows.
The system supports natural language queries over both structured and unstructured data by combining:

- Structured customer and ticket data stored in SQLite
- Unstructured policy documents (PDFs) indexed using vector embeddings
- A deterministic multi-agent orchestration layer using LangGraph
- Gemini 2.5 Flash for low-latency reasoning
- Model Context Protocol (MCP) for standardized tool exposure

The system is explicitly designed to minimize hallucinations, enforce grounded answers, and provide traceability suitable for enterprise environments.

---

## 2. High-Level Architecture

User (Streamlit UI)
→ Router Agent (LangGraph)
→ Policy RAG Agent (PDF → Chunks → Embeddings → Chroma)
→ SQL Agent (Natural Language → SQL → SQLite Execution)
→ Final Answer Composition

Key design principles:
- Deterministic routing
- Evidence-based answers
- Execution-grounded structured queries
- Fail-safe refusal when information is missing

---

## 3. Repository Structure

```text
GenAI-Multi-Agent-Support-Assistant/
├── app_streamlit.py        # Streamlit UI
├── mcp_server.py           # MCP tool server
├── agents/
│   ├── graph.py            # LangGraph orchestration
│   ├── policy_rag.py       # PDF ingestion + RAG logic
│   ├── sql_agent.py        # Safe NL-to-SQL execution
│   └── prompts.py          # System prompts
├── data/
│   ├── seed_customers.py   # SQLite seed script
│   └── customers.db
├── policy_uploads/         # Uploaded policy PDFs
├── storage/
│   └── chroma/             # Persistent vector store
├── requirements.txt
├── .env.example
└── README.md
```


## 4. Environment Setup

### 4.1 Python Environment
Python 3.10+ is recommended.

python -m venv .venv

Activate:
- Windows: .venv\Scripts\activate
- macOS/Linux: source .venv/bin/activate

### 4.2 Install Dependencies

pip install -r requirements.txt

### 4.3 Environment Variables

Create a `.env` file in the project root:

GOOGLE_API_KEY=YOUR_GEMINI_API_KEY

Verify:
python -c "from dotenv import load_dotenv; load_dotenv('.env'); import os; print(os.getenv('GOOGLE_API_KEY'))"

---

## 5. Data Initialization

### 5.1 Seed Structured Database

python data/seed_customers.py

Creates deterministic customers and tickets tables.

### 5.2 Policy PDF Ingestion

- Upload PDFs via Streamlit UI
- PDFs are parsed, chunked, embedded, and stored in Chroma
- Embedding is batched and rate-limited to avoid API quota issues

---

## 6. Running the Application

### 6.1 Start UI

streamlit run app_streamlit.py

### 6.2 Run MCP Server (Optional)

python mcp_server.py

Exposed MCP tools:
- ingest_policy_pdfs
- ask_policy
- ask_customer
- ask_router

---

## 7. Multi-Agent Flow

### Router Agent
- Outputs strict JSON routing decisions
- Enforced by LangGraph state machine

### Policy RAG Agent
- Answers only from retrieved PDF text
- Mandatory citations
- Refuses when evidence is insufficient

### SQL Agent
- Generates SELECT-only SQL
- Executes against SQLite
- Summarizes real query results

---

## 8. Hallucination Mitigation

- Retrieval-based grounding
- Similarity threshold enforcement
- Citation checks
- SQL execution validation
- Low-temperature inference

---

## 9. Testing Strategy

- Policy-only questions
- SQL-only questions
- Combined reasoning questions
- Negative tests (out-of-scope queries)

---

## 10. Design Tradeoffs

- SQLite for simplicity and determinism
- Chroma for local persistence
- Gemini 2.5 Flash for speed/cost balance
- MCP for future extensibility

---

## 11. Conclusion

This system demonstrates safe, explainable, and production-aware GenAI design with multi-agent orchestration and grounded reasoning.
