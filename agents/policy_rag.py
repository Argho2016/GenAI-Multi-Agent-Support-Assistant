from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
import time
from dataclasses import dataclass
from typing import List

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_message

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_chroma import Chroma

from .prompts import POLICY_SYSTEM

CHROMA_DIR = os.path.join("storage", "chroma")
COLLECTION_NAME = "policies"


@dataclass
class PolicyAnswer:
    answer: str
    sources: List[dict]  # [{"file":..., "page":..., "snippet":...}]


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
    )


def _get_vs() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def ingest_pdfs(pdf_paths: List[str], batch_size: int = 16, sleep_s: float = 0.75) -> dict:
    """
    Loads PDFs, splits into chunks, embeds in small batches with throttling,
    stores in Chroma. Designed to avoid free-tier 429 during ingestion.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,     
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )

    docs: List[Document] = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        loaded = loader.load()
        for d in loaded:
            d.metadata["source_file"] = os.path.basename(path)
            if "page" in d.metadata:
                d.metadata["page_human"] = int(d.metadata["page"]) + 1
        docs.extend(loaded)

    chunks = splitter.split_documents(docs)
    vs = _get_vs()

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(8),
        retry=retry_if_exception_message("RESOURCE_EXHAUSTED"),
        reraise=True,
    )
    def _add_batch(batch: List[Document]):
        vs.add_documents(batch)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        _add_batch(batch)
        time.sleep(sleep_s)

    #vs.persist()

    return {
        "pdfs": len(pdf_paths),
        "pages": len(docs),
        "chunks": len(chunks),
        "batched": True,
        "batch_size": batch_size,
        "sleep_s": sleep_s,
        "collection": COLLECTION_NAME,
        "persist_directory": CHROMA_DIR,
    }


def retrieve(question: str, k: int = 5) -> List[Document]:
    vs = _get_vs()
    retriever = vs.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(question)



def answer_policy(question: str, k: int = 5) -> PolicyAnswer:
    """
    Answers a policy question using RAG over ingested PDFs.
    """
    llm = _get_llm()
    docs = retrieve(question, k=k)

    context_blocks: List[str] = []
    sources: List[dict] = []

    for d in docs:
        file_ = d.metadata.get("source_file", "unknown")
        page = d.metadata.get("page_human", d.metadata.get("page", "unknown"))
        text = (d.page_content or "").strip()

        snippet = text[:350].replace("\n", " ")
        sources.append({"file": file_, "page": page, "snippet": snippet})

        context_blocks.append(f"[source: {file_}, page {page}]\n{text}\n")

    context = "\n---\n".join(context_blocks) if context_blocks else "(no context retrieved)"

    messages = [
        SystemMessage(content=POLICY_SYSTEM),
        HumanMessage(content=f"Question: {question}\n\nContext:\n{context}"),
    ]
    out = llm.invoke(messages).content

    return PolicyAnswer(answer=out, sources=sources)
