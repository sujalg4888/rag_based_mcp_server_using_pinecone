# server.py
# FastMCP server exposing a search_docs tool backed by Pinecone.
# Run: uv run python main.py

import os
import time

from dotenv import load_dotenv
from fastmcp import FastMCP
from pinecone import Pinecone

from models import embed_texts, generate_answer

load_dotenv()

QUERY_INSTRUCTION = "Given a search query, retrieve relevant passages that answer the query"
RERANK_MODEL = "bge-reranker-v2-m3"
RERANK_FETCH_MULTIPLIER = 4
RERANK_FETCH_MAX = 50

pc = Pinecone(api_key=os.environ["PINECONE_KEY"])
index = pc.Index(os.environ["PINECONE_INDEX"])

mcp = FastMCP("rag-pinecone")


def _retrieve(query: str, top_k: int) -> list[dict]:
    embedding = embed_texts([query], instruction=QUERY_INSTRUCTION)[0]
    fetch_k = min(max(top_k * RERANK_FETCH_MULTIPLIER, top_k), RERANK_FETCH_MAX)
    results = index.query(vector=embedding, top_k=fetch_k, include_metadata=True)

    candidates = [
        {
            "score": match["score"],
            "source": match["metadata"].get("source"),
            "page": match["metadata"].get("page"),
            "text": match["metadata"].get("text"),
        }
        for match in results["matches"]
    ]
    if not candidates:
        return candidates

    reranked = pc.inference.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=[{"text": c["text"]} for c in candidates],
        top_n=top_k,
    )
    return [
        {**candidates[doc.index], "score": doc.score}
        for doc in reranked.data
    ]


@mcp.tool()
def search_docs(query: str, top_k: int = 5) -> list[dict]:
    """Search the ingested documents for chunks relevant to the query."""
    return _retrieve(query, top_k)


@mcp.tool()
def ask_docs(query: str, top_k: int = 3) -> dict:
    """Answer a question using the ingested documents, generated locally via Qwen."""
    t0 = time.time()
    matches = _retrieve(query, top_k)
    t1 = time.time()
    context = "\n\n".join(f"[{m['source']} p.{m['page']}] {m['text']}" for m in matches)
    answer = generate_answer(query, context)
    t2 = time.time()
    print(f"[ask_docs] retrieve={t1 - t0:.2f}s generate={t2 - t1:.2f}s total={t2 - t0:.2f}s", flush=True)

    return {
        "answer": answer,
        "sources": sorted({m["source"] for m in matches}),
    }


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8000)
