# server.py
# FastMCP server exposing search_docs/ask_docs tools backed by Pinecone +
# Ollama, wired together through the Embedder/Generator/VectorSearcher
# abstractions (interfaces.py).
# Run: uv run python main.py

import logging
import time

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

import config
from ollama_models import OllamaEmbedder, OllamaGenerator
from pinecone_store import PineconeStore
from retrieval import Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("rag-pinecone")

_embedder = OllamaEmbedder()
_generator = OllamaGenerator()
_store = PineconeStore(config.PINECONE_KEY, config.PINECONE_INDEX)
_retriever = Retriever(_embedder, _store)


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


@mcp.tool()
def search_docs(query: str, top_k: int = 5) -> list[dict]:
    """Search the ingested documents for chunks relevant to the query."""
    return _retriever.retrieve(query, top_k)


@mcp.tool()
def ask_docs(query: str, top_k: int = 3) -> dict:
    """Answer a question using the ingested documents, generated locally via Qwen."""
    t0 = time.time()
    matches = _retriever.retrieve(query, top_k)
    t1 = time.time()
    context = "\n\n".join(f"[{m['source']} p.{m['page']}] {m['text']}" for m in matches)
    answer = _generator.generate(query, context)
    t2 = time.time()
    logger.info("ask_docs retrieve=%.2fs generate=%.2fs total=%.2fs", t1 - t0, t2 - t1, t2 - t0)

    return {
        "answer": answer,
        "sources": sorted({m["source"] for m in matches}),
    }


if __name__ == "__main__":
    mcp.run(transport="http", host=config.MCP_HOST, port=config.MCP_PORT)
