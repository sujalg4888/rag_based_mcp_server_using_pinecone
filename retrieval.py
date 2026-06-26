# retrieval.py
# Embed -> vector search -> rerank, against the Embedder/VectorSearcher
# abstractions. Doesn't know or care whether those are backed by
# Ollama/Pinecone or something else.

from interfaces import Embedder, VectorSearcher

QUERY_INSTRUCTION = "Given a search query, retrieve relevant passages that answer the query"
RERANK_FETCH_MULTIPLIER = 4
RERANK_FETCH_MAX = 50


class Retriever:
    def __init__(self, embedder: Embedder, searcher: VectorSearcher):
        self._embedder = embedder
        self._searcher = searcher

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        embedding = self._embedder.embed([query], instruction=QUERY_INSTRUCTION)[0]
        fetch_k = min(max(top_k * RERANK_FETCH_MULTIPLIER, top_k), RERANK_FETCH_MAX)
        candidates = self._searcher.query(embedding, fetch_k)
        if not candidates:
            return candidates

        return self._searcher.rerank(query, candidates, top_k)
