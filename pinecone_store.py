# pinecone_store.py
# Pinecone-backed implementation of VectorSearcher + VectorIndexer
# (interfaces.py). Owns index lifecycle, querying, and reranking.

import time

from pinecone import Pinecone, ServerlessSpec

RERANK_MODEL = "bge-reranker-v2-m3"


class PineconeStore:
    def __init__(self, api_key: str, index_name: str):
        self._pc = Pinecone(api_key=api_key)
        self._index_name = index_name
        self._index = self._pc.Index(index_name)

    def query(self, vector: list[float], top_k: int) -> list[dict]:
        results = self._index.query(vector=vector, top_k=top_k, include_metadata=True)
        return [
            {
                "score": match["score"],
                "source": match["metadata"].get("source"),
                "page": match["metadata"].get("page"),
                "text": match["metadata"].get("text"),
            }
            for match in results["matches"]
        ]

    def rerank(self, query: str, documents: list[dict], top_n: int) -> list[dict]:
        if not documents:
            return documents

        reranked = self._pc.inference.rerank(
            model=RERANK_MODEL,
            query=query,
            documents=[{"text": d["text"]} for d in documents],
            top_n=top_n,
        )
        return [{**documents[d.index], "score": d.score} for d in reranked.data]

    def ensure_index(self, dimension: int) -> None:
        existing = {index["name"]: index for index in self._pc.list_indexes()}
        if self._index_name in existing and existing[self._index_name]["dimension"] != dimension:
            self._pc.delete_index(self._index_name)
            del existing[self._index_name]

        if self._index_name not in existing:
            self._pc.create_index(
                name=self._index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not self._pc.describe_index(self._index_name).status["ready"]:
                time.sleep(1)

        self._index = self._pc.Index(self._index_name)

    def upsert(self, vectors: list[dict]) -> None:
        self._index.upsert(vectors=vectors)
