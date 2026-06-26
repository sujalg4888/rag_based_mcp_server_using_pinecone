# interfaces.py
# Abstractions the rest of the codebase depends on, instead of concrete
# Ollama/Pinecone implementations. Lets retrieval/server/eval code swap
# embedder, generator, or vector store without touching their logic.

from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: list[str], instruction: str | None = None) -> list[list[float]]: ...


class Generator(Protocol):
    def generate(self, query: str, context: str) -> str: ...


class VectorSearcher(Protocol):
    def query(self, vector: list[float], top_k: int) -> list[dict]: ...
    def rerank(self, query: str, documents: list[dict], top_n: int) -> list[dict]: ...


class VectorIndexer(Protocol):
    def ensure_index(self, dimension: int) -> None: ...
    def upsert(self, vectors: list[dict]) -> None: ...
