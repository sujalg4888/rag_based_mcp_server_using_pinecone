# ollama_models.py
# Embeddings (mxbai-embed-large) and answer generation (qwen2.5:1.5b) via
# Ollama, running as a local daemon. No external API key, no in-process
# model weights.
#
# Implements the Embedder/Generator protocols from interfaces.py. Swapping
# to a different backend means adding a new class here (or elsewhere) and
# wiring it in at the composition root (server.py/ingest.py) -- retrieval.py
# and the MCP tools never need to change.

import ollama

EMBED_DIM = 1024


class OllamaEmbedder:
    def __init__(self, model_name: str = "mxbai-embed-large", client: ollama.Client | None = None):
        self.model_name = model_name
        self._client = client or ollama.Client()

    def embed(self, texts: list[str], instruction: str | None = None) -> list[list[float]]:
        if instruction:
            texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]

        response = self._client.embed(model=self.model_name, input=texts)
        return response["embeddings"]


class OllamaGenerator:
    def __init__(self, model_name: str = "qwen2.5:1.5b", client: ollama.Client | None = None):
        self.model_name = model_name
        self._client = client or ollama.Client()

    def chat(self, messages: list[dict], **options) -> dict:
        return self._client.chat(model=self.model_name, messages=messages, options=options) # pyright: ignore[reportReturnType]

    def generate(self, query: str, context: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the question using only the provided context. "
                    "Cite the source filenames you used. "
                    "If the context doesn't contain the answer, say so plainly."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
        response = self.chat(messages, temperature=0.2, num_predict=250, repeat_penalty=1.1)
        content = response["message"]["content"]
        return content.strip() if content else ""
