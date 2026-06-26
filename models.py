# models.py
# Embeddings (mxbai-embed-large) and answer generation (qwen2.5:1.5b) via
# Ollama, running as a local daemon. No external API key, no in-process
# model weights.

import ollama

EMBED_MODEL_NAME = "mxbai-embed-large"
GEN_MODEL_NAME = "qwen2.5:1.5b"
EMBED_DIM = 1024

_client = ollama.Client()


def embed_texts(texts: list[str], instruction: str | None = None) -> list[list[float]]:
    if instruction:
        texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]

    response = _client.embed(model=EMBED_MODEL_NAME, input=texts)
    return response["embeddings"]


def generate_answer(query: str, context: str) -> str:
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
    response = _client.chat(
        model=GEN_MODEL_NAME,
        messages=messages,
        options={"temperature": 0.2, "num_predict": 250, "repeat_penalty": 1.1},
    )
    content = response["message"]["content"]
    return content.strip() if content else ""
