# models.py
# Local Qwen models: embeddings (Qwen3-Embedding-0.6B) via transformers,
# answer generation (Qwen2.5-1.5B-Instruct) via llama.cpp GGUF for fast CPU inference.
# No external API key required - runs fully on-device.

import torch
from llama_cpp import Llama
from llama_cpp.llama_types import ChatCompletionRequestMessage, CreateChatCompletionResponse
from transformers import AutoModel, AutoTokenizer

EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
GEN_MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
GEN_MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
EMBED_DIM = 1024

N_THREADS = 10  # physical cores (i5-1334U: 10 cores / 12 logical, hybrid P+E)

torch.set_num_threads(N_THREADS)

_embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME, padding_side="left")
_embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
_embed_model.eval()

_gen_model = Llama.from_pretrained(
    repo_id=GEN_MODEL_REPO,
    filename=GEN_MODEL_FILE,
    n_ctx=2048,
    n_threads=N_THREADS,
    n_batch=512,
    use_mlock=True,
    verbose=False,
)


def _last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size), sequence_lengths]


def embed_texts(texts: list[str], instruction: str | None = None) -> list[list[float]]:
    if instruction:
        texts = [f"Instruct: {instruction}\nQuery:{t}" for t in texts]

    batch = _embed_tokenizer(texts, padding=True, truncation=True, max_length=8192, return_tensors="pt")
    with torch.no_grad():
        outputs = _embed_model(**batch)

    embeddings = _last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.tolist()


def generate_answer(query: str, context: str) -> str:
    messages: list[ChatCompletionRequestMessage] = [
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
    response: CreateChatCompletionResponse = _gen_model.create_chat_completion(
        messages=messages, max_tokens=250, temperature=0.2, repeat_penalty=1.1
    )  # type: ignore[assignment]
    content = response["choices"][0]["message"]["content"]
    return content.strip() if content else ""
