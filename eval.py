# eval.py
# Ragas evaluation of the RAG pipeline (server.py: retrieve + generate_answer).
# Reuses the already-loaded local Qwen models (no second model load, no API key).
# Run: uv run python eval.py

import asyncio
import threading

from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue
from llama_cpp.llama_types import CreateChatCompletionResponse
from ragas import EvaluationDataset, evaluate
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM
from ragas.metrics import AnswerRelevancy, Faithfulness
from ragas.run_config import RunConfig

from models import _gen_model, embed_texts
from server import _retrieve, generate_answer

TOP_K = 3

# Questions to probe the ingested doc(s). No ground-truth answers needed for
# Faithfulness/AnswerRelevancy. Add "ground_truth" per item later to enable
# ContextPrecision/ContextRecall.
TEST_SET = [
    {"question": "What is the Orion Pedagogy Framework designed to ensure?"},
    {"question": "How does the framework determine if a student has truly learned something?"},
    {"question": "What role does human judgment play in the framework?"},
]


_LLAMA_LOCK = threading.Lock()  # llama.cpp's context is not safe for concurrent calls


class LocalLlamaLLM(BaseRagasLLM):
    """Wraps the already-loaded llama.cpp model (models._gen_model) for ragas."""

    def generate_text(
        self, prompt: PromptValue, n: int = 1, temperature: float = 1e-8, stop=None, callbacks=None
    ) -> LLMResult:
        text = prompt.to_string()
        generations = []
        with _LLAMA_LOCK:
            for _ in range(n):
                response: CreateChatCompletionResponse = _gen_model.create_chat_completion(
                    messages=[{"role": "user", "content": text}],
                    max_tokens=512,
                    temperature=temperature,
                    stop=stop,
                )  # type: ignore[assignment]
                content = response["choices"][0]["message"]["content"] or ""
                generations.append(Generation(text=content.strip()))
        return LLMResult(generations=[generations])

    async def agenerate_text(
        self, prompt: PromptValue, n: int = 1, temperature=None, stop=None, callbacks=None
    ) -> LLMResult:
        if temperature is None:
            temperature = self.get_temperature(n)
        return await asyncio.to_thread(self.generate_text, prompt, n, temperature, stop, callbacks)


class LocalEmbeddings(BaseRagasEmbeddings):
    """Wraps models.embed_texts (Qwen3-Embedding-0.6B) for ragas."""

    def embed_query(self, text: str) -> list[float]:
        return embed_texts([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)


def build_dataset(top_k: int = TOP_K) -> EvaluationDataset:
    rows = []
    for case in TEST_SET:
        question = case["question"]
        matches = _retrieve(question, top_k)
        contexts = [m["text"] for m in matches]
        answer = generate_answer(question, "\n\n".join(contexts))
        rows.append(
            {
                "user_input": question,
                "retrieved_contexts": contexts,
                "response": answer,
            }
        )
    return EvaluationDataset.from_list(rows)


def main():
    dataset = build_dataset()
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), AnswerRelevancy()],
        llm=LocalLlamaLLM(),
        embeddings=LocalEmbeddings(),
        # llama.cpp's context is not thread-safe for concurrent calls.
        run_config=RunConfig(max_workers=1),
    )
    print(result)
    result.to_pandas().to_csv("eval_results.csv", index=False)


if __name__ == "__main__":
    main()
