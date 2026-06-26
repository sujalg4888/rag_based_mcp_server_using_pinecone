# eval.py
# Ragas evaluation of the RAG pipeline (server.py: retrieve + generate).
# Calls the same Ollama-backed models as the server (no API key needed).
# Run: uv run python eval.py

import asyncio

# nest_asyncio 1.6.0 doesn't support Python 3.14's asyncio internals: its
# patch breaks current_task() tracking inside asyncio.timeout, which ragas's
# per-metric wait_for relies on, causing "Timeout should be used inside a
# task" and every metric scoring as nan. We never run inside a nested event
# loop here, so the patch isn't needed — disable it before ragas imports it.
import nest_asyncio

nest_asyncio.apply = lambda *args, **kwargs: None

from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue
from ragas import EvaluationDataset, evaluate
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM
from ragas.metrics import AnswerRelevancy, Faithfulness
from ragas.run_config import RunConfig

from server import _embedder, _generator, _retriever

TOP_K = 3

# Questions to probe the ingested doc(s). No ground-truth answers needed for
# Faithfulness/AnswerRelevancy. Add "ground_truth" per item later to enable
# ContextPrecision/ContextRecall.
TEST_SET = [
    {"question": "What is the Orion Pedagogy Framework designed to ensure?"},
    {"question": "How does the framework determine if a student has truly learned something?"},
    {"question": "What role does human judgment play in the framework?"},
]


class LocalLlamaLLM(BaseRagasLLM):
    """Wraps a Generator's chat() (interfaces.py) for ragas."""

    def __init__(self, generator):
        self._generator = generator

    def generate_text(
        self, prompt: PromptValue, n: int = 1, temperature: float = 1e-8, stop=None, callbacks=None
    ) -> LLMResult:
        text = prompt.to_string()
        generations = []
        for _ in range(n):
            response = self._generator.chat(
                [{"role": "user", "content": text}],
                temperature=temperature,
                num_predict=512,
                stop=stop,
            )
            content = response["message"]["content"] or ""
            generations.append(Generation(text=content.strip()))
        return LLMResult(generations=[generations])

    async def agenerate_text(
        self, prompt: PromptValue, n: int = 1, temperature=None, stop=None, callbacks=None
    ) -> LLMResult:
        if temperature is None:
            temperature = self.get_temperature(n)
        return await asyncio.to_thread(self.generate_text, prompt, n, temperature, stop, callbacks)


class LocalEmbeddings(BaseRagasEmbeddings):
    """Wraps an Embedder (interfaces.py) for ragas."""

    def __init__(self, embedder):
        self._embedder = embedder

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)


def build_dataset(top_k: int = TOP_K) -> EvaluationDataset:
    rows = []
    for case in TEST_SET:
        question = case["question"]
        matches = _retriever.retrieve(question, top_k)
        contexts = [m["text"] for m in matches]
        answer = _generator.generate(question, "\n\n".join(contexts))
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
        llm=LocalLlamaLLM(_generator),
        embeddings=LocalEmbeddings(_embedder),
        # llama.cpp's context is not thread-safe for concurrent calls.
        run_config=RunConfig(max_workers=1),
    )
    print(result)
    result.to_pandas().to_csv("eval_results.csv", index=False)


if __name__ == "__main__":
    main()
