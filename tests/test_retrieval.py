from retrieval import RERANK_FETCH_MAX, RERANK_FETCH_MULTIPLIER, Retriever


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, texts, instruction=None):
        self.calls.append((texts, instruction))
        return [[1.0, 0.0] for _ in texts]


class FakeSearcher:
    def __init__(self, candidates):
        self._candidates = candidates
        self.query_calls = []
        self.rerank_calls = []

    def query(self, vector, top_k):
        self.query_calls.append((vector, top_k))
        return self._candidates[:top_k]

    def rerank(self, query, documents, top_n):
        self.rerank_calls.append((query, documents, top_n))
        return documents[:top_n]


def make_candidates(n):
    return [{"score": 1.0, "source": f"doc{i}.pdf", "page": 1, "text": f"text {i}"} for i in range(n)]


def test_retrieve_embeds_query_with_instruction():
    embedder = FakeEmbedder()
    searcher = FakeSearcher(make_candidates(5))
    retriever = Retriever(embedder, searcher)

    retriever.retrieve("what is x", top_k=3)

    assert embedder.calls[0][0] == ["what is x"]
    assert embedder.calls[0][1] is not None


def test_retrieve_fetches_more_than_top_k_for_rerank():
    embedder = FakeEmbedder()
    searcher = FakeSearcher(make_candidates(20))
    retriever = Retriever(embedder, searcher)

    retriever.retrieve("query", top_k=3)

    fetched_top_k = searcher.query_calls[0][1]
    assert fetched_top_k == min(3 * RERANK_FETCH_MULTIPLIER, RERANK_FETCH_MAX)


def test_retrieve_caps_fetch_at_max():
    embedder = FakeEmbedder()
    searcher = FakeSearcher(make_candidates(60))
    retriever = Retriever(embedder, searcher)

    retriever.retrieve("query", top_k=30)

    fetched_top_k = searcher.query_calls[0][1]
    assert fetched_top_k == RERANK_FETCH_MAX


def test_retrieve_skips_rerank_when_no_candidates():
    embedder = FakeEmbedder()
    searcher = FakeSearcher([])
    retriever = Retriever(embedder, searcher)

    result = retriever.retrieve("query", top_k=5)

    assert result == []
    assert searcher.rerank_calls == []


def test_retrieve_passes_top_k_to_rerank():
    embedder = FakeEmbedder()
    searcher = FakeSearcher(make_candidates(10))
    retriever = Retriever(embedder, searcher)

    retriever.retrieve("query", top_k=4)

    assert searcher.rerank_calls[0][2] == 4
