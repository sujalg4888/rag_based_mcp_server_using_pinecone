# RAG MCP Server (Pinecone)

FastMCP server exposing `search_docs` and `ask_docs` tools backed by a Pinecone
vector index. Embeddings (Qwen3-Embedding-0.6B) and answer generation
(Qwen2.5-1.5B-Instruct GGUF) run locally via `transformers` / `llama.cpp` — no
LLM API key required, only a Pinecone account.

## Setup

1. Copy `.env.example` to `.env` and fill in:
   ```
   PINECONE_KEY=your_pinecone_api_key
   PINECONE_INDEX=your_index_name
   ```
2. Drop PDFs into `data/`.
3. Install deps and ingest:
   ```bash
   uv sync
   uv run python ingest.py
   ```
4. Run the server:
   ```bash
   uv run python main.py
   ```
5. Query it:
   ```bash
   uv run python client.py "your question"
   ```

## Docker

Build and run with Docker Compose (recommended — persists the HuggingFace
model cache in a named volume so models aren't re-downloaded on every
restart):

```bash
docker compose up --build
```

The server listens on `http://localhost:8000/mcp`. `data/` is mounted as a
volume, so PDFs added on the host are visible inside the container.

To ingest PDFs into Pinecone from inside the running container:

```bash
docker compose exec rag-mcp-server uv run python ingest.py
```

### Without Compose

```bash
docker build -t rag-mcp-server .
docker run --rm -p 8000:8000 --env-file .env -v "$(pwd)/data:/app/data" rag-mcp-server
```

### Notes

- The image installs build tools to compile `llama-cpp-python`; first build
  is slow, subsequent ones are cached.
- Models are downloaded from HuggingFace on first run, not baked into the
  image — mount a volume over `/root/.cache/huggingface` (already done in
  `docker-compose.yml`) to avoid re-downloading.
- `MCP_HOST`/`MCP_PORT` env vars override the listen address (default
  `0.0.0.0:8000` in the container, `127.0.0.1:8000` for local `uv run`).

## Files

- `ingest.py` — chunk PDFs from `data/`, embed, upsert into Pinecone.
- `server.py` — MCP tools `search_docs` (retrieval + rerank) and `ask_docs`
  (retrieval + local generation).
- `models.py` — local embedding/generation models.
- `main.py` — server entrypoint.
- `client.py` — example MCP client for manual testing.
- `eval.py` — Ragas evaluation (Faithfulness, AnswerRelevancy) of the RAG
  pipeline using the local Qwen models; writes `eval_results.csv`.
