# ingest.py
# Run once (or whenever data/ changes) to load PDFs into Pinecone:
#   uv run python ingest.py

import logging
from pathlib import Path

import config
from ollama_models import EMBED_DIM, OllamaEmbedder
from pdf_chunking import chunk_pages, extract_pdf, strip_repeated_lines
from pinecone_store import PineconeStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


def ingest_pdf(path, embedder: OllamaEmbedder, store: PineconeStore) -> None:
    pages = strip_repeated_lines(extract_pdf(path))
    chunks = chunk_pages(pages)
    embeddings = embedder.embed([chunk for chunk, _ in chunks])

    vectors = [
        {
            "id": f"{path.stem}-{i}",
            "values": embedding,
            "metadata": {"text": chunk, "source": path.name, "page": page_num},
        }
        for i, ((chunk, page_num), embedding) in enumerate(zip(chunks, embeddings))
    ]

    store.upsert(vectors)
    logger.info("Upserted %d chunks from %s", len(vectors), path.name)


def main():
    embedder = OllamaEmbedder()
    store = PineconeStore(config.PINECONE_KEY, config.PINECONE_INDEX)
    store.ensure_index(EMBED_DIM)

    for pdf_path in DATA_DIR.glob("*.pdf"):
        ingest_pdf(pdf_path, embedder, store)


if __name__ == "__main__":
    main()
