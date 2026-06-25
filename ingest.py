# ingest.py
# Run once (or whenever data/ changes) to load PDFs into Pinecone:
#   uv run python ingest.py

import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader

from models import EMBED_DIM, embed_texts

load_dotenv()

CHUNK_SIZE = 140
CHUNK_OVERLAP = 25
DATA_DIR = Path(__file__).parent / "data"

pc = Pinecone(api_key=os.environ["PINECONE_KEY"])
INDEX_NAME = os.environ["PINECONE_INDEX"]


def extract_pdf(path) -> list[str]:
    reader = PdfReader(path)
    return [page.extract_text() for page in reader.pages]


def strip_repeated_lines(pages: list[str]) -> list[str]:
    """Drop lines that recur across most pages (running headers/footers)."""
    if len(pages) < 2:
        return pages

    page_lines = [[l.strip() for l in p.split("\n") if l.strip()] for p in pages]
    counts: dict[str, int] = {}
    for lines in page_lines:
        for line in set(lines):
            counts[line] = counts.get(line, 0) + 1

    # Require >=3 words so stray single tokens (PDF wrap artifacts like "and")
    # that happen to repeat across pages aren't mistaken for headers/footers.
    threshold = max(2, len(pages) // 2)
    repeated = {line for line, c in counts.items() if c >= threshold and len(line.split()) >= 3}
    return ["\n".join(l for l in lines if l not in repeated) for lines in page_lines]


def split_sentences(text):
    # pypdf inserts stray spaces/newlines mid-phrase in some PDFs; collapse
    # all whitespace before splitting so that noise doesn't break sentences.
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def chunk_pages(pages: list[str], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[tuple[str, int]]:
    """Chunk sentence-aware across pages, tracking the source page per chunk."""
    tagged = [
        (sentence, page_num)
        for page_num, page_text in enumerate(pages, start=1)
        for sentence in split_sentences(page_text)
    ]

    chunks: list[tuple[str, int]] = []
    current: list[str] = []
    current_words = 0
    current_page = 1

    for sentence, page_num in tagged:
        sentence_words = len(sentence.split())

        if current and current_words + sentence_words > chunk_size:
            chunks.append((" ".join(current), current_page))

            kept, kept_words = [], 0
            for s in reversed(current):
                w = len(s.split())
                if kept_words + w > overlap:
                    break
                kept.insert(0, s)
                kept_words += w
            current, current_words = kept, kept_words

        if not current:
            current_page = page_num
        current.append(sentence)
        current_words += sentence_words

    if current:
        chunks.append((" ".join(current), current_page))

    return chunks


def ensure_index():
    existing = {index["name"]: index for index in pc.list_indexes()}
    if INDEX_NAME in existing and existing[INDEX_NAME]["dimension"] != EMBED_DIM:
        pc.delete_index(INDEX_NAME)
        del existing[INDEX_NAME]

    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)


def ingest_pdf(path):
    pages = strip_repeated_lines(extract_pdf(path))
    chunks = chunk_pages(pages)
    embeddings = embed_texts([chunk for chunk, _ in chunks])

    vectors = [
        {
            "id": f"{path.stem}-{i}",
            "values": embedding,
            "metadata": {"text": chunk, "source": path.name, "page": page_num},
        }
        for i, ((chunk, page_num), embedding) in enumerate(zip(chunks, embeddings))
    ]

    index = pc.Index(INDEX_NAME)
    index.upsert(vectors=vectors)
    print(f"Upserted {len(vectors)} chunks from {path.name}")


def main():
    ensure_index()

    for pdf_path in DATA_DIR.glob("*.pdf"):
        ingest_pdf(pdf_path)


if __name__ == "__main__":
    main()
