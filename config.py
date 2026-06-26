# config.py
# Loads + validates required env vars once, at import time, so missing
# config fails fast with a clear message instead of a bare KeyError deep
# inside a request.

import os

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


PINECONE_KEY = _require("PINECONE_KEY")
PINECONE_INDEX = _require("PINECONE_INDEX")
MCP_HOST = os.environ.get("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.environ.get("MCP_PORT", "8000"))
