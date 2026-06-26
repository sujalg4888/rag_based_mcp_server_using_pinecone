import importlib
import sys

import pytest


def _reload_config(monkeypatch, **env):
    # Don't let a real .env file in the repo backfill vars we just deleted.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("PINECONE_KEY", raising=False)
    monkeypatch.delenv("PINECONE_INDEX", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    sys.modules.pop("config", None)
    return importlib.import_module("config")


def test_config_loads_with_required_vars(monkeypatch):
    config = _reload_config(monkeypatch, PINECONE_KEY="key", PINECONE_INDEX="idx")
    assert config.PINECONE_KEY == "key"
    assert config.PINECONE_INDEX == "idx"


def test_config_raises_when_missing_pinecone_key(monkeypatch):
    with pytest.raises(RuntimeError, match="PINECONE_KEY"):
        _reload_config(monkeypatch, PINECONE_INDEX="idx")


def test_config_defaults_for_host_and_port(monkeypatch):
    config = _reload_config(monkeypatch, PINECONE_KEY="key", PINECONE_INDEX="idx")
    assert config.MCP_HOST == "127.0.0.1"
    assert config.MCP_PORT == 8000
