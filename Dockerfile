FROM python:3.14-slim

# build-essential + cmake needed to compile llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# llama-cpp-python's ggml shared libs don't declare libstdc++ as a NEEDED
# dependency, so dlopen can't resolve RTTI symbols on import
# (undefined symbol: _ZTVN10...class_type_infoE) unless it's preloaded.
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8000
EXPOSE 8000

CMD ["uv", "run", "python", "main.py"]
