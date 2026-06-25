import asyncio
import sys
import time

from fastmcp import Client

SERVER_URL = "http://127.0.0.1:8000/mcp"


async def main(query: str, top_k: int = 5):
    async with Client(SERVER_URL, timeout=1000) as client:
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])

        start = time.time()
        result = await client.call_tool("ask_docs", {"query": query, "top_k": top_k})
        elapsed = time.time() - start

        print(f"\n{result.data['answer']}\n")
        print("Sources:", ", ".join(result.data["sources"]))
        print(f"\n[took {elapsed:.2f}s]")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "test query"
    asyncio.run(main(query))
