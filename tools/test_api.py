#!/usr/bin/env python3
"""Live, small paid API smoke test; no game, GPU, or MineLLM required."""
import argparse
import math
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "MP5_agent/agent"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO / "configs/openrouter.yaml"))
    args = parser.parse_args()
    from config import load_config, resolve_base_url
    from llm_factory import create_chat_openai, create_openai_embeddings
    from langchain.schema import HumanMessage
    from langchain.vectorstores import Chroma
    from chromadb.config import Settings

    cfg = load_config(args.config)
    print("Chat endpoint:", resolve_base_url(cfg["models"]["planner"]["base_url"]), flush=True)
    print("Embedding endpoint:", resolve_base_url(cfg["embeddings"]["base_url"]), flush=True)
    print("1/3 Chat request...", flush=True)
    chat = create_chat_openai(cfg["models"]["planner"])
    chat.request_timeout = 60
    chat.max_retries = 0
    chat.max_tokens = 32
    reply = chat([HumanMessage(content="Reply with the single word OK.")])
    assert isinstance(reply.content, str) and reply.content.strip(), "Empty chat response"
    print("PASS chat:", reply.content[:100], flush=True)

    print("2/3 Embedding request...", flush=True)
    embeddings = create_openai_embeddings(cfg["embeddings"])
    embeddings.request_timeout = 60
    embeddings.max_retries = 1
    vector = embeddings.embed_query("Mine three wood logs.")
    assert vector and all(math.isfinite(v) for v in vector), "Invalid embedding vector"
    print("PASS embedding: dimensions =", len(vector), flush=True)

    print("3/3 Chroma write, persist, reload, retrieve...", flush=True)
    with tempfile.TemporaryDirectory(prefix="mp5-api-") as td:
        settings = Settings(anonymized_telemetry=False, is_persistent=True, persist_directory=td)
        db = Chroma(collection_name="mp5_api_smoke", embedding_function=embeddings,
                    persist_directory=td, client_settings=settings)
        db.add_texts(["Mine three wood logs."], ids=["wood"], metadatas=[{"task_name": "wood"}])
        db.persist()
        reopened = Chroma(collection_name="mp5_api_smoke", embedding_function=embeddings,
                          persist_directory=td, client_settings=settings)
        docs = reopened.similarity_search("Collect wood from a tree.", k=1)
        assert len(docs) == 1 and docs[0].metadata["task_name"] == "wood", "Retrieval failed"
        reopened.delete_collection()
    print("PASS Chroma round trip (one record; not a retrieval quality benchmark)")
    print("ALL API SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
