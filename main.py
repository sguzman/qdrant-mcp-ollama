from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastmcp import FastMCP

# -------------------------
# Config (env vars)
# -------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qllama/bge-small-en-v1.5")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_DISTANCE = os.getenv(
    "QDRANT_DISTANCE", "Cosine"
)  # Cosine / Dot / Euclid / Manhattan

DEFAULT_TOP_K = int(os.getenv("TOP_K", "8"))
TIMEOUT_S = float(os.getenv("HTTP_TIMEOUT_S", "30"))

# Optional: default collection if caller doesn't specify one.
DEFAULT_COLLECTION = os.getenv("QDRANT_DEFAULT_COLLECTION", "")


# -------------------------
# Helpers
# -------------------------
def _log(event: str, **fields: Any) -> None:
    payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **fields}
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _qdrant_headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        h["api-key"] = QDRANT_API_KEY
    return h


def _require_collection(collection: Optional[str]) -> str:
    c = (collection or "").strip()
    if c:
        return c
    if DEFAULT_COLLECTION.strip():
        return DEFAULT_COLLECTION.strip()
    raise ValueError("collection is required (or set QDRANT_DEFAULT_COLLECTION)")


def _ollama_embed(texts: List[str]) -> List[List[float]]:
    """
    Ollama embeddings endpoint:
      POST /api/embed  { model, input }
    Returns:
      { embeddings: number[][], ... }
    """
    url = f"{OLLAMA_URL.rstrip('/')}/api/embed"
    body = {"model": OLLAMA_MODEL, "input": texts}
    _log("ollama.embed.request", url=url, model=OLLAMA_MODEL, n=len(texts))
    r = requests.post(url, json=body, timeout=TIMEOUT_S)
    r.raise_for_status()
    data = r.json()
    emb = data.get("embeddings")
    if not isinstance(emb, list) or not emb:
        raise RuntimeError(f"Ollama /api/embed returned no embeddings: {data}")
    return emb


def _qdrant_collection_info(name: str) -> Tuple[bool, Optional[int]]:
    """Return (exists, vector_size_if_known)."""
    url = f"{QDRANT_URL.rstrip('/')}/collections/{name}"
    r = requests.get(url, headers=_qdrant_headers(), timeout=TIMEOUT_S)
    if r.status_code != 200:
        return (False, None)
    data = r.json()
    # Qdrant returns something like: {"result":{"config":{"params":{"vectors":{"size":...}}}}}
    try:
        size = data["result"]["config"]["params"]["vectors"]["size"]
        return (True, int(size))
    except Exception:
        return (True, None)


def _qdrant_create_collection(name: str, vector_size: int) -> None:
    url = f"{QDRANT_URL.rstrip('/')}/collections/{name}"
    body = {"vectors": {"size": vector_size, "distance": QDRANT_DISTANCE}}
    _log(
        "qdrant.collection.create",
        name=name,
        size=vector_size,
        distance=QDRANT_DISTANCE,
    )
    r = requests.put(url, headers=_qdrant_headers(), json=body, timeout=TIMEOUT_S)
    r.raise_for_status()


def _qdrant_ensure_collection(name: str, vector_size: int) -> None:
    exists, existing_size = _qdrant_collection_info(name)
    if not exists:
        _qdrant_create_collection(name, vector_size)
        return
    if existing_size is not None and existing_size != vector_size:
        raise RuntimeError(
            f"Collection '{name}' exists with vector size {existing_size}, "
            f"but current embedding model outputs size {vector_size}. "
            f"Use a different collection or recreate it."
        )


def _qdrant_upsert(collection: str, points: List[Dict[str, Any]]) -> Dict[str, Any]:
    url = f"{QDRANT_URL.rstrip('/')}/collections/{collection}/points?wait=true"
    body = {"points": points}
    _log("qdrant.points.upsert", collection=collection, n=len(points))
    r = requests.put(url, headers=_qdrant_headers(), json=body, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def _qdrant_search(
    collection: str, query_vector: List[float], limit: int
) -> Dict[str, Any]:
    url = f"{QDRANT_URL.rstrip('/')}/collections/{collection}/points/search"
    body = {
        "vector": query_vector,
        "limit": limit,
        "with_payload": True,
        "with_vector": False,
    }
    _log("qdrant.points.search", collection=collection, limit=limit)
    r = requests.post(url, headers=_qdrant_headers(), json=body, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()


# -------------------------
# MCP server
# -------------------------
mcp = FastMCP("ollama-qdrant")


@mcp.tool
def embed_text(text: str) -> Dict[str, Any]:
    """Embed a single text using Ollama (/api/embed)."""
    vec = _ollama_embed([text])[0]
    return {"model": OLLAMA_MODEL, "dim": len(vec), "embedding": vec}


@mcp.tool
def upsert_texts(
    texts: List[str],
    collection: Optional[str] = None,
    ids: Optional[List[str]] = None,
    metadatas: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Embed texts with Ollama and upsert into Qdrant.

    Args:
      texts: list of strings
      collection: qdrant collection name (optional if QDRANT_DEFAULT_COLLECTION is set)
      ids: optional list of point ids (strings). If not provided, UUIDs are generated.
      metadatas: optional list of payload dicts, same length as texts
    """
    if not texts:
        return {"ok": True, "upserted": 0}

    if ids is not None and len(ids) != len(texts):
        raise ValueError("ids must be same length as texts (or omitted)")
    if metadatas is not None and len(metadatas) != len(texts):
        raise ValueError("metadatas must be same length as texts (or omitted)")

    c = _require_collection(collection)

    embs = _ollama_embed(texts)
    vector_size = len(embs[0])
    _qdrant_ensure_collection(c, vector_size)

    points: List[Dict[str, Any]] = []
    for i, (t, v) in enumerate(zip(texts, embs)):
        pid = ids[i] if ids else str(uuid.uuid4())
        payload = (metadatas[i] if metadatas else {}) | {
            "text": t,
            "embedding_model": OLLAMA_MODEL,
        }
        points.append({"id": pid, "vector": v, "payload": payload})

    res = _qdrant_upsert(c, points)
    return {"ok": True, "collection": c, "upserted": len(points), "result": res}


@mcp.tool
def search_text(
    query: str,
    collection: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
) -> Dict[str, Any]:
    """Embed query with Ollama and run Qdrant vector search."""
    c = _require_collection(collection)

    vec = _ollama_embed([query])[0]
    _qdrant_ensure_collection(c, len(vec))
    res = _qdrant_search(c, vec, int(top_k))
    return {"ok": True, "collection": c, "top_k": int(top_k), "result": res}


if __name__ == "__main__":
    _log(
        "server.start",
        ollama_url=OLLAMA_URL,
        ollama_model=OLLAMA_MODEL,
        qdrant_url=QDRANT_URL,
        default_collection=DEFAULT_COLLECTION,
    )
    mcp.run()
