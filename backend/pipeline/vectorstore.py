"""Thin wrappers around Chroma persistent client and OpenAI batch embeddings."""
import chromadb
from openai import OpenAI


def get_collection(path: str, name: str = "videos"):
    """Return a persistent Chroma collection configured for cosine similarity."""
    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def embed_batch(texts: list[str], model: str, client: OpenAI, batch_size: int = 100) -> list[list[float]]:
    """Embed a list of texts, batching to avoid hitting the per-request input cap."""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        resp = client.embeddings.create(model=model, input=texts[i:i + batch_size])
        out.extend([d.embedding for d in resp.data])
    return out
