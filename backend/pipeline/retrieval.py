"""Retrieval: embed the question, search Chroma, optionally rerank with Cohere."""
from openai import OpenAI

from ..config import settings
from .vectorstore import embed_batch


def retrieve(coll, question: str, oai: OpenAI, k: int = 15) -> list[dict]:
    """Return the top-k chunks from the vector store for a given question."""
    q_emb = embed_batch([question], settings.embedding_model, oai)[0]
    r = coll.query(query_embeddings=[q_emb], n_results=k)

    hits: list[dict] = []
    if not r["ids"] or not r["ids"][0]:
        return hits

    for i in range(len(r["ids"][0])):
        meta = r["metadatas"][0][i] or {}
        # parent_child strategy stores the larger parent_text in metadata; prefer it
        # for the LLM context while keeping the small child for embedding.
        text = meta.pop("parent_text", None) or r["documents"][0][i]
        hits.append({
            "text": text,
            **meta,
            "distance": r["distances"][0][i] if r.get("distances") else None,
        })
    return hits


def rerank(question: str, hits: list[dict], top_n: int = 4) -> list[dict]:
    """Rerank with Cohere if an API key is configured; otherwise pass through top-N."""
    if not hits:
        return []
    if not settings.cohere_api_key:
        return hits[:top_n]

    import cohere
    co = cohere.Client(settings.cohere_api_key)
    resp = co.rerank(
        model="rerank-english-v3.0",
        query=question,
        documents=[h["text"] for h in hits],
        top_n=min(top_n, len(hits)),
    )
    return [hits[r.index] for r in resp.results]
