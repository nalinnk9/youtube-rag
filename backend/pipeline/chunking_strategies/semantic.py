"""Embedding-based semantic chunker.

Embeds each sentence, computes cosine distance between consecutive sentence
embeddings, and splits where distances exceed the configured percentile (a
"semantic break"). Falls back to grouping every ~5 sentences if embedding fails.
"""
from openai import OpenAI

from ...config import settings
from ..vectorstore import embed_batch
from .sentence_window import _split_sentences_with_ts


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - (dot / (na * nb))


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def chunk_semantic(
    segments: list[dict],
    percentile: float = 95.0,
    oai: OpenAI | None = None,
    **_,
) -> list[dict]:
    if not segments:
        return []
    sentences = _split_sentences_with_ts(segments)
    if len(sentences) <= 1:
        return [{
            "start": segments[0]["start"],
            "end": segments[-1]["end"],
            "text": " ".join(s["text"] for s in segments),
        }]

    if oai is None:
        oai = OpenAI(api_key=settings.openai_api_key)

    try:
        embeddings = embed_batch([s["text"] for s in sentences], settings.embedding_model, oai)
    except Exception:
        # Fallback: group every 5 sentences
        boundaries = list(range(5, len(sentences), 5))
    else:
        distances = [_cosine_distance(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]
        threshold = _percentile(distances, percentile)
        boundaries = [i + 1 for i, d in enumerate(distances) if d >= threshold]

    chunks: list[dict] = []
    prev = 0
    for b in boundaries + [len(sentences)]:
        group = sentences[prev:b]
        if not group:
            continue
        chunks.append({
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "text": " ".join(s["text"] for s in group),
        })
        prev = b

    return chunks
