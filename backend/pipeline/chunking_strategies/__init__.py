"""Registry of chunking strategies. Each strategy maps {start,end,text} segments
to chunks of {start,end,text} (parent-child also adds parent_text). Callers select
a strategy by name; the orchestrator (ingest, compare) iterates over the registry.
"""
from typing import Callable

from ...config import settings
from .fixed_time import chunk_fixed_time
from .parent_child import chunk_parent_child
from .recursive_token import chunk_recursive_token
from .semantic import chunk_semantic
from .sentence_window import chunk_sentence_window
from .sliding_window import chunk_sliding_window


def _sliding_window(segments):
    return chunk_sliding_window(
        segments,
        target_chars=settings.chunk_target_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )


def _recursive_token(segments):
    return chunk_recursive_token(
        segments,
        target_tokens=settings.recursive_token_size,
        overlap_tokens=settings.recursive_token_overlap,
    )


def _sentence_window(segments):
    return chunk_sentence_window(
        segments,
        window_size=settings.sentence_window_size,
        overlap=settings.sentence_window_overlap,
    )


def _semantic(segments):
    return chunk_semantic(segments, percentile=settings.semantic_percentile)


def _fixed_time(segments):
    return chunk_fixed_time(segments, window_seconds=settings.time_window_seconds)


def _parent_child(segments):
    return chunk_parent_child(
        segments,
        parent_chars=settings.parent_chunk_chars,
        parent_overlap=settings.parent_chunk_overlap,
        child_chars=settings.child_chunk_chars,
    )


STRATEGIES: dict[str, Callable[[list[dict]], list[dict]]] = {
    "sliding_window": _sliding_window,
    "recursive_token": _recursive_token,
    "sentence_window": _sentence_window,
    "semantic": _semantic,
    "fixed_time": _fixed_time,
    "parent_child": _parent_child,
}


def collection_name_for(strategy: str, base: str = "videos") -> str:
    return f"{base}_{strategy}"


__all__ = ["STRATEGIES", "collection_name_for"]
