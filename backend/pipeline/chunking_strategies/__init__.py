"""Registry of chunking strategies. Each strategy maps {start,end,text} segments
to chunks of {start,end,text} (parent-child also adds parent_text). Callers select
a strategy by name; the orchestrator (ingest, compare) iterates over the registry.
"""
from typing import Callable

from ...config import settings
from .fixed_time import chunk_fixed_time
from .parent_child import chunk_parent_child
from .recursive_token import chunk_recursive_token
from .section_aware import chunk_section_aware
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


def _section_aware(segments):
    return chunk_section_aware(
        segments,
        max_chars=settings.section_max_chars,
        min_chars=settings.section_min_chars,
    )


STRATEGIES: dict[str, Callable[[list[dict]], list[dict]]] = {
    "sliding_window": _sliding_window,
    "recursive_token": _recursive_token,
    "sentence_window": _sentence_window,
    "semantic": _semantic,
    "fixed_time": _fixed_time,
    "parent_child": _parent_child,
    "section_aware": _section_aware,
}

# Some strategies only make sense for certain source types. `fixed_time` requires
# real-valued timestamps; `section_aware` requires a `section` field on each
# segment. The ingest orchestrator filters strategies via this table.
SOURCE_STRATEGIES: dict[str, list[str]] = {
    "youtube": ["sliding_window", "recursive_token", "sentence_window", "semantic", "fixed_time", "parent_child"],
    "pdf":     ["sliding_window", "recursive_token", "sentence_window", "semantic", "parent_child", "section_aware"],
}


def strategies_for_source(source_type: str) -> list[str]:
    enabled = set(settings.strategy_list)
    allowed = SOURCE_STRATEGIES.get(source_type, list(STRATEGIES.keys()))
    return [s for s in allowed if s in enabled]


def collection_name_for(strategy: str, base: str = "videos") -> str:
    return f"{base}_{strategy}"


__all__ = ["STRATEGIES", "SOURCE_STRATEGIES", "strategies_for_source", "collection_name_for"]
