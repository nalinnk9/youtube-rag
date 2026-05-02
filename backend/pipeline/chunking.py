"""Backward-compatibility shim. The original sliding-window logic now lives in
`backend.pipeline.chunking_strategies.sliding_window`. This module re-exports it
so any callers importing `chunk_segments` continue to work.
"""
from .chunking_strategies.sliding_window import chunk_sliding_window as chunk_segments

__all__ = ["chunk_segments"]
