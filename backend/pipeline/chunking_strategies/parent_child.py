"""Parent-child hierarchical chunker.

Builds two levels:
  - Parent chunks (~1500 chars) — the units sent to the LLM at answer time.
  - Child chunks (~300 chars) — the units indexed and embedded.

Each child carries its parent's full text in metadata as `parent_text` so retrieval
can swap the small embed-friendly chunk for a larger context-rich block before
generation.
"""
from .sliding_window import chunk_sliding_window


def chunk_parent_child(
    segments: list[dict],
    parent_chars: int = 1500,
    parent_overlap: int = 200,
    child_chars: int = 300,
    **_,
) -> list[dict]:
    if not segments:
        return []

    parents = chunk_sliding_window(segments, target_chars=parent_chars, overlap_chars=parent_overlap)
    out: list[dict] = []

    for parent in parents:
        # Slice the parent into child-sized pieces by character count, walking words
        words = parent["text"].split()
        if not words:
            continue
        children: list[str] = []
        cur = ""
        for w in words:
            candidate = (cur + " " + w) if cur else w
            if len(candidate) > child_chars and cur:
                children.append(cur)
                cur = w
            else:
                cur = candidate
        if cur:
            children.append(cur)

        if len(children) == 1:
            # Parent already small enough; index it as-is
            out.append({
                "start": parent["start"],
                "end": parent["end"],
                "text": children[0],
                "parent_text": parent["text"],
            })
            continue

        # Distribute parent's time range proportionally across children
        total_chars = sum(len(c) for c in children)
        running = 0
        duration = parent["end"] - parent["start"]
        for child in children:
            frac_start = running / total_chars if total_chars else 0
            running += len(child)
            frac_end = running / total_chars if total_chars else 1
            out.append({
                "start": parent["start"] + duration * frac_start,
                "end": parent["start"] + duration * frac_end,
                "text": child,
                "parent_text": parent["text"],
            })

    return out
