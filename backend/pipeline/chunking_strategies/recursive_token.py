"""Token-aware recursive splitter using tiktoken.

Joins all transcript segments into a single text and splits recursively on
paragraph -> newline -> sentence -> word -> char boundaries until each piece is
within `target_tokens`. Timestamps are reattached by mapping each chunk's character
range back to the segments it covers.
"""
import tiktoken

_ENC = None


def _enc():
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


def _split_recursive(text: str, target_tokens: int, separators: list[str]) -> list[str]:
    enc = _enc()
    if len(enc.encode(text)) <= target_tokens:
        return [text]

    if not separators:
        # Hard char-level split as last resort
        token_ids = enc.encode(text)
        pieces = []
        for i in range(0, len(token_ids), target_tokens):
            pieces.append(enc.decode(token_ids[i:i + target_tokens]))
        return pieces

    sep, *rest = separators
    parts = text.split(sep) if sep else list(text)

    out: list[str] = []
    cur = ""
    for p in parts:
        candidate = (cur + sep + p) if cur else p
        if len(enc.encode(candidate)) <= target_tokens:
            cur = candidate
        else:
            if cur:
                out.append(cur)
            if len(enc.encode(p)) > target_tokens:
                out.extend(_split_recursive(p, target_tokens, rest))
                cur = ""
            else:
                cur = p
    if cur:
        out.append(cur)
    return out


def _attach_timestamps(text_chunk: str, segments: list[dict], full_text: str, char_offset: int) -> tuple[float, float, int]:
    """Find which segments overlap the chunk in the joined text and return (start, end, new_offset)."""
    start_char = full_text.find(text_chunk, char_offset)
    if start_char < 0:
        start_char = char_offset
    end_char = start_char + len(text_chunk)

    # Walk segments accumulating their lengths to find overlap
    pos = 0
    first_seg = None
    last_seg = None
    for seg in segments:
        seg_start = pos
        seg_end = pos + len(seg["text"]) + 1  # +1 for join space
        if seg_end > start_char and seg_start < end_char:
            if first_seg is None:
                first_seg = seg
            last_seg = seg
        pos = seg_end
        if seg_start >= end_char:
            break

    if first_seg is None:
        first_seg = segments[0]
        last_seg = segments[-1]
    return first_seg["start"], last_seg["end"], end_char


def chunk_recursive_token(
    segments: list[dict],
    target_tokens: int = 256,
    overlap_tokens: int = 50,
    **_,
) -> list[dict]:
    if not segments:
        return []

    full_text = " ".join(s["text"] for s in segments)
    pieces = _split_recursive(full_text, target_tokens, ["\n\n", "\n", ". ", " ", ""])

    enc = _enc()
    if overlap_tokens > 0 and len(pieces) > 1:
        with_overlap: list[str] = []
        for i, p in enumerate(pieces):
            if i == 0:
                with_overlap.append(p)
                continue
            prev_tokens = enc.encode(pieces[i - 1])
            tail = enc.decode(prev_tokens[-overlap_tokens:]) if len(prev_tokens) > overlap_tokens else pieces[i - 1]
            with_overlap.append(tail + " " + p)
        pieces = with_overlap

    chunks: list[dict] = []
    offset = 0
    for piece in pieces:
        start, end, offset = _attach_timestamps(piece, segments, full_text, offset)
        chunks.append({"start": start, "end": end, "text": piece})

    return chunks
