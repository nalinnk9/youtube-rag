"""Sliding character window chunker — original strategy preserved as a baseline.

Combines fine-grained transcript segments into chunks of approximately `target_chars`
characters, keeping a tail of recent segments as overlap so answers spanning a chunk
boundary are still retrievable. Each chunk preserves the start time of its first
segment and the end time of its last for deep-linking.
"""


def chunk_sliding_window(
    segments: list[dict],
    target_chars: int = 1000,
    overlap_chars: int = 150,
    **_,
) -> list[dict]:
    if not segments:
        return []

    chunks: list[dict] = []
    buf: list[dict] = []
    buf_len = 0

    for seg in segments:
        buf.append(seg)
        buf_len += len(seg["text"]) + 1

        if buf_len >= target_chars:
            chunks.append({
                "start": buf[0]["start"],
                "end": buf[-1]["end"],
                "text": " ".join(s["text"] for s in buf),
            })
            tail: list[dict] = []
            tail_len = 0
            for s in reversed(buf):
                if tail_len + len(s["text"]) > overlap_chars:
                    break
                tail.insert(0, s)
                tail_len += len(s["text"]) + 1
            buf = tail
            buf_len = tail_len

    if buf:
        chunks.append({
            "start": buf[0]["start"],
            "end": buf[-1]["end"],
            "text": " ".join(s["text"] for s in buf),
        })

    return chunks
