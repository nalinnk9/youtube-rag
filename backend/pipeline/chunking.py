"""Group fine-grained transcript segments into retrieval chunks with preserved timestamps."""


def chunk_segments(
    segments: list[dict],
    target_chars: int = 1000,
    overlap_chars: int = 150,
) -> list[dict]:
    """Combine transcript segments into chunks of approximately `target_chars` characters.

    Each chunk carries the start time of its first segment and the end time of its
    last, so the retrieved chunk can deep-link back to the exact moment in the video.

    Uses a character overlap between consecutive chunks to reduce the chance that an
    answer straddling a chunk boundary gets missed at retrieval time.
    """
    if not segments:
        return []

    chunks: list[dict] = []
    buf: list[dict] = []
    buf_len = 0

    for seg in segments:
        buf.append(seg)
        buf_len += len(seg["text"]) + 1  # +1 for the joining space

        if buf_len >= target_chars:
            chunks.append({
                "start": buf[0]["start"],
                "end": buf[-1]["end"],
                "text": " ".join(s["text"] for s in buf),
            })
            # Carry a tail of recent segments forward for overlap
            tail: list[dict] = []
            tail_len = 0
            for s in reversed(buf):
                if tail_len + len(s["text"]) > overlap_chars:
                    break
                tail.insert(0, s)
                tail_len += len(s["text"]) + 1
            buf = tail
            buf_len = tail_len

    # Flush anything left over
    if buf:
        chunks.append({
            "start": buf[0]["start"],
            "end": buf[-1]["end"],
            "text": " ".join(s["text"] for s in buf),
        })

    return chunks
