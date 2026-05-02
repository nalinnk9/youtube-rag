"""Fixed time-window chunker.

Bucket transcript segments into fixed-duration windows (e.g., 60 seconds each)
based on segment start time. Useful for video content because it produces
uniform-length temporal slices regardless of speech rate.
"""


def chunk_fixed_time(
    segments: list[dict],
    window_seconds: float = 60.0,
    **_,
) -> list[dict]:
    if not segments:
        return []

    chunks: list[dict] = []
    buf: list[dict] = []
    window_start = segments[0]["start"]

    for seg in segments:
        if seg["start"] - window_start >= window_seconds and buf:
            chunks.append({
                "start": buf[0]["start"],
                "end": buf[-1]["end"],
                "text": " ".join(s["text"] for s in buf),
            })
            buf = []
            window_start = seg["start"]
        buf.append(seg)

    if buf:
        chunks.append({
            "start": buf[0]["start"],
            "end": buf[-1]["end"],
            "text": " ".join(s["text"] for s in buf),
        })

    return chunks
