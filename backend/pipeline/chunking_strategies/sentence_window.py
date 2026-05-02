"""Sentence-window chunker.

Splits the transcript into sentences (regex on .!? with abbreviation guard), then
groups N sentences per chunk with a configurable overlap of M sentences. Each
sentence carries its source segment's timestamp so chunk start/end remain accurate.
"""
import re

_ABBREV = {"mr", "mrs", "ms", "dr", "st", "vs", "etc", "i.e", "e.g", "fig", "ph.d"}
_SENT_END = re.compile(r"([.!?])\s+")


def _split_sentences_with_ts(segments: list[dict]) -> list[dict]:
    """Walk segments, emit (text, start, end) per sentence."""
    sentences: list[dict] = []
    buf = ""
    buf_start: float | None = None
    buf_end: float | None = None

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        if buf_start is None:
            buf_start = seg["start"]
        buf_end = seg["end"]
        buf = (buf + " " + text).strip() if buf else text

        # Try to flush completed sentences from buf
        while True:
            m = _SENT_END.search(buf)
            if not m:
                break
            end_idx = m.end()
            candidate = buf[:end_idx].strip()
            tail_word = candidate.rsplit(" ", 1)[-1].rstrip(".!?").lower()
            if tail_word in _ABBREV:
                # Skip abbreviation, keep scanning further
                next_search = _SENT_END.search(buf, end_idx)
                if not next_search:
                    break
                end_idx = next_search.end()
                candidate = buf[:end_idx].strip()
            sentences.append({"text": candidate, "start": buf_start, "end": buf_end})
            buf = buf[end_idx:].lstrip()
            buf_start = seg["start"] if buf else None

    if buf and buf_start is not None and buf_end is not None:
        sentences.append({"text": buf.strip(), "start": buf_start, "end": buf_end})

    return sentences


def chunk_sentence_window(
    segments: list[dict],
    window_size: int = 5,
    overlap: int = 1,
    **_,
) -> list[dict]:
    if not segments:
        return []
    sentences = _split_sentences_with_ts(segments)
    if not sentences:
        return []

    chunks: list[dict] = []
    step = max(1, window_size - overlap)
    for i in range(0, len(sentences), step):
        window = sentences[i:i + window_size]
        if not window:
            break
        chunks.append({
            "start": window[0]["start"],
            "end": window[-1]["end"],
            "text": " ".join(s["text"] for s in window),
        })
        if i + window_size >= len(sentences):
            break

    return chunks
