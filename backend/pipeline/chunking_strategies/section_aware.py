"""Section-aware chunker. Designed for research papers but works on any text
whose `segments` carry a `section` field.

Behavior:
  - Group consecutive segments by `section`. Each section becomes a chunk.
  - If a section's combined text exceeds `max_chars`, subdivide on sentence
    boundaries while keeping the section title in every sub-chunk's text prefix
    (so the embedding picks up the section context).
  - Tiny tail sections (under `min_chars`) get merged into the previous one to
    avoid stub chunks for "References" boilerplate, page numbers, etc.

Each output chunk carries:
  - start, end: from first/last segment in the chunk (page numbers for PDFs)
  - text: the section's text (or a slice of it), with the section title prepended
  - section: the section title verbatim, for retrieval metadata + UI display
"""
import re

_SENT_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _flush_section(buf: list[dict], chunks: list[dict], max_chars: int, last_section: str) -> None:
    if not buf:
        return
    text = " ".join(s["text"].strip() for s in buf if s.get("text"))
    if not text:
        return
    start = buf[0]["start"]
    end = buf[-1]["end"]
    title_prefix = f"[{last_section}]\n\n" if last_section else ""

    if len(text) <= max_chars:
        chunks.append({"start": start, "end": end, "text": title_prefix + text, "section": last_section})
        return

    # Subdivide on sentence boundaries while keeping section title prefix
    sentences = _SENT_BREAK.split(text)
    cur = ""
    for sent in sentences:
        candidate = (cur + " " + sent).strip() if cur else sent
        if len(candidate) > max_chars and cur:
            chunks.append({"start": start, "end": end, "text": title_prefix + cur.strip(), "section": last_section})
            cur = sent
        else:
            cur = candidate
    if cur:
        chunks.append({"start": start, "end": end, "text": title_prefix + cur.strip(), "section": last_section})


def chunk_section_aware(
    segments: list[dict],
    max_chars: int = 1800,
    min_chars: int = 200,
    **_,
) -> list[dict]:
    if not segments:
        return []

    chunks: list[dict] = []
    buf: list[dict] = []
    current_section = segments[0].get("section", "") or ""

    for seg in segments:
        sec = seg.get("section", "") or ""
        if sec != current_section:
            _flush_section(buf, chunks, max_chars, current_section)
            buf = []
            current_section = sec
        buf.append(seg)
    _flush_section(buf, chunks, max_chars, current_section)

    # Merge tiny tail / micro chunks back into their predecessor
    merged: list[dict] = []
    for c in chunks:
        if merged and len(c["text"]) < min_chars:
            prev = merged[-1]
            prev["text"] = prev["text"].rstrip() + "\n\n" + c["text"].strip()
            prev["end"] = c["end"]
        else:
            merged.append(c)

    return merged
