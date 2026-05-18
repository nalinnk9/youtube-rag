"""LLM answer generation with inline [n] citations that map back to source snippets."""
import re

from ..config import settings


def format_timestamp(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _build_prompt(question: str, hits: list[dict]) -> tuple[str, str]:
    has_pdf = any((h.get("source_type") or "youtube") == "pdf" for h in hits)
    base = (
        "You answer questions strictly from the provided snippets. "
        "Every factual claim must be followed by a citation marker like [1] or [2] "
        "matching the snippet numbers. Prefer concise, direct answers. "
        "If the snippets do not contain an answer, say exactly: "
        '"The indexed library doesn\'t cover this." Do not guess.'
    )
    paper_addendum = (
        "\n\nSome snippets are from research papers. For paper snippets: prefer direct "
        "quotes for empirical claims (numbers, results, definitions); never extrapolate "
        "beyond what the paper explicitly states; preserve technical terms and equations "
        "verbatim; if you summarize a method, label it as such."
    ) if has_pdf else ""
    system = base + paper_addendum

    context_blocks = []
    for i, h in enumerate(hits, start=1):
        if (h.get("source_type") or "youtube") == "pdf":
            section = h.get("section", "") or ""
            page = int(h.get("start") or 1)
            sec_part = f", §{section}" if section else ""
            context_blocks.append(
                f"[{i}] Paper: \"{h.get('title','Untitled')}\" (p. {page}{sec_part})\n{h.get('text','')}"
            )
        else:
            ts = format_timestamp(h.get("start") or 0)
            context_blocks.append(
                f"[{i}] Video: \"{h.get('title','Untitled')}\" (starts at {ts})\n{h.get('text','')}"
            )
    context = "\n\n".join(context_blocks)
    user = f"Question: {question}\n\nSnippets:\n{context}"
    return system, user


def _call_anthropic(system: str, user: str, max_tokens: int = 800, model: str | None = None) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    msg = client.messages.create(
        model=model or settings.llm_model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


def _call_openai(system: str, user: str, model: str | None = None) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=model or settings.llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def call_llm(system: str, user: str, max_tokens: int = 800) -> str:
    """Shared LLM-call helper. Routes to Anthropic or OpenAI based on settings."""
    provider = settings.llm_provider.lower()
    if provider == "anthropic":
        return _call_anthropic(system, user, max_tokens=max_tokens)
    if provider == "openai":
        return _call_openai(system, user)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")


def _source_from_pdf_hit(n: int, h: dict) -> dict:
    page = int(h.get("start") or 1)
    doc_id = h.get("doc_id", "")
    return {
        "n": n,
        "source_type": "pdf",
        "doc_id": doc_id,
        "title": h.get("title", "Untitled document"),
        "authors": h.get("authors", ""),
        "section": h.get("section", ""),
        "page": page,
        "start": page,
        "end": int(h.get("end") or page),
        "page_label": f"p. {page}",
        "url": f"/pdf_view?doc_id={doc_id}#page={page}" if doc_id else "",
        "thumbnail": f"/pdf_page?doc_id={doc_id}&page={page}" if doc_id else "",
        "snippet": h.get("text", ""),
    }


def _source_from_youtube_hit(n: int, h: dict) -> dict:
    start = int(h.get("start") or 0)
    video_id = h.get("video_id", "")
    return {
        "n": n,
        "source_type": "youtube",
        "video_id": video_id,
        "title": h.get("title", "Untitled"),
        "channel": h.get("channel", ""),
        "start": start,
        "end": int(h.get("end") or start),
        "timestamp": format_timestamp(h.get("start") or 0),
        "thumbnail": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
        "url": f"https://youtube.com/watch?v={video_id}&t={start}s",
        "snippet": h.get("text", ""),
    }


def build_sources_from_citations(text: str, hits: list[dict]) -> list[dict]:
    """Parse [n] markers out of `text` and resolve them to source dicts for the UI.

    Shared between `/ask` (which gets citations in prose) and `/visualize` (which
    gets them inside structured visual objects). The output shape matches the
    existing /ask response so the frontend renders citations identically.
    """
    cited_numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    sources: list[dict] = []
    for n in cited_numbers:
        if 1 <= n <= len(hits):
            h = hits[n - 1]
            if (h.get("source_type") or "youtube") == "pdf":
                sources.append(_source_from_pdf_hit(n, h))
            else:
                sources.append(_source_from_youtube_hit(n, h))
    return sources


def generate_answer(question: str, hits: list[dict]) -> dict:
    """Given a question and a list of retrieved hits, produce a cited answer."""
    if not hits:
        return {
            "answer": "The indexed library doesn't cover this (no relevant snippets found).",
            "sources": [],
        }

    system, user = _build_prompt(question, hits)
    text = call_llm(system, user, max_tokens=800)
    sources = build_sources_from_citations(text, hits)
    return {"answer": text, "sources": sources}
