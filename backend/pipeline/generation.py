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
    system = (
        "You answer questions strictly from the provided video transcript snippets. "
        "Every factual claim must be followed by a citation marker like [1] or [2] "
        "matching the snippet numbers. Prefer concise, direct answers. "
        "If the snippets do not contain an answer, say exactly: "
        '"The indexed videos don\'t cover this." Do not guess.'
    )
    context_blocks = []
    for i, h in enumerate(hits, start=1):
        ts = format_timestamp(h["start"])
        context_blocks.append(
            f"[{i}] Video: \"{h['title']}\" (starts at {ts})\n{h['text']}"
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
            start = int(h["start"])
            sources.append({
                "n": n,
                "video_id": h["video_id"],
                "title": h["title"],
                "channel": h.get("channel", ""),
                "start": start,
                "end": int(h["end"]),
                "timestamp": format_timestamp(h["start"]),
                "thumbnail": f"https://img.youtube.com/vi/{h['video_id']}/hqdefault.jpg",
                "url": f"https://youtube.com/watch?v={h['video_id']}&t={start}s",
                "snippet": h["text"],
            })
    return sources


def generate_answer(question: str, hits: list[dict]) -> dict:
    """Given a question and a list of retrieved hits, produce a cited answer."""
    if not hits:
        return {
            "answer": "The indexed videos don't cover this (no relevant snippets found).",
            "sources": [],
        }

    system, user = _build_prompt(question, hits)
    text = call_llm(system, user, max_tokens=800)
    sources = build_sources_from_citations(text, hits)
    return {"answer": text, "sources": sources}
