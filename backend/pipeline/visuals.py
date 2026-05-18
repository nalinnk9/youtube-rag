"""Generate structured visual explanations (cards, diagrams) from retrieved chunks.

Returns up to 4 visual elements, each carrying inline citations [n] that map back
to the retrieved hits the same way text answers do. The frontend renders concept
cards, step lists, comparison tables, and Mermaid diagrams from this output.

Faithfulness is enforced via prompting: every visual must derive from the
provided snippets and include the citation numbers it relies on.
"""
import json
import re

from .generation import build_sources_from_citations, call_llm, format_timestamp


_SYSTEM = (
    "You are a learning-content designer. From the user's question and the provided "
    "video transcript snippets, produce 2 to 4 structured visual elements that help "
    "explain the answer. Visuals MUST derive from the snippets — never invent facts "
    "or details that aren't supported there.\n\n"
    "Output STRICT JSON in this exact shape:\n"
    "{\n"
    "  \"visuals\": [\n"
    "    { \"type\": \"concept_card\", \"title\": \"...\", \"body\": \"... [n] ...\", \"citations\": [n, ...] },\n"
    "    { \"type\": \"key_steps\",    \"title\": \"...\", \"steps\": [\"step one [n]\", \"step two [n]\"], \"citations\": [n, ...] },\n"
    "    { \"type\": \"comparison_table\", \"title\": \"...\", \"columns\": [\"col1\", \"col2\"], \"rows\": [[\"a\", \"b\"], [\"c\", \"d\"]], \"citations\": [n, ...] },\n"
    "    { \"type\": \"mermaid\", \"title\": \"...\", \"code\": \"flowchart LR\\n  A-->B\", \"citations\": [n, ...] }\n"
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "- Pick the visual types that best fit the question. Skip types that don't fit.\n"
    "- Use Mermaid only for processes, flows, hierarchies, or sequences. Supported diagram kinds: flowchart, sequenceDiagram, mindmap, classDiagram. Keep it under 12 nodes for readability.\n"
    "- Every `body` and every `step` string SHOULD contain at least one inline [n] citation referencing the snippets that support it.\n"
    "- Every visual MUST include a `citations` array listing all [n] numbers used.\n"
    "- Concept_card body uses markdown but stays under ~300 words.\n"
    "- Comparison table rows must all match the `columns` length.\n"
    "- If the snippets don't support any useful visual, return {\"visuals\": []}.\n"
    "- Return ONLY the JSON object — no prose, no markdown fence around the JSON."
)


def _format_user(question: str, hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        ts = format_timestamp(h["start"])
        blocks.append(f"[{i}] Video: \"{h['title']}\" (at {ts})\n{h['text']}")
    return f"Question: {question}\n\nSnippets:\n" + "\n\n".join(blocks)


def _parse_json(text: str) -> dict:
    """Tolerant JSON extraction: strips ```json fences and finds the first object."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    obj = re.search(r"\{.*\}", text, re.DOTALL)
    if obj:
        return json.loads(obj.group(0))
    raise ValueError(f"No JSON object found in visuals response: {text[:300]}")


_ALLOWED_TYPES = {"concept_card", "key_steps", "comparison_table", "mermaid"}


def _validate(visuals: list[dict]) -> list[dict]:
    """Drop any visual that doesn't match the schema. Defense against bad LLM output."""
    out: list[dict] = []
    for v in visuals:
        if not isinstance(v, dict):
            continue
        t = v.get("type")
        if t not in _ALLOWED_TYPES:
            continue
        if not v.get("title"):
            continue
        if t == "concept_card" and not v.get("body"):
            continue
        if t == "key_steps":
            steps = v.get("steps") or []
            if not isinstance(steps, list) or not steps:
                continue
        if t == "comparison_table":
            cols = v.get("columns") or []
            rows = v.get("rows") or []
            if not isinstance(cols, list) or not isinstance(rows, list) or not cols or not rows:
                continue
            if any(not isinstance(r, list) or len(r) != len(cols) for r in rows):
                continue
        if t == "mermaid" and not v.get("code"):
            continue
        if not isinstance(v.get("citations"), list):
            v["citations"] = []
        out.append(v)
    return out


def _collect_citations(visuals: list[dict]) -> str:
    """Concatenate every [n] mention across all visuals so build_sources_from_citations
    can resolve them to the same source-dict shape /ask returns."""
    parts: list[str] = []
    for v in visuals:
        if v.get("body"):
            parts.append(v["body"])
        for s in v.get("steps", []) or []:
            parts.append(str(s))
        for r in v.get("rows", []) or []:
            parts.append(" ".join(str(c) for c in r))
        if v.get("code"):
            parts.append(v["code"])
        for n in v.get("citations", []) or []:
            parts.append(f"[{n}]")
    return "\n".join(parts)


def generate_visuals(question: str, hits: list[dict]) -> dict:
    """Return {visuals: [...], sources: [...]}.

    `sources` mirrors the /ask response shape so the frontend can resolve [n]
    citation pills the same way. An empty `visuals` array means the model judged
    the snippets insufficient to support any faithful visual — UI shows a hint.
    """
    if not hits:
        return {"visuals": [], "sources": []}

    user = _format_user(question, hits)
    raw = call_llm(_SYSTEM, user, max_tokens=2000)

    try:
        parsed = _parse_json(raw)
    except (ValueError, json.JSONDecodeError):
        return {"visuals": [], "sources": [], "error": "Visuals response was not valid JSON."}

    visuals = _validate(parsed.get("visuals") or [])
    sources = build_sources_from_citations(_collect_citations(visuals), hits)
    return {"visuals": visuals, "sources": sources}
