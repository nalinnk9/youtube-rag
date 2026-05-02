"""LLM-as-judge for comparing chunking strategies on the same question.

Three modes:
  - "answer":    score each strategy's final answer (faithfulness, relevance,
                 completeness, citation quality).
  - "retrieval": score each strategy's retrieved chunks for relevance to the
                 question, ignoring generation quality.
  - "both":      run both judges (answer + retrieval) and return both score sets.

A single judge call evaluates all strategies at once so scoring is internally
consistent. Returns per-strategy score dicts plus a winner and reasoning.
"""
import json
import re

from ..config import settings


def _call_judge(system: str, user: str) -> str:
    provider = settings.judge_provider.lower()
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        msg = client.messages.create(
            model=settings.judge_model,
            max_tokens=1500,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=settings.judge_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""
    raise ValueError(f"Unknown judge_provider: {settings.judge_provider!r}")


def _parse_json(text: str) -> dict:
    """Pull a JSON object out of the LLM response, tolerating prose around it."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    obj = re.search(r"\{.*\}", text, re.DOTALL)
    if obj:
        return json.loads(obj.group(0))
    raise ValueError(f"No JSON object found in judge response: {text[:300]}")


_ANSWER_SYSTEM = (
    "You are an impartial RAG evaluator comparing answers from different chunking strategies. "
    "Score each strategy's answer on a 1-5 integer scale across four axes:\n"
    "  - faithfulness:    does every claim follow from the cited snippets?\n"
    "  - relevance:       does the answer address the user's question?\n"
    "  - completeness:    does it cover the question fully (not partial)?\n"
    "  - citation:        are [n] markers used correctly and pointing to the right source?\n"
    "Return strict JSON with this shape:\n"
    "{\"scores\": {\"<strategy>\": {\"faithfulness\": int, \"relevance\": int, \"completeness\": int, \"citation\": int, \"total\": int}}, "
    "\"winner\": \"<strategy>\", \"reasoning\": \"<2-3 sentences>\"}\n"
    "`total` must equal the sum of the four axes. Pick the highest-total strategy as winner; break ties by faithfulness then relevance."
)


_RETRIEVAL_SYSTEM = (
    "You are an impartial retrieval evaluator. For each chunking strategy you will see the question and "
    "the chunks it retrieved. Score each strategy's chunk set on a 1-5 integer scale:\n"
    "  - relevance:   how relevant are the retrieved chunks to the question?\n"
    "  - coverage:    do the chunks together contain enough info to answer the question?\n"
    "  - precision:   what fraction of the chunks are useful (vs. noise)?\n"
    "  - granularity: are the chunks the right size (not too fragmented, not too bloated)?\n"
    "Return strict JSON:\n"
    "{\"scores\": {\"<strategy>\": {\"relevance\": int, \"coverage\": int, \"precision\": int, \"granularity\": int, \"total\": int}}, "
    "\"winner\": \"<strategy>\", \"reasoning\": \"<2-3 sentences>\"}\n"
    "Pick the highest-total strategy as winner; break ties by relevance then coverage."
)


def _format_answer_block(question: str, results: list[dict]) -> str:
    parts = [f"Question: {question}\n"]
    for r in results:
        snippet_lines = []
        for i, src in enumerate(r.get("sources", []), start=1):
            snip = (src.get("snippet") or "")[:300]
            snippet_lines.append(f"  [{i}] \"{src.get('title', '')}\" @ {src.get('timestamp', '')}: {snip}")
        snippets_text = "\n".join(snippet_lines) if snippet_lines else "  (no sources)"
        parts.append(
            f"---\nStrategy: {r['strategy']}\n"
            f"Answer: {r['answer']}\n"
            f"Sources used:\n{snippets_text}"
        )
    return "\n\n".join(parts)


def _format_retrieval_block(question: str, results: list[dict]) -> str:
    parts = [f"Question: {question}\n"]
    for r in results:
        chunk_lines = []
        for i, h in enumerate(r.get("hits", []), start=1):
            text = (h.get("text") or "")[:400]
            chunk_lines.append(f"  [{i}] {text}")
        chunks_text = "\n".join(chunk_lines) if chunk_lines else "  (no chunks retrieved)"
        parts.append(f"---\nStrategy: {r['strategy']}\nRetrieved chunks:\n{chunks_text}")
    return "\n\n".join(parts)


def judge_answers(question: str, results: list[dict]) -> dict:
    user = _format_answer_block(question, results)
    raw = _call_judge(_ANSWER_SYSTEM, user)
    parsed = _parse_json(raw)
    parsed["mode"] = "answer"
    return parsed


def judge_retrieval(question: str, results: list[dict]) -> dict:
    user = _format_retrieval_block(question, results)
    raw = _call_judge(_RETRIEVAL_SYSTEM, user)
    parsed = _parse_json(raw)
    parsed["mode"] = "retrieval"
    return parsed


def run_judge(mode: str, question: str, results: list[dict]) -> dict:
    """Dispatch to the requested judge mode. `results` must include the fields
    each judge needs: `answer`+`sources` for answer mode, `hits` for retrieval mode."""
    mode = (mode or "answer").lower()
    if mode == "answer":
        return judge_answers(question, results)
    if mode == "retrieval":
        return judge_retrieval(question, results)
    if mode == "both":
        return {
            "mode": "both",
            "answer": judge_answers(question, results),
            "retrieval": judge_retrieval(question, results),
        }
    raise ValueError(f"Unknown judge mode: {mode!r}")
