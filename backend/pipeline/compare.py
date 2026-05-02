"""Fan-out orchestration: run a single question across every chunking strategy in parallel,
then run the chosen judge over the combined results."""
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from ..config import settings
from .chunking_strategies import collection_name_for
from .generation import generate_answer
from .judge import run_judge
from .retrieval import rerank, retrieve
from .vectorstore import get_collection


def _run_one(strategy: str, question: str, oai: OpenAI) -> dict:
    t0 = time.time()
    coll = get_collection(settings.chroma_path, collection_name_for(strategy))
    hits = retrieve(coll, question, oai, k=settings.top_k_retrieve)
    ranked = rerank(question, hits, top_n=settings.top_k_rerank)
    answer_obj = generate_answer(question, ranked)
    latency_ms = int((time.time() - t0) * 1000)
    return {
        "strategy": strategy,
        "answer": answer_obj["answer"],
        "sources": answer_obj["sources"],
        "hits": ranked,
        "latency_ms": latency_ms,
        "n_retrieved": len(hits),
        "n_used": len(ranked),
    }


def ask_compare(question: str, judge_mode: str, oai: OpenAI, strategies: list[str] | None = None) -> dict:
    strategies = strategies or settings.strategy_list

    with ThreadPoolExecutor(max_workers=len(strategies)) as pool:
        futures = {pool.submit(_run_one, s, question, oai): s for s in strategies}
        results = []
        for fut in futures:
            try:
                results.append(fut.result())
            except Exception as e:
                strategy = futures[fut]
                results.append({
                    "strategy": strategy,
                    "answer": f"Error: {type(e).__name__}: {e}",
                    "sources": [],
                    "hits": [],
                    "latency_ms": 0,
                    "n_retrieved": 0,
                    "n_used": 0,
                    "error": str(e),
                })

    results.sort(key=lambda r: strategies.index(r["strategy"]))

    judge_payload = None
    judgeable = [r for r in results if r.get("answer") and not r.get("error")]
    if len(judgeable) >= 2:
        try:
            judge_payload = run_judge(judge_mode, question, judgeable)
        except Exception as e:
            judge_payload = {"mode": judge_mode, "error": f"{type(e).__name__}: {e}"}

    # Strip raw `hits` from the API response — they were only needed for the judge
    response_results = [{k: v for k, v in r.items() if k != "hits"} for r in results]

    return {"results": response_results, "judge": judge_payload}
