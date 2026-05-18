"""FastAPI app exposing the ingestion + query pipeline and serving the frontend."""
import os

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

from .config import settings
from .ingest import ingest_playlist
from .pipeline.chunking_strategies import collection_name_for
from .pipeline.compare import ask_compare
from .pipeline.generation import generate_answer
from .pipeline.judge import judge_visuals
from .pipeline.retrieval import rerank, retrieve
from .pipeline.vectorstore import get_collection
from .pipeline.visuals import generate_visuals


def _startup_check():
    if not settings.openai_api_key:
        print("⚠️  OPENAI_API_KEY not set — embeddings will fail.")
    if settings.llm_provider == "anthropic" and not settings.anthropic_api_key:
        print("⚠️  ANTHROPIC_API_KEY not set — /ask will fail.")
    if settings.llm_provider == "openai" and not settings.openai_api_key:
        print("⚠️  OPENAI_API_KEY not set — /ask will fail.")
    if not settings.cohere_api_key:
        print("ℹ️  COHERE_API_KEY not set — reranking disabled (falling back to top-K).")
    print(f"   LLM: {settings.llm_provider} / {settings.llm_model}")
    print(f"   Judge: {settings.judge_provider} / {settings.judge_model}")
    print(f"   Embeddings: {settings.embedding_model}")
    print(f"   Chroma path: {settings.chroma_path}")
    print(f"   Strategies: {', '.join(settings.strategy_list)}")


_startup_check()

app = FastAPI(title="YouTube RAG")

_oai = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None


def _default_collection():
    return get_collection(settings.chroma_path, collection_name_for(settings.default_strategy))


class AskRequest(BaseModel):
    question: str
    strategy: str | None = None


class AskCompareRequest(BaseModel):
    question: str
    judge_mode: str = "answer"


class IngestRequest(BaseModel):
    url: str


class VisualizeRequest(BaseModel):
    question: str
    strategy: str | None = None
    judge: bool = False


@app.post("/ask")
def ask(req: AskRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(400, "question cannot be empty")
    if _oai is None:
        raise HTTPException(500, "OPENAI_API_KEY not configured")

    strategy = req.strategy or settings.default_strategy
    if strategy not in settings.strategy_list:
        raise HTTPException(400, f"unknown strategy: {strategy}")

    coll = get_collection(settings.chroma_path, collection_name_for(strategy))
    hits = retrieve(coll, q, _oai, k=settings.top_k_retrieve)
    ranked = rerank(q, hits, top_n=settings.top_k_rerank)
    return generate_answer(q, ranked)


@app.post("/visualize")
def visualize(req: VisualizeRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(400, "question cannot be empty")
    if _oai is None:
        raise HTTPException(500, "OPENAI_API_KEY not configured")

    strategy = req.strategy or settings.default_strategy
    if strategy not in settings.strategy_list:
        raise HTTPException(400, f"unknown strategy: {strategy}")

    coll = get_collection(settings.chroma_path, collection_name_for(strategy))
    hits = retrieve(coll, q, _oai, k=settings.top_k_retrieve)
    ranked = rerank(q, hits, top_n=settings.top_k_rerank)

    out = generate_visuals(q, ranked)
    if req.judge and out.get("visuals"):
        try:
            out["judge"] = judge_visuals(q, out["visuals"], ranked)
        except Exception as e:
            out["judge"] = {"mode": "visuals", "error": f"{type(e).__name__}: {e}"}
    return out


@app.post("/ask_compare")
def ask_compare_endpoint(req: AskCompareRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(400, "question cannot be empty")
    if req.judge_mode not in {"answer", "retrieval", "both"}:
        raise HTTPException(400, f"unknown judge_mode: {req.judge_mode}")
    if _oai is None:
        raise HTTPException(500, "OPENAI_API_KEY not configured")

    return ask_compare(q, req.judge_mode, _oai)


@app.get("/strategies")
def strategies():
    out = []
    for s in settings.strategy_list:
        coll = get_collection(settings.chroma_path, collection_name_for(s))
        all_ = coll.get()
        video_ids = {m["video_id"] for m in (all_.get("metadatas") or []) if m.get("video_id")}
        out.append({
            "name": s,
            "videos": len(video_ids),
            "chunks": len(all_.get("ids") or []),
            "is_default": s == settings.default_strategy,
        })
    return {"strategies": out, "default": settings.default_strategy}


@app.post("/ingest")
def ingest(req: IngestRequest, bg: BackgroundTasks):
    if not req.url.strip():
        raise HTTPException(400, "url cannot be empty")
    bg.add_task(ingest_playlist, req.url)
    return {"status": "started", "url": req.url, "hint": "check the server terminal for progress"}


@app.get("/stats")
def stats():
    coll = _default_collection()
    all_ = coll.get()
    video_ids = {m["video_id"] for m in (all_.get("metadatas") or []) if m.get("video_id")}
    return {"videos": len(video_ids), "chunks": len(all_.get("ids") or [])}


@app.get("/videos")
def list_videos():
    coll = _default_collection()
    all_ = coll.get()
    seen: dict[str, dict] = {}
    for m in (all_.get("metadatas") or []):
        vid = m.get("video_id")
        if vid and vid not in seen:
            seen[vid] = {
                "video_id": vid,
                "title": m.get("title", "Untitled"),
                "channel": m.get("channel", ""),
                "thumbnail": f"https://img.youtube.com/vi/{vid}/default.jpg",
            }
    return list(seen.values())


_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
