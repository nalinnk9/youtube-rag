"""FastAPI app exposing the ingestion + query pipeline and serving the frontend."""
import os

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

from .config import settings
from .ingest import ingest_playlist
from .pipeline.generation import generate_answer
from .pipeline.retrieval import rerank, retrieve
from .pipeline.vectorstore import get_collection


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
    print(f"   Embeddings: {settings.embedding_model}")
    print(f"   Chroma path: {settings.chroma_path}")


_startup_check()

app = FastAPI(title="YouTube RAG")

# Shared resources
_coll = get_collection(settings.chroma_path, settings.collection_name)
_oai = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None


class AskRequest(BaseModel):
    question: str


class IngestRequest(BaseModel):
    url: str


@app.post("/ask")
def ask(req: AskRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(400, "question cannot be empty")
    if _oai is None:
        raise HTTPException(500, "OPENAI_API_KEY not configured")

    hits = retrieve(_coll, q, _oai, k=settings.top_k_retrieve)
    ranked = rerank(q, hits, top_n=settings.top_k_rerank)
    return generate_answer(q, ranked)


@app.post("/ingest")
def ingest(req: IngestRequest, bg: BackgroundTasks):
    if not req.url.strip():
        raise HTTPException(400, "url cannot be empty")
    bg.add_task(ingest_playlist, req.url)
    return {"status": "started", "url": req.url, "hint": "check the server terminal for progress"}


@app.get("/stats")
def stats():
    all_ = _coll.get()
    video_ids = {m["video_id"] for m in (all_.get("metadatas") or []) if m.get("video_id")}
    return {"videos": len(video_ids), "chunks": len(all_.get("ids") or [])}


@app.get("/videos")
def list_videos():
    all_ = _coll.get()
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


# Serve frontend/index.html as the root
_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
