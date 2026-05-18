"""FastAPI app exposing the ingestion + query pipeline and serving the frontend."""
import os

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

from .config import settings
from .ingest import ingest_playlist
from .ingest_pdf import ingest_pdf_file
from .pipeline.chunking_strategies import collection_name_for
from .pipeline.compare import ask_compare
from .pipeline.generation import generate_answer
from .pipeline.judge import judge_visuals
from .pipeline.pdf import available_extractors as _available_extractors
from .pipeline.pdf import render_page as _render_pdf_page
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


@app.get("/extractors")
def list_extractors():
    return {"extractors": _available_extractors()}


@app.post("/ingest_pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    extractor: str = Form(default=""),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "only .pdf files are accepted")
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "uploaded file is empty")
    if len(raw) > settings.pdf_max_bytes:
        raise HTTPException(413, f"file exceeds {settings.pdf_max_bytes} bytes")

    chosen = (extractor or settings.pdf_default_extractor).lower()
    valid = {e["name"] for e in _available_extractors() if e["available"]}
    if chosen not in valid:
        raise HTTPException(400, f"extractor '{chosen}' is not available. Available: {sorted(valid)}")

    try:
        result = ingest_pdf_file(raw, file.filename, extractor=chosen)
    except ImportError as e:
        raise HTTPException(400, str(e))
    return result


@app.get("/pdf_page")
def pdf_page(doc_id: str, page: int = 1):
    doc_coll = get_collection(settings.chroma_path, "documents")
    res = doc_coll.get(ids=[doc_id])
    if not res.get("ids"):
        raise HTTPException(404, f"unknown doc_id: {doc_id}")
    path = (res.get("metadatas") or [{}])[0].get("path")
    if not path or not os.path.exists(path):
        raise HTTPException(404, "PDF file missing on disk")

    os.makedirs(settings.pdf_renders_dir, exist_ok=True)
    cache_dir = os.path.join(settings.pdf_renders_dir, doc_id)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"page-{page}.png")

    if not os.path.exists(cache_path):
        try:
            png = _render_pdf_page(path, page, scale=settings.pdf_render_scale)
        except Exception as e:
            raise HTTPException(500, f"render failed: {type(e).__name__}: {e}")
        with open(cache_path, "wb") as f:
            f.write(png)
    return FileResponse(cache_path, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/pdf_view")
def pdf_view(doc_id: str):
    """Serve the raw PDF for in-browser viewing (uses #page=N anchor)."""
    doc_coll = get_collection(settings.chroma_path, "documents")
    res = doc_coll.get(ids=[doc_id])
    if not res.get("ids"):
        raise HTTPException(404, f"unknown doc_id: {doc_id}")
    path = (res.get("metadatas") or [{}])[0].get("path")
    if not path or not os.path.exists(path):
        raise HTTPException(404, "PDF file missing on disk")
    return FileResponse(path, media_type="application/pdf")


@app.get("/documents")
def list_documents():
    doc_coll = get_collection(settings.chroma_path, "documents")
    all_ = doc_coll.get()
    docs = []
    for i, _id in enumerate(all_.get("ids") or []):
        m = (all_.get("metadatas") or [])[i] or {}
        docs.append({
            "doc_id": _id,
            "title": m.get("title", "Untitled"),
            "authors": m.get("authors", ""),
            "num_pages": m.get("num_pages", 0),
            "extractor": m.get("extractor", ""),
            "num_assets": m.get("num_assets", 0),
            "original_name": m.get("original_name", ""),
        })
    return {"documents": docs}


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
