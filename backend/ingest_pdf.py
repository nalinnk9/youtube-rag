"""PDF ingestion: upload → extract → chunk under each PDF-compatible strategy → embed → Chroma.

Mirrors `backend.ingest` but for PDFs instead of YouTube videos. Each document is
stored under its own `doc_id` and chunks are tagged with `source_type='pdf'` and
`section` metadata so the frontend can render PDF citations differently from
video citations.
"""
import hashlib
import os
import uuid
from dataclasses import asdict

from openai import OpenAI

from .config import settings
from .pipeline.chunking_strategies import STRATEGIES, collection_name_for, strategies_for_source
from .pipeline.pdf import extract as pdf_extract
from .pipeline.vectorstore import embed_batch, get_collection


def save_upload(file_bytes: bytes, original_name: str) -> tuple[str, str]:
    """Persist an uploaded PDF to disk under a content-hashed name. Returns (doc_id, path)."""
    os.makedirs(settings.pdf_uploads_dir, exist_ok=True)
    sha = hashlib.sha1(file_bytes).hexdigest()[:16]
    doc_id = f"pdf_{sha}"
    out_path = os.path.join(settings.pdf_uploads_dir, f"{doc_id}.pdf")
    if not os.path.exists(out_path):
        with open(out_path, "wb") as f:
            f.write(file_bytes)
    return doc_id, out_path


def _segments_from_pages(pages: list[dict]) -> list[dict]:
    """Convert extracted pages into the {start, end, text, section} shape that all
    chunking strategies consume. `start` and `end` are page numbers (floats) so
    timestamp-style code paths continue to work.

    Each non-empty paragraph becomes its own segment so sentence_window and
    semantic chunkers have something fine-grained to work with.
    """
    segments: list[dict] = []
    for page in pages:
        page_num = float(page["page_num"])
        section = page.get("section", "") or ""
        text = (page.get("text") or "").strip()
        if not text:
            continue
        # Split page into paragraphs on double-newline; fall back to whole page.
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text]
        for para in paragraphs:
            segments.append({
                "start": page_num,
                "end": page_num,
                "text": para,
                "section": section,
            })
    return segments


def _already_ingested(coll, doc_id: str) -> bool:
    res = coll.get(where={"doc_id": doc_id}, limit=1)
    return bool(res.get("ids"))


def _ingest_strategy_for_doc(coll, oai: OpenAI, doc_id: str, meta: dict, segments: list[dict], strategy_name: str) -> int:
    if _already_ingested(coll, doc_id):
        return 0
    chunker = STRATEGIES[strategy_name]
    chunks = chunker(segments)
    if not chunks:
        return 0

    ids = [f"{doc_id}_{strategy_name}_{i}" for i in range(len(chunks))]
    docs = [c["text"] for c in chunks]
    metas: list[dict] = []
    for c in chunks:
        m = {
            "source_type": "pdf",
            "doc_id": doc_id,
            "title": meta.get("title", ""),
            "authors": meta.get("authors", ""),
            "extractor": meta.get("extractor", ""),
            "start": float(c["start"]),
            "end": float(c["end"]),
            "section": c.get("section", "") or "",
            "strategy": strategy_name,
        }
        if "parent_text" in c:
            m["parent_text"] = c["parent_text"]
        metas.append(m)

    embeddings = embed_batch(docs, settings.embedding_model, oai)
    coll.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
    return len(chunks)


def ingest_pdf_file(
    file_bytes: bytes,
    original_name: str,
    extractor: str = "pypdfium2",
) -> dict:
    """Save the upload, run the chosen extractor, then chunk + embed under every
    PDF-compatible strategy."""
    doc_id, path = save_upload(file_bytes, original_name)
    result = pdf_extract(path, extractor=extractor, doc_id=doc_id)

    meta = asdict(result.metadata)
    pages = [asdict(p) for p in result.pages]
    segments = _segments_from_pages(pages)

    oai = OpenAI(api_key=settings.openai_api_key)
    strategies = strategies_for_source("pdf")
    counts: dict[str, int] = {}
    for strat in strategies:
        coll = get_collection(settings.chroma_path, collection_name_for(strat))
        try:
            counts[strat] = _ingest_strategy_for_doc(coll, oai, doc_id, meta, segments, strat)
        except Exception as e:
            counts[strat] = 0
            print(f"  [{strat}] FAILED on {doc_id}: {type(e).__name__}: {e}")

    # Persist a lightweight doc record alongside chunks: write to a "documents"
    # collection so /documents endpoint can list ingested PDFs without scanning
    # every chunk collection.
    doc_coll = get_collection(settings.chroma_path, "documents")
    doc_coll.upsert(
        ids=[doc_id],
        documents=[meta.get("title", "Untitled")],
        # Chroma demands a non-null embedding; we never search this collection, so
        # a zero vector is fine.
        embeddings=[[0.0] * 1536],
        metadatas=[{
            "doc_id": doc_id,
            "source_type": "pdf",
            "title": meta.get("title", "Untitled"),
            "authors": meta.get("authors", ""),
            "num_pages": meta.get("num_pages", 0),
            "extractor": meta.get("extractor", ""),
            "path": path,
            "num_assets": len(result.assets),
            "original_name": original_name,
        }],
    )

    return {
        "doc_id": doc_id,
        "title": meta.get("title", "Untitled"),
        "authors": meta.get("authors", ""),
        "num_pages": meta.get("num_pages", 0),
        "extractor": meta.get("extractor", ""),
        "by_strategy": counts,
        "total_chunks": sum(counts.values()),
        "num_assets": len(result.assets),
    }
