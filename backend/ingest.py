"""Ingestion orchestration: playlist URL → chunks → embeddings → Chroma."""
from openai import OpenAI
from tqdm import tqdm

from .config import settings
from .playlist import get_playlist_videos
from .pipeline.chunking import chunk_segments
from .pipeline.transcripts import get_segments
from .pipeline.vectorstore import embed_batch, get_collection


def _already_ingested(coll, video_id: str) -> bool:
    res = coll.get(where={"video_id": video_id}, limit=1)
    return bool(res.get("ids"))


def ingest_video(coll, oai: OpenAI, video: dict) -> int:
    """Ingest a single video. Returns number of chunks added (0 if skipped)."""
    vid = video["video_id"]
    if _already_ingested(coll, vid):
        return 0

    segments = get_segments(
        vid,
        languages=settings.language_list,
        whisper_model=settings.whisper_model,
    )
    if not segments:
        return 0

    chunks = chunk_segments(
        segments,
        target_chars=settings.chunk_target_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )
    if not chunks:
        return 0

    ids = [f"{vid}_{i}" for i in range(len(chunks))]
    docs = [c["text"] for c in chunks]
    metas = [
        {
            "video_id": vid,
            "title": video["title"],
            "channel": video.get("channel", ""),
            "start": float(c["start"]),
            "end": float(c["end"]),
        }
        for c in chunks
    ]
    embeddings = embed_batch(docs, settings.embedding_model, oai)
    coll.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
    return len(chunks)


def ingest_playlist(url: str) -> dict:
    """Enumerate a playlist (or single video) URL and ingest every video."""
    videos = get_playlist_videos(url)
    print(f"Found {len(videos)} video(s) at {url}")

    coll = get_collection(settings.chroma_path, settings.collection_name)
    oai = OpenAI(api_key=settings.openai_api_key)

    totals = {"ingested": 0, "chunks": 0, "skipped": 0, "failed": 0}

    for video in tqdm(videos, desc="Ingesting"):
        title_short = (video["title"] or "")[:60]
        try:
            n = ingest_video(coll, oai, video)
            if n == 0:
                totals["skipped"] += 1
                tqdm.write(f"  skipped: {title_short}")
            else:
                totals["ingested"] += 1
                totals["chunks"] += n
                tqdm.write(f"  +{n} chunks: {title_short}")
        except Exception as e:
            totals["failed"] += 1
            tqdm.write(f"  FAILED ({type(e).__name__}): {title_short} — {e}")

    print(
        f"\nDone. {totals['ingested']} videos ingested, "
        f"{totals['chunks']} chunks indexed, "
        f"{totals['skipped']} skipped, "
        f"{totals['failed']} failed."
    )
    return totals
