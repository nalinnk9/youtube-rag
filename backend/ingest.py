"""Ingestion orchestration: playlist URL → segments → (per-strategy chunks) → embeddings → Chroma."""
from openai import OpenAI
from tqdm import tqdm

from .config import settings
from .pipeline.chunking_strategies import STRATEGIES, collection_name_for, strategies_for_source
from .pipeline.transcripts import get_segments
from .pipeline.vectorstore import embed_batch, get_collection
from .playlist import get_playlist_videos


def _already_ingested(coll, video_id: str) -> bool:
    res = coll.get(where={"video_id": video_id}, limit=1)
    return bool(res.get("ids"))


def _ingest_strategy(coll, oai: OpenAI, video: dict, segments: list[dict], strategy_name: str) -> int:
    if _already_ingested(coll, video["video_id"]):
        return 0
    chunker = STRATEGIES[strategy_name]
    chunks = chunker(segments)
    if not chunks:
        return 0

    vid = video["video_id"]
    ids = [f"{vid}_{strategy_name}_{i}" for i in range(len(chunks))]
    docs = [c["text"] for c in chunks]
    metas = []
    for c in chunks:
        meta = {
            "source_type": "youtube",
            "video_id": vid,
            "title": video["title"],
            "channel": video.get("channel", ""),
            "start": float(c["start"]),
            "end": float(c["end"]),
            "strategy": strategy_name,
        }
        if "parent_text" in c:
            meta["parent_text"] = c["parent_text"]
        metas.append(meta)

    embeddings = embed_batch(docs, settings.embedding_model, oai)
    coll.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
    return len(chunks)


def ingest_video(video: dict, oai: OpenAI, strategies: list[str] | None = None) -> dict:
    """Ingest one video under each enabled strategy. Returns per-strategy chunk counts."""
    strategies = strategies or strategies_for_source("youtube")
    counts: dict[str, int] = {}

    # Skip the transcript fetch entirely if every strategy already has this video
    all_done = True
    for s in strategies:
        coll = get_collection(settings.chroma_path, collection_name_for(s))
        if not _already_ingested(coll, video["video_id"]):
            all_done = False
            break
    if all_done:
        return {s: 0 for s in strategies}

    segments = get_segments(
        video["video_id"],
        languages=settings.language_list,
        whisper_model=settings.whisper_model,
    )
    if not segments:
        return {s: 0 for s in strategies}

    for s in strategies:
        coll = get_collection(settings.chroma_path, collection_name_for(s))
        try:
            counts[s] = _ingest_strategy(coll, oai, video, segments, s)
        except Exception as e:
            tqdm.write(f"  [{s}] FAILED on {video['video_id']}: {type(e).__name__}: {e}")
            counts[s] = 0

    return counts


def ingest_playlist(url: str) -> dict:
    """Enumerate a playlist URL and ingest every video under every enabled strategy."""
    videos = get_playlist_videos(url)
    print(f"Found {len(videos)} video(s) at {url}")
    print(f"Strategies: {', '.join(settings.strategy_list)}")

    oai = OpenAI(api_key=settings.openai_api_key)
    totals = {"videos": 0, "by_strategy": {s: 0 for s in settings.strategy_list}, "failed": 0}

    for video in tqdm(videos, desc="Ingesting"):
        title_short = (video["title"] or "")[:60]
        try:
            counts = ingest_video(video, oai)
            if any(counts.values()):
                totals["videos"] += 1
                for s, n in counts.items():
                    totals["by_strategy"][s] += n
                summary = ", ".join(f"{s}:{n}" for s, n in counts.items() if n)
                tqdm.write(f"  {title_short} — {summary}")
            else:
                tqdm.write(f"  skipped: {title_short}")
        except Exception as e:
            totals["failed"] += 1
            tqdm.write(f"  FAILED ({type(e).__name__}): {title_short} — {e}")

    print(f"\nDone. {totals['videos']} videos ingested, {totals['failed']} failed.")
    for s, n in totals["by_strategy"].items():
        print(f"  {s}: {n} chunks")
    return totals
