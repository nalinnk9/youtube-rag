#!/usr/bin/env python
"""CLI: re-index already-ingested videos under additional chunking strategies.

Reads video IDs from any existing per-strategy collection (or the legacy `videos`
collection), re-fetches their transcripts, and writes chunks into the requested
strategies' collections.

Usage:
    python -m scripts.reindex                     # re-index all videos under all enabled strategies
    python -m scripts.reindex --strategies semantic,fixed_time
    python -m scripts.reindex --source videos     # read source video list from legacy collection
"""
import argparse
import sys

from openai import OpenAI
from tqdm import tqdm

from backend.config import settings
from backend.ingest import ingest_video
from backend.pipeline.chunking_strategies import collection_name_for
from backend.pipeline.vectorstore import get_collection


def _collect_videos(source_collection: str) -> list[dict]:
    coll = get_collection(settings.chroma_path, source_collection)
    all_ = coll.get()
    seen: dict[str, dict] = {}
    for m in (all_.get("metadatas") or []):
        vid = m.get("video_id")
        if vid and vid not in seen:
            seen[vid] = {
                "video_id": vid,
                "title": m.get("title", ""),
                "channel": m.get("channel", ""),
            }
    return list(seen.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategies",
        default=",".join(settings.strategy_list),
        help="Comma-separated strategies to re-index into",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Source collection to read video IDs from (default: first enabled strategy, then 'videos')",
    )
    args = parser.parse_args()

    target_strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    for s in target_strategies:
        if s not in settings.strategy_list:
            print(f"WARNING: '{s}' is not in enabled_strategies; will still write a collection for it.")

    sources_to_try = []
    if args.source:
        sources_to_try.append(args.source)
    else:
        sources_to_try.append(collection_name_for(settings.default_strategy))
        sources_to_try.append("videos")

    videos: list[dict] = []
    used_source = None
    for src in sources_to_try:
        videos = _collect_videos(src)
        if videos:
            used_source = src
            break

    if not videos:
        print("No videos found in any source collection. Run `ingest_playlist` first.")
        sys.exit(1)

    print(f"Found {len(videos)} video(s) in '{used_source}'")
    print(f"Re-indexing under: {', '.join(target_strategies)}")

    oai = OpenAI(api_key=settings.openai_api_key)
    totals = {s: 0 for s in target_strategies}

    for video in tqdm(videos, desc="Re-indexing"):
        title_short = (video["title"] or "")[:60]
        try:
            counts = ingest_video(video, oai, strategies=target_strategies)
            for s, n in counts.items():
                totals[s] += n
            summary = ", ".join(f"{s}:{n}" for s, n in counts.items() if n) or "skipped"
            tqdm.write(f"  {title_short} — {summary}")
        except Exception as e:
            tqdm.write(f"  FAILED ({type(e).__name__}): {title_short} — {e}")

    print("\nDone.")
    for s, n in totals.items():
        print(f"  {s}: +{n} chunks")


if __name__ == "__main__":
    main()
