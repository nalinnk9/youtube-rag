#!/usr/bin/env python
"""CLI: ingest a YouTube playlist (or single video) URL into the vector DB.

Usage:
    python -m scripts.ingest_playlist "https://www.youtube.com/playlist?list=PLxxxx"
    python -m scripts.ingest_playlist "https://www.youtube.com/watch?v=xxxx"
"""
import sys

from backend.ingest import ingest_playlist


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.ingest_playlist <playlist_or_video_url>")
        sys.exit(1)
    url = sys.argv[1]
    ingest_playlist(url)


if __name__ == "__main__":
    main()
