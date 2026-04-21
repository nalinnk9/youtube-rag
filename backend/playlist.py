"""Extract video metadata from a YouTube playlist or single video URL using yt-dlp."""
import yt_dlp


def get_playlist_videos(url: str) -> list[dict]:
    """Return [{video_id, title, channel, duration}, ...] for any playlist or single-video URL.

    Uses `extract_flat=True` so no videos are actually downloaded — this is purely a
    metadata enumeration call.
    """
    ydl_opts = {
        "extract_flat": True,
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    # Playlist result
    if info.get("_type") == "playlist" or "entries" in info:
        entries = info.get("entries") or []
        out = []
        for e in entries:
            if e is None:
                continue
            out.append({
                "video_id": e.get("id"),
                "title": e.get("title") or "Untitled",
                "channel": e.get("channel") or e.get("uploader") or info.get("uploader") or "",
                "duration": e.get("duration"),
            })
        return [v for v in out if v["video_id"]]

    # Single video result
    return [{
        "video_id": info["id"],
        "title": info.get("title") or "Untitled",
        "channel": info.get("channel") or info.get("uploader") or "",
        "duration": info.get("duration"),
    }]
