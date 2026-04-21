"""Hybrid transcript extraction: try YouTube captions first, fall back to Whisper ASR."""
import os
import tempfile
import time
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import yt_dlp

# Lazy-loaded Whisper model (expensive; only load on first fallback)
_whisper_model = None


def _get_whisper(model_name: str):
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        print(f"[whisper] loading model '{model_name}' (first run downloads the weights)…")
        _whisper_model = WhisperModel(model_name, device="cpu", compute_type="int8")
    return _whisper_model


def _try_captions(video_id: str, languages: list[str]) -> list[dict] | None:
    """Return segments from YouTube captions, or None if unavailable."""
    for attempt in range(3):
        try:
            raw = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
            return [
                {
                    "start": s["start"],
                    "end": s["start"] + s["duration"],
                    "text": s["text"].strip(),
                }
                for s in raw
                if s["text"].strip()
            ]
        except (TranscriptsDisabled, NoTranscriptFound):
            return None
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            print(f"  [captions error] {video_id}: {e}")
            return None
    return None


def _transcribe_with_whisper(video_id: str, whisper_model: str) -> list[dict]:
    """Download audio and transcribe locally with faster-whisper."""
    with tempfile.TemporaryDirectory() as tmp:
        out_template = os.path.join(tmp, f"{video_id}.%(ext)s")
        ydl_opts = {
            "format": "bestaudio[ext=m4a]/bestaudio",
            "outtmpl": out_template,
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://youtube.com/watch?v={video_id}"])

        # Find whatever file yt-dlp actually wrote
        audio_path = None
        for f in os.listdir(tmp):
            audio_path = os.path.join(tmp, f)
            break
        if not audio_path:
            raise RuntimeError(f"Audio download failed for {video_id}")

        model = _get_whisper(whisper_model)
        segments, _info = model.transcribe(audio_path, vad_filter=True)
        return [
            {"start": s.start, "end": s.end, "text": s.text.strip()}
            for s in segments
            if s.text.strip()
        ]


def get_segments(
    video_id: str,
    languages: list[str] | None = None,
    whisper_model: str = "small.en",
) -> list[dict]:
    """Return a list of {start, end, text} segments for a video.

    Strategy:
      1. Try YouTube auto-captions in the requested languages.
      2. If captions are disabled / missing, download the audio and run Whisper.
    """
    if languages is None:
        languages = ["en"]

    captioned = _try_captions(video_id, languages)
    if captioned:
        return captioned

    print(f"  [whisper] falling back to ASR for {video_id}")
    return _transcribe_with_whisper(video_id, whisper_model)
