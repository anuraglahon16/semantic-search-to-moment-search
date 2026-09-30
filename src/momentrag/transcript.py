"""YouTube transcript acquisition — the knowledge source for both RAG systems.

fetch_cues() pulls the caption track with youtube-transcript-api and normalizes it
into timed cues [{text, t_start, t_end}] (seconds). Auto-generated YouTube captions
are "rolling": each cue's duration overlaps the next cue's start, so t_end is clamped
to the next cue's start. That gives a clean, non-overlapping timeline — the same
shape Moment Search's src/ingest/transcript.py produces from json3 captions.

fetch_chapters() pulls the uploader's chapter markers (yt-dlp metadata). Chapters are
an optional structural prior for moment segmentation; a video without them still
segments on semantic shifts alone.

Everything is cached under data/ so the pipeline, tests and evaluation run offline
after the first fetch.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _clean(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\[(music|applause|laughter)\]", " ", text, flags=re.I)
    return " ".join(text.split())


def normalize_cues(raw: list[dict]) -> list[dict]:
    """[{text, start, duration}] -> [{text, t_start, t_end}] with no overlaps."""
    rows = [(float(c["start"]), float(c.get("duration", 0.0)), _clean(c["text"]))
            for c in raw]
    rows = [r for r in rows if r[2]]
    rows.sort(key=lambda r: r[0])
    cues = []
    for i, (start, dur, text) in enumerate(rows):
        end = start + dur
        if i + 1 < len(rows):
            end = min(end, rows[i + 1][0])       # rolling captions overlap the next cue
        cues.append({"text": text, "t_start": round(start, 2),
                     "t_end": round(max(end, start + 0.01), 2)})
    return cues


def fetch_cues(video_id: str, languages: tuple[str, ...] = ("en",),
               cache: bool = True) -> list[dict]:
    """Timed cues for a YouTube video, cached to data/<video_id>.cues.json."""
    path = DATA_DIR / f"{video_id}.cues.json"
    if cache and path.exists():
        return json.loads(path.read_text())
    from youtube_transcript_api import YouTubeTranscriptApi

    fetched = YouTubeTranscriptApi().fetch(video_id, languages=list(languages))
    raw = [{"text": s.text, "start": s.start, "duration": s.duration} for s in fetched]
    cues = normalize_cues(raw)
    DATA_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(cues, indent=0))
    return cues


def fetch_chapters(video_id: str, cache: bool = True) -> list[dict]:
    """[{title, t_start, t_end}] from the video's chapter markers ([] when none)."""
    path = DATA_DIR / f"{video_id}.chapters.json"
    if cache and path.exists():
        return json.loads(path.read_text())
    try:
        import yt_dlp

        with yt_dlp.YoutubeDL({"skip_download": True, "quiet": True}) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}",
                                    download=False)
        chapters = [{"title": c["title"], "t_start": float(c["start_time"]),
                     "t_end": float(c["end_time"])} for c in info.get("chapters") or []]
    except Exception as exc:                      # chapters are optional, never fatal
        print(f"[transcript] chapters unavailable ({type(exc).__name__}: {exc})")
        chapters = []
    DATA_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(chapters, indent=0))
    return chapters


def full_text(cues: list[dict]) -> str:
    return " ".join(c["text"] for c in cues)


def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def deeplink(video_id: str, seconds: float) -> str:
    return f"https://www.youtube.com/watch?v={video_id}&t={int(seconds)}s"
