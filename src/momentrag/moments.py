"""Part 2 — turn a timed transcript into "moments".

Two granularities, both carrying real timestamps (never invented by an LLM):

  window  ~20s of consecutive caption cues. The retrieval atom — small enough that a
          hit pinpoints *where* in the video the answer is said. Mirrors Moment
          Search's TRANSCRIPT_CHUNK_SECONDS=20 transcript chunks.
  moment  a run of consecutive windows about ONE thing (typically 40-150s). The
          reading unit — big enough to hold a complete explanation, so the LLM gets
          the whole thought instead of a fragment cut at a word count.

Segmentation (segment_moments):
  1. Chapter markers, when the video has them, are hard boundaries (snapped to the
     nearest window gap) — the uploader's own table of contents.
  2. Inside each chapter, TextTiling on window embeddings: at every gap compare the
     mean embedding of the W windows before vs after it; a "depth" dip in that
     similarity marks a topic shift. Cut at deep dips, and keep cutting any segment
     longer than MAX_S at its deepest remaining dip, never leaving a side < MIN_S.
  3. Segments shorter than MIN_S merge into their more similar neighbour.

Enrichment (enrich_moments): each moment gets a title, a one-sentence summary and
keywords — from the LLM when one is configured, else extractively (chapter title +
TF-IDF keywords). The enriched "moment card" is embedded as its own search branch,
the transcript analogue of Moment Search's enrich stage.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter

import numpy as np

from . import models
from .transcript import DATA_DIR, fmt_ts

WINDOW_S = 20.0
TILING_W = 3          # windows per side when scoring a gap (~60s of speech)
MIN_S = 40.0
MAX_S = 150.0
DEPTH_STD = 0.5       # a gap is a boundary when depth > mean + DEPTH_STD * std


def build_windows(cues: list[dict], window_s: float = WINDOW_S) -> list[dict]:
    """Group cues into ~window_s windows (cues are never split)."""
    windows: list[dict] = []
    buf: list[dict] = []
    for c in cues:
        if buf and c["t_end"] - buf[0]["t_start"] > window_s:
            windows.append(_window(len(windows), buf))
            buf = []
        buf.append(c)
    if buf:
        windows.append(_window(len(windows), buf))
    return windows


def _window(i: int, cues: list[dict]) -> dict:
    return {"id": i, "text": " ".join(c["text"] for c in cues),
            "t_start": cues[0]["t_start"], "t_end": cues[-1]["t_end"]}


# --------------------------------------------------------------------------- segmentation

def gap_similarities(vecs: np.ndarray, w: int = TILING_W) -> np.ndarray:
    """sim[g] = cosine(mean(vecs[g-w+1..g]), mean(vecs[g+1..g+w])) for gap g (after window g)."""
    n = len(vecs)
    sims = np.ones(max(n - 1, 0))
    for g in range(n - 1):
        left = vecs[max(0, g - w + 1):g + 1].mean(0)
        right = vecs[g + 1:min(n, g + 1 + w)].mean(0)
        sims[g] = float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-9))
    return sims


def depth_scores(sims: np.ndarray) -> np.ndarray:
    """TextTiling depth: how far each gap's similarity dips below the peaks around it."""
    depth = np.zeros_like(sims)
    for g in range(len(sims)):
        # nearest peaks: climb while similarity keeps rising on each side
        i = g
        while i > 0 and sims[i - 1] >= sims[i]:
            i -= 1
        lpeak = sims[i]
        j = g
        while j < len(sims) - 1 and sims[j + 1] >= sims[j]:
            j += 1
        rpeak = sims[j]
        depth[g] = (lpeak - sims[g]) + (rpeak - sims[g])
    return depth


def _chapter_cuts(windows: list[dict], chapters: list[dict]) -> set[int]:
    """Chapter starts -> gap indices (gap g = boundary after window g)."""
    cuts = set()
    for ch in chapters[1:]:
        t = ch["t_start"]
        g = min(range(len(windows) - 1),
                key=lambda k: abs(windows[k]["t_end"] - t), default=None)
        if g is not None:
            cuts.add(g)
    return cuts


def segment_moments(windows: list[dict], chapters: list[dict] | None = None,
                    vecs: np.ndarray | None = None, *, use_chapters: bool = True,
                    use_semantic: bool = True, fixed_s: float | None = None,
                    min_s: float = MIN_S, max_s: float = MAX_S) -> list[dict]:
    """Group windows into moments. Returns [{id, window_ids, t_start, t_end, text, chapter}].

    fixed_s set -> naive fixed-duration moments (an ablation, ignores everything else).
    """
    n = len(windows)
    if n == 0:
        return []
    dur = lambda a, b: windows[b]["t_end"] - windows[a]["t_start"]   # windows a..b inclusive

    if fixed_s:
        cuts: set[int] = set()
        start = 0
        for g in range(n - 1):
            if dur(start, g) >= fixed_s:
                cuts.add(g)
                start = g + 1
        return _assemble(windows, sorted(cuts), chapters)

    cuts = _chapter_cuts(windows, chapters) if (use_chapters and chapters) else set()
    if use_semantic:
        if vecs is None:
            vecs = models.embed_docs([w["text"] for w in windows])
        depth = depth_scores(gap_similarities(vecs))
        thr = depth.mean() + DEPTH_STD * depth.std()

        def ok(a: int, g: int, b: int) -> bool:        # both sides of gap g long enough
            return dur(a, g) >= min_s and dur(g + 1, b) >= min_s

        def split(a: int, b: int) -> None:             # recursively cut segment a..b
            cands = [g for g in range(a, b) if ok(a, g, b)]
            if not cands:
                return
            g = max(cands, key=lambda k: depth[k])
            if depth[g] > thr or dur(a, b) > max_s:
                cuts.add(g)
                split(a, g)
                split(g + 1, b)

        bounds = [-1] + sorted(cuts) + [n - 1]
        for a, b in zip(bounds[:-1], bounds[1:]):
            split(a + 1, b)
        cuts = _merge_short(windows, sorted(cuts), vecs, min_s)
    return _assemble(windows, sorted(cuts), chapters)


def _merge_short(windows: list[dict], cuts: list[int], vecs: np.ndarray,
                 min_s: float) -> set[int]:
    """Drop the cut that isolates a too-short segment, joining it to its closer neighbour."""
    cuts = list(cuts)
    changed = True
    while changed and cuts:
        changed = False
        bounds = [-1] + cuts + [len(windows) - 1]
        for s in range(len(bounds) - 1):
            a, b = bounds[s] + 1, bounds[s + 1]
            if windows[b]["t_end"] - windows[a]["t_start"] >= min_s:
                continue
            seg = vecs[a:b + 1].mean(0)
            left = vecs[bounds[s - 1] + 1:a].mean(0) if s > 0 else None
            right = vecs[b + 1:bounds[s + 2] + 1].mean(0) if s + 2 < len(bounds) else None
            cos = lambda u: -1.0 if u is None else float(
                seg @ u / (np.linalg.norm(seg) * np.linalg.norm(u) + 1e-9))
            drop = bounds[s] if cos(left) >= cos(right) else bounds[s + 1]
            if drop in cuts:
                cuts.remove(drop)
                changed = True
                break
    return set(cuts)


def _assemble(windows: list[dict], cuts: list[int], chapters: list[dict] | None) -> list[dict]:
    moments = []
    bounds = [-1] + cuts + [len(windows) - 1]
    for a, b in zip(bounds[:-1], bounds[1:]):
        ws = windows[a + 1:b + 1]
        if not ws:
            continue
        t0, t1 = ws[0]["t_start"], ws[-1]["t_end"]
        moments.append({"id": len(moments), "window_ids": [w["id"] for w in ws],
                        "t_start": t0, "t_end": t1,
                        "text": " ".join(w["text"] for w in ws),
                        "chapter": _chapter_at(chapters, (t0 + t1) / 2)})
    return moments


def _chapter_at(chapters: list[dict] | None, t: float) -> str | None:
    for ch in chapters or []:
        if ch["t_start"] <= t < ch["t_end"]:
            return ch["title"]
    return None


# --------------------------------------------------------------------------- enrichment

_STOP = set("""a about above after again all also am an and any are as at be because been
before being below between both but by can could did do does doing don down during each few
for from further had has have having he her here hers him his how i if in into is it its
itself just kind know like ll me more most my no nor not now of off on once only or other our
out over own re really right s same she should so some sort such t than that the their them
then there these they this those through to too uh um under until up us very was we were what
when where which while who whom why will with would you your going basically actually thing
things okay one get got see want think way lot make also say said let well go""".split())

_ENRICH_SYSTEM = (
    "You index segments of a video transcript for search. The transcript is "
    "auto-generated: no punctuation and some misheard words. For the segment you are "
    "given, reply with ONLY this JSON: {\"title\": \"<4-8 word title>\", \"summary\": "
    "\"<one sentence: what is explained or claimed, with any specific names/numbers>\", "
    "\"keywords\": [\"<5-8 key terms, correctly spelled>\"]}"
)


def _tfidf_keywords(moments: list[dict], top: int = 8) -> list[list[str]]:
    toks = [[w for w in re.findall(r"[a-z][a-z0-9\-]+", m["text"].lower())
             if w not in _STOP and len(w) > 2] for m in moments]
    df = Counter(w for t in toks for w in set(t))
    out = []
    for t in toks:
        tf = Counter(t)
        score = {w: c * math.log(len(toks) / df[w]) for w, c in tf.items()}
        out.append([w for w, _ in sorted(score.items(), key=lambda x: -x[1])[:top]])
    return out


def _parse_card(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not str(data.get("title", "")).strip():
        return None
    kws = data.get("keywords") if isinstance(data.get("keywords"), list) else []
    return {"title": str(data["title"]).strip(), "summary": str(data.get("summary", "")).strip(),
            "keywords": [str(k).strip() for k in kws if str(k).strip()][:8]}


def enrich_moments(moments: list[dict], llm=None) -> list[dict]:
    """Attach title/summary/keywords/card to each moment (LLM, else extractive)."""
    kw = _tfidf_keywords(moments)
    for m, kws in zip(moments, kw):
        card = None
        if llm is not None:
            try:
                card = _parse_card(llm.complete(_ENRICH_SYSTEM, m["text"], max_tokens=400))
            except Exception as exc:
                print(f"[enrich] moment {m['id']}: LLM failed ({type(exc).__name__}) — extractive")
        if card is None:
            card = {"title": m.get("chapter") or " ".join(kws[:4]),
                    "summary": " ".join(m["text"].split()[:40]) + " …",
                    "keywords": kws}
            m["enriched_by"] = "extractive"
        else:
            m["enriched_by"] = f"{llm.backend}:{llm.model}"
        m.update(card)
        m["card"] = (f"{m['title']}. {m['summary']} Keywords: {', '.join(m['keywords'])}."
                     + (f" Chapter: {m['chapter']}." if m.get("chapter") else ""))
    return moments


# --------------------------------------------------------------------------- build + cache

def build_moments(video_id: str, cues: list[dict], chapters: list[dict], llm=None,
                  cache: bool = True, **seg_kwargs) -> tuple[list[dict], list[dict]]:
    """windows + enriched moments for a video, cached per segmentation/enrichment config."""
    windows = build_windows(cues)
    tag = json.dumps({"seg": seg_kwargs, "n": len(windows),
                      "llm": f"{llm.backend}:{llm.model}" if llm else "extractive"},
                     sort_keys=True)
    key = hashlib.sha1(tag.encode()).hexdigest()[:10]
    path = DATA_DIR / f"{video_id}.moments.{key}.json"
    if cache and path.exists():
        return windows, json.loads(path.read_text())["moments"]
    vecs = models.embed_docs([w["text"] for w in windows])
    moments = enrich_moments(segment_moments(windows, chapters, vecs, **seg_kwargs), llm)
    DATA_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps({"config": json.loads(tag), "moments": moments}, indent=1))
    return windows, moments


def describe(moments: list[dict]) -> str:
    return "\n".join(f"[{m['id']:>2}] {fmt_ts(m['t_start'])}-{fmt_ts(m['t_end'])} "
                     f"({m['t_end'] - m['t_start']:>5.0f}s) {m['title']}" for m in moments)
