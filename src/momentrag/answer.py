"""Prompts, context formatting and citation handling for both systems.

The two prompts differ only in what the retrieved context *is*: anonymous text chunks
(baseline) vs titled, timestamped moments (Moment RAG). Both demand grounded answers
with [n] citations, so the comparison isolates retrieval and context structure.
"""
from __future__ import annotations

import re

from .transcript import deeplink, fmt_ts

_RULES = (
    "Answer ONLY from the numbered context below. The transcript is auto-generated, so "
    "expect missing punctuation and misheard words (e.g. 'Llama 270b' means 'Llama 2 70B'); "
    "read through them. Cite every claim with the number of the context it came from, "
    "like [1] or [2, 3]. If the context does not contain the answer, say you couldn't find "
    "it in the video; do not use outside knowledge. Be concise: 1-4 sentences."
)

SYSTEM_BASELINE = ("You answer questions about a YouTube talk using excerpts of its "
                   "transcript. " + _RULES)

SYSTEM_MOMENT = ("You answer questions about a YouTube talk using moments of it: each is a "
                 "titled, timestamped segment of the transcript that covers one idea. "
                 + _RULES)


def format_baseline_context(question: str, hits: list[dict]) -> str:
    ctx = "\n\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, 1))
    return f"Context:\n{ctx}\n\nQuestion: {question}"


def format_moment_context(question: str, hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        head = (f"[{i}] Moment \"{h.get('title') or 'untitled'}\" "
                f"({fmt_ts(h['t_start'])}-{fmt_ts(h['t_end'])}, "
                f"most relevant at {fmt_ts(h['anchor_t'])})")
        blocks.append(f"{head}\n{h['text']}")
    return "Context:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def validate_citations(answer: str, n: int) -> str:
    """Strip [k] references to contexts the model was never shown (Moment Search's
    _validate_citations)."""
    def fix(m: re.Match) -> str:
        nums = [int(x) for x in re.split(r"\s*,\s*", m.group(1))]
        ok = [str(x) for x in nums if 1 <= x <= n]
        return f"[{', '.join(ok)}]" if ok else ""
    return _CITE.sub(fix, answer)


def cited(answer: str) -> set[int]:
    return {int(x) for m in _CITE.finditer(answer or "") for x in re.split(r"\s*,\s*", m.group(1))}


def render_sources(video_id: str, hits: list[dict], moment: bool) -> list[str]:
    """Human-readable source list. Timestamps come from the retrieval payload only."""
    lines = []
    for i, h in enumerate(hits, 1):
        if moment:
            lines.append(f"[{i}] {h.get('title')} — {fmt_ts(h['t_start'])}-{fmt_ts(h['t_end'])} "
                         f"· jump to {fmt_ts(h['anchor_t'])}: {deeplink(video_id, h['anchor_t'])}")
        else:
            lines.append(f"[{i}] \"{' '.join(h['text'].split()[:14])} …\" (no timestamp)")
    return lines
