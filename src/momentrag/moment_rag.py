"""Part 2 — Moment RAG: retrieve moments, not chunks.

Read path (mirrors Moment Search's src/rag/search.py, transcript-only):

  1 · Retrieve — three branches, in parallel, always (no router):
        dense   question -> bge -> nearest ~20s windows          (what is *meant*)
        lexical BM25 over the same windows                        (exact names, numbers,
                                                                   misheard ASR words)
        card    question -> bge -> enriched moment cards          (what a moment is *about*)
  2 · Fuse — Reciprocal Rank Fusion: each branch is ranked on its own and scored
        1/(RRF_K + rank), because raw cosine and BM25 scores are incomparable. Window
        hits collapse into the moment that contains them (Moment Search collapses hits
        within FUSION_WINDOW_S; here the moment boundary is the join key). A moment keeps
        only its BEST hit per branch, and is boosted when >= 2 branches agree.
  3 · Rerank — a cross-encoder re-reads every window of the top candidate moments;
        a moment's relevance is its best window's, blended with its normalized RRF.
        The best window becomes the moment's *anchor*: the timestamp the answer cites.
  4 · Gate — abstain with no LLM call only when NEITHER signal is confident: the best
        reranked window is below GATE_REL AND the best dense cosine is below DENSE_GATE
        (Moment Search abstains only when neither the visual nor the text branch clears
        its bar). Both thresholds come from eval/calibration.json, never the test set.
  5 · Read + answer — the LLM reads each whole moment (the complete thought), cites
        [n], invalid citations are stripped, and timestamps/deep links come from the
        retrieved payload — the LLM never invents a time. When the anchor sits in a
        moment's first or last window, the neighbouring window is added (Moment Search's
        CONTEXT_PAD_S) so an answer straddling a boundary is read whole.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from . import models
from .answer import SYSTEM_MOMENT, format_moment_context, validate_citations

RRF_K = 60
BRANCH_K = 20          # candidates per branch before fusion
AGREE_BOOST = 1.25     # >= 2 branches landing on the same moment
RERANK_POOL = 8        # fused moments whose windows the cross-encoder re-reads
RERANK_WEIGHT = 0.7    # blend: cross-encoder relevance vs normalized RRF
GATE_REL = 0.10        # cross-encoder relevance (0-1) below which the reranker "sees nothing"
DENSE_GATE = 0.62      # bge cosine: calibration answerable min 0.648, unanswerable max 0.596
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "to", "and", "in", "is", "it", "that", "what", "how",
         "does", "do", "he", "is", "for", "on", "with", "about", "which", "why", "when",
         "was", "are", "this", "they", "you", "i", "uh", "um", "so", "like", "say"}


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class MomentRAG:
    video_id: str
    windows: list[dict]
    moments: list[dict]
    use_dense: bool = True
    use_bm25: bool = True
    use_cards: bool = True
    use_rerank: bool = True
    use_gate: bool = True
    name: str = "moment"
    _w2m: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._w2m = {wid: m["id"] for m in self.moments for wid in m["window_ids"]}
        self._win = {w["id"]: w for w in self.windows}
        wv = models.embed_docs([w["text"] for w in self.windows])
        self._widx = faiss.IndexFlatIP(wv.shape[1])
        self._widx.add(wv)
        self._bm25 = BM25Okapi([_tokens(w["text"]) for w in self.windows])
        cv = models.embed_docs([m.get("card") or m["text"] for m in self.moments])
        self._cidx = faiss.IndexFlatIP(cv.shape[1])
        self._cidx.add(cv)

    # ------------------------------------------------------------------ retrieval

    def _branches(self, question: str) -> dict[str, list[tuple[int, float]]]:
        """branch -> [(window_id | moment_id, raw score)] best-first."""
        out: dict[str, list[tuple[int, float]]] = {}
        q = models.embed_query(question)[None, :]
        if self.use_dense:
            s, ids = self._widx.search(q, min(BRANCH_K, len(self.windows)))
            out["dense"] = [(int(i), float(x)) for x, i in zip(s[0], ids[0]) if i >= 0]
        if self.use_bm25:
            scores = self._bm25.get_scores(_tokens(question))
            top = np.argsort(-scores)[:BRANCH_K]
            out["bm25"] = [(int(i), float(scores[i])) for i in top if scores[i] > 0]
        if self.use_cards:
            s, ids = self._cidx.search(q, min(BRANCH_K, len(self.moments)))
            out["card"] = [(int(i), float(x)) for x, i in zip(s[0], ids[0]) if i >= 0]
        return out

    def _fuse(self, branches: dict[str, list[tuple[int, float]]]) -> list[dict]:
        cands: dict[int, dict] = {}
        for branch, hits in branches.items():
            for rank, (idx, raw) in enumerate(hits):
                mid = idx if branch == "card" else self._w2m[idx]
                c = cands.setdefault(mid, {"moment_id": mid, "best": {}, "win_rrf": {}})
                rrf = 1.0 / (RRF_K + rank)
                if branch not in c["best"]:                 # best hit per branch only
                    c["best"][branch] = {"rrf": rrf, "raw": raw}
                if branch != "card":                        # window-level evidence
                    c["win_rrf"][idx] = c["win_rrf"].get(idx, 0.0) + rrf
        for c in cands.values():
            c["rrf"] = sum(b["rrf"] for b in c["best"].values())
            if len(c["best"]) >= 2:
                c["rrf"] *= AGREE_BOOST
            m = self.moments[c["moment_id"]]
            c["anchor_wid"] = (max(c["win_rrf"], key=c["win_rrf"].get) if c["win_rrf"]
                               else m["window_ids"][0])
        return sorted(cands.values(), key=lambda c: c["rrf"], reverse=True)

    def _rerank(self, question: str, fused: list[dict]) -> list[dict]:
        pool = fused[:RERANK_POOL]
        pairs = [(c, wid) for c in pool for wid in self.moments[c["moment_id"]]["window_ids"]]
        rel = models.rerank_scores(question, [self._win[w]["text"] for _, w in pairs])
        best: dict[int, tuple[float, int]] = {}
        for (c, wid), r in zip(pairs, rel):
            if c["moment_id"] not in best or r > best[c["moment_id"]][0]:
                best[c["moment_id"]] = (r, wid)
        top = max((c["rrf"] for c in fused), default=0.0) or 1.0
        for c in pool:
            r, wid = best[c["moment_id"]]
            c["rel"], c["anchor_wid"] = r, wid
            c["score"] = RERANK_WEIGHT * r + (1 - RERANK_WEIGHT) * c["rrf"] / top
        for c in fused[RERANK_POOL:]:
            c["score"] = (1 - RERANK_WEIGHT) * c["rrf"] / top
        return sorted(fused, key=lambda c: c["score"], reverse=True)

    def _read_text(self, m: dict, anchor_wid: int) -> tuple[str, float, float]:
        """The moment's text, padded by one neighbouring window when the anchor is at an edge."""
        wids = list(m["window_ids"])
        if anchor_wid == wids[0] and anchor_wid - 1 in self._win:
            wids.insert(0, anchor_wid - 1)
        if anchor_wid == wids[-1] and anchor_wid + 1 in self._win:
            wids.append(anchor_wid + 1)
        ws = [self._win[w] for w in wids]
        return " ".join(w["text"] for w in ws), ws[0]["t_start"], ws[-1]["t_end"]

    def retrieve(self, question: str, k: int = 3) -> dict:
        branches = self._branches(question)
        best_dense = branches["dense"][0][1] if branches.get("dense") else 0.0
        fused = self._fuse(branches)
        if self.use_rerank:
            fused = self._rerank(question, fused)
        else:
            for c in fused:
                c["score"] = c["rrf"]
        hits = []
        for rank, c in enumerate(fused[:k]):
            m = self.moments[c["moment_id"]]
            a = self._win[c["anchor_wid"]]
            text, r0, r1 = self._read_text(m, c["anchor_wid"])
            hits.append({"rank": rank, "moment_id": m["id"], "title": m.get("title"),
                         "chapter": m.get("chapter"), "t_start": m["t_start"],
                         "t_end": m["t_end"], "anchor_t": a["t_start"],
                         "anchor_text": a["text"], "text": text,
                         "read_start": r0, "read_end": r1,
                         "score": round(c["score"], 4), "rel": c.get("rel"),
                         "branches": sorted(c["best"])})
        best_rel = max((c.get("rel", 0.0) for c in fused[:RERANK_POOL]), default=0.0)
        abstain = bool(self.use_gate and self.use_rerank
                       and best_rel < GATE_REL and best_dense < DENSE_GATE)
        return {"hits": hits, "best_rel": best_rel, "best_dense": best_dense,
                "abstain": abstain,
                "candidates": len(fused)}

    # ------------------------------------------------------------------ answer

    def ask(self, question: str, llm=None, k: int = 3) -> dict:
        r = self.retrieve(question, k)
        hits = [] if r["abstain"] else r["hits"]
        out = {"question": question, "system": self.name, "hits": hits,
               "retrieved": r["hits"], "abstain": r["abstain"], "best_rel": r["best_rel"],
               "best_dense": r["best_dense"],
               "context_words": sum(len(h["text"].split()) for h in hits)}
        if r["abstain"]:
            out["answer"] = ("I couldn't find that in this video — no moment of the talk "
                             "looks related to the question.")
        elif llm is not None:
            raw = llm.complete(SYSTEM_MOMENT, format_moment_context(question, hits))
            out["answer"] = validate_citations(raw, len(hits))
        return out
