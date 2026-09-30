"""Offline tests — no network, no model downloads, no LLM.

Everything model-dependent is exercised with synthetic vectors or stubs; the real
models are covered by eval/run_eval.py.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from momentrag import moment_rag as mr  # noqa: E402
from momentrag import moments as M  # noqa: E402
from momentrag.answer import cited, validate_citations  # noqa: E402
from momentrag.baseline import chunk_words  # noqa: E402
from momentrag.transcript import deeplink, fmt_ts, normalize_cues  # noqa: E402


def _cues(n: int, secs: float = 4.0, words: str = "alpha beta gamma delta") -> list[dict]:
    return [{"text": f"{words} {i}", "t_start": i * secs, "t_end": (i + 1) * secs}
            for i in range(n)]


# ---------------------------------------------------------------- transcript

def test_normalize_clamps_rolling_caption_overlap():
    raw = [{"text": "hello there", "start": 0.0, "duration": 4.0},
           {"text": "[Music]", "start": 1.0, "duration": 1.0},
           {"text": "general\nkenobi", "start": 2.0, "duration": 4.0}]
    cues = normalize_cues(raw)
    assert [c["text"] for c in cues] == ["hello there", "general kenobi"]
    assert cues[0]["t_end"] == 2.0                       # clamped to next cue start
    assert cues[1]["t_end"] == 6.0


def test_timestamp_helpers():
    assert fmt_ts(75) == "01:15" and fmt_ts(3725) == "1:02:05"
    assert deeplink("abc", 90.7) == "https://www.youtube.com/watch?v=abc&t=90s"


# ---------------------------------------------------------------- baseline

def test_baseline_chunks_have_fixed_size_and_overlap():
    chunks = chunk_words(_cues(100), size=50, overlap=10)
    words = [c["text"].split() for c in chunks]
    assert all(len(w) == 50 for w in words[:-1])
    assert words[0][-10:] == words[1][:10]               # overlap carried over
    assert chunks[0]["t_start"] == 0.0 and chunks[1]["t_start"] > 0


# ---------------------------------------------------------------- moments

def test_windows_respect_duration_and_keep_cues_whole():
    ws = M.build_windows(_cues(30, secs=4.0), window_s=20.0)
    assert all(w["t_end"] - w["t_start"] <= 20.0 for w in ws)
    assert sum(len(w["text"].split()) for w in ws) == 30 * 5


def _two_topic_vectors(n_a: int, n_b: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    a, b = np.eye(8)[0], np.eye(8)[1]
    vecs = [a + 0.05 * rng.standard_normal(8) for _ in range(n_a)] + \
           [b + 0.05 * rng.standard_normal(8) for _ in range(n_b)]
    return np.array([v / np.linalg.norm(v) for v in vecs], dtype="float32")


def test_semantic_segmentation_cuts_at_topic_shift():
    ws = M.build_windows(_cues(60, secs=4.0), window_s=20.0)       # 12 windows x 20s
    vecs = _two_topic_vectors(6, len(ws) - 6)
    ms = M.segment_moments(ws, [], vecs, use_chapters=False, min_s=40, max_s=400)
    assert [m["window_ids"][0] for m in ms] == [0, 6]              # boundary at the shift


def test_segmentation_honours_min_and_max_duration():
    ws = M.build_windows(_cues(150, secs=4.0), window_s=20.0)      # 600s, one topic
    vecs = _two_topic_vectors(len(ws), 0)
    ms = M.segment_moments(ws, [], vecs, use_chapters=False, min_s=40, max_s=150)
    durs = [m["t_end"] - m["t_start"] for m in ms]
    assert len(ms) > 1 and all(40 <= d <= 150 for d in durs)


def test_chapters_are_hard_boundaries():
    ws = M.build_windows(_cues(60, secs=4.0), window_s=20.0)
    chapters = [{"title": "A", "t_start": 0, "t_end": 100},
                {"title": "B", "t_start": 100, "t_end": 240}]
    ms = M.segment_moments(ws, chapters, use_semantic=False)
    assert [m["chapter"] for m in ms] == ["A", "B"]
    assert abs(ms[1]["t_start"] - 100) <= 10


def test_fixed_duration_ablation():
    ws = M.build_windows(_cues(90, secs=4.0), window_s=20.0)       # 360s
    ms = M.segment_moments(ws, [], fixed_s=120)
    assert len(ms) == 3


def test_enrichment_falls_back_to_extractive_when_llm_fails():
    class Broken:
        backend, model = "x", "y"

        def complete(self, *a, **k):
            raise RuntimeError("down")

    ms = [{"id": 0, "text": "transformer attention heads attention", "chapter": "Arch"},
          {"id": 1, "text": "jailbreak prompt injection attack", "chapter": None}]
    out = M.enrich_moments(ms, llm=Broken())
    assert out[0]["title"] == "Arch" and out[0]["enriched_by"] == "extractive"
    assert "attention" in out[0]["keywords"] and out[1]["card"]


def test_enrichment_parses_llm_json_card():
    class Good:
        backend, model = "stub", "m"

        def complete(self, *a, **k):
            return 'Sure! {"title": "Scaling laws", "summary": "N and D predict loss.", ' \
                   '"keywords": ["scaling", "parameters"]}'

    out = M.enrich_moments([{"id": 0, "text": "scaling", "chapter": None}], llm=Good())
    assert out[0]["title"] == "Scaling laws" and out[0]["enriched_by"] == "stub:m"


# ---------------------------------------------------------------- citations

def test_invented_citations_are_stripped():
    assert validate_citations("A [1] B [4] C [2, 7]", 3) == "A [1] B  C [2]"
    assert cited("x [1] y [2, 3]") == {1, 2, 3}


# ---------------------------------------------------------------- moment retrieval (stubbed models)

@pytest.fixture
def stub_models(monkeypatch):
    vocab = ["jailbreak", "grandmother", "scaling", "parameters", "valuation", "scale"]

    def emb(texts):
        out = np.array([[t.lower().count(w) + 0.01 for w in vocab] for t in texts], "float32")
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    monkeypatch.setattr(mr.models, "embed_docs", emb)
    monkeypatch.setattr(mr.models, "embed_query", lambda q: emb([q])[0])
    monkeypatch.setattr(mr.models, "rerank_scores", lambda q, docs: [
        0.9 if any(w in d.lower() and w in q.lower() for w in vocab) else 0.0 for d in docs])


def _corpus():
    texts = ["scaling laws parameters"] * 3 + ["jailbreak grandmother story"] * 3 + \
            ["scale valuation plot"] * 3
    windows = [{"id": i, "text": t, "t_start": 20.0 * i, "t_end": 20.0 * (i + 1)}
               for i, t in enumerate(texts)]
    moments = [{"id": j, "window_ids": [3 * j, 3 * j + 1, 3 * j + 2], "t_start": 60.0 * j,
                "t_end": 60.0 * (j + 1), "text": " ".join(texts[3 * j:3 * j + 3]),
                "title": f"m{j}", "chapter": None, "card": texts[3 * j]} for j in range(3)]
    return windows, moments


def test_moment_rag_returns_the_right_moment_with_real_timestamps(stub_models):
    windows, moments = _corpus()
    rag = mr.MomentRAG("vid", windows, moments)
    r = rag.retrieve("tell me about the grandmother jailbreak", k=2)
    top = r["hits"][0]
    assert top["moment_id"] == 1 and not r["abstain"]
    assert 60.0 <= top["anchor_t"] < 120.0                 # anchor from payload, inside moment
    assert set(top["branches"]) >= {"dense", "bm25"}


def test_gate_abstains_only_when_both_signals_are_weak(stub_models):
    windows, moments = _corpus()
    rag = mr.MomentRAG("vid", windows, moments, dense_gate=1.01)   # dense never clears it
    assert rag.retrieve("unrelated cooking question", k=2)["abstain"]
    assert not rag.retrieve("scaling parameters", k=2)["abstain"]   # reranker still confident
    rag.dense_gate = 0.0                                             # dense always clears it
    assert not rag.retrieve("unrelated cooking question", k=2)["abstain"]


def test_dense_gate_calibrates_to_midpoint_or_keeps_default(stub_models):
    windows, moments = _corpus()
    rag = mr.MomentRAG("vid", windows, moments)
    fit = mr.calibrate_dense_gate(rag, ["grandmother jailbreak"], ["cooking recipe"])
    assert fit["separated"]
    assert fit["unanswerable_max"] < fit["dense_gate"] < fit["answerable_min"]
    overlap = mr.calibrate_dense_gate(rag, ["cooking recipe"], ["grandmother jailbreak"])
    assert not overlap["separated"] and overlap["dense_gate"] == mr.DENSE_GATE


def test_edge_anchor_pads_with_neighbouring_window(stub_models):
    windows, moments = _corpus()
    rag = mr.MomentRAG("vid", windows, moments)
    text, t0, t1 = rag._read_text(moments[1], anchor_wid=3)      # first window of moment 1
    assert t0 == 40.0 and t1 == 120.0 and text.startswith("scaling")
    text, t0, t1 = rag._read_text(moments[1], anchor_wid=4)      # middle: no padding
    assert (t0, t1) == (60.0, 120.0)
