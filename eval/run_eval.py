"""Compare baseline chunk RAG with Moment RAG on the labeled question set.

Retrieval metrics (no LLM needed):
  evidence@k   every answer-keyword group appears in the context the LLM would read
               (top-k units concatenated): did retrieval deliver the answer?
  span MRR     1/rank of the first retrieved unit whose time span overlaps a gold span
  pinpoint@1   the rank-1 result's pointer time (moment anchor; for the baseline, the
               chunk's first-word time, which it never actually shows the user) falls in
               [gold_start - 20s, gold_end]: after "jump to", is the answer heard within
               one 20s window?
  ctx words    words of context sent to the LLM (the cost / noise side of the trade)
  abstain      unanswerable: retrieval refused (correct); answerable: refused (wrong)

Answer metrics (--answers, needs an LLM backend; see src/momentrag/llm.py):
  cited        the answer cites at least one [n]
  refused      the answer says it could not find it (right for unanswerable)
  Correctness is graded separately by eval/judge.py against reference answers — the
  LLM corrects the transcript's misspellings, so keyword matching undercounts it.

Usage:
  python eval/run_eval.py                 # retrieval metrics + ablations
  python eval/run_eval.py --answers       # also generate + score answers (slow locally)
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from momentrag import llm as llm_mod                           # noqa: E402
from momentrag import moment_rag as mr                         # noqa: E402
from momentrag.answer import cited                              # noqa: E402
from momentrag.baseline import BaselineRAG                     # noqa: E402
from momentrag.moment_rag import MomentRAG                     # noqa: E402
from momentrag.moments import build_moments                    # noqa: E402
from momentrag.transcript import fetch_chapters, fetch_cues    # noqa: E402

RESULTS = ROOT / "results"
_REFUSAL = re.compile(r"couldn.?t find|could not find|not (?:mentioned|covered|discussed|"
                      r"contain|provided|in the (?:video|context))|no (?:information|mention)|"
                      r"does not (?:say|state|mention|discuss|contain|provide)|doesn.?t (?:say|state|mention|"
                      r"discuss|contain|provide)", re.I)


def has_evidence(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return bool(keywords) and all(re.search(k, t) for k in keywords)


def overlaps(a0: float, a1: float, spans: list[list[float]]) -> bool:
    return any(a0 <= e and a1 >= s for s, e in spans)


def score_retrieval(q: dict, hits: list[dict], abstain: bool) -> dict:
    """hits: ranked units with text, t_start, t_end and 'pointer' (the jump-to time)."""
    row = {"id": q["id"], "type": q["type"], "abstain": abstain,
           "ctx_words": 0 if abstain else sum(len(h["text"].split()) for h in hits)}
    if q["type"] == "unanswerable":
        return row
    shown = [] if abstain else hits
    row["evidence"] = has_evidence(" ".join(h["text"] for h in shown), q["keywords"])
    rr = 0.0
    for r, h in enumerate(shown, 1):
        if overlaps(h["t_start"], h["t_end"], q["gold"]):
            rr = 1.0 / r
            break
    row["mrr"] = rr
    row["pinpoint"] = bool(shown) and any(s - 20 <= shown[0]["pointer"] <= e for s, e in q["gold"])
    row["pointer_err_s"] = (min(abs(shown[0]["pointer"] - s) for s, _ in q["gold"])
                            if shown else None)
    return row


def summarize(name: str, rows: list[dict], latency_ms: float) -> dict:
    ans = [r for r in rows if r["type"] != "unanswerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]
    pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else None
    out = {"system": name,
           "evidence@k %": pct([r["evidence"] for r in ans]),
           "span MRR": round(statistics.mean(r["mrr"] for r in ans), 3),
           "pinpoint@1 %": pct([r["pinpoint"] for r in ans]),
           "median ptr err s": round(statistics.median(
               r["pointer_err_s"] for r in ans if r["pointer_err_s"] is not None), 1),
           "avg ctx words": round(statistics.mean(r["ctx_words"] for r in ans)),
           "false abstain %": pct([r["abstain"] for r in ans]),
           "unanswerable abstain %": pct([r["abstain"] for r in una]),
           "latency ms": round(latency_ms)}
    for t in ("fact", "explain", "locate"):
        sub = [r for r in ans if r["type"] == t]
        out[f"evidence {t} %"] = pct([r["evidence"] for r in sub])
    return out


def run_baseline(name: str, rag: BaselineRAG, questions: list[dict], k: int):
    rag.retrieve(questions[0]["question"], k)             # warm-up: exclude model loading
    rows, t0 = [], time.perf_counter()
    for q in questions:
        hits = [{**h, "pointer": h["t_start"]} for h in rag.retrieve(q["question"], k)]
        rows.append(score_retrieval(q, hits, abstain=False) | {"system": name,
                    "top": [round(h["t_start"]) for h in hits]})
    return rows, 1000 * (time.perf_counter() - t0) / len(questions)


def run_moment(name: str, rag: MomentRAG, questions: list[dict], k: int):
    rag.retrieve(questions[0]["question"], k)             # warm-up: exclude model loading
    rows, t0 = [], time.perf_counter()
    for q in questions:
        r = rag.retrieve(q["question"], k)
        hits = [{**h, "pointer": h["anchor_t"]} for h in r["hits"]]
        rows.append(score_retrieval(q, hits, abstain=r["abstain"]) | {
            "system": name, "best_rel": round(r["best_rel"], 3),
            "best_dense": round(r["best_dense"], 3),
            "top": [f"{round(h['t_start'])}-{round(h['t_end'])}@{round(h['anchor_t'])}"
                    for h in hits]})
    return rows, 1000 * (time.perf_counter() - t0) / len(questions)


def calibrate_gate(rag: MomentRAG) -> dict:
    """Set rag.dense_gate from the held-out calibration questions (never the test set) and
    record every calibration question's scores and gate outcome."""
    cal = json.loads((ROOT / "eval" / "calibration.json").read_text())
    fit = mr.calibrate_dense_gate(rag, cal["answerable"], cal["unanswerable"])
    rag.dense_gate = fit["dense_gate"]
    out = {"GATE_REL": mr.GATE_REL, **fit}
    for grp in ("answerable", "unanswerable"):
        rs = [rag.retrieve(q, 3) for q in cal[grp]]
        out[grp] = [{"q": q, "rel": round(r["best_rel"], 4), "dense": round(r["best_dense"], 4),
                     "abstain": r["abstain"]} for q, r in zip(cal[grp], rs)]
    return out


def to_markdown(rows: list[dict]) -> str:
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |"
              for r in rows]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true", help="generate + score LLM answers")
    ap.add_argument("--enrich-llm", action="store_true",
                    help="LLM-enriched moment cards (default: extractive, keyless)")
    args = ap.parse_args()

    spec = json.loads((ROOT / "eval" / "questions.json").read_text())
    vid, questions = spec["video_id"], spec["questions"]
    cues, chapters = fetch_cues(vid), fetch_chapters(vid)
    llm = llm_mod.get_llm()
    enrich_llm = llm if args.enrich_llm else None
    global RESULTS
    RESULTS = ROOT / "results" / ("llm-cards" if enrich_llm else "extractive-cards")
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"LLM: {llm} · moment cards: {'LLM' if enrich_llm else 'extractive'} · -> {RESULTS}")

    windows, moments = build_moments(vid, cues, chapters, llm=enrich_llm)
    full = MomentRAG(vid, windows, moments, name="Moment RAG (full)")
    cal = calibrate_gate(full)
    print(f"dense gate calibrated to {cal['dense_gate']} (answerable min "
          f"{cal['answerable_min']}, unanswerable max {cal['unanswerable_max']})")
    base = BaselineRAG(cues)

    systems = [
        ("Baseline chunks k=3", lambda: run_baseline("Baseline chunks k=3", base, questions, 3)),
        ("Baseline chunks k=5", lambda: run_baseline("Baseline chunks k=5", base, questions, 5)),
        ("Moment RAG (full)", lambda: run_moment("Moment RAG (full)", full, questions, 3)),
    ]
    # Component ablations: add one Moment RAG idea at a time.
    ablations = [
        ("A1 moments, dense only", dict(use_bm25=False, use_cards=False, use_rerank=False, use_gate=False)),
        ("A2 + BM25", dict(use_cards=False, use_rerank=False, use_gate=False)),
        ("A3 + moment cards", dict(use_rerank=False, use_gate=False)),
        ("A4 + rerank/anchor", dict(use_gate=False)),
    ]
    for name, kw in ablations:
        rag = MomentRAG(vid, windows, moments, name=name, **kw)
        systems.append((name, lambda n=name, g=rag: run_moment(n, g, questions, 3)))
    # Segmentation ablations: same full retriever, different moment boundaries.
    segs = [("S fixed 120s moments", dict(fixed_s=120)),
            ("S chapters only", dict(use_semantic=False)),
            ("S semantic only", dict(use_chapters=False))]
    for name, kw in segs:
        w, m = build_moments(vid, cues, chapters, llm=enrich_llm, **kw)
        rag = MomentRAG(vid, w, m, name=name, dense_gate=full.dense_gate)
        systems.append((name, lambda n=name, g=rag: run_moment(n, g, questions, 3)))

    summary, per_q = [], []
    for name, fn in systems:
        rows, lat = fn()
        per_q += rows
        summary.append(summarize(name, rows, lat))
        print(f"{name:26} evidence={summary[-1]['evidence@k %']}% "
              f"MRR={summary[-1]['span MRR']} pinpoint={summary[-1]['pinpoint@1 %']}% "
              f"ctx={summary[-1]['avg ctx words']}w "
              f"unans-abstain={summary[-1]['unanswerable abstain %']}%")

    print("gate calibration: answerable abstained",
          sum(x["abstain"] for x in cal["answerable"]), "/", len(cal["answerable"]),
          "· unanswerable abstained", sum(x["abstain"] for x in cal["unanswerable"]),
          "/", len(cal["unanswerable"]))
    (RESULTS / "retrieval_summary.json").write_text(json.dumps(summary, indent=1))
    (RESULTS / "retrieval_per_question.json").write_text(json.dumps(per_q, indent=1))
    (RESULTS / "gate_calibration.json").write_text(json.dumps(cal, indent=1))
    (RESULTS / "retrieval_summary.md").write_text(to_markdown(summary) + "\n")
    (RESULTS / "moments.md").write_text(
        "| # | start | end | secs | title | chapter |\n|---|---|---|---|---|---|\n" +
        "\n".join(f"| {m['id']} | {int(m['t_start'])} | {int(m['t_end'])} | "
                  f"{int(m['t_end'] - m['t_start'])} | {m['title']} | {m.get('chapter') or ''} |"
                  for m in moments) + "\n")

    if args.answers:
        if llm is None:
            sys.exit("--answers needs an LLM backend (OPENAI_API_KEY or a running Ollama)")
        run_answers(questions, base, full, llm)


def run_answers(questions: list[dict], base: BaselineRAG, full: MomentRAG, llm) -> None:
    path = RESULTS / "answers.json"
    done = {(a["id"], a["system"]): a for a in
            (json.loads(path.read_text()) if path.exists() else [])
            if a.get("model") == llm.model}
    for q in questions:
        for sysname, rag in (("baseline", base), ("moment", full)):
            if (q["id"], sysname) in done:
                continue
            t0 = time.perf_counter()
            out = rag.ask(q["question"], llm=llm, k=3)
            ans = out.get("answer", "")
            done[(q["id"], sysname)] = {
                "id": q["id"], "type": q["type"], "system": sysname, "model": llm.model,
                "question": q["question"], "answer": ans,
                "cited": bool(cited(ans)), "refused": bool(out.get("abstain") or _REFUSAL.search(ans)),
                "seconds": round(time.perf_counter() - t0, 1),
                "sources": ([f"{round(h['t_start'])}-{round(h['t_end'])}@{round(h['anchor_t'])}"
                             for h in out["hits"]] if sysname == "moment" else
                            [round(h["t_start"]) for h in out["hits"]])}
            print(f"[{sysname:8}] {q['id']} {done[(q['id'], sysname)]['seconds']}s")
            path.write_text(json.dumps(list(done.values()), indent=1))   # resumable
    rows = list(done.values())
    table = []
    for sysname in ("baseline", "moment"):
        rs = [r for r in rows if r["system"] == sysname]
        ans = [r for r in rs if r["type"] != "unanswerable"]
        una = [r for r in rs if r["type"] == "unanswerable"]
        table.append({"system": sysname, "model": llm.model,
                      "cites sources %": round(100 * sum(r["cited"] for r in ans) / len(ans), 1),
                      "wrongly refused %": round(100 * sum(r["refused"] for r in ans) / len(ans), 1),
                      "unanswerable refused %": round(100 * sum(r["refused"] for r in una) / len(una), 1),
                      "avg seconds": round(statistics.mean(r["seconds"] for r in rs), 1)})
    (RESULTS / "answer_summary.md").write_text(to_markdown(table) + "\n")
    (RESULTS / "answer_summary.json").write_text(json.dumps(table, indent=1))
    print(to_markdown(table))


if __name__ == "__main__":
    main()
