"""Blind LLM judge for generated answers (results/*/answers.json).

Keyword matching is fine for retrieval (the transcript's own spellings are known), but
the LLM rewrites "merily feifer" as "Mary Lee Pfeiffer", so answers are graded against
hand-written reference answers instead. The judge never sees which system wrote an
answer, and the pairs are shuffled.

Verdicts: correct | partial | incorrect.
  answerable    correct = states the reference's key facts, nothing contradicting them
  locate        correct = gives a time within ~30s of the reference time
  unanswerable  correct = declines / says the video doesn't cover it

Usage:  python eval/judge.py results/llm-cards [results/extractive-cards ...]
"""
from __future__ import annotations

import json
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from momentrag.llm import get_llm  # noqa: E402

_SYSTEM = (
    "You grade answers to questions about a video talk against a reference answer. "
    "Reply with ONLY JSON: {\"verdict\": \"correct\"|\"partial\"|\"incorrect\", "
    "\"reason\": \"<one short sentence>\"}.\n"
    "- correct: states the reference's key facts (paraphrase and extra detail are fine) and "
    "contradicts nothing in it.\n"
    "- partial: some key facts right, others missing or wrong.\n"
    "- incorrect: wrong, missing the key facts, or declines to answer.\n"
    "For a 'where/when in the video' question, the answer is correct only if it gives a "
    "timestamp within about 30 seconds of the reference time; a description without a time "
    "is at most partial.\n"
    "If the reference says the video does not cover it, the answer is correct only if it "
    "declines or says the video does not cover it, and incorrect if it asserts an answer."
)


def judge_dir(d: Path, llm) -> dict:
    answers = json.loads((d / "answers.json").read_text())
    qs = {q["id"]: q for q in json.loads((ROOT / "eval/questions.json").read_text())["questions"]}
    cache_path = d / "judged.json"
    cache = {(j["id"], j["system"]): j for j in
             (json.loads(cache_path.read_text()) if cache_path.exists() else [])}
    todo = [a for a in answers if (a["id"], a["system"]) not in cache]
    random.Random(0).shuffle(todo)                      # blind: order carries no system signal
    for a in todo:
        q = qs[a["id"]]
        user = (f"Question: {q['question']}\nReference answer: {q['reference']}\n"
                f"Answer to grade: {a['answer'] or '(empty)'}")
        raw = llm.complete(_SYSTEM, user, max_tokens=200)
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            v = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            v = {}
        verdict = v.get("verdict") if v.get("verdict") in ("correct", "partial", "incorrect") else "incorrect"
        cache[(a["id"], a["system"])] = {"id": a["id"], "type": a["type"], "system": a["system"],
                                         "verdict": verdict, "reason": v.get("reason", raw[:200])}
        cache_path.write_text(json.dumps(list(cache.values()), indent=1))

    rows = []
    for sysname in ("baseline", "moment"):
        js = [j for j in cache.values() if j["system"] == sysname]
        score = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
        row = {"system": sysname}
        for t in ("fact", "explain", "locate", "unanswerable"):
            sub = [j for j in js if j["type"] == t]
            row[f"{t} correct"] = f"{sum(j['verdict'] == 'correct' for j in sub)}/{len(sub)}"
        ans = [j for j in js if j["type"] != "unanswerable"]
        row["answerable correct %"] = round(100 * sum(j["verdict"] == "correct" for j in ans) / len(ans), 1)
        row["answerable score %"] = round(100 * statistics.mean(score[j["verdict"]] for j in ans), 1)
        rows.append(row)
    (d / "judge_summary.json").write_text(json.dumps(rows, indent=1))
    cols = list(rows[0])
    md = "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                   ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows])
    (d / "judge_summary.md").write_text(md + "\n")
    return {"dir": d.name, "table": md}


def main() -> None:
    llm = get_llm()
    if llm is None:
        sys.exit("judge needs an LLM backend")
    for arg in sys.argv[1:] or ["results/llm-cards"]:
        out = judge_dir(ROOT / arg, llm)
        print(f"\n{out['dir']} (judge: {llm.model})\n{out['table']}")


if __name__ == "__main__":
    main()
