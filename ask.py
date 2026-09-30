"""Ask both systems the same question, side by side.

  python ask.py "Why does he advise against opening a talk with a joke?"
  python ask.py --video <youtube_id> "question"      # any video with captions
  python ask.py --moments                            # print the video's moments and exit
  python ask.py --retrieval-only "question"          # no LLM call
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from momentrag.answer import render_sources                      # noqa: E402
from momentrag.baseline import BaselineRAG                       # noqa: E402
from momentrag.llm import get_llm                                # noqa: E402
from momentrag.moment_rag import MomentRAG                       # noqa: E402
from momentrag.moments import build_moments, cached_llm_moments, describe  # noqa: E402
from momentrag.transcript import fetch_chapters, fetch_cues      # noqa: E402

DEFAULT_VIDEO = "Unzc731iCUY"   # Patrick Winston (MIT OpenCourseWare) — How to Speak


def _block(title: str, out: dict, video_id: str, moment: bool) -> None:
    print(f"\n{'=' * 78}\n{title}   ({out['context_words']} context words)\n{'-' * 78}")
    if out.get("answer"):
        print(textwrap.fill(out["answer"], 78))
    print("\nSources:")
    for line in render_sources(video_id, out["hits"], moment) or ["(none — abstained)"]:
        print("  " + line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--moments", action="store_true")
    ap.add_argument("--retrieval-only", action="store_true")
    args = ap.parse_args()

    cues, chapters = fetch_cues(args.video), fetch_chapters(args.video)
    # Prefer cached LLM-written moment cards (built by `eval/run_eval.py --enrich-llm`);
    # the eval shows extractive cards can hurt. Never enriches implicitly.
    windows, moments = (cached_llm_moments(args.video, cues)
                        or build_moments(args.video, cues, chapters))
    if args.moments or not args.question:
        print(describe(moments))
        return
    llm = None if args.retrieval_only else get_llm()
    print(f"LLM: {llm.backend + ':' + llm.model if llm else 'none (retrieval only)'}")
    _block("BASELINE — fixed 200-word chunks",
           BaselineRAG(cues).ask(args.question, llm=llm, k=args.k), args.video, False)
    _block("MOMENT RAG — timestamped moments",
           MomentRAG(args.video, windows, moments).ask(args.question, llm=llm, k=args.k),
           args.video, True)


if __name__ == "__main__":
    main()
