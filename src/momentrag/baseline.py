"""Part 1 — baseline semantic-search RAG over a YouTube transcript.

The standard tutorial pipeline:
  transcript text -> fixed-size word chunks (size/overlap) -> embeddings -> FAISS
  (inner product on normalized vectors = cosine) -> top-k chunks -> LLM answer.

The transcript is flattened to one string before chunking, which is exactly the
weakness Moment RAG targets: chunk edges fall wherever the word count says, mid-thought,
and the answer carries no timestamp. Each chunk still records the time span its words
came from, but only so the evaluation can score it — the baseline never shows it to
the LLM or the user.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import faiss

from . import models
from .answer import SYSTEM_BASELINE, format_baseline_context

CHUNK_WORDS = 200
OVERLAP_WORDS = 40


def chunk_words(cues: list[dict], size: int = CHUNK_WORDS,
                overlap: int = OVERLAP_WORDS) -> list[dict]:
    """Flatten the transcript and cut fixed-size word chunks with overlap."""
    words: list[tuple[str, float, float]] = []
    for c in cues:
        for w in c["text"].split():
            words.append((w, c["t_start"], c["t_end"]))
    step = max(1, size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        span = words[i:i + size]
        if not span:
            break
        chunks.append({"id": len(chunks), "text": " ".join(w for w, _, _ in span),
                       "t_start": span[0][1], "t_end": span[-1][2]})
        if i + size >= len(words):
            break
    return chunks


@dataclass
class BaselineRAG:
    cues: list[dict]
    chunk_size: int = CHUNK_WORDS
    overlap: int = OVERLAP_WORDS
    chunks: list[dict] = field(init=False)
    index: faiss.Index = field(init=False)

    def __post_init__(self) -> None:
        self.chunks = chunk_words(self.cues, self.chunk_size, self.overlap)
        vecs = models.embed_docs([c["text"] for c in self.chunks])
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    def retrieve(self, question: str, k: int = 3) -> list[dict]:
        q = models.embed_query(question)[None, :]
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [{**self.chunks[i], "score": float(s), "rank": r}
                for r, (s, i) in enumerate(zip(scores[0], ids[0])) if i >= 0]

    def ask(self, question: str, llm=None, k: int = 3) -> dict:
        hits = self.retrieve(question, k)
        out = {"question": question, "system": "baseline", "hits": hits,
               "context_words": sum(len(h["text"].split()) for h in hits)}
        if llm is not None:
            out["answer"] = llm.complete(SYSTEM_BASELINE,
                                         format_baseline_context(question, hits))
        return out
