"""Local, keyless models shared by BOTH systems — so any difference in the comparison
comes from how the transcript is structured and searched, never from the embedder.

  embedder  BAAI/bge-small-en-v1.5   (384-d, normalized; query-side instruction prefix)
  reranker  cross-encoder/ms-marco-MiniLM-L-6-v2  (Moment RAG only, like Moment
            Search's fastembed cross-encoder in src/rag/rerank.py)
"""
from __future__ import annotations

import math
import os
from functools import lru_cache

import numpy as np

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL)


def embed_docs(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    return _embedder().encode(texts, batch_size=32, normalize_embeddings=True,
                              show_progress_bar=False).astype("float32")


def embed_query(text: str) -> np.ndarray:
    prefix = _QUERY_PREFIX if "bge" in EMBED_MODEL else ""
    return _embedder().encode([prefix + text], normalize_embeddings=True,
                              show_progress_bar=False)[0].astype("float32")


def rerank_scores(query: str, docs: list[str]) -> list[float]:
    """Cross-encoder relevance squashed to 0-1 (sigmoid of the raw logit)."""
    if not docs:
        return []
    raw = _cross_encoder().predict([(query, d) for d in docs], show_progress_bar=False)
    return [1.0 / (1.0 + math.exp(-float(s))) for s in raw]
