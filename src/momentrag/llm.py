"""Answer-generation backend, env-switched (like Moment Search's LLM_* config).

  LLM_BACKEND=openai  OPENAI_API_KEY + LLM_MODEL (default gpt-5.6-luna)
  LLM_BACKEND=ollama  local Ollama server, OLLAMA_MODEL (default deepseek-r1:8b)
  LLM_BACKEND=none    retrieval-only: no generated answer

With LLM_BACKEND unset: openai when OPENAI_API_KEY is set, else ollama when the local
server answers, else none. Both RAG systems always use the same backend and model.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")   # this project's .env only

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
_THINK = re.compile(r"<think>.*?</think>", re.S)


@dataclass
class LLM:
    backend: str
    model: str

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        if self.backend == "openai":
            from openai import OpenAI

            resp = OpenAI().chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            return (resp.choices[0].message.content or "").strip()
        if self.backend == "ollama":
            body = json.dumps({
                "model": self.model, "stream": False,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "options": {"temperature": 0, "num_ctx": 8192, "seed": 7},
            }).encode()
            req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                text = json.loads(r.read())["message"]["content"]
            return _THINK.sub("", text).strip()      # reasoning models emit <think>…</think>
        raise RuntimeError("no LLM backend configured (LLM_BACKEND=none)")


def _ollama_up() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2):
            return True
    except Exception:
        return False


def get_llm() -> LLM | None:
    backend = os.getenv("LLM_BACKEND", "").strip().lower()
    if not backend:
        backend = ("openai" if os.getenv("OPENAI_API_KEY")
                   else "ollama" if _ollama_up() else "none")
    if backend == "openai":
        return LLM("openai", os.getenv("LLM_MODEL", "gpt-5.6-luna"))
    if backend == "ollama":
        return LLM("ollama", os.getenv("OLLAMA_MODEL", "deepseek-r1:8b"))
    return None
