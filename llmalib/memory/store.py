"""
memory/store.py

Key-value store for results, examples, and any cross-run state.
Two backends: InMemoryStore (default) and FileStore (JSON on disk).
Both expose the same interface.

BM25-powered get_relevant() — no external library, ~50 lines.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class StoreEntry:
    key: str
    value: Any
    score: float = 1.0
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


@runtime_checkable
class Store(Protocol):
    def set(
        self, key: str, value: Any, score: float = 1.0, ttl: float | None = None
    ) -> None: ...
    def get(self, key: str) -> StoreEntry | None: ...
    def delete(self, key: str) -> None: ...
    def list(self, prefix: str = "") -> list[str]: ...
    def get_relevant(self, query: str, top_k: int = 5) -> list[StoreEntry]: ...


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------


class InMemoryStore:
    def __init__(self) -> None:
        self._data: dict[str, StoreEntry] = {}

    def set(
        self, key: str, value: Any, score: float = 1.0, ttl: float | None = None
    ) -> None:
        expires_at = time.time() + ttl if ttl is not None else None
        self._data[key] = StoreEntry(
            key=key, value=value, score=score, expires_at=expires_at
        )

    def get(self, key: str) -> StoreEntry | None:
        entry = self._data.get(key)
        if entry is None or entry.is_expired():
            return None
        return entry

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def list(self, prefix: str = "") -> list[str]:
        return [
            k
            for k, v in self._data.items()
            if k.startswith(prefix) and not v.is_expired()
        ]

    def get_relevant(self, query: str, top_k: int = 5) -> list[StoreEntry]:
        live_entries = [v for v in self._data.values() if not v.is_expired()]
        return _bm25_rank(query, live_entries, top_k)


def make_memory_store() -> InMemoryStore:
    return InMemoryStore()


# ---------------------------------------------------------------------------
# File-backed backend
# ---------------------------------------------------------------------------


class FileStore:
    def __init__(self, path: str) -> None:
        self._path = path
        self._data: dict[str, StoreEntry] = {}
        self._load()

    def set(
        self, key: str, value: Any, score: float = 1.0, ttl: float | None = None
    ) -> None:
        expires_at = time.time() + ttl if ttl is not None else None
        self._data[key] = StoreEntry(
            key=key, value=value, score=score, expires_at=expires_at
        )
        self._save()

    def get(self, key: str) -> StoreEntry | None:
        entry = self._data.get(key)
        if entry is None or entry.is_expired():
            return None
        return entry

    def delete(self, key: str) -> None:
        self._data.pop(key, None)
        self._save()

    def list(self, prefix: str = "") -> list[str]:
        return [
            k
            for k, v in self._data.items()
            if k.startswith(prefix) and not v.is_expired()
        ]

    def get_relevant(self, query: str, top_k: int = 5) -> list[StoreEntry]:
        live_entries = [v for v in self._data.values() if not v.is_expired()]
        return _bm25_rank(query, live_entries, top_k)

    def _load(self) -> None:
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for item in raw:
                e = StoreEntry(**item)
                if not e.is_expired():
                    self._data[e.key] = e
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, TypeError):
            # Corrupted file — start fresh
            self._data = {}

    def _save(self) -> None:
        live = [
            {
                "key": e.key,
                "value": e.value,
                "score": e.score,
                "created_at": e.created_at,
                "expires_at": e.expires_at,
            }
            for e in self._data.values()
            if not e.is_expired()
        ]
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(live, f, indent=2, default=str)


def make_file_store(path: str) -> FileStore:
    return FileStore(path)


# ---------------------------------------------------------------------------
# BM25 ranking (no external library)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_rank(
    query: str,
    entries: list[StoreEntry],
    top_k: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[StoreEntry]:
    if not entries:
        return []

    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return entries[:top_k]

    # Build corpus: key + serialised value for each entry
    corpus: list[list[str]] = []
    for e in entries:
        text = e.key + " " + json.dumps(e.value, default=str)
        corpus.append(_tokenize(text))

    N = len(corpus)
    avg_dl = sum(len(doc) for doc in corpus) / N

    # IDF per query term
    idf: dict[str, float] = {}
    for term in query_tokens:
        df = sum(1 for doc in corpus if term in doc)
        idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

    # BM25 score × entry quality score
    scored: list[tuple[float, StoreEntry]] = []
    for doc, entry in zip(corpus, entries):
        dl = len(doc)
        term_freq: dict[str, int] = {}
        for t in doc:
            term_freq[t] = term_freq.get(t, 0) + 1

        bm25 = 0.0
        for term in query_tokens:
            tf = term_freq.get(term, 0)
            bm25 += idf.get(term, 0) * (
                tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg_dl))
            )

        scored.append((bm25 * entry.score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:top_k]]
