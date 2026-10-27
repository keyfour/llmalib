"""
memory/store.py

Key-value store for results, examples, and any cross-run state.
Thin adapter over `llmalib._rust`: both backends (in-memory dict, JSON file
on disk) and the BM25 relevance ranking run in Rust behind `PyStore`.
Both expose the same interface as before.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from llmalib import _rust
from llmalib._bridge import value_to_dict
from llmalib.core.task import Example


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
# Shared implementation (wraps the Rust PyStore)
# ---------------------------------------------------------------------------


class _RustBackedStore:
    """Common wrapper behaviour; subclasses pick the Rust backend."""

    _inner: _rust.PyStore

    def set(
        self, key: str, value: Any, score: float = 1.0, ttl: float | None = None
    ) -> None:
        # Value policy: pydantic models / Example records are converted to
        # JSON-friendly dicts at the boundary (the Rust store holds JSON).
        # Stored Examples are deserialised back into pydantic Instances by
        # select_examples downstream.
        self._inner.set(key, value_to_dict(value), score, ttl)

    def get(self, key: str) -> StoreEntry | None:
        entry = self._inner.get(key)
        if entry is None:
            return None
        return _entry_from_dict(entry)

    def delete(self, key: str) -> None:
        self._inner.delete(key)

    def list(self, prefix: str = "") -> list[str]:
        return self._inner.list(prefix)

    def get_relevant(self, query: str, top_k: int = 5) -> list[StoreEntry]:
        return [_entry_from_dict(e) for e in self._inner.get_relevant(query, top_k)]


class InMemoryStore(_RustBackedStore):
    def __init__(self) -> None:
        self._inner = _rust.make_memory_store()


def make_memory_store() -> InMemoryStore:
    return InMemoryStore()


# ---------------------------------------------------------------------------
# File-backed backend
# ---------------------------------------------------------------------------


class FileStore(_RustBackedStore):
    def __init__(self, path: str) -> None:
        try:
            self._inner = _rust.make_file_store(path)
        except _rust.StoreError as exc:
            raise OSError(str(exc)) from None


def make_file_store(path: str) -> FileStore:
    return FileStore(path)


# ---------------------------------------------------------------------------
# BM25 ranking (delegates to Rust)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return _rust.tokenize(text)


def _entry_from_dict(d: dict[str, Any]) -> StoreEntry:
    value = d["value"]
    # Examples stored through .set() come back as their spec dicts —
    # reshape them into a pydantic instance when we can. The schema is not
    # known here, so the output stays a dict; select_examples re-hydrates.
    if (
        isinstance(value, dict)
        and "input_text" in value
        and "output" in value
    ):
        value = Example(
            input_text=value["input_text"],
            output=value.get("output", {}),
            description=value.get("description", ""),
        )
    return StoreEntry(
        key=d["key"],
        value=value,
        score=d["score"],
        created_at=d["created_at"],
        expires_at=d["expires_at"],
    )


def _entry_to_dict(e: StoreEntry) -> dict[str, Any]:
    return {
        "key": e.key,
        "value": value_to_dict(e.value),
        "score": e.score,
        "created_at": e.created_at,
        "expires_at": e.expires_at,
    }


def _bm25_rank(
    query: str,
    entries: list[StoreEntry],
    top_k: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[StoreEntry]:
    if not entries:
        return []
    # Entries marshal as dicts (the Rust layer holds JSON-only values).
    return [
        _entry_from_dict(e)
        for e in _rust.bm25_rank(query, [_entry_to_dict(x) for x in entries], top_k)
    ]
