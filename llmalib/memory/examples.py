"""
memory/examples.py

Select and format few-shot examples for prompt injection.
Thin adapter over `llmalib._rust`: selection (inline task.examples priority +
BM25 relevance from the store) and formatting run in Rust.
Examples are placed at the end of the system prompt for maximum recency effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llmalib import _rust
from llmalib._bridge import spec_from_task
from llmalib.core.task import Task
from llmalib.memory.store import Store


@dataclass(frozen=True)
class FormattedExample:
    text: str  # Rendered string ready for prompt injection
    source: str  # "inline" | "store" — for tracing


def select_examples(
    task: Task,
    ctx_vars: dict[str, Any],
    store: Store | None = None,
    max_examples: int = 4,
) -> list[FormattedExample]:
    """
    Select up to max_examples few-shot examples.

    Priority:
    1. Inline task.examples (always included first, up to limit)
    2. Store examples (fill remaining slots via BM25 relevance)

    Most relevant example is placed last (recency bias per research).
    """
    # Protocol-typed store: the Rust-backed attribute is looked up defensively.
    inner = getattr(store, "_inner", None) if store is not None else None
    out = _rust.select_examples(spec_from_task(task), ctx_vars, inner, max_examples)
    return [FormattedExample(text=d["text"], source=d["source"]) for d in out]


def format_examples_block(examples: list[FormattedExample]) -> str:
    """
    Render a list of FormattedExample into a single block for the system prompt.
    Returns empty string if no examples.
    """
    if not examples:
        return ""
    return _rust.format_examples_block(
        [{"text": ex.text, "source": ex.source} for ex in examples]
    )
