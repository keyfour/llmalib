"""
memory/examples.py

Select and format few-shot examples for prompt injection.
Sources: inline task.examples (priority) + store.get_relevant() (dynamic).
Examples are placed at the end of the system prompt for maximum recency effect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from llmalib.core.task import Example, Task
from llmalib.memory.store import Store


@dataclass(frozen=True)
class FormattedExample:
    text: str  # Rendered string ready for prompt injection
    source: str  # "inline" | "store" — for tracing


def select_examples(
    task: Task,
    ctx_vars: dict,
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
    selected: list[FormattedExample] = []

    # Inline examples — always included
    for ex in task.examples[:max_examples]:
        selected.append(_format_example(ex, "inline"))

    remaining_slots = max_examples - len(selected)

    # Store examples — fill remaining slots
    if store is not None and remaining_slots > 0:
        query = _build_query(ctx_vars)
        entries = store.get_relevant(
            query, top_k=remaining_slots + 2
        )  # fetch a few extra, filter below
        for entry in entries:
            if len(selected) >= max_examples:
                break
            value = entry.value
            if isinstance(value, Example):
                selected.append(_format_example(value, "store"))
            elif isinstance(value, dict) and "input_text" in value:
                # Deserialise dicts stored from previous runs
                try:
                    ex = Example(
                        input_text=value["input_text"],
                        output=_dict_to_output(value.get("output", {}), task),
                        description=value.get("description", ""),
                    )
                    selected.append(_format_example(ex, "store"))
                except Exception:
                    pass  # Skip malformed entries silently

    # Put highest-relevance example last (recency effect)
    # Inline examples come first; store examples are already ordered by relevance
    return selected


def format_examples_block(examples: list[FormattedExample]) -> str:
    """
    Render a list of FormattedExample into a single block for the system prompt.
    Returns empty string if no examples.
    """
    if not examples:
        return ""
    lines = ["", "--- Examples ---"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"\nExample {i}:")
        lines.append(ex.text)
    lines.append("--- End Examples ---")
    return "\n".join(lines)


def _format_example(ex: Example, source: str) -> FormattedExample:
    try:
        output_str = ex.output.model_dump_json(indent=2)
    except Exception:
        output_str = json.dumps(str(ex.output), indent=2)

    text = f"Input: {ex.input_text}\nOutput:\n{output_str}"
    if ex.description:
        text = f"[{ex.description}]\n{text}"
    return FormattedExample(text=text, source=source)


def _build_query(ctx_vars: dict) -> str:
    """Build a BM25 query string from context variables."""
    parts = []
    for v in ctx_vars.values():
        if isinstance(v, str):
            parts.append(v[:200])  # Cap to avoid very long queries
    return " ".join(parts) if parts else "example"


def _dict_to_output(data: dict, task: Task):
    """Try to reconstruct a Pydantic model instance from a plain dict."""
    return task.output_schema.model_validate(data)
