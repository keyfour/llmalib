"""
reliability/guards.py

Lightweight post-generation checks applied to parsed output before acceptance.
A Guard is a callable: (value: BaseModel, ctx: Context) -> list[str]
Empty return = passed. Non-empty = errors fed into the reflection loop.

No ML models or embeddings — these are fast rule-based heuristics.

The factory functions return `llmalib._rust.NativeGuard` instances: plain
Python-callable handles whose rule logic lives in the Rust crates. Accepting
`(pydantic value, ctx)` callables and returning error strings is unchanged —
the reflection loop and pipeline bridges need no adaptation.
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel

from llmalib import _rust

Guard = Callable[[BaseModel, Any], list[str]]


def field_in_set(field: str, allowed: set[Any]) -> Guard:
    """
    Check that value.<field> is one of the allowed values.

    Example: field_in_set("label", {"positive", "negative", "neutral"})
    Catches: label="POSITIVE" (wrong case), label="mixed" (invented value)
    """
    # Membership is decided Rust-side against stringified allowed values.
    expected = sorted({str(v) for v in allowed})
    return _rust.guard_field_in_set(field, expected)


def float_in_range(field: str, min_val: float, max_val: float) -> Guard:
    """
    Check that value.<field> is between min_val and max_val (inclusive).

    Example: float_in_range("confidence", 0.0, 1.0)
    """
    return _rust.guard_float_in_range(field, float(min_val), float(max_val))


def max_length(field: str, max_chars: int) -> Guard:
    """
    Check that str(value.<field>) does not exceed max_chars.
    Catches runaway generation in summary/explanation fields.
    """
    return _rust.guard_max_length(field, int(max_chars))


def no_content_from_outside_context(
    response_field: str,
    context_field: str,
    threshold: float = 0.3,
) -> Guard:
    """
    Lightweight grounding check: verifies that response_field shares
    significant token overlap with context_field from ctx.vars.

    Uses Jaccard similarity on lowercased word tokens — no embeddings needed.
    A low overlap suggests the model generated from parametric knowledge
    rather than the provided input.

    threshold: minimum Jaccard similarity (0.0–1.0).
      0.3 is permissive (default). Raise to 0.5+ for stricter grounding.
    """
    return _rust.guard_no_content_from_outside_context(
        response_field, context_field, float(threshold)
    )
