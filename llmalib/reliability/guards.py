"""
reliability/guards.py

Lightweight post-generation checks applied to parsed output before acceptance.
A Guard is a callable: (value: BaseModel, ctx: Context) -> list[str]
Empty return = passed. Non-empty = errors fed into the reflection loop.

No ML models or embeddings — these are fast rule-based heuristics.
"""

from __future__ import annotations

from typing import Any, Callable
from pydantic import BaseModel

Guard = Callable[[BaseModel, Any], list[str]]


def field_in_set(field: str, allowed: set) -> Guard:
    """
    Check that value.<field> is one of the allowed values.

    Example: field_in_set("label", {"positive", "negative", "neutral"})
    Catches: label="POSITIVE" (wrong case), label="mixed" (invented value)
    """

    def check(value: BaseModel, ctx: Any) -> list[str]:
        actual = getattr(value, field, None)
        if actual not in allowed:
            return [f"Field '{field}' must be one of {sorted(allowed)}, got {actual!r}"]
        return []

    return check


def float_in_range(field: str, min_val: float, max_val: float) -> Guard:
    """
    Check that value.<field> is between min_val and max_val (inclusive).

    Example: float_in_range("confidence", 0.0, 1.0)
    """

    def check(value: BaseModel, ctx: Any) -> list[str]:
        actual = getattr(value, field, None)
        if actual is None:
            return [f"Field '{field}' is missing or None"]
        try:
            f = float(actual)
        except (TypeError, ValueError):
            return [f"Field '{field}' is not numeric: {actual!r}"]
        if not (min_val <= f <= max_val):
            return [f"Field '{field}' must be between {min_val} and {max_val}, got {f}"]
        return []

    return check


def max_length(field: str, max_chars: int) -> Guard:
    """
    Check that str(value.<field>) does not exceed max_chars.
    Catches runaway generation in summary/explanation fields.
    """

    def check(value: BaseModel, ctx: Any) -> list[str]:
        actual = getattr(value, field, None)
        if actual is None:
            return []
        length = len(str(actual))
        if length > max_chars:
            return [
                f"Field '{field}' is too long: {length} chars "
                f"(max {max_chars}). Provide a more concise response."
            ]
        return []

    return check


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

    def check(value: BaseModel, ctx: Any) -> list[str]:
        response_text = str(getattr(value, response_field, "") or "")
        context_text = ""
        if hasattr(ctx, "vars") and isinstance(ctx.vars, dict):
            context_text = str(ctx.vars.get(context_field, "") or "")
        elif isinstance(ctx, dict):
            context_text = str(ctx.get(context_field, "") or "")

        if not context_text.strip():
            # No context to compare against — skip the check
            return []

        response_tokens = set(_tokenize(response_text))
        context_tokens = set(_tokenize(context_text))

        if not response_tokens:
            return [f"Field '{response_field}' is empty — no content to ground."]

        intersection = response_tokens & context_tokens
        union = response_tokens | context_tokens
        jaccard = len(intersection) / len(union) if union else 0.0

        if jaccard < threshold:
            return [
                f"Field '{response_field}' does not appear grounded in the "
                f"provided '{context_field}' (similarity={jaccard:.2f}, "
                f"required>={threshold}). "
                f"Base your answer only on the provided input text."
            ]
        return []

    return check


# --- helpers ---


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenisation — no dependencies."""
    import re

    return re.findall(r"[a-z0-9]+", text.lower())
