"""
memory/context_window.py

Token budget manager: trim messages to fit within a task's declared budget.
Thin adapter over `llmalib._rust` (estimation + trimming strategies run in
Rust). Uses truncation (not LLM summarisation) — faster, cheaper, and
competitive in quality per arxiv:2508.21433.

Never silently overflows. Raises BudgetExceededError if the irreducible
minimum (system + final user turn) already exceeds the budget.
"""

from __future__ import annotations

from typing import Any, Callable

from llmalib import _rust

# The Rust engine raises this exact class on the boundary — alias it so
# `pytes.raises(BudgetExceededError)` and user `except` clauses work.
BudgetExceededError = _rust.BudgetExceededError


def count_tokens(text: str, tokenizer: Callable[[str], int] | None = None) -> int:
    """
    Estimate token count.
    Default: len(text) // 4  (conservative — errs toward over-trimming).
    Pass a model-specific tokenizer for precision.
    """
    return _rust.count_tokens(text, tokenizer)


def trim_to_budget(
    messages: list[dict[str, Any]],
    budget: int,
    tokenizer: Callable[[str], int] | None = None,
) -> list[dict[str, Any]]:
    """
    Return a (possibly trimmed) copy of messages that fits within budget tokens.

    Trim order (most expendable first):
    1. Middle assistant turns — keep first + last sentence
    2. Long user context blocks — truncate to remaining budget
    3. Never trim: system prompt or final user turn

    Raises BudgetExceededError if system + final user turn alone exceed budget.
    Never mutates the input list.
    """
    return _rust.trim_to_budget(messages, budget, tokenizer)


def _keep_first_and_last_sentence(text: str) -> str:
    """Reduce a block of text to its first and last sentence."""
    return _rust.keep_first_and_last_sentence(text)
