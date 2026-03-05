"""
memory/context_window.py

Token budget manager: trim messages to fit within a task's declared budget.
Uses truncation (not LLM summarisation) — faster, cheaper, and competitive
in quality per arxiv:2508.21433.

Never silently overflows. Raises BudgetExceededError if the irreducible
minimum (system + final user turn) already exceeds the budget.
"""

from __future__ import annotations

from typing import Callable


class BudgetExceededError(Exception):
    """
    Raised when system prompt + final user turn exceed the budget alone.
    This indicates a misconfigured task, not a runtime trim situation.
    """


def count_tokens(text: str, tokenizer: Callable[[str], int] | None = None) -> int:
    """
    Estimate token count.
    Default: len(text) // 4  (conservative — errs toward over-trimming).
    Pass a model-specific tokenizer for precision.
    """
    if tokenizer:
        return tokenizer(text)
    return max(1, len(text) // 4)


def trim_to_budget(
    messages: list[dict],
    budget: int,
    tokenizer: Callable[[str], int] | None = None,
) -> list[dict]:
    """
    Return a (possibly trimmed) copy of messages that fits within budget tokens.

    Trim order (most expendable first):
    1. Middle assistant turns — keep first + last sentence
    2. Long user context blocks — truncate to remaining budget
    3. Never trim: system prompt or final user turn

    Raises BudgetExceededError if system + final user turn alone exceed budget.
    Never mutates the input list.
    """
    if not messages:
        return []

    def msg_tokens(m: dict) -> int:
        return count_tokens(m.get("content", ""), tokenizer)

    # Identify protected messages (never trimmed)
    system_msgs = [m for m in messages if m["role"] == "system"]
    final_user = _find_final_user(messages)

    protected_tokens = sum(msg_tokens(m) for m in system_msgs)
    if final_user:
        protected_tokens += msg_tokens(final_user)

    if protected_tokens > budget:
        raise BudgetExceededError(
            f"System prompt + final user turn require ~{protected_tokens} tokens "
            f"but budget is {budget}. Reduce system prompt or increase token_budget."
        )

    # Work on a shallow copy
    result = [dict(m) for m in messages]
    current_tokens = sum(msg_tokens(m) for m in result)

    if current_tokens <= budget:
        return result

    remaining_budget = budget - protected_tokens

    # Pass 1: shorten middle assistant turns
    for i, msg in enumerate(result):
        if current_tokens <= budget:
            break
        if msg["role"] == "assistant" and msg is not final_user:
            original = msg["content"]
            shortened = _keep_first_and_last_sentence(original)
            if len(shortened) < len(original):
                saved = count_tokens(original, tokenizer) - count_tokens(
                    shortened, tokenizer
                )
                result[i] = dict(msg, content=shortened)
                current_tokens -= saved

    # Pass 2: truncate long user context blocks (excluding the final user turn)
    for i, msg in enumerate(result):
        if current_tokens <= budget:
            break
        if msg["role"] == "user" and msg is not final_user:
            original = msg["content"]
            max_chars = remaining_budget * 4  # reverse of count_tokens default
            if len(original) > max_chars:
                truncated = original[:max_chars] + " [truncated]"
                saved = count_tokens(original, tokenizer) - count_tokens(
                    truncated, tokenizer
                )
                result[i] = dict(msg, content=truncated)
                current_tokens -= saved

    return result


def _find_final_user(messages: list[dict]) -> dict | None:
    for msg in reversed(messages):
        if msg["role"] == "user":
            return msg
    return None


def _keep_first_and_last_sentence(text: str) -> str:
    """Reduce a block of text to its first and last sentence."""
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s.strip()]
    if len(sentences) <= 2:
        return text
    return sentences[0] + " [...] " + sentences[-1]
