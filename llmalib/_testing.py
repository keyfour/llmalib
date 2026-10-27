"""
llmalib/_testing.py

Test seam for the Rust extension: install a fake model-call transport
instead of patching `llmalib.reliability.retry.call` / httpx.

Usage in tests:

    from llmalib._testing import set_caller, sequence_caller, raising_caller

    set_caller(sequence_caller('{"label": "positive", "confidence": 0.9}'))
    set_caller(raising_caller(ClientError("network down")))
    set_caller(None)        # reset (restores the real HTTP transport)
"""

from __future__ import annotations

from typing import Any, Callable

from llmalib._rust import set_caller


def sequence_caller(
    *responses: str,
) -> Callable[[list[dict[str, Any]], dict[str, Any]], str]:
    """
    Caller that hands out the given responses in order, one per call.
    The final response is repeated once the sequence is exhausted — this
    mirrors `unittest.mock(return_value=...)` semantics for tests that
    expect the same response on every attempt.
    """
    seq: list[str] = list(responses)
    if not seq:
        raise ValueError("sequence_caller needs at least one response")

    def caller(messages: list[dict[str, Any]], config: dict[str, Any]) -> str:
        if len(seq) > 1:
            return seq.pop(0)
        return seq[0]

    return caller


def raising_caller(
    exception: BaseException,
) -> Callable[[list[dict[str, Any]], dict[str, Any]], str]:
    """Caller that raises the given exception on every call."""

    def caller(messages: list[dict[str, Any]], config: dict[str, Any]) -> str:
        raise exception

    return caller


def inspecting_caller(
    response: str, recorder: dict[str, Any]
) -> Callable[[list[dict[str, Any]], dict[str, Any]], str]:
    """
    Caller that always returns `response` while recording the last
    call's messages and config into `recorder`.
    """

    def caller(messages: list[dict[str, Any]], config: dict[str, Any]) -> str:
        recorder["messages"] = messages
        recorder["config"] = config
        return response

    return caller


__all__ = ["set_caller", "sequence_caller", "raising_caller", "inspecting_caller"]
