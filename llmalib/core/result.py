"""
core/result.py

Typed result envelope for every task execution.
Never raises on model-level failures — errors are values.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pydantic import BaseModel


@dataclass
class Attempt:
    """Record of a single LLM call within a task execution."""

    attempt_number: int
    rendered_prompt: str  # Exact messages serialised as string (for tracing/replay)
    raw_response: str  # Exact string returned by the model
    parse_error: str | None  # Pydantic validation error, if any
    guard_errors: list[str]  # Guard check failures (empty = all passed)
    duration_ms: float


@dataclass
class Result:
    """
    Outcome of one task execution.

    ok=True  → value is a validated Pydantic instance, error is None
    ok=False → value is None, error describes the final failure
    trace holds every attempt regardless of outcome.
    """

    task_name: str
    ok: bool
    value: BaseModel | None
    error: str | None
    attempts: int
    trace: list[Attempt] = field(default_factory=list)


def make_ok_result(task_name: str, value: BaseModel, trace: list[Attempt]) -> Result:
    return Result(
        task_name=task_name,
        ok=True,
        value=value,
        error=None,
        attempts=len(trace),
        trace=trace,
    )


def make_error_result(task_name: str, error: str, trace: list[Attempt]) -> Result:
    return Result(
        task_name=task_name,
        ok=False,
        value=None,
        error=error,
        attempts=len(trace),
        trace=trace,
    )


def make_attempt(
    attempt_number: int,
    rendered_prompt: str,
    raw_response: str,
    parse_error: str | None,
    guard_errors: list[str],
    started_at: float,
) -> Attempt:
    return Attempt(
        attempt_number=attempt_number,
        rendered_prompt=rendered_prompt,
        raw_response=raw_response,
        parse_error=parse_error,
        guard_errors=guard_errors,
        duration_ms=(time.monotonic() - started_at) * 1000,
    )
