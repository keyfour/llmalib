"""
reliability/retry.py

Reflection-based retry loop (engine in `llmalib._rust.run_with_retry`).
On validation or guard failure, a specific error turn is appended to the
conversation history before the next attempt.

Every attempt is recorded in the returned Result regardless of outcome.
"""

from __future__ import annotations

from llmalib import _rust
from llmalib._bridge import (
    config_to_dict,
    ctx_to_dict,
    result_from_dict,
    spec_from_task,
)
from llmalib.core.client import ClientConfig, ClientError
from llmalib.core.result import Result
from llmalib.core.task import Task


def run_with_retry(
    task: Task,
    messages: list[dict],
    config: ClientConfig,
    ctx=None,  # Context — passed to guards; optional to avoid circular import
) -> Result:
    """
    Execute task with reflection-based retry on validation or guard failure.

    Each failed attempt appends a reflection turn to the message history
    so the model can see what went wrong and correct it.
    Returns Result — never raises on model-level failures.
    Raises ClientError only on unrecoverable network/HTTP failures.
    """
    ctx_dict = ctx_to_dict(ctx) if ctx is not None else None
    try:
        out = _rust.run_with_retry(
            spec_from_task(task), messages, config_to_dict(config), ctx_dict
        )
    except _rust.ClientError as exc:
        raise ClientError(str(exc)) from None
    return result_from_dict(out, task.output_schema)


def build_reflection_message(
    parse_error: str | None,
    guard_errors: list[str],
) -> dict:
    """
    Build the user-turn message that feeds failure details back to the model.

    Specific + actionable: lists each error individually with a clear directive.
    Concise: small models lose focus with long error context.
    """
    return _rust.build_reflection_message(parse_error, guard_errors)
