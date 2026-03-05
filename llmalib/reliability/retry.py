"""
reliability/retry.py

Reflection-based retry loop.
On validation or guard failure, builds a specific error turn and appends it
to the conversation history before the next attempt.

Every attempt is recorded in the returned Result regardless of outcome.
"""

from __future__ import annotations

import json
import time

from llmalib.core.client import ClientConfig, ClientError, call
from llmalib.core.result import (
    Attempt,
    Result,
    make_attempt,
    make_error_result,
    make_ok_result,
)
from llmalib.core.task import Task
from llmalib.reliability.validator import parse_response


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
    trace: list[Attempt] = []
    current_messages = list(messages)  # local copy — don't mutate caller's list

    for attempt_num in range(1, task.max_retries + 1):
        started_at = time.monotonic()
        rendered = _render_messages(current_messages)

        try:
            raw = call(current_messages, config)
        except ClientError:
            # Network/HTTP errors are not retried here — propagate up
            raise

        parse_result = parse_response(raw, task.output_schema)

        guard_errors: list[str] = []
        if parse_result.ok and task.guards:
            for guard in task.guards:
                guard_errors.extend(guard(parse_result.value, ctx))

        attempt = make_attempt(
            attempt_number=attempt_num,
            rendered_prompt=rendered,
            raw_response=raw,
            parse_error=parse_result.error,
            guard_errors=guard_errors,
            started_at=started_at,
        )
        trace.append(attempt)

        if parse_result.ok and not guard_errors:
            return make_ok_result(task.name, parse_result.value, trace)

        # Build reflection message for next attempt
        if attempt_num < task.max_retries:
            reflection = build_reflection_message(parse_result.error, guard_errors)
            current_messages = current_messages + [
                {"role": "assistant", "content": raw},
                reflection,
            ]

    # Exhausted retries — report the last failure
    last = trace[-1]
    all_errors = []
    if last.parse_error:
        all_errors.append(last.parse_error)
    all_errors.extend(last.guard_errors)
    final_error = f"Failed after {task.max_retries} attempts. " + " | ".join(all_errors)

    return make_error_result(task.name, final_error, trace)


def build_reflection_message(
    parse_error: str | None,
    guard_errors: list[str],
) -> dict:
    """
    Build the user-turn message that feeds failure details back to the model.

    Specific + actionable: lists each error individually with a clear directive.
    Concise: small models lose focus with long error context.
    """
    lines = ["Your previous response was invalid. Fix the following errors:"]

    if parse_error:
        for line in parse_error.strip().splitlines():
            lines.append(f"  - {line.strip()}")

    for err in guard_errors:
        lines.append(f"  - {err}")

    lines.append("")
    lines.append(
        "Respond ONLY with valid JSON matching the required schema. "
        "No prose, no code fences, no explanation."
    )

    return {"role": "user", "content": "\n".join(lines)}


def _render_messages(messages: list[dict]) -> str:
    """Serialise messages list to a readable string for trace storage."""
    return json.dumps(messages, indent=2, ensure_ascii=False)
