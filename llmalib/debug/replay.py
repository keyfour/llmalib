"""
debug/replay.py

Replay a specific recorded task attempt against a different model or prompt.
Primary tool for iterating on prompts and schema design without re-running
the full pipeline.
"""

from __future__ import annotations

import json
import sys
from pydantic import BaseModel

from llmalib.core.client import ClientConfig, call
from llmalib.core.result import Result, make_attempt, make_error_result, make_ok_result
from llmalib.core.task import Task
from llmalib.debug.tracer import Tracer
from llmalib.reliability.validator import parse_response
import time


def replay_task(
    tracer: Tracer,
    task_name: str,
    task: Task,
    attempt_number: int = 1,
    model: str | None = None,
    base_url: str | None = None,
    prompt_override: str | None = None,
    schema_override: type[BaseModel] | None = None,
    diff: bool = True,
) -> Result:
    """
    Re-run a recorded task attempt, optionally with overrides.

    Loads the exact messages from the recorded attempt, applies any overrides,
    calls the model, and returns a new Result.

    diff=True prints a side-by-side comparison of original vs new response.
    """
    # Find the task trace
    tt = next((t for t in tracer.task_traces if t.task_name == task_name), None)
    if tt is None:
        available = [t.task_name for t in tracer.task_traces]
        raise ValueError(
            f"Task '{task_name}' not found in trace. Available: {available}"
        )

    if attempt_number < 1 or attempt_number > len(tt.attempts):
        raise ValueError(
            f"attempt_number must be between 1 and {len(tt.attempts)}, "
            f"got {attempt_number}"
        )

    original_attempt = tt.attempts[attempt_number - 1]

    # Reconstruct messages from the rendered prompt
    try:
        messages: list[dict] = json.loads(original_attempt.rendered_prompt)
    except json.JSONDecodeError:
        # Fallback: wrap as a single user message if not valid JSON
        messages = [{"role": "user", "content": original_attempt.rendered_prompt}]

    # Apply prompt override to the final user turn
    if prompt_override is not None:
        messages = list(messages)
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                messages[i] = dict(messages[i], content=prompt_override)
                break

    schema = schema_override or task.output_schema
    config = ClientConfig(
        base_url=base_url or task.base_url,
        model=model or task.model,
        temperature=task.temperature,
        timeout=task.timeout,
        max_tokens=task.max_tokens,
    )

    started_at = time.monotonic()
    raw = call(messages, config)
    parse_result = parse_response(raw, schema)

    guard_errors: list[str] = []
    if parse_result.ok and task.guards:
        for guard in task.guards:
            guard_errors.extend(guard(parse_result.value, None))

    attempt = make_attempt(
        attempt_number=1,
        rendered_prompt=json.dumps(messages, indent=2),
        raw_response=raw,
        parse_error=parse_result.error,
        guard_errors=guard_errors,
        started_at=started_at,
    )

    if diff:
        _print_diff(original_attempt.raw_response, raw, attempt_number)

    if parse_result.ok and not guard_errors:
        return make_ok_result(task_name, parse_result.value, [attempt])
    else:
        errors = []
        if parse_result.error:
            errors.append(parse_result.error)
        errors.extend(guard_errors)
        return make_error_result(task_name, " | ".join(errors), [attempt])


def _print_diff(original: str, replayed: str, attempt_number: int) -> None:
    sep = "─" * 60
    print(f"\n{sep}", file=sys.stderr)
    print(f"── Original (attempt {attempt_number})", file=sys.stderr)
    print(original[:500], file=sys.stderr)
    print(f"\n── Replay", file=sys.stderr)
    print(replayed[:500], file=sys.stderr)

    # Simple line-level diff
    orig_lines = original.splitlines()
    new_lines = replayed.splitlines()
    if orig_lines != new_lines:
        print(f"\n── Diff", file=sys.stderr)
        for line in orig_lines:
            if line not in new_lines:
                print(f"- {line}", file=sys.stderr)
        for line in new_lines:
            if line not in orig_lines:
                print(f"+ {line}", file=sys.stderr)
    else:
        print("\n── Diff: (no change)", file=sys.stderr)
    print(sep, file=sys.stderr)
