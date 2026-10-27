"""
debug/replay.py

Replay a specific recorded task attempt against a different model or prompt.
Primary tool for iterating on prompts and schema design without re-running
the full pipeline.

The Python layer validates the trace select, mirrors the recorded attempts
into a Rust tracer, and lets `llmalib._rust.replay_task` do the engine work
(call + parse + guards, optional diff output to stderr).
"""

from __future__ import annotations

from pydantic import BaseModel

from llmalib import _rust
from llmalib._bridge import result_from_dict, spec_from_task
from llmalib.core.client import ClientError
from llmalib.core.result import Result
from llmalib.core.task import Task
from llmalib.debug.tracer import TaskTrace, Tracer


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
    # --- trace selection (same error messages as before) ---
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

    # --- mirror the recorded attempts into the Rust tracer ---
    rust_tracer = _rust.make_tracer(tracer.run_id)
    for trace in tracer.task_traces:
        rust_tracer.record(_trace_to_result_dict(trace))

    schema = schema_override or task.output_schema
    try:
        out = _rust.replay_task(
            rust_tracer,
            task_name,
            spec_from_task(task),
            attempt_number,
            model,
            base_url,
            prompt_override,
            schema.model_json_schema() if schema_override is not None else None,
            diff,
        )
    except _rust.ClientError as exc:
        raise ClientError(str(exc)) from None

    return result_from_dict(out, schema)


def _trace_to_result_dict(trace: TaskTrace) -> dict:
    """Serialise a TaskTrace into the TaskResult shape `_rust` expects."""
    return {
        "task_name": trace.task_name,
        "ok": trace.ok,
        "value": trace.final_value,
        "error": trace.final_error,
        "attempts": len(trace.attempts),
        "trace": [
            {
                "attempt_number": a.attempt_number,
                "rendered_prompt": a.rendered_prompt,
                "raw_response": a.raw_response,
                "parse_error": a.parse_error,
                "guard_errors": a.guard_errors,
                "duration_ms": a.duration_ms,
            }
            for a in trace.attempts
        ],
    }
