"""
debug/tracer.py

Always-on recorder for every LLM call in a pipeline run.
Overhead is negligible (string copies). The debugging value is not.

A thin wrapper over `llmalib._rust.PyTracer`: recording, summaries and
to_dict/to_file/load_trace serialization run Rust-side under the same
formats as the engine. `started_at` stays a Python monotonic float (it is
the display clock for trace inspection; the persisted `started_at`,
serialized by Rust, is an RFC3339 timestamp).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from llmalib import _rust
from llmalib._bridge import attempt_from_dict, task_result_to_dict
from llmalib.core.result import Attempt, Result


@dataclass
class TaskTrace:
    task_name: str
    attempts: list[Attempt]
    ok: bool
    final_value: dict[str, Any] | None
    final_error: str | None


class Tracer:
    """Wrapper around the Rust tracer; API unchanged (record/summary/to_dict/…)."""

    def __init__(self, run_id: str) -> None:
        self._inner = _rust.make_tracer(run_id)
        # Python-side display clock for inspection (not persisted).
        self.started_at: float = time.monotonic()

    @property
    def run_id(self) -> str:
        return self._inner.run_id

    @run_id.setter
    def run_id(self, value: str) -> None:
        self._inner.run_id = value

    @property
    def task_traces(self) -> list[TaskTrace]:
        return [_hydrate_trace(d) for d in self._inner.task_traces]

    def record(self, result: Result) -> None:
        """Append a TaskTrace from a completed Result."""
        value = result.value.model_dump() if result.value is not None else None
        self._inner.record(
            task_result_to_dict(
                result.task_name, result.ok, value, result.error, result.trace
            )
        )

    def summary(self) -> str:
        return self._inner.summary()

    def to_dict(self) -> dict[str, Any]:
        return self._inner.to_dict()

    def to_file(self, path: str) -> None:
        self._inner.to_file(path)


def make_tracer(run_id: str) -> Tracer:
    return Tracer(run_id=run_id)


def load_trace(path: str) -> Tracer:
    """Load a trace from a JSON file for inspection or replay."""
    wrapped = Tracer.__new__(Tracer)
    wrapped._inner = _rust.load_trace(path)
    wrapped.started_at = time.monotonic()
    return wrapped


def _hydrate_trace(d: dict[str, Any]) -> TaskTrace:
    """Rebuild a Python `TaskTrace` from the Rust tracer's dict."""
    return TaskTrace(
        task_name=d["task_name"],
        attempts=[attempt_from_dict(a) for a in d["attempts"]],
        ok=d["ok"],
        final_value=d.get("final_value"),
        final_error=d.get("final_error"),
    )
