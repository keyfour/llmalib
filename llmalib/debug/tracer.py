"""
debug/tracer.py

Always-on recorder for every LLM call in a pipeline run.
Overhead is negligible (string copies). The debugging value is not.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from llmalib.core.result import Attempt, Result


@dataclass
class TaskTrace:
    task_name: str
    attempts: list[Attempt]
    ok: bool
    final_value: dict | None
    final_error: str | None


@dataclass
class Tracer:
    run_id: str
    task_traces: list[TaskTrace] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)

    def record(self, result: Result) -> None:
        """Append a TaskTrace from a completed Result."""
        self.task_traces.append(
            TaskTrace(
                task_name=result.task_name,
                attempts=result.trace,
                ok=result.ok,
                final_value=result.value.model_dump() if result.value else None,
                final_error=result.error,
            )
        )

    def summary(self) -> str:
        elapsed = f"{time.monotonic() - self.started_at:.1f}s"
        parts = [f"run={self.run_id} | {elapsed} |"]
        for tt in self.task_traces:
            icon = "✓" if tt.ok else "✗"
            parts.append(f"{tt.task_name} {icon}({len(tt.attempts)})")
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "tasks": [
                {
                    "task_name": tt.task_name,
                    "ok": tt.ok,
                    "final_value": tt.final_value,
                    "final_error": tt.final_error,
                    "attempts": [
                        {
                            "attempt_number": a.attempt_number,
                            "rendered_prompt": a.rendered_prompt,
                            "raw_response": a.raw_response,
                            "parse_error": a.parse_error,
                            "guard_errors": a.guard_errors,
                            "duration_ms": a.duration_ms,
                        }
                        for a in tt.attempts
                    ],
                }
                for tt in self.task_traces
            ],
        }

    def to_file(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def make_tracer(run_id: str) -> Tracer:
    return Tracer(run_id=run_id)


def load_trace(path: str) -> Tracer:
    """Load a trace from a JSON file for inspection or replay."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tracer = Tracer(run_id=data["run_id"], started_at=data.get("started_at", 0))

    for task_data in data.get("tasks", []):
        attempts = [
            Attempt(
                attempt_number=a["attempt_number"],
                rendered_prompt=a["rendered_prompt"],
                raw_response=a["raw_response"],
                parse_error=a.get("parse_error"),
                guard_errors=a.get("guard_errors", []),
                duration_ms=a.get("duration_ms", 0),
            )
            for a in task_data.get("attempts", [])
        ]
        tracer.task_traces.append(
            TaskTrace(
                task_name=task_data["task_name"],
                attempts=attempts,
                ok=task_data["ok"],
                final_value=task_data.get("final_value"),
                final_error=task_data.get("final_error"),
            )
        )

    return tracer
