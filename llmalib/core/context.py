"""
core/context.py

Shared mutable state for a single pipeline run.
Upstream task results are merged into vars so downstream templates
can reference them directly with {{ field_name }}.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from llmalib.core.result import Result


@dataclass
class Context:
    vars: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Result] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


def make_context(**vars: Any) -> Context:
    """Convenience constructor: make_context(text='hello', lang='en')"""
    return Context(vars=dict(vars))


def update_context(ctx: Context, task_name: str, result: Result) -> Context:
    """
    Record result and, if successful, merge its value fields into ctx.vars
    so downstream templates can reference them directly.

    Mutates ctx in place and returns it for chaining.
    Field name collisions are resolved last-write-wins — use unique field
    names or namespace them (e.g. task_a_summary) to avoid surprises.
    """
    ctx.results[task_name] = result

    if result.ok and result.value is not None:
        # Merge Pydantic model fields into vars
        ctx.vars.update(result.value.model_dump())

    return ctx


def get_result(ctx: Context, task_name: str) -> Result | None:
    return ctx.results.get(task_name)
