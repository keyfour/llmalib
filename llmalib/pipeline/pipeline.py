"""
pipeline/pipeline.py

Sequential pipeline runner (engine in `llmalib._rust.run_pipeline`).
Executes tasks in order, threading Context through each step.
Supports optional Router for conditional branching.

The Python layer marshals tasks/context into the Rust engine and rehydrates
result values into pydantic instances afterwards; context mutation
(vars/results) is written back onto the SAME Context object the caller
passed in, exactly like the original implementation.
"""

from __future__ import annotations

from typing import Callable

from llmalib import _rust
from llmalib._bridge import (
    adapt_router,
    json_schema_to_model,
    result_from_dict,
    spec_from_task,
)
from llmalib.core.client import ClientError
from llmalib.core.context import Context
from llmalib.core.result import Result
from llmalib.core.task import Task
from llmalib.debug.inspector import print_result
from llmalib.debug.tracer import Tracer, make_tracer
from llmalib.memory.store import Store

# Router: callable that may modify the remaining task list after each result
Router = Callable[[Result, list[Task]], list[Task]]

# Defensive fallback for results whose task is not in the pool.
_EMPTY_SCHEMA = json_schema_to_model({"type": "object", "properties": {}})


def run_pipeline(
    tasks: list[Task],
    ctx: Context,
    router: Router | None = None,
    store: Store | None = None,
    tracer: Tracer | None = None,
    debug: bool = False,
) -> list[Result]:
    """
    Execute tasks sequentially.

    Each task sees all results of previous tasks via ctx.
    Upstream result fields are auto-merged into ctx.vars for template access.

    Returns list[Result] — all results including failures.
    Raises ClientError on unrecoverable network failures.
    """
    if tracer is None:
        tracer = make_tracer(ctx.run_id)

    try:
        out = _rust.run_pipeline(
            [spec_from_task(t) for t in tasks],
            vars=dict(ctx.vars),
            run_id=ctx.run_id,
            router=adapt_router(router, tasks) if router is not None else None,
            store=store._inner if store is not None else None,
            debug=False,
        )
    except _rust.ClientError as exc:
        raise ClientError(str(exc)) from None
    except _rust.BudgetExceededError:
        raise

    if tracer.run_id != out["run_id"]:
        tracer.run_id = out["run_id"]
    ctx.vars = dict(out["vars"])

    schema_by_name = {t.name: t.output_schema for t in tasks}
    results = [
        result_from_dict(d, schema_by_name.get(d["task_name"], _EMPTY_SCHEMA))
        for d in out["results"]
    ]
    for result in results:
        ctx.results[result.task_name] = result
        tracer.record(result)
        if debug:
            print_result(result)

    return results
