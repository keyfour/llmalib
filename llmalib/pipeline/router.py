"""
pipeline/router.py

Router factories for conditional pipeline flow.
A Router is just a callable: (Result, list[Task]) -> list[Task]

The routing *decisions* run in Rust (`llmalib._rust.route_*`); this module
keeps the task-list plumbing because the public router contract operates on
Python `Result`/`Task` objects (tests assert the returned lists are the
original objects, so re-marshalling tasks would break object identity).
"""

from __future__ import annotations

from typing import Callable

from llmalib import _rust
from llmalib.core.result import Result
from llmalib.core.task import Task

Router = Callable[[Result, list[Task]], list[Task]]


def stop_on_failure() -> Router:
    """
    Stop the pipeline (return empty list) when a task fails.
    Default pipeline behaviour is to continue regardless of failure.
    """

    def route(result: Result, remaining: list[Task]) -> list[Task]:
        if _rust.route_stop_on_failure(result.ok):
            return remaining
        return []

    return route


def branch_on_field(field: str, branches: dict[str, list[Task]]) -> Router:
    """
    Route to a different task list based on the value of a field in result.value.

    Example:
        branch_on_field("label", {
            "urgent":  [escalation_task],
            "normal":  [standard_task],
        })

    If the field value has no matching branch, the remaining list is unchanged.
    If result.ok is False, the remaining list is unchanged.
    """
    # Python-str keys decide branch membership Rust-side; the lookup stays on
    # the original dict (non-str branch keys never route, as before).
    keys = [str(k) for k in branches]

    def route(result: Result, remaining: list[Task]) -> list[Task]:
        key = _rust.route_branch_key(field, result.ok, result.value, keys)
        if key is None:
            return remaining
        branch_tasks = branches.get(key)
        if branch_tasks is not None:
            return list(branch_tasks)
        return remaining

    return route


def compose_routers(*routers: Router) -> Router:
    """
    Combine multiple routers. Each router sees the remaining list
    as modified by the previous router.
    """

    def route(result: Result, remaining: list[Task]) -> list[Task]:
        current = remaining
        for r in routers:
            current = r(result, current)
        return current

    return route
