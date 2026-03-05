"""
pipeline/router.py

Router factories for conditional pipeline flow.
A Router is just a callable: (Result, list[Task]) -> list[Task]
"""

from __future__ import annotations

from typing import Callable
from llmalib.core.result import Result
from llmalib.core.task import Task

Router = Callable[[Result, list[Task]], list[Task]]


def stop_on_failure() -> Router:
    """
    Stop the pipeline (return empty list) when a task fails.
    Default pipeline behaviour is to continue regardless of failure.
    """

    def route(result: Result, remaining: list[Task]) -> list[Task]:
        if not result.ok:
            return []
        return remaining

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

    def route(result: Result, remaining: list[Task]) -> list[Task]:
        if not result.ok or result.value is None:
            return remaining
        value = getattr(result.value, field, None)
        if value is None:
            return remaining
        branch_tasks = branches.get(str(value))
        if branch_tasks is not None:
            # Replace remaining with the branch tasks
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
