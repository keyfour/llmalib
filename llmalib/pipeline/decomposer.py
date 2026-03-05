"""
pipeline/decomposer.py

Convert a freeform user prompt into an ordered list of Tasks
by asking the model once to select and order from a registry.

TODO: Add support for using a larger/smarter model for the planner
      while tasks run on a smaller local model (two-model P-t-E pattern).
TODO: Add re-planning support — call decompose() again if a task fails
      and the pipeline needs to adapt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from pydantic import BaseModel

from llmalib.core.client import ClientConfig, call
from llmalib.core.task import Task
from llmalib.reliability.validator import parse_response


class DecompositionError(Exception):
    """Raised when the model cannot produce a valid decomposition plan."""


# ---------------------------------------------------------------------------
# Schema for the model's response
# ---------------------------------------------------------------------------


class _TaskSpec(BaseModel):
    task_name: str
    parameter_overrides: dict = {}


class _DecompositionPlan(BaseModel):
    tasks: list[_TaskSpec]
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decompose(
    prompt: str,
    task_registry: dict[str, Task],
    config: ClientConfig,
    max_tasks: int = 8,
) -> list[Task]:
    """
    Ask the model to select and order tasks from task_registry for the given prompt.
    Returns an ordered list of Task instances ready for run_pipeline().

    Applies up to 2 reflection retries if the response is invalid.
    Raises DecompositionError after exhausting retries.
    """
    if not task_registry:
        raise DecompositionError("task_registry is empty — no tasks to decompose into.")

    system_prompt = _build_decomposition_system_prompt(task_registry, max_tasks)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Goal: {prompt}"},
    ]

    last_error: str = "Unknown error"
    for attempt in range(1, 3):  # Max 2 attempts for decomposition
        raw = call(messages, config)
        parse_result = parse_response(raw, _DecompositionPlan)

        if not parse_result.ok:
            last_error = parse_result.error or "parse failed"
            messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"Invalid response. Error: {last_error}\n"
                        f"Respond only with JSON matching the schema."
                    ),
                },
            ]
            continue

        plan: _DecompositionPlan = parse_result.value
        tasks, validation_error = _validate_plan(plan, task_registry)

        if validation_error:
            last_error = validation_error
            messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"Invalid plan. Error: {last_error}\n"
                        f"Only use task names from the registry: "
                        f"{list(task_registry.keys())}"
                    ),
                },
            ]
            continue

        return tasks

    raise DecompositionError(
        f"Decomposition failed after 2 attempts. Last error: {last_error}"
    )


def _validate_plan(
    plan: _DecompositionPlan,
    task_registry: dict[str, Task],
) -> tuple[list[Task], str | None]:
    """
    Validate that all task names in the plan exist in the registry.
    Apply any parameter overrides.
    Returns (tasks, None) on success or ([], error_message) on failure.
    """
    tasks: list[Task] = []
    unknown = []

    for spec in plan.tasks:
        if spec.task_name not in task_registry:
            unknown.append(spec.task_name)
            continue

        base_task = task_registry[spec.task_name]

        if spec.parameter_overrides:
            # Apply overrides to allowed fields only
            allowed = {
                "model",
                "max_retries",
                "temperature",
                "token_budget",
                "max_tokens",
            }
            safe_overrides = {
                k: v for k, v in spec.parameter_overrides.items() if k in allowed
            }
            if safe_overrides:
                base_task = _apply_overrides(base_task, safe_overrides)

        tasks.append(base_task)

    if unknown:
        valid = list(task_registry.keys())
        return [], f"Unknown task names: {unknown}. Valid names: {valid}"

    if not tasks:
        return [], "Plan contained no tasks."

    return tasks, None


def _apply_overrides(task: Task, overrides: dict) -> Task:
    """Return a new Task with the given field overrides applied."""
    import dataclasses

    return dataclasses.replace(task, **overrides)


def _build_decomposition_system_prompt(
    task_registry: dict[str, Task],
    max_tasks: int,
) -> str:
    task_list = "\n".join(
        f'  - "{name}": {_task_description(t)}' for name, t in task_registry.items()
    )

    example_output = json.dumps(
        {
            "tasks": [
                {"task_name": list(task_registry.keys())[0], "parameter_overrides": {}},
            ],
            "reasoning": "Start with the first step because...",
        },
        indent=2,
    )

    return f"""You are a task planner. Given a goal, select and order tasks from the registry below.

Available tasks:
{task_list}

Rules:
1. Select only task names from the registry above (exact spelling).
2. Order tasks logically to achieve the goal.
3. Use at most {max_tasks} tasks.
4. Respond ONLY with valid JSON — no prose, no fences.

Schema:
{{"tasks": [{{"task_name": "<str>", "parameter_overrides": {{}}}}], "reasoning": "<str>"}}

Example:
{example_output}"""


def _task_description(task: Task) -> str:
    """Build a one-line description of a task for the decomposition prompt."""
    schema_fields = list(task.output_schema.model_fields.keys())
    return f"produces {{{', '.join(schema_fields)}}}"
