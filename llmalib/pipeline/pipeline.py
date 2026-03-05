"""
pipeline/pipeline.py

Sequential pipeline runner.
Executes tasks in order, threading Context through each step.
Supports optional Router for conditional branching.
"""

from __future__ import annotations

import json
from typing import Callable

from jinja2 import BaseLoader, Environment, TemplateError

from llmalib.core.client import ClientConfig, config_from_task
from llmalib.core.context import Context, update_context
from llmalib.core.result import Result
from llmalib.core.task import Task
from llmalib.debug.inspector import print_result
from llmalib.debug.tracer import Tracer, make_tracer
from llmalib.memory.context_window import trim_to_budget
from llmalib.memory.examples import format_examples_block, select_examples
from llmalib.memory.store import Store
from llmalib.reliability.retry import run_with_retry
from llmalib.reliability.validator import format_schema_hint

# Router: callable that may modify the remaining task list after each result
Router = Callable[[Result, list[Task]], list[Task]]

_jinja_env = Environment(loader=BaseLoader())


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

    results: list[Result] = []
    remaining = list(tasks)

    while remaining:
        task = remaining.pop(0)
        result = _run_task(task, ctx, store)

        update_context(ctx, task.name, result)
        tracer.record(result)
        results.append(result)

        if debug:
            print_result(result)

        if router is not None:
            remaining = router(result, remaining)

    return results


def _run_task(task: Task, ctx: Context, store: Store | None) -> Result:
    """Build messages for a task and execute with retry."""
    # 1. Select few-shot examples
    examples = select_examples(task, ctx.vars, store)
    examples_block = format_examples_block(examples)

    # 2. Build system prompt
    system_content = _build_system_prompt(task, examples_block)

    # 3. Render user prompt from Jinja2 template
    try:
        user_content = _render_template(task.prompt_template, ctx.vars)
    except TemplateError as exc:
        from llmalib.core.result import make_error_result

        return make_error_result(
            task.name,
            f"Template render error: {exc}",
            [],
        )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    # 4. Trim to token budget
    messages = trim_to_budget(messages, task.token_budget)

    # 5. Execute with reflection retry
    config = config_from_task(task)
    return run_with_retry(task, messages, config, ctx)


def _build_system_prompt(task: Task, examples_block: str) -> str:
    if task.system_prompt:
        base = task.system_prompt
    else:
        schema_hint = format_schema_hint(task.output_schema)
        base = (
            f"You are a precise assistant. "
            f"Respond ONLY with valid JSON matching this schema exactly:\n{schema_hint}\n"
            f"No prose, no code fences, no explanation — only the JSON object."
        )

    if examples_block:
        return base + "\n" + examples_block

    return base


def _render_template(template_str: str, vars: dict) -> str:
    try:
        tmpl = _jinja_env.from_string(template_str)
        return tmpl.render(**vars)
    except Exception as exc:
        raise TemplateError(str(exc))
