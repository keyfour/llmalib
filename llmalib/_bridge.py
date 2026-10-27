"""
llmalib/_bridge.py

Private marshalling layer between the Python package and `llmalib._rust`.

Everything here exists to keep the public API byte-for-byte identical
(dataclasses, pydantic schemas, guard callables on Python values) while the
engine runs Rust-side:
- tasks marshal to flat spec dicts (see crates/llmalib-py/src/convert.rs),
- pydantic schemas marshal via `model_json_schema()` and result values are
  re-hydrated into pydantic instances with the same class,
- Python guard callables are wrapped so they see pydantic values again
  (the Rust bridge passes plain dicts); wrappers are marked with
  `_llmalib_original` so round-trips never double-wrap,
- routers are wrapped the same way (result dicts re-hydrated into Result/
  Task objects and results marshalled back to specs).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional

from pydantic import BaseModel, create_model

from llmalib._rust import ContextView
from llmalib.core.client import ClientConfig
from llmalib.core.context import Context
from llmalib.core.result import Attempt, Result
from llmalib.core.task import Example, Task

# ---------------------------------------------------------------------------
# Task → spec dict
# ---------------------------------------------------------------------------


def schema_to_json(schema: type[BaseModel]) -> dict[str, Any]:
    """Pydantic model class → JSON Schema dict for the Rust engine."""
    return schema.model_json_schema()


def spec_from_task(task: Task) -> dict[str, Any]:
    """Marshal a Python Task into the flat spec dict `_rust` understands."""
    return {
        "name": task.name,
        "prompt_template": task.prompt_template,
        "output_schema": schema_to_json(task.output_schema),
        "model": task.model,
        "base_url": task.base_url,
        "temperature": task.temperature,
        "max_tokens": task.max_tokens,
        "timeout": task.timeout,
        "max_retries": task.max_retries,
        "token_budget": task.token_budget,
        "system_prompt": task.system_prompt,
        "guards": [adapt_guard(g, task.output_schema) for g in task.guards],
        "examples": [example_to_dict(ex) for ex in task.examples],
    }


def example_to_dict(ex: Example) -> dict[str, Any]:
    output = ex.output
    # output is normally a pydantic instance; tolerate plain values (when a
    # stored Example was reshaped without schema information).
    dumped = output.model_dump() if hasattr(output, "model_dump") else output
    return {
        "input_text": ex.input_text,
        "output": dumped,
        "description": ex.description,
    }


def adapt_guard(
    guard: Callable, schema: type[BaseModel]
) -> Callable[[dict[str, Any], ContextView], list[str]]:
    """
    Wrap a Python guard so the Rust bridge (which hands over a plain dict)
    sees a pydantic instance again, exactly like the original signature
    `(value: BaseModel, ctx) -> list[str]`.

    Already-adapted guards pass through unchanged — important because task
    specs round-trip back from Rust (routers, decompose) and must not be
    wrapped twice.
    """
    if getattr(guard, "_llmalib_original", None) is not None:
        return guard

    def adapted(
        value_dict: dict[str, Any],
        ctx: Any,
        _guard: Callable = guard,
        _schema: type[BaseModel] = schema,
    ) -> list[str]:
        return _guard(hydrate_value(value_dict, _schema), ctx)

    adapted._llmalib_original = guard  # type: ignore[attr-defined]
    return adapted


# ---------------------------------------------------------------------------
# spec dict → Task
# ---------------------------------------------------------------------------

# JSON Schema type → annotation (used for tasks reconstructed from specs —
# branch tasks in routers, decompose outputs). Constraints beyond type are
# not reconstructed; see .plan.md notes.
_JSON_TYPES: dict[str, Any] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "array": list[Any],
    "object": dict[str, Any],
    "null": type(None),
}


def _annotation_from_json_schema(prop: dict[str, Any]) -> Any:
    types: Any = prop.get("type")
    if isinstance(types, list):
        # Bearer of the first non-null member, py39-compatible union
        non_null = [t for t in types if t != "null"]
        if non_null:
            return _annotation_from_json_schema({"type": non_null[0]})
        return Optional[str]
    if types in _JSON_TYPES:
        return _JSON_TYPES[types]
    return Any


def json_schema_to_model(schema: dict[str, Any]) -> type[BaseModel]:
    """
    Rebuild a minimal pydantic model class from a JSON Schema dict
    (fields + required only). Used when a Task must be reconstructed from a
    spec dict and the original pydantic class is unavailable.
    """
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])

    fields_spec: dict[str, Any] = {}
    for name, prop in props.items():
        annotation = _annotation_from_json_schema(prop)
        if name in required:
            fields_spec[name] = (annotation, ...)
        else:
            fields_spec[name] = (Optional[annotation], None)
    return create_model(schema.get("title") or "RebuiltSchema", **fields_spec)


def task_from_spec(spec: dict[str, Any]) -> Task:
    """Reconstruct a Task from a spec dict returned by `_rust`."""
    schema = json_schema_to_model(spec["output_schema"])
    guards = tuple(
        guard._llmalib_original  # type: ignore[attr-defined]
        if getattr(guard, "_llmalib_original", None) is not None
        else guard
        for guard in (spec.get("guards") or [])
    )
    examples = tuple(example_from_dict(ex, schema) for ex in spec.get("examples") or [])
    return Task(
        name=spec["name"],
        prompt_template=spec["prompt_template"],
        output_schema=schema,
        model=spec["model"],
        base_url=spec["base_url"],
        temperature=spec["temperature"],
        max_tokens=spec["max_tokens"],
        timeout=spec["timeout"],
        max_retries=spec["max_retries"],
        token_budget=spec["token_budget"],
        system_prompt=spec["system_prompt"],
        guards=guards,
        examples=examples,
    )


def example_from_dict(d: dict[str, Any], schema: type[BaseModel]) -> Example:
    return Example(
        input_text=d["input_text"],
        output=hydrate_value(d.get("output", {}), schema),
        description=d.get("description", ""),
    )


def value_to_dict(value: Any) -> Any:
    """Best-effort JSON-friendly conversion for user-supplied values."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, Example):
        return example_to_dict(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return value


# ---------------------------------------------------------------------------
# Results / attempts
# ---------------------------------------------------------------------------


def hydrate_value(
    value: Any, schema: type[BaseModel]
) -> BaseModel | None:
    """Re-hydrate a parsed dict back into the task's pydantic model."""
    if value is None:
        return None
    return schema.model_validate(value)


def attempt_from_dict(d: dict[str, Any]) -> Attempt:
    return Attempt(
        attempt_number=d["attempt_number"],
        rendered_prompt=d["rendered_prompt"],
        raw_response=d["raw_response"],
        parse_error=d["parse_error"],
        guard_errors=list(d["guard_errors"]),
        duration_ms=d["duration_ms"],
    )


def result_from_dict(d: dict[str, Any], schema: type[BaseModel]) -> Result:
    return Result(
        task_name=d["task_name"],
        ok=d["ok"],
        value=hydrate_value(d.get("value"), schema) if d["ok"] else None,
        error=d.get("error"),
        attempts=d["attempts"],
        trace=[attempt_from_dict(a) for a in d.get("trace") or []],
    )


# ---------------------------------------------------------------------------
# Context / config
# ---------------------------------------------------------------------------


def ctx_to_dict(ctx: Context) -> dict[str, Any]:
    return {"vars": dict(ctx.vars), "run_id": ctx.run_id}


def config_to_dict(config: ClientConfig) -> dict[str, Any]:
    return {
        "base_url": config.base_url,
        "model": config.model,
        "temperature": config.temperature,
        "timeout": config.timeout,
        "max_tokens": config.max_tokens,
    }


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------


def adapt_router(
    router: Callable, task_pool: list[Task]
) -> Callable[[dict[str, Any], list[dict[str, Any]]], list[dict[str, Any]]]:
    """
    Wrap a Python router `(Result, list[Task]) -> list[Task]` for the Rust
    bridge, which hands over `(result_dict, remaining_specs)`.

    `remaining` specs are matched back to the original Task objects by name;
    foreign tasks (router branches) are rebuilt with `task_from_spec`.
    """

    def adapted(
        result_dict: dict[str, Any],
        remaining_specs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        schema = _pool_schema(result_dict, task_pool)
        result = result_from_dict(result_dict, schema)
        remaining = _tasks_from_specs(remaining_specs, task_pool)
        new_tasks = router(result, remaining)
        return [spec_from_task(t) for t in new_tasks]

    return adapted


def _pool_schema(
    result_dict: dict[str, Any], task_pool: list[Task]
) -> type[BaseModel]:
    name = result_dict.get("task_name")
    for t in task_pool:
        if t.name == name:
            return t.output_schema
    # Unknown producing task (defensive): an empty schema hydrates nothing.
    return json_schema_to_model({"type": "object", "properties": {}})


def _tasks_from_specs(
    specs: list[dict[str, Any]], task_pool: list[Task]
) -> list[Task]:
    by_name: dict[str, Task] = {}
    for t in task_pool:
        by_name.setdefault(t.name, t)
    out: list[Task] = []
    for spec in specs:
        match = by_name.get(spec["name"])
        out.append(match if match is not None else task_from_spec(spec))
    return out
