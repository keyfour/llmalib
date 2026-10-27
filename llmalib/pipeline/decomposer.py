"""
pipeline/decomposer.py

Convert a freeform user prompt into an ordered list of Tasks
by asking the model once to select and order from a registry.
Engine in `llmalib._rust.decompose` (planning schema, reflection retries,
parameter overrides); the Python layer marshals the registry and rehydrates
the output into Task instances.

TODO: Add support for using a larger/smarter model for the planner
      while tasks run on a smaller local model (two-model P-t-E pattern).
TODO: Add re-planning support — call decompose() again if a task fails
      and the pipeline needs to adapt.
"""

from __future__ import annotations

from llmalib import _rust
from llmalib._bridge import spec_from_task, task_from_spec
from llmalib.core.client import ClientConfig, ClientError
from llmalib.core.task import Task


class DecompositionError(Exception):
    """Raised when the model cannot produce a valid decomposition plan."""


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
    config_dict = {
        "base_url": config.base_url,
        "model": config.model,
        "temperature": config.temperature,
        "timeout": config.timeout,
        "max_tokens": config.max_tokens,
    }
    registry_dicts = {
        name: spec_from_task(task) for name, task in task_registry.items()
    }
    try:
        specs = _rust.decompose(prompt, registry_dicts, config_dict, max_tasks)
    except _rust.DecompositionError as exc:
        raise DecompositionError(str(exc)) from None
    except _rust.ClientError as exc:
        raise ClientError(str(exc)) from None
    return [task_from_spec(s) for s in specs]
