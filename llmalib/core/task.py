"""
core/task.py

Task is the atomic unit of work: what to ask, what to expect back,
and how to handle failures. Immutable — tasks are declarations, not state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel

if TYPE_CHECKING:
    # Avoid circular imports — Guard and Example are defined in reliability/guards.py
    # and memory/examples.py respectively.
    pass

# Guard: callable that checks a parsed value against the run context.
# Returns a list of error strings (empty = passed).
Guard = Callable[[BaseModel, Any], list[str]]  # (value, ctx) -> [errors]


@dataclass(frozen=True)
class Example:
    """A single few-shot example attached to a task or stored in the example store."""

    input_text: str
    output: BaseModel  # Must match the task's output_schema type
    description: str = ""  # Optional label for tracing / debugging


@dataclass(frozen=True)
class Task:
    """
    Immutable declaration of a single unit of LLM work.

    prompt_template is a Jinja2 template string.
    Variables are filled from Context.vars at render time.
    output_schema is a Pydantic BaseModel subclass that defines
    the exact shape the model must produce.
    """

    # --- Required ---
    name: str
    prompt_template: str
    output_schema: type[BaseModel]

    # --- Model config ---
    model: str = "llama3.2"
    base_url: str = "http://localhost:11434/v1"
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout: float = 120.0

    # --- Reliability ---
    max_retries: int = 3
    guards: tuple[Guard, ...] = field(default_factory=tuple)

    # --- Few-shot ---
    examples: tuple[Example, ...] = field(default_factory=tuple)

    # --- Context budget ---
    # Conservative default: most 7B models degrade beyond 2048 tokens in practice
    token_budget: int = 2048

    # --- Optional system prompt prefix ---
    # If None, a default system prompt is generated from the schema hint
    system_prompt: str | None = None
