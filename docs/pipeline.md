# pipeline — Orchestration Layer

The `pipeline` package contains three concerns: running a sequence of tasks, decomposing a
freeform prompt into that sequence, and routing the output of one task to determine what
runs next.

---

## Research Background

The Plan-then-Execute (P-t-E) pattern (arxiv:2509.08646) emerged as a robust architecture
for agentic workflows: a planner LLM produces a structured plan once, then an executor
handles individual steps. This separation means the expensive reasoning call happens once
and the per-step calls are cheap and focused.

For small models specifically, the planning step is critical. A 7B model asked to both plan
*and* execute in a single prompt will reliably lose track of the goal. Separating them into
two distinct model calls — with the plan result validated and stored — produces far more
consistent behaviour.

`llmalib` implements this via `Decomposer` (the planner) and `Pipeline` (the executor).

---

## `pipeline.py` — Sequential Runner

### Purpose

`run_pipeline` takes an ordered list of `Task` objects and a `Context`, executes them in
sequence, and returns the list of `Result` objects. It is a plain function, not a class.

### Requirements

- Accept `tasks: list[Task]`, `ctx: Context`, optional `router: Router | None`, and
  `debug: bool = False`.
- Execute tasks in order, passing `ctx` to each.
- After each task, call `update_context(ctx, task.name, result)` to make the result
  available to subsequent tasks.
- If a `router` is provided, call `router(result, remaining_tasks)` after each task to
  allow dynamic reordering or task skipping.
- If `debug=True`, pass each `Result` to `inspector.print_result()` immediately after
  the task completes.
- Never catch exceptions silently. A `ClientError` (network failure) propagates up.
  A `ValidationError` or guard failure is captured inside `Result(ok=False)` by the
  retry layer — the pipeline runner never sees it.
- Return `list[Result]` — all results including failed ones.

### Interface

```python
def run_pipeline(
    tasks: list[Task],
    ctx: Context,
    router: Router | None = None,
    debug: bool = False,
) -> list[Result]:
    """
    Execute tasks in order.
    Each task sees the accumulated results of all previous tasks in ctx.
    """
```

### Internal Flow per Task

```
1. context_window.trim(ctx, task)     → trim ctx.vars to fit task.token_budget
2. examples.select(task, ctx)         → pick up to N few-shot examples
3. render_prompt(task, ctx, examples) → Jinja2 render → messages list
4. run_with_retry(task, messages, config) → Result
5. update_context(ctx, task.name, result)
6. if router: adjust remaining task list
7. if debug: inspector.print_result(result)
```

### Design Notes

**Why sequential only?**
Parallel execution requires async or threading, both of which obscure errors and make
debugging harder. Local models are typically single-GPU anyway; concurrent calls queue
behind each other at the server. Sequential execution is faster to understand and easier
to trace.

Future versions can add `run_pipeline_parallel` for independent tasks (DAG-style), but
the default should always be sequential.

**Why return all results, including failures?**
A pipeline where task 3 fails but tasks 1 and 2 succeeded should give the caller access
to all three results. Raising an exception on first failure would throw away partial work.
The caller decides whether a failed task is fatal.

---

## `decomposer.py` — Prompt → Task List

### Purpose

`decompose` takes a freeform user prompt and a registry of available `Task` templates,
calls the LLM once, and returns an ordered list of `Task` instances configured for the
goal.

### Requirements

- Accept `prompt: str`, `task_registry: dict[str, Task]`, `config: ClientConfig`, and
  optional `max_tasks: int = 8`.
- Produce a structured decomposition request that instructs the model to select and order
  tasks from the registry.
- Parse the model's response into a `DecompositionPlan` Pydantic model.
- Apply a reflection retry (max 2) if the response cannot be parsed or references unknown
  task names.
- Return `list[Task]` — the ordered tasks to execute, with any per-task parameter overrides
  applied.
- Raise `DecompositionError` if decomposition fails after retries.

### Interface

```python
@dataclass(frozen=True)
class TaskSpec:
    task_name: str                    # Must match a key in task_registry
    parameter_overrides: dict = field(default_factory=dict)

@dataclass(frozen=True)
class DecompositionPlan:
    tasks: list[TaskSpec]
    reasoning: str                    # Model's explanation — logged but not used

def decompose(
    prompt: str,
    task_registry: dict[str, Task],
    config: ClientConfig,
    max_tasks: int = 8,
) -> list[Task]:
    """
    Ask the model to select and order tasks from the registry.
    Returns instantiated Task objects ready for run_pipeline().
    """
```

### Decomposition Prompt Design

The decomposition prompt is itself a template. It must:

1. List all available tasks with their names and one-line descriptions.
2. Instruct the model to respond with JSON only — a list of `{task_name, overrides}` objects.
3. Constrain `max_tasks` explicitly in the prompt to prevent the model from generating
   arbitrarily long plans.
4. Include one worked example of valid JSON output (few-shot).

Research finding: prompts that include few-shot demonstrations bootstrapped
from correct predictions, with instructions and demonstrations jointly optimized, consistently
outperform instruction-only prompts. The decomposition prompt ships with one
hardcoded example that demonstrates valid JSON decomposition output.

### Design Notes

**Why use a registry instead of free-form task generation?**
A model asked to invent arbitrary task definitions will hallucinate field names, schema
types, and parameter values. Constraining selection to a predefined registry makes the
decomposition output verifiable. The model picks from a known set; `llmalib` validates
the picks.

**Why only one LLM call for decomposition?**
The most expensive component — the large, reasoning-focused Planner LLM —
is invoked sparingly, perhaps only once at the beginning of a task, and occasionally for
re-planning if the workflow supports it. Decomposition is called once. If the
plan is wrong, the caller re-calls with a more specific prompt — there is no auto-replanning.

---

## `router.py` — Conditional Routing

### Purpose

A `Router` decides, after each task's `Result`, which task (if any) to run next.
The default router simply advances in order. Custom routers enable branching logic.

### Requirements

- A `Router` is a callable: `(result: Result, remaining: list[Task]) -> list[Task]`.
- Returning the unchanged `remaining` list means "continue as planned".
- Returning an empty list means "stop the pipeline here".
- Returning a modified list means "skip, reorder, or inject tasks".
- Provide two built-in router factories: `stop_on_failure()` and `branch_on_field()`.

### Interface

```python
# Type alias
Router = Callable[[Result, list[Task]], list[Task]]

def stop_on_failure() -> Router:
    """
    Returns a router that stops the pipeline (returns []) if result.ok is False.
    Default behaviour when no router is provided is to continue regardless.
    """

def branch_on_field(field: str, branches: dict[str, list[Task]]) -> Router:
    """
    Routes based on the value of a field in result.value.

    Example:
        branch_on_field("label", {
            "urgent": [escalation_task],
            "normal": [standard_reply_task],
        })
    """
```

### Design Notes

Routers are plain callables rather than a class hierarchy. Any function with the right
signature is a valid router. This means routing logic can be tested in isolation with
mock `Result` objects, without instantiating any pipeline infrastructure.

Complex routing (e.g., loop back to an earlier task on failure) is achieved by returning
a `remaining` list that includes tasks already run. The pipeline runner does not track
"visited" tasks — it simply executes whatever the router returns. The caller is responsible
for preventing infinite loops via `max_iterations` if needed.
