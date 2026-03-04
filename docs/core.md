# core — Foundational Types and HTTP Client

The `core` package contains the four building blocks every other module depends on.
There are no operations in this package — only data shapes and one thin I/O function.

---

## Research Background

Small models are most reliable when each call is **narrow and schema-constrained**.
AutoPDL (arxiv:2504.04365) found accuracy gains of up to 67.5 percentage points on 3B–70B
models simply by selecting the right prompting pattern per model. The common thread across
all high-performing configurations is that the model is told exactly what shape to produce.

`core` enforces this by making the output schema a required field of every `Task`.
There is no way to create a task without declaring what you expect back.

---

## `client.py` — HTTP Wrapper

### Purpose

Send a rendered prompt string to any OpenAI-compatible local inference server and return
the raw response string. Nothing more.

### Requirements

- Accept a `base_url`, `model` name, rendered prompt string, and optional generation parameters.
- Return a plain `str` (the model's response content).
- Raise a typed `ClientError` on HTTP failure, timeout, or empty response — never swallow errors silently.
- Support configurable `timeout` (default: 120 s) because local models are slow.
- Be a plain function: `call(prompt, model, base_url, **kwargs) -> str`.
- Do not manage conversation history — callers own the message list.

### Interface

```python
@dataclass(frozen=True)
class ClientConfig:
    base_url: str = "http://localhost:11434/v1"
    model: str = "llama3.2"
    temperature: float = 0.1      # Low temperature for structured tasks
    timeout: float = 120.0
    max_tokens: int = 1024

def call(messages: list[dict], config: ClientConfig) -> str:
    """
    POST to /v1/chat/completions.
    Returns content of first choice message.
    Raises ClientError on any failure.
    """
```

### Design Notes

- Temperature defaults to `0.1`. Small models are more consistent at low temperature for
  structured output tasks. Raise it only for generative / creative tasks.
- The function accepts a `messages` list (not a single string) so that the retry loop in
  `reliability/retry.py` can append reflection turns without rebuilding from scratch.
- Connection pooling is handled by `httpx` automatically; do not add manual retry logic here.
  Retries belong in `reliability/retry.py` at the semantic level, not the network level.

---

## `task.py` — Task Definition

### Purpose

A `Task` is the atomic unit of work in `llmalib`. It describes **what to ask**, **what to
expect back**, and **how to handle failures**.

### Requirements

- Be a `@dataclass(frozen=True)` — tasks are immutable declarations, not runtime objects.
- Require `name`, `prompt_template` (Jinja2 string), and `output_schema` (a Pydantic `BaseModel` subclass).
- Have sensible defaults for all operational parameters so a task can be created with three fields.
- Support per-task model override so different tasks in a pipeline can use different models.
- Support an optional list of `Guard` instances for post-generation checks.
- Support an optional list of `Example` instances for few-shot injection.

### Interface

```python
@dataclass(frozen=True)
class Task:
    # Required
    name: str
    prompt_template: str           # Jinja2 template; vars filled from Context
    output_schema: type[BaseModel] # Defines expected response shape

    # Model config (optional overrides)
    model: str = "llama3.2"
    base_url: str = "http://localhost:11434/v1"
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout: float = 120.0

    # Reliability
    max_retries: int = 3
    guards: tuple[Guard, ...] = ()

    # Few-shot examples (injected into system prompt)
    examples: tuple[Example, ...] = ()

    # Token budget for the full rendered prompt (system + history + user)
    token_budget: int = 2048
```

### Design Notes

**Why `frozen=True`?**
Tasks are defined once and reused across runs. Mutating a task during a pipeline run would
be a source of subtle bugs. Immutability forces all state into `Context`.

**Why Jinja2 templates?**
Templates are readable, debuggable, and familiar. The rendered prompt is always inspectable
in the `Trace` before it is sent to the model. String formatting (`f""`) would not provide
this separation.

**Token budget**
Research on context degradation (arxiv:2509.21361; arxiv:2508.21433) shows that performance
drops measurably beyond the model's effective context window, which for most 7B models is
2000–4000 tokens in practice even when the technical limit is higher. Setting an explicit
budget per task is the primary defence against silent quality degradation.

---

## `result.py` — Result Envelope

### Purpose

Every task execution produces a `Result`. It is the single type that flows through a pipeline.
It carries either a successful parsed value or a structured error — never raises an exception
on normal failure.

### Requirements

- Be a `@dataclass` (not frozen — the pipeline runner may annotate it).
- Carry the task name, success flag, parsed value or error string, attempt count, and full trace.
- The `value` field is typed as `BaseModel | None`; `error` is `str | None`.
- Never raise. A failed parse or guard check produces `Result(ok=False, error="...")`, not an exception.
- Include a `trace: list[Attempt]` so every prompt/response pair is preserved.

### Interface

```python
@dataclass
class Attempt:
    attempt_number: int
    rendered_prompt: str       # Exact string sent to the model
    raw_response: str          # Exact string returned by the model
    parse_error: str | None    # Pydantic validation error, if any
    guard_errors: list[str]    # Guard check failures, if any
    duration_ms: float

@dataclass
class Result:
    task_name: str
    ok: bool
    value: BaseModel | None
    error: str | None          # Human-readable summary of final failure
    attempts: int
    trace: list[Attempt]
```

### Design Notes

Making `Result` a non-raising envelope is the most important reliability decision in the
library. It means `pipeline.run()` never crashes on a bad model response — it returns a
`Result(ok=False)` that the caller can inspect, log, or route around. This makes pipelines
composable without wrapping every call in `try/except`.

---

## `context.py` — Pipeline Context

### Purpose

`Context` is the shared mutable state passed through an entire pipeline run. It holds input
variables, accumulated results, and metadata about the current run.

### Requirements

- Be a regular `@dataclass` (mutable — the pipeline writes results into it).
- Hold `vars: dict[str, Any]` — the template variables available to all tasks.
- Hold `results: dict[str, Result]` — keyed by `task.name`, populated as tasks complete.
- Hold `run_id: str` — a UUID generated at creation time for trace correlation.
- Provide a `get(task_name) -> Result | None` helper.
- Provide an `update(task_name, result)` helper that also copies `result.value` fields
  into `vars` so downstream tasks can reference upstream outputs in their templates.

### Interface

```python
@dataclass
class Context:
    vars: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Result] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: str(uuid4()))

def make_context(**vars) -> Context:
    """Convenience constructor: llmalib.make_context(text="hello", ...)"""

def update_context(ctx: Context, task_name: str, result: Result) -> Context:
    """
    Adds result to ctx.results.
    If result.ok, merges result.value model fields into ctx.vars
    so downstream templates can reference {{ field_name }} directly.
    Returns the same ctx (mutated in place for simplicity).
    """
```

### Design Notes

**Merging result fields into `vars`**
This is what makes chained tasks ergonomic. If `task_a` produces
`SummaryResult(summary="...", keywords=[...])`, the next task's template can use
`{{ summary }}` and `{{ keywords }}` directly without explicit plumbing.

The convention is: field names from the Pydantic model are merged as-is.
Name collisions between tasks are resolved by last-write-wins — pipeline authors
should use unique field names or namespace them (`task_a_summary`, etc.).

**Why not immutable context?**
Immutable context would require returning a new `Context` from every task, making the
pipeline runner more complex for marginal benefit. The context is local to one `run_pipeline()`
call; it is never shared across threads or concurrent runs.
