# debug — Observability

The `debug` package makes every aspect of a pipeline run inspectable without modifying
any application code. It records all LLM calls, renders them for human reading, and
allows failed runs to be replayed against different models or prompts.

---

## Design Philosophy

The most common cause of inexplicable behaviour in LLM pipelines is not bugs in application
code — it is unexpected model output at some intermediate step that silently propagates as
bad data into subsequent steps. Debugging this requires seeing the exact prompt that was
sent and the exact response that was received, for every attempt, for every task.

`llmalib` records this information unconditionally. The `Tracer` is always active; there
is no "production mode" that disables it. The overhead is negligible (a few string copies).
The debugging value is enormous.

---

## `tracer.py` — Execution Recorder

### Purpose

`Tracer` records every LLM call made during a pipeline run. It is passed into the pipeline
runner and updated automatically — application code does not call it directly.

### Requirements

- Be a `@dataclass` that accumulates `TaskTrace` records.
- A `TaskTrace` contains the task name, all `Attempt` records (from `result.trace`), and
  the final outcome.
- Support `to_dict()` for JSON serialisation (for export and replay).
- Support `to_file(path)` to write the full trace as a JSON file.
- Be initialised with the run's `Context.run_id` for correlation.
- Provide `summary() -> str` — a compact one-line-per-task summary of the run.

### Interface

```python
@dataclass
class TaskTrace:
    task_name: str
    attempts: list[Attempt]          # From Result.trace
    ok: bool
    final_value: dict | None         # result.value.model_dump() if ok
    final_error: str | None

@dataclass
class Tracer:
    run_id: str
    task_traces: list[TaskTrace] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def record(self, result: Result) -> None:
        """Append a TaskTrace from a completed Result."""

    def summary(self) -> str:
        """
        Returns a compact summary, e.g.:
        run=abc123 | 3 tasks | classify_intent ✓ (1 attempt) | extract_facts ✓ (2 attempts) | synthesize ✗
        """

    def to_dict(self) -> dict:
        """Full JSON-serialisable representation of the trace."""

    def to_file(self, path: str) -> None:
        """Write trace as JSON to path."""

def load_trace(path: str) -> Tracer:
    """Load a trace from a JSON file for inspection or replay."""
```

### What Is Recorded Per Attempt

Every `Attempt` (defined in `core/result.py`) contains:

| Field | Contents |
|---|---|
| `attempt_number` | 1-indexed attempt counter |
| `rendered_prompt` | The exact messages list, serialised as a string |
| `raw_response` | The exact string returned by the model |
| `parse_error` | Pydantic validation error message, or `None` |
| `guard_errors` | List of guard failure messages, or `[]` |
| `duration_ms` | Wall-clock time from request to response |

This is sufficient to reconstruct exactly what happened at every step without re-running
the pipeline.

---

## `inspector.py` — Console Output

### Purpose

`inspector` renders pipeline results and traces to the terminal in a human-readable format
using `rich` (with graceful fallback to plain text if `rich` is not installed).

### Requirements

- Provide `print_result(result: Result)` — prints one task's outcome immediately after
  it completes (used by `run_pipeline(..., debug=True)`).
- Provide `print_trace(tracer: Tracer)` — prints a full run summary after all tasks complete.
- Provide `print_attempt(attempt: Attempt)` — prints a single attempt's prompt and response
  for deep inspection.
- Use colour coding: green for success, red for failure, yellow for retry.
- All output goes to `stderr` to avoid polluting `stdout` in pipeline scripts.
- Gracefully degrade to plain text if `rich` is not available.

### Output Format

**`print_result` (per-task, shown live during debug run):**
```
✓  classify_intent   [attempt 1/3]  142ms
   label="urgent"  confidence=0.91  reason="..."

✗  extract_facts     [attempt 3/3]  FAILED
   Error: Missing required field 'source_paragraph'
```

**`print_trace` (end-of-run summary):**
```
────────────────────────────────────────────────────
 Pipeline Run: run_id=abc123
 Duration: 4.2s  |  Tasks: 3  |  Success: 2  |  Failed: 1
────────────────────────────────────────────────────
 Task 1: classify_intent     ✓  1 attempt   142ms
 Task 2: extract_facts       ✗  3 attempts  1840ms
   └─ Attempt 1: parse error — missing field 'source_paragraph'
   └─ Attempt 2: guard error — answer not grounded in input
   └─ Attempt 3: parse error — response was not valid JSON
 Task 3: synthesize          ✓  1 attempt   890ms
────────────────────────────────────────────────────
```

**`print_attempt` (deep inspection on demand):**
```
─── Attempt 2 ──────────────────────────────────────
PROMPT (messages):
  [system] You are a fact extraction assistant...
           Expected output: {"source_paragraph": "string", ...}
  [user]   Extract the key fact from: ...
  [assistant] {"source_paragraph": null}
  [user]   Your previous response was invalid. Errors:
           - 'source_paragraph' must not be null

RESPONSE:
  {"source_paragraph": "The study found that...", ...}

PARSE: ✓
GUARDS: ✓
────────────────────────────────────────────────────
```

### Design Notes

**Why `stderr`?**
Debug output to `stdout` contaminates scripts that pipe pipeline output to downstream
processes. `stderr` is the conventional channel for operational output.

**Why `rich` as optional?**
`rich` is an excellent library but it is a dependency. If a project already has `rich`,
the output is beautiful. If not, the output is still useful plain text. The fallback
is automatic — no configuration required.

---

## `replay.py` — Trace Replay

### Purpose

`replay` re-runs a specific task from a saved trace, optionally with a different model,
prompt, or schema. This is the primary tool for iterating on prompt and schema design.

### Requirements

- Accept a `Tracer` (loaded from file) and a `task_name`.
- Re-run only that task using the recorded prompt from a specified attempt.
- Accept optional overrides: `model`, `prompt_override`, `schema_override`.
- Return a new `Result` (not a `Tracer` — replay is a single task operation).
- Print a side-by-side diff of the original and new responses if `diff=True`.

### Interface

```python
def replay_task(
    tracer: Tracer,
    task_name: str,
    attempt_number: int = 1,         # Which attempt's prompt to replay
    model: str | None = None,        # Override model
    base_url: str | None = None,
    prompt_override: str | None = None,  # Replace the final user turn
    schema_override: type[BaseModel] | None = None,
    diff: bool = True,
) -> Result:
    """
    Re-run a recorded task attempt, optionally with overrides.
    Useful for testing prompt changes and model swaps without re-running
    the full pipeline.
    """
```

### Typical Workflow

```python
# 1. Run the pipeline, save the trace
tracer = Tracer(run_id=ctx.run_id)
results = run_pipeline(tasks, ctx, tracer=tracer)
tracer.to_file("run_2024_03_04.json")

# 2. A task failed. Inspect it.
inspector.print_attempt(tracer.task_traces[1].attempts[2])

# 3. Try a better prompt for the failing task
tracer = load_trace("run_2024_03_04.json")
result = replay_task(
    tracer,
    task_name="extract_facts",
    attempt_number=3,
    prompt_override="Extract the main claim and supporting evidence. "
                    "The source must be a direct quote from the input text.",
    diff=True,
)
# → Shows diff between original response and new response
```

### Design Notes

**Why replay from a specific attempt, not always the last?**
Sometimes the first attempt's prompt is cleaner than a later one (which has been
modified by reflection messages). Replaying from attempt 1 lets you test a new
base prompt from a clean state.

**Diff output format**
```
── Original (attempt 3) ─────────────────────────────
{"source_paragraph": "In 2023, the report stated..."}

── Replay (new prompt) ──────────────────────────────
{"source_paragraph": "The study found that 73% of..."}

── Diff ─────────────────────────────────────────────
- "source_paragraph": "In 2023, the report stated..."
+ "source_paragraph": "The study found that 73% of..."
```

The diff is character-level on the JSON output, not on the prompt. This makes it
easy to see exactly what changed in the model's response.
