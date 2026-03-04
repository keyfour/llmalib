# memory — Context and Example Management

The `memory` package handles three distinct problems: keeping prompts within a model's
effective context window, persisting state across pipeline runs, and selecting relevant
few-shot examples.

---

## Research Background

### Context degradation is not just a theoretical concern

Multiple independent studies confirm that context quality degrades well before the
technical context limit is reached:

- The "lost in the middle" effect (Liu et al., 2024): models reliably recall information
  near the beginning and end of the context but ignore the middle.
- Context rot (arxiv:2509.21361): LLM performance on real tasks drops measurably as
  input length increases, even when retrieval accuracy is held constant.
- Context length alone hurts performance (arxiv:2510.05381): even with perfect retrieval
  of relevant information, longer contexts degrade reasoning quality.

For small local models — 3B to 13B parameters — these effects are pronounced. The
effective context window for reliable task performance is often 2000–4000 tokens,
even when the model's technical limit is 8000–32000.

The practical implication: `llmalib` must enforce context budgets per task and trim
aggressively. Soft warnings are not enough.

### Memory inflation in long-running agents

Research on long-running agents (arxiv:2509.25250) documents the "self-degradation" 
problem: agents that accumulate all prior history degrade over time as flawed memories
pollute the context. The solution is selective retention: store only high-quality,
task-relevant results.

`llmalib` applies this at the pipeline level via `store.py`: results are stored with a
relevance score and an optional TTL, and `store.get_relevant()` returns only the
highest-scoring matches.

### Few-shot example selection

Research on prompting small models (arxiv:2504.04365, arxiv:2507.14241) consistently
finds that:

1. Few-shot examples have 2–3x more influence when placed near the end of the prompt
   (recency bias).
2. Example diversity matters more than raw quantity — 3 diverse examples outperform
   10 similar ones.
3. Formatting consistency across examples is critical — inconsistent formatting reduces
   few-shot effectiveness.

`llmalib` uses BM25 for example retrieval (no embeddings required), enforces consistent
formatting, and always injects selected examples at the end of the system prompt.

---

## `context_window.py` — Token Budget Manager

### Purpose

`trim_to_budget` takes a task's rendered messages and trims them to fit within the task's
declared `token_budget`. It never silently overflows.

### Requirements

- Accept `messages: list[dict]`, `budget: int`, and an optional `tokenizer` callable.
- Default tokenizer: `len(text) // 4` (conservative 4-chars-per-token estimate). No
  dependency on `tiktoken` or any tokenizer model.
- Trim strategy (in order, most expendable first):
  1. Shorten middle `assistant` turns in conversation history (keep first + last sentence).
  2. Truncate long `user` context blocks to their first `N` characters + `"[truncated]"` marker.
  3. Never truncate the system prompt or the final user turn.
- Raise `BudgetExceededError` if the system prompt + final user turn alone exceed the budget
  (indicates a misconfigured task, not a runtime trim situation).
- Log a warning (via `tracer`) whenever trimming occurs, including how many tokens were removed.

### Interface

```python
def count_tokens(text: str, tokenizer: Callable[[str], int] | None = None) -> int:
    """
    Estimate token count. Defaults to len(text) // 4.
    Replace with tiktoken or model-specific tokenizer for precision.
    """

def trim_to_budget(
    messages: list[dict],
    budget: int,
    tokenizer: Callable[[str], int] | None = None,
) -> list[dict]:
    """
    Return a (possibly trimmed) copy of messages that fits within budget tokens.
    Raises BudgetExceededError if system + final user turn exceed budget alone.
    Never mutates the input list.
    """
```

### Trim Priority

| Priority | Message type | Strategy |
|---|---|---|
| 1 (first to trim) | Middle `assistant` turns | Keep first + last sentence |
| 2 | Long `user` context blocks | Truncate to 80% of remaining budget |
| 3 (last resort) | Any remaining | Hard truncate with `[truncated]` marker |
| Never | System prompt | Preserved entirely |
| Never | Final user turn | Preserved entirely |

### Design Notes

**Why a conservative 4-chars-per-token default?**
Under-counting tokens means under-trimming, which leads to silent context overflow.
Over-counting tokens means over-trimming, which is visible and correctable. The
conservative default errs toward over-trimming. Projects that need precision should
inject their model's actual tokenizer via the `tokenizer` parameter.

**Why not use LLM summarisation for trimming?**
Research (arxiv:2508.21433) found that simple observation masking (truncation) is
competitive with LLM summarisation in terms of solve rate, and substantially cheaper
in terms of tokens and latency. Summarisation also adds a second model call per trim
operation, which can cascade into budget problems of its own. `llmalib` uses truncation
by default and leaves summarisation as a future extension.

---

## `store.py` — Key-Value Store

### Purpose

`Store` provides persistent key-value storage for results, intermediate outputs, and
any data that needs to survive beyond a single `run_pipeline` call.

### Requirements

- Implement two backends: `InMemoryStore` (default) and `FileStore` (JSON file on disk).
- Both backends implement the same interface so they are interchangeable.
- Support `set(key, value, score=1.0)`, `get(key)`, `delete(key)`, `list(prefix)`, and
  `get_relevant(query, top_k)`.
- `get_relevant` uses BM25 over stored keys and values to return the `top_k` most
  relevant entries. This allows the store to function as a lightweight long-term memory
  without vector embeddings.
- Support optional TTL (time-to-live in seconds) on `set()` — entries expire silently.
- Be thread-safe for reads (writes are assumed single-threaded in sequential pipelines).

### Interface

```python
@dataclass
class StoreEntry:
    key: str
    value: Any
    score: float = 1.0
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None

# Store protocol (both backends implement this)
class Store(Protocol):
    def set(self, key: str, value: Any, score: float = 1.0, ttl: float | None = None) -> None: ...
    def get(self, key: str) -> StoreEntry | None: ...
    def delete(self, key: str) -> None: ...
    def list(self, prefix: str = "") -> list[str]: ...
    def get_relevant(self, query: str, top_k: int = 5) -> list[StoreEntry]: ...

def make_memory_store() -> Store:
    """Create an in-memory store. Data is lost when the process exits."""

def make_file_store(path: str) -> Store:
    """Create a file-backed store. Data persists across runs."""
```

### BM25 in `get_relevant`

BM25 is implemented from scratch (40 lines) — no external library. It tokenises
keys and serialised values on `set()` and builds an inverted index. `get_relevant`
tokenises the query and scores entries by term frequency × inverse document frequency.

This is sufficient for the primary use case: retrieving relevant past results and
few-shot examples from a store with dozens to low hundreds of entries. For stores
with thousands of entries, users should integrate a proper vector store.

### Design Notes

**Why not use SQLite?**
SQLite adds a dependency and file locking complexity for a use case that is typically
single-process. JSON file persistence is transparent, human-readable, and trivially
debuggable. A future `SqliteStore` backend can be added without changing the interface.

**Score field**
The `score` field lets callers mark some entries as higher quality than others.
`get_relevant` uses BM25 score × entry score as the final ranking criterion.
This allows the pipeline to demote results from failed tasks or promote
hand-crafted gold examples.

---

## `examples.py` — Few-Shot Example Selection

### Purpose

`select_examples` retrieves the most relevant few-shot examples for a given task
and context, formats them consistently, and returns them ready for prompt injection.

### Requirements

- Accept `task: Task`, `ctx: Context`, `store: Store | None`, and `max_examples: int = 4`.
- Use two sources of examples:
  1. `task.examples` — examples declared inline on the Task (always included, up to limit).
  2. `store.get_relevant()` — dynamically retrieved from the store if provided.
- Inline task examples take priority over store examples.
- Format each example as a structured block:
  ```
  Example N:
  Input: <input text>
  Output: <JSON schema output>
  ```
- Inject examples at the **end** of the system prompt (recency bias: models weight
  the last examples most heavily per arxiv:2507.14241).
- Return `list[FormattedExample]` — the rendered strings ready for prompt injection.

### Interface

```python
@dataclass(frozen=True)
class Example:
    input_text: str
    output: BaseModel           # Must match the task's output_schema
    description: str = ""       # Optional label for debugging

@dataclass(frozen=True)
class FormattedExample:
    text: str                   # Rendered string for prompt injection
    source: str                 # "inline" | "store" — for tracing

def select_examples(
    task: Task,
    ctx: Context,
    store: Store | None = None,
    max_examples: int = 4,
) -> list[FormattedExample]:
    """
    Select and format up to max_examples few-shot examples.
    Inline task examples take priority. Store examples fill remaining slots.
    """
```

### Formatting Rules

Based on research findings (arxiv:2507.14241):

1. **Consistent format**: Every example uses the same template. Never mix JSON and prose
   examples for the same task.
2. **Most relevant last**: BM25 similarity to the current input determines order.
   The highest-scoring example is placed last (maximum recency effect).
3. **Diversity over volume**: `max_examples=4` by default. More examples rarely help and
   consume token budget.
4. **Output is exact JSON**: The output block in each example is the JSON serialisation of
   the `Example.output` model. This shows the model exactly what valid output looks like.

### Example Store Workflow

```python
store = make_file_store("examples.json")

# Store a gold example after a successful run
result = run_pipeline([task], ctx)
if result[0].ok:
    store.set(
        key=f"example:{task.name}:{hash(ctx.vars['text'])}",
        value=Example(input_text=ctx.vars["text"], output=result[0].value),
        score=1.0,
    )

# Next run: examples are automatically retrieved and injected
results = run_pipeline([task], ctx, store=store)
```

This pattern means that every successful run can automatically improve future runs —
without fine-tuning.
