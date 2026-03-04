# reliability — Correctness and Robustness

The `reliability` package contains the three mechanisms that make small model outputs
trustworthy: schema validation, reflection-based retries, and lightweight hallucination
guards.

---

## Research Background

### Why structured output is the primary defence

Hallucination surveys (arxiv:2510.06265, arxiv:2601.09929) categorise mitigation strategies
into: prompt-based, retrieval-based, reasoning-based, and model-training-based.
For local inference without fine-tuning access, only the first two are available.

Of the prompt-based strategies, structured output constraints are the most tractable:

> "Instruction-based prompts, when coupled with techniques such as CoT reasoning or
> iterative refinement, significantly reduces hallucinations by constraining model
> responses to align with facts and rationality."

The key insight is that a schema-validated response cannot hallucinate *structure* —
only *content*. Content hallucination is addressed by guards and retrieval grounding.

### Why reflection works for small models

The Reflexion paper (Shinn et al., NeurIPS 2024) demonstrated that feeding a model its
own error as a new user turn — verbal reinforcement — is more effective than simply
retrying with the same prompt. The model sees what went wrong and has a chance to correct
specifically.

For small models, this matters more than for large ones. A 7B model may simply not follow
a complex instruction on the first attempt, but *will* correct a specific, concise error
message on the second or third attempt.

### Why guards are heuristics, not ML models

The SLM+LLM hallucination detection framework (arxiv:2408.12748) uses a small model for
detection and a large model for explanation. This is effective but expensive and complex.

For `llmalib`'s use case — local inference, no cloud dependency — the equivalent is
rule-based guards: fast heuristics that catch common failure modes without requiring a
second model call. When a guard fires, the error is fed back through the reflection loop.

---

## `validator.py` — Schema Validation

### Purpose

`parse_response` takes the model's raw string output and a Pydantic schema, attempts to
extract and parse JSON, and returns either the validated model instance or a structured
parse error.

### Requirements

- Accept `raw: str` and `schema: type[BaseModel]`.
- Attempt JSON extraction using three strategies in order:
  1. Direct `json.loads(raw)` — for clean JSON responses.
  2. Extract from markdown code fences (` ```json ... ``` `).
  3. Extract the first `{...}` or `[...]` block via regex.
- On successful JSON extraction, validate against the schema with `schema.model_validate()`.
- Return `ParseResult(ok=True, value=instance)` or `ParseResult(ok=False, error=str)`.
- The error string must be human-readable and specific enough to include in a reflection
  prompt. E.g.: `"Missing required field 'confidence'. Got keys: ['label', 'reason']"`.
- Never raise. All errors are returned as `ParseResult(ok=False)`.

### Interface

```python
@dataclass(frozen=True)
class ParseResult:
    ok: bool
    value: BaseModel | None
    error: str | None

def parse_response(raw: str, schema: type[BaseModel]) -> ParseResult:
    """
    Attempt to extract and validate JSON from raw model output.
    Returns ParseResult — never raises.
    """

def format_schema_hint(schema: type[BaseModel]) -> str:
    """
    Produce a concise JSON schema description for inclusion in prompts.
    E.g.: '{"label": "string", "confidence": "float", "reason": "string"}'
    """
```

### Design Notes

**Three-strategy extraction**
Small models frequently wrap their JSON in explanatory text or markdown fences even when
instructed not to. The three-strategy approach handles all common patterns. If all three
fail, the error message explains what was found, giving the reflection loop actionable
information.

**Schema hint injection**
`format_schema_hint` is used by the prompt renderer to include a compact schema
description in every system prompt. Research (AutoPDL, arxiv:2504.04365) shows that
including the expected output format directly in the prompt is one of the highest-impact
prompting strategies for small models.

---

## `retry.py` — Reflection Loop

### Purpose

`run_with_retry` wraps a single task execution. On validation or guard failure, it
constructs a reflection message, appends it to the conversation history, and calls
the model again — up to `task.max_retries` times.

### Requirements

- Accept `task: Task`, `messages: list[dict]`, `config: ClientConfig`.
- On each attempt:
  1. Call `client.call(messages, config)`.
  2. Call `validator.parse_response(raw, task.output_schema)`.
  3. If parse succeeds, call each guard in `task.guards`.
  4. If all guards pass, return `Result(ok=True, ...)`.
  5. On any failure, build a reflection message and append to `messages`.
  6. Increment attempt counter; stop at `task.max_retries`.
- Record every attempt in `Result.trace` regardless of outcome.
- Return `Result(ok=False, error=last_error)` after exhausting retries.

### Interface

```python
def run_with_retry(
    task: Task,
    messages: list[dict],
    config: ClientConfig,
) -> Result:
    """
    Execute task with reflection-based retry on validation/guard failure.
    Returns Result — never raises on model-level failures.
    """

def build_reflection_message(
    parse_error: str | None,
    guard_errors: list[str],
) -> dict:
    """
    Build the user-turn message that feeds the error back to the model.

    Example output:
    {
        "role": "user",
        "content": (
            "Your previous response was invalid. Errors:\\n"
            "- Missing required field 'confidence'\\n"
            "- Answer references facts not present in the input\\n"
            "Please respond again with valid JSON matching the schema."
        )
    }
    """
```

### Reflection Message Design

The reflection message must be:

1. **Specific**: List each error individually, not as a generic "invalid response".
2. **Actionable**: Tell the model exactly what to fix, not just what was wrong.
3. **Concise**: Small models lose focus with long error explanations. Two to four bullet
   points is the target.

Research (Reflexion, NeurIPS 2024): verbal reinforcement works because the model
"remembers" the prior attempt in its context and can contrast its new response against
the error. Generic messages like "try again" do not provide this signal.

### Attempt Recording

Every attempt is recorded as an `Attempt` dataclass with:
- The exact rendered prompt (messages list serialised as a string)
- The exact raw response
- The parse error (if any)
- The guard errors (if any)
- Duration in milliseconds

This full recording is what makes `replay.py` possible.

---

## `guards.py` — Hallucination Heuristics

### Purpose

Guards are post-generation checks applied to parsed output before it is accepted.
They are fast, rule-based, and return a list of error strings (empty = pass).

### Requirements

- A `Guard` is a callable: `(value: BaseModel, ctx: Context) -> list[str]`.
- Empty return means the guard passed.
- Non-empty return means the guard failed; each string is an error message for reflection.
- Provide four built-in guard factories.
- Guards are composable — a task can have multiple guards; all are run; all failures are collected.

### Built-in Guards

```python
def field_in_set(field: str, allowed: set) -> Guard:
    """
    Checks that value.<field> is one of the allowed values.

    Example: field_in_set("label", {"positive", "negative", "neutral"})
    Catches: label="POSITIVE" (wrong case), label="mixed" (invented value)
    """

def float_in_range(field: str, min_val: float, max_val: float) -> Guard:
    """
    Checks that value.<field> is between min_val and max_val.

    Example: float_in_range("confidence", 0.0, 1.0)
    Catches: confidence=1.5 (out of range), confidence=-0.1
    """

def no_content_from_outside_context(
    response_field: str,
    context_field: str,
    threshold: float = 0.3,
) -> Guard:
    """
    Lightweight grounding check: verifies that the response_field text shares
    significant token overlap with context_field text.

    Uses Jaccard similarity on word tokens — no embeddings, no model call.
    A low overlap score suggests the model may have hallucinated content
    not present in the provided input.

    threshold: minimum Jaccard similarity (0.0–1.0). Default 0.3 is permissive;
    raise to 0.5+ for stricter grounding.
    """

def max_length(field: str, max_chars: int) -> Guard:
    """
    Checks that str(value.<field>) does not exceed max_chars.
    Catches runaway generation in summary/explanation fields.
    """
```

### Custom Guards

Any function with the signature `(BaseModel, Context) -> list[str]` is a valid guard:

```python
def no_urls_in_summary(value: MySchema, ctx: Context) -> list[str]:
    if "http" in value.summary:
        return ["Summary must not contain URLs"]
    return []

task = Task(
    ...,
    guards=(no_urls_in_summary,),
)
```

### Design Notes

**Why Jaccard similarity for grounding, not embeddings?**
Embedding-based grounding checks require loading a second model or calling an embedding
API. Jaccard on word tokens is O(n) and has zero dependencies. It is a coarse heuristic —
not a guarantee of faithfulness — but it catches the most common failure mode: a model
that ignores the provided context entirely and generates from parametric knowledge.

For high-stakes grounding requirements, users should replace this guard with a proper
retrieval-based check or an LLM-as-judge call. The guard interface (`BaseModel, Context -> list[str]`)
accommodates any implementation.

**Guards run after parsing, not before**
Guards operate on the validated Pydantic instance, not on the raw string. This means they
can access structured fields cleanly and the schema is already validated when guards run.
A parse failure short-circuits to the reflection loop before guards are reached.
