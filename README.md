# llmalib

**A minimal, transparent Python library for building agentic pipelines with local small language models.**

---

## Motivation

Most agentic frameworks were designed around GPT-4-class models with large context windows, reliable instruction following, and fast inference. When you run a 3B–13B model locally — via Ollama, llama.cpp, or vLLM — these assumptions collapse. Abstractions that hide LLM calls behind graph edges and node callbacks become debugging nightmares the moment the model produces unexpected output.

Research confirms this mismatch. Work on SLM-first architectures (arxiv:2506.02153) shows that small models excel when given **narrow, focused tasks** rather than open-ended instructions. Studies on context degradation ("context rot", arxiv:2509.21361; "lost in the middle", Liu et al. 2024) show that even models with large context windows degrade significantly beyond a few thousand tokens. Hallucination surveys (arxiv:2510.06265) confirm that structured output constraints and iterative refinement are the most effective mitigation strategies available without fine-tuning.

`llmalib` is built from these findings up. Every design decision is traceable to an empirical result.

---

## Design Principles

### 1. Narrow tasks over broad prompts

A single unstructured prompt fed to a 7B model is unreliable. The same goal broken into three focused sub-tasks — each with a tight schema and token budget — is not. `llmalib` makes task decomposition the primary abstraction.

> *"Invocations of tools and language models during an agentic process are often accompanied by careful prompting that focuses the language model on delivering the narrow functionality that is required at the time."*
> — arxiv:2506.02153

### 2. Structured output as the reliability layer

Hallucination detection after the fact is expensive and unreliable. Preventing hallucination by constraining output format via Pydantic schemas is cheap and deterministic. Every task in `llmalib` declares its output shape; the library enforces it.

> Instruction-based prompts coupled with iterative refinement *"significantly reduces hallucinations by constraining model responses to align with facts and rationality."*
> — arxiv:2510.06265

### 3. Transparent execution

Every LLM call — its rendered prompt, raw response, parse result, retry count, and guard outcomes — is recorded in a `Trace`. Nothing is hidden. You can replay any run, diff prompts across retries, and export traces for debugging.

### 4. Explicit context budgets

Context rot is real. Feeding an ever-growing history into each task call silently degrades quality. `llmalib` enforces a per-task token budget and trims context deterministically rather than letting the model "figure it out".

> Observation masking — explicitly trimming verbose context — is consistently more efficient than LLM-based summarization, often achieving better solve rates at lower cost.
> — arxiv:2508.21433

### 5. Functional Python, no class hierarchies

Tasks are data (`@dataclass`). Operations are functions. Pipelines are lists. There is no `BaseAgent` to inherit from, no lifecycle hooks to implement, no dependency injection container to configure.

### 6. Minimal dependencies

`pydantic`, `httpx`, `jinja2`, `rich` (optional). That is the entire dependency tree.

---

## Architecture Overview

```
llmalib/
│
├── core/               # Foundational types and HTTP client
│   ├── client.py       # Thin wrapper: render prompt → HTTP → raw string
│   ├── task.py         # Task dataclass: schema + template + config
│   ├── result.py       # Result envelope: value | error | trace
│   └── context.py      # Shared mutable state for a pipeline run
│
├── pipeline/           # Orchestration layer
│   ├── pipeline.py     # Sequential runner with router support
│   ├── decomposer.py   # Freeform prompt → ordered list of Tasks
│   └── router.py       # Route a Result to the next Task
│
├── reliability/        # Correctness and robustness
│   ├── validator.py    # Pydantic schema validation + custom rules
│   ├── retry.py        # Reflection loop: feed errors back as context
│   └── guards.py       # Lightweight hallucination heuristics
│
├── memory/             # Context and example management
│   ├── context_window.py  # Token budget: trim to fit, never silently overflow
│   ├── store.py           # KV store: in-memory + optional file persistence
│   └── examples.py        # Few-shot example selection via BM25
│
└── debug/              # Observability
    ├── tracer.py       # Records every attempt for every task
    ├── inspector.py    # Rich-formatted console output
    └── replay.py       # Re-run a recorded trace against a new model/prompt
```

---

## Data Flow

```
User prompt (unstructured text)
         │
         ▼
   [ Decomposer ]
   Calls the LLM once to produce an ordered list of Tasks.
   Each Task has a name, template, output schema, and config.
         │
         ▼
   [ Pipeline.run() ]
   Iterates over Tasks in order (or as routed).
         │
    ┌────▼────────────────────────────────────────────┐
    │  For each Task:                                  │
    │                                                  │
    │  1. context_window.trim()   ← enforce token budget│
    │  2. examples.select()       ← inject few-shot    │
    │  3. template.render()       ← fill Jinja2 vars   │
    │  4. client.call()           ← HTTP to local LLM  │
    │  5. validator.parse()       ← Pydantic schema     │
    │  6. guards.check()          ← heuristic checks   │
    │                                                  │
    │  On failure → retry.reflect()                    │
    │    Appends error to history, calls again (max N) │
    │                                                  │
    │  Records every attempt in Trace                  │
    └────┬────────────────────────────────────────────┘
         │
    Result(ok, value, error, attempts, trace)
         │
    router.next() → next Task or end
         │
         ▼
   Context accumulates all Results
         │
         ▼
   Final output + complete Trace
```

---

## Quick Start

```python
from pydantic import BaseModel
from llmalib.core.task import Task
from llmalib.pipeline.pipeline import run_pipeline
from llmalib.core.context import Context

class SentimentResult(BaseModel):
    label: str       # "positive" | "negative" | "neutral"
    confidence: float
    reason: str

task = Task(
    name="classify_sentiment",
    prompt_template="Classify the sentiment of: {{ text }}",
    output_schema=SentimentResult,
    model="llama3.2",
    max_retries=3,
)

ctx = Context(vars={"text": "The product exceeded my expectations."})
results = run_pipeline([task], ctx, debug=True)
print(results[0].value)
# SentimentResult(label='positive', confidence=0.92, reason='...')
```

---

## Documentation Index

| File | Contents |
|---|---|
| [core.md](core.md) | `client`, `task`, `result`, `context` |
| [pipeline.md](pipeline.md) | `pipeline`, `decomposer`, `router` |
| [reliability.md](reliability.md) | `validator`, `retry`, `guards` |
| [memory.md](memory.md) | `context_window`, `store`, `examples` |
| [debug.md](debug.md) | `tracer`, `inspector`, `replay` |
| [research.md](research.md) | Arxiv findings that informed design |

---

## Non-Goals

| Temptation | Why skipped |
|---|---|
| Async pipeline runner | Local models are sequential; async adds complexity without throughput gain |
| Plugin / extension system | Just subclass or wrap; plugin systems become frameworks |
| Built-in vector store | BM25 covers 90% of few-shot selection; add Chroma only when you need it |
| Streaming responses | Nice-to-have; add later when the core is stable |
| Cloud model support | OpenAI-compatible base URL means it already works |
| Automatic model selection | User knows their hardware; routing by capability is a future feature |
