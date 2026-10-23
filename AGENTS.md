# AGENTS.md - llmalib Development Guide

This document provides essential information for agents working with the `llmalib` codebase.

## Project Overview

`llmalib` is a minimal, transparent library for building agentic pipelines with local small language models. It's designed around empirical research findings about small model behavior and focuses on narrow, schema-constrained tasks.

The project is currently in transition from Python to Rust, with both implementations coexisting. The Python version (`llmalib/` directory) is the original implementation and is considered stable. The Rust version (`crates/` directory) is an ongoing migration effort — see `.plan.md` (local, untracked) for the phase plan; core/reliability/debug/memory basics have been migrated, pipeline and bindings remain in progress.

**Key Design Principles:**

1. Narrow tasks over broad prompts
2. Structured output as the reliability layer  
3. Transparent execution with full tracing
4. Explicit context budgets to prevent rot
5. Functional approach (no class hierarchies)
6. Minimal dependencies

## Project Structure

```
llmalib/
├── llmalib/                    # Original Python package
│   ├── core/                   # Foundational types and HTTP client
│   │   ├── client.py          # HTTP wrapper for local LLM inference
│   │   ├── task.py            # Task dataclass + schema enforcement
│   │   ├── result.py          # Result envelope + attempt tracking
│   │   └── context.py         # Shared mutable state for pipeline runs
│   ├── pipeline/               # Orchestration layer
│   │   ├── pipeline.py        # Sequential task execution
│   │   ├── decomposer.py      # Freeform prompt → Task list
│   │   └── router.py          # Conditional task routing
│   ├── reliability/            # Correctness and robustness
│   │   ├── validator.py       # JSON extraction + Pydantic validation
│   │   ├── retry.py           # Reflection-based retry loops
│   │   └── guards.py          # Lightweight hallucination heuristics
│   ├── memory/                 # Context and example management
│   │   ├── context_window.py  # Token budget enforcement
│   │   ├── store.py           # KV store for persistence
│   │   └── examples.py        # Few-shot example selection
│   └── debug/                  # Observability tools
│       ├── tracer.py          # Records every attempt
│       ├── inspector.py       # Rich console output
│       └── replay.py          # Trace replay capability
├── crates/                     # Rust workspace (migration target)
│   ├── llmalib-core/           # Core: Task, Result, Context, client, Guard trait
│   │   └── src/
│   │       ├── client.rs       # HTTP client wrapper (reqwest blocking)
│   │       ├── context.rs      # Shared mutable state
│   │       ├── guard.rs        # Guard trait definition
│   │       ├── result.rs       # Result envelope (TaskResult / Attempt)
│   │       ├── task.rs         # Task definition + builder
│   │       └── lib.rs          # Package exports
│   ├── llmalib-memory/         # KV stores, token budgeting, few-shot examples
│   │   └── src/ {context_window,examples,store,lib}.rs
│   ├── llmalib-reliability/    # Validation, guards, reflection-based retry
│   │   └── src/ {guard,retry,validator,lib}.rs
│   ├── llmalib-debug/          # Tracer, inspector, replay
│   │   └── src/ {inspector,replay,tracer,lib}.rs
│   └── llmalib-pipeline/       # Sequential pipeline orchestration
│       └── src/ {pipeline,lib}.rs
├── tests/                      # Python test suite
├── docs/                       # Documentation
│   ├── core.md                # Python core module docs
│   ├── debug.md               # Python debug module docs
│   ├── memory.md              # Python memory module docs
│   ├── pipeline.md            # Python pipeline module docs
│   ├── reliability.md         # Python reliability module docs
│   ├── research.md            # Research foundations
│   └── MIGRATION.md           # Rust migration guide (to be created)
└── ...                        # Config files (pyproject.toml, Cargo.toml, etc.)
```

## Essential Commands

### Development Setup

```bash
# Install Python dependencies
uv sync

# Install Python development dependencies
uv sync --extra dev

# Install Rust toolchain (if not already installed)
rustup update

# Build Rust packages
cargo build --workspace

# Run Rust tests
cargo test --workspace

# Run specific Rust package tests
cargo test -p llmalib-core
```

### Testing

```bash
# Run all Python tests with coverage
pytest --cov=llmalib --cov-report=term-missing --cov-report=html

# Run specific Python test file
pytest tests/test_core.py

# Run with verbose output
pytest -v

# Run tests matching pattern
pytest -k "test_task"

# Run all Rust tests
cargo test --workspace

# Run Rust tests for specific package
cargo test -p llmalib-core

# Run Rust tests with verbose output
cargo test -- --nocapture
```

### Code Quality

```bash
# Lint Python code with ruff
ruff check llmalib/

# Format Python code with ruff
ruff format llmalib/

# Type checking with mypy
mypy llmalib/

# Check Rust code formatting
cargo fmt -- --check

# Format Rust code
cargo fmt

# Clippy linting for Rust
cargo clippy --workspace -- -D warnings

# Run all quality checks
ruff check && ruff format && mypy && cargo fmt -- --check && cargo clippy --workspace -- -D warnings
```

### Building

```bash
# Build with hatch (defined in pyproject.toml)
# No explicit build command needed - uses standard Python packaging
```

## Code Organization and Patterns

### Python Implementation

**Core Data Structures**

**Task** (`llmalib/core/task.py`):

- Immutable `@dataclass(frozen=True)` - all tasks are declarations, not state
- Required: `name`, `prompt_template`, `output_schema`
- Model config: `model`, `base_url`, `temperature`, `max_tokens`, `timeout`
- Reliability: `max_retries`, `guards`
- Context: `examples`, `token_budget`, `system_prompt`

**Result** (`llmalib/core/result.py`):

- Envelope pattern: `ok: bool`, `value: BaseModel | None`, `error: str | None`
- Always includes `attempts: list[Attempt]` for full tracing
- Never raises exceptions - errors are captured in the envelope

**Context** (`llmalib/core/context.py`):

- Mutable state container for pipeline runs
- `vars: dict` for template variables (populated from previous results)
- `results: dict` mapping task names to their results
- `run_id: str` for trace correlation

### Rust Implementation (in progress)

**Core Data Structures**

**Task** (`crates/llmalib-core/src/task.rs`):

- Immutable struct with all fields public (no dataclass decorator needed)
- Required: `name`, `prompt_template`, `output_schema`
- Model config: `model`, `base_url`, `temperature`, `max_tokens`, `timeout`
- Reliability: `max_retries` (total attempt count, like Python), `guards` (`Vec<Arc<dyn Guard>>`)
- Context: `examples`, `token_budget`, `system_prompt` (Option<String>)
- Create via `Task::new(name, prompt_template, output_schema)` — applies documented Python defaults —
  or override optional fields with `Task::builder(...)`
- `Task` is `Clone`; clones share guard implementations via `Arc` (like Python shallow copies)

**Result** (`crates/llmalib-core/src/result.rs`):

- Envelope pattern similar to Python: `ok`, `value`, `error`
- Includes attempt tracking for tracing
- Uses `TaskResult` type with associated methods

**Context** (`crates/llmalib-core/src/context.rs`):

- Shared mutable state container for pipeline runs
- `vars`: serde_json::Map<String, serde_json::Value> for template variables
- `results`: serde_json::Map<String, serde_json::Value> mapping task names
- `run_id`: String (8-char uuid prefix, auto-generated; override via `with_run_id`)
- Only successful results merge fields into vars under their **plain field names**
  (prevents error propagation; matches Python's `{{ field }}` template access)
- Provides `merge_result()`, `update_context()`, and `make_context(vars)`.

### Code Patterns (Both Implementations)

1. **Functional Style**: Operations are functions, not methods on core types
2. **Immutable Data**: Core types are immutable/frozen (Python: @dataclass(frozen=True), Rust: struct with no mut methods)
3. **Schema Validation**: Every task declares output shape upfront (Python: Pydantic, Rust: serde_json::Value)
4. **Templating**: User prompts use template variables (Python: Jinja2, Rust: likely similar approach)
5. **Error Handling**: Never raises - returns error envelopes/types
6. **Explicit Dependencies**: All dependencies are injected, not hidden
7. **Token Budget**: Every task has token_budget for context window enforcement
8. **Retry Strategy**: Reflection-based retry that appends error context
9. **Guard System**: Lightweight hallucination heuristics pluggable per task

## Dependencies and Configuration

### Python Dependencies

**Core Dependencies**

- `pydantic>=2.0` - Schema validation and data models
- `httpx>=0.25.0` - HTTP client for local LLM inference
- `jinja2>=3.1.0` - Template rendering for prompts
- `rich>=13.0.0` - Optional rich console output

**Development Dependencies**

- `pytest>=7.0.0` - Testing framework
- `pytest-cov>=4.0.0` - Coverage reporting
- `ruff>=0.4.0` - Linting and formatting
- `mypy>=1.0.0` - Type checking

### Rust Dependencies

**Core Dependencies** (see crates/*/Cargo.toml)

- `serde` - Serialization framework
- `serde_json` - JSON handling
- `thiserror` - Error handling
- `uuid` - Run id generation (llmalib-core)
- `reqwest` - HTTP client (blocking and JSON features, llmalib-core)
- `regex` - JSON extraction / tokenization (reliability, memory)
- `chrono` - Timestamps (llmalib-debug)

**Development Dependencies**

- `rustfmt` - Code formatting
- `clippy` - Linting

### Configuration Files

**pyproject.toml** (Python):

- Line length: 88 characters
- Target Python: 3.9+
- Ruff rules: E, W, F, I, B, C4, UP
- MyPy: strict mode enabled
- Pytest: coverage enabled by default

**Cargo.toml** (Rust workspace):

- Defines workspace members: core, memory, reliability, pipeline, debug (crates/*)
- Shared `[workspace.package]` provides version 0.1.0 and edition 2021 to all crates
- Resolver = "2"

**Rust package Cargo.toml examples** (llmalib-core):

- Dependencies as listed above
- Library path set to mod.rs

**.gitignore**:

- Standard Python ignores
- Standard Rust ignores (target/, Cargo.lock)
- Coverage reports
- Lock files

## Important Gotchas and Patterns

### 1. Task Immutability

- Tasks are frozen dataclasses - cannot be modified after creation
- Override defaults by passing arguments to constructor:

```python
task = Task(
    name="my_task",
    prompt_template="Process: {{ input }}",
    output_schema=MySchema,
    model="mistral",  # Override default "llama3.2"
    max_retries=5    # Override default 3
)
```

### 2. Context Variable Merging

- Successful results automatically merge their fields into `ctx.vars`
- Failed results do NOT merge fields (prevents error propagation)
- Use `update_context()` explicitly for custom merging

### 3. Token Budget Enforcement

- Every task has a `token_budget` (default: 2048)
- Context window automatically trims content to fit
- Conservative defaults prevent context rot in small models

### 4. Retry Strategy

- Uses reflection-based retry, not simple retry
- Appends error context to next attempt prompt
- Maximum retries configurable via `max_retries`

### 5. JSON Extraction

- Three-strategy extraction for robustness
- Handles clean JSON, markdown fences, and regex extraction
- Never raises - returns ParseResult with error state

### 6. HTTP Client Behavior

- Requires OpenAI-compatible local server (Ollama, llama.cpp, vLLM)
- Default base_url: `http://localhost:11434/v1`
- Timeout configurable (default: 120s) for slow local models
- Raises `ClientError` on all failures

### 7. Template Rendering

- Jinja2 templates with automatic context variable injection
- Templates can reference previous task results as variables
- System prompts auto-generated from schema if not provided

## Working with Local Models

### Model Requirements

- Must be OpenAI-compatible API
- Common setup: Ollama (`localhost:11434`), llama.cpp server, vLLM
- Supported models: Llama 3.2, Mistral, Mixtral, etc.

### Configuration Example

```python
task = Task(
    name="classify",
    prompt_template="Classify: {{ text }}",
    output_schema=ClassificationResult,
    model="llama3.2",           # Model name as known to server
    base_url="http://localhost:11434/v1",
    temperature=0.1,           # Low for structured tasks
    max_tokens=1024,          # Conservative budget
    timeout=120.0,            # Local models are slow
)
```

### Debugging Tips

- Use `debug=True` in `run_pipeline()` for rich output
- Inspect traces with `tracer.dump()` or `inspector.print_result()`
- Replay traces with different models using `replay.run_trace()`
- Check token usage with `context_window.trim()` logging

## Development Workflow

1. **Adding New Features**: Follow existing patterns - functional style with dataclasses
2. **Testing**: Add comprehensive tests for all new functionality
3. **Documentation**: Update module docs in `docs/` directory
4. **Code Quality**: Run `ruff check && ruff format && mypy` before commits
5. **Breaking Changes**: This is alpha software - breaking changes are expected

## Research Context

The library is built on specific research findings:

- Small models excel at narrow, focused tasks (arxiv:2506.02153)
- Context rotation degrades quality beyond few thousand tokens
- Structured output constraints prevent hallucinations effectively
- Iterative reflection improves reliability more than simple retry

See `docs/research.md` for detailed references.