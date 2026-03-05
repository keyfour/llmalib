# AGENTS.md - llmalib Development Guide

This document provides essential information for agents working with the `llmalib` codebase.

## Project Overview

`llmalib` is a minimal, transparent Python library for building agentic pipelines with local small language models. It's designed around empirical research findings about small model behavior and focuses on narrow, schema-constrained tasks.

**Key Design Principles:**

1. Narrow tasks over broad prompts
2. Structured output as the reliability layer  
3. Transparent execution with full tracing
4. Explicit context budgets to prevent rot
5. Functional Python approach (no class hierarchies)
6. Minimal dependencies

## Project Structure

```
llmalib/
├── llmalib/                    # Main package
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
├── tests/                      # Comprehensive test suite
└── docs/                       # Module documentation
```

## Essential Commands

### Development Setup

```bash
# Install dependencies
uv sync

# Install with development dependencies
uv sync --extra dev

# Run development server (if needed)
# No built-in server - depends on local LLM setup
```

### Testing

```bash
# Run all tests with coverage
pytest --cov=llmalib --cov-report=term-missing --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run with verbose output
pytest -v

# Run tests matching pattern
pytest -k "test_task"
```

### Code Quality

```bash
# Lint with ruff
ruff check llmalib/

# Format code with ruff
ruff format llmalib/

# Type checking with mypy
mypy llmalib/

# Run all quality checks
ruff check && ruff format && mypy
```

### Building

```bash
# Build with hatch (defined in pyproject.toml)
# No explicit build command needed - uses standard Python packaging
```

## Code Organization and Patterns

### Core Data Structures

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

### Code Patterns

1. **Functional Style**: Operations are functions, not methods. No inheritance hierarchies.
2. **Immutable Data**: Core types are frozen dataclasses.
3. **Pydantic Schemas**: Every task declares its output shape upfront.
4. **Jinja2 Templates**: User prompts are templates with context variables.
5. **Error Handling**: Never raises - returns error envelopes.
6. **Explicit Dependencies**: All dependencies are injected, not hidden.

### Import Patterns

```python
# Core types
from llmalib.core.task import Task
from llmalib.core.result import Result, make_ok_result
from llmalib.core.context import Context, make_context

# Pipeline execution
from llmalib.pipeline.pipeline import run_pipeline

# HTTP client
from llmalib.core.client import call, ClientConfig

# Reliability
from llmalib.reliability.validator import parse_response
from llmalib.reliability.retry import run_with_retry
```

## Testing Approach

### Test Structure

- Comprehensive test suite in `tests/` directory
- Tests organized by module (`test_core.py`, `test_pipeline.py`, etc.)
- Heavy use of `unittest.mock` for HTTP client mocking
- Pydantic models for test data structures

### Test Patterns

```python
# Common test pattern
def test_task_defaults(self):
    task = make_sample_task()
    assert task.model == "llama3.2"
    assert task.temperature == 0.1

# Mock HTTP responses
mock_response = MagicMock()
mock_response.status_code = 200
mock_response.json.return_value = {
    "choices": [{"message": {"content": "hello world"}}]
}
with patch("httpx.post", return_value=mock_response):
    result = call([{"role": "user", "content": "hi"}], config)
```

### Test Coverage

- Current coverage target: 100% via `--cov=llmalib`
- Coverage reports generated in HTML and terminal
- Tests cover all modules including error paths

## Dependencies and Configuration

### Core Dependencies

- `pydantic>=2.0` - Schema validation and data models
- `httpx>=0.25.0` - HTTP client for local LLM inference
- `jinja2>=3.1.0` - Template rendering for prompts
- `rich>=13.0.0` - Optional rich console output

### Development Dependencies

- `pytest>=7.0.0` - Testing framework
- `pytest-cov>=4.0.0` - Coverage reporting
- `ruff>=0.4.0` - Linting and formatting
- `mypy>=1.0.0` - Type checking

### Configuration Files

**pyproject.toml**:

- Line length: 88 characters
- Target Python: 3.9+
- Ruff rules: E, W, F, I, B, C4, UP
- MyPy: strict mode enabled
- Pytest: coverage enabled by default

**.gitignore**:

- Standard Python ignores
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

