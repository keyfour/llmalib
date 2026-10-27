# AGENTS.md - llmalib Development Guide

This document provides essential information for agents working with the `llmalib` codebase.

## Project Overview

`llmalib` is a minimal, transparent library for building agentic pipelines with local small language models. It's designed around empirical research findings about small model behavior and focuses on narrow, schema-constrained tasks.

**The Python → Rust migration is complete.** The Python package (`llmalib/`) is a
thin wrapper over the `llmalib._rust` pyo3 extension module built from `crates/`;
every rule / engine behavior lives in the Rust crates. The public Python API
(module paths, signatures, kwargs, defaults, error strings, result types) is
unchanged. See `docs/MIGRATION.md` for the architecture and boundary contract,
and `.plan.md` (local, untracked) for the phase history and divergence notes.

**Key Design Principles:**

1. Narrow tasks over broad prompts
2. Structured output as the reliability layer
3. Transparent execution with full tracing
4. Explicit context budgets to prevent rot
5. Functional approach (no class hierarchies)
6. Minimal dependencies

## Project Structure

```
llmalib/                        # repo root
├── crates/                     # Rust workspace — the engine (source of truth)
│   ├── llmalib-core/           # Task, TaskResult/Attempt, Context,
│   │                           # Caller trait + reqwest HTTP default, Guard trait
│   ├── llmalib-memory/         # context_window (token budget), store (KV+BM25+TTL),
│   │                           # examples (few-shot selection)
│   ├── llmalib-reliability/    # validator (JSON-Schema via `jsonschema`),
│   │                           # retry (reflection), guard.rs (native guard rules)
│   ├── llmalib-debug/          # tracer (Rust-native), replay engine
│   ├── llmalib-pipeline/       # run_pipeline, routers, decomposer
│   └── llmalib-py/             # pyo3 bindings (cdylib) → `llmalib._rust`
│       └── src/ {api,bridges,classes,convert,lib}.rs
├── llmalib/                    # Python wrapper package (same public API)
│   ├── _rust.abi3.so           # built artifact (gitignored, maturin)
│   ├── _rust.pyi               # mypy stubs for the FFI surface
│   ├── _bridge.py              # marshalling: Task↔spec dicts, value hydration,
│   │                           # guard/router adapters, guard `_llmalib_original`
│   ├── _testing.py             # test seam: sequence/raising/inspecting callers
│   ├── core/                   # task/result/context dataclasses; client adapter
│   ├── pipeline/               # pipeline, router (decision delegation), decomposer
│   ├── reliability/            # validator, retry, guards (→ _rust NativeGuard)
│   ├── memory/                 # store (Rust-backed), examples, context_window
│   └── debug/                  # tracer (_rust.PyTracer wrapper), replay, inspector
├── tests/                      # Python test suite (behavioral parity; caller seam)
│   └── conftest.py             # autouse fixture resetting the global caller
├── docs/                       # module docs + MIGRATION.md (architecture/boundary)
├── scripts/cleanup.sh          # safe cache/artifact cleanup (--dry-run/--target)
├── pyproject.toml              # maturin backend; pydantic + rich runtime deps
├── Cargo.toml                  # workspace: crates/*, shared [workspace.package]
└── rust-toolchain.toml         # stable
```

## Essential Commands

### Development Setup

```bash
# Install Python env + build the llmalib._rust extension (maturin backend)
uv sync --extra dev

# Rebuild the extension after editing any crate
#   (uv caches builds by version — --reinstall-package forces it)
uv sync --extra dev --reinstall-package llmalib

# Rust toolchain (stable, pinned in rust-toolchain.toml)
rustup update

# Build Rust packages without going through maturin
cargo build --workspace
```

### Testing (two suites)

```bash
# Rust engine test suite (per-crate integration tests + unit tests)
cargo test --workspace
cargo test -p llmalib-core           # one crate
cargo test -- --nocapture            # verbose

# Python parity test suite (115 tests, exercises the FFI)
uv run pytest tests/ -q -o addopts=""   # -o addopts="" disables coverage addopts
uv run pytest tests/test_pipeline.py    # one file
uv run pytest -k "test_task"            # pattern

# Python tests never touch HTTP: llmalib._testing.set_caller(...) installs a
# fake Caller; tests/conftest.py autouse-fixture resets it per test.
```

### Code Quality

```bash
uv run ruff check llmalib/
uv run ruff format llmalib/
uv run mypy llmalib/                 # strict-ish; _rust.pyi covers the FFI

cargo fmt -- --check
cargo fmt
cargo clippy --workspace -- -D warnings
cargo clippy --fix --allow-dirty --workspace

# Full pre-commit gate (all of it must be green)
cargo fmt -- --check && cargo clippy --workspace -- -D warnings \
  && cargo test --workspace
uv run ruff check llmalib/ && uv run ruff format --check llmalib/ \
  && uv run mypy llmalib/ && uv run pytest tests/ -q -o addopts=""

# Clean caches / stray build artifacts (also removes the leftover probe crate)
scripts/cleanup.sh [--dry-run] [--target] [--rebuild]
```

### Building

The Python package build is driven by **maturin** from `pyproject.toml`
(`build-backend = "maturin"`, abi3 py39 wheels; `crate-type = cdylib+rlib`
on `crates/llmalib-py`). No manual build step is normally needed —
`uv sync` runs it.

## Code Organization and Patterns

### Python side — thin wrappers only

- Public surface stays identical: same module paths, signatures, kwargs,
  defaults, error strings. Frozen `@dataclass` `Task`/`Attempt`/`Result`,
  pydantic schemas for output validation types, `Example`.
- `llmalib/_bridge.py` is the marshalling layer: `spec_from_task` /
  `task_from_spec` (flat spec dicts ↔ Tasks), `schema_to_json`
  (`model_json_schema()`), value hydration (`hydrate_value`,
  `attempt_from_dict`, `result_from_dict`, `task_result_to_dict`).
- Guard/router bridges re-hydrate into pydantic instances/Result objects so
  user callables always see the original shapes; wrappers are marked with
  `_llmalib_original` so specs that round-trip back from Rust (routers,
  decompose) are never double-wrapped.
- Factories in `guards.py` return `llmalib._rust.NativeGuard` instances —
  plain `(value, ctx) -> list[str]` callables whose rule logic runs in Rust.
- Router factories (`stop_on_failure`, `branch_on_field`) delegate the
  *decision* to `_rust.route_*` but keep Python task-list plumbing: the
  public router contract passes `Task` objects and returns the original
  lists (tests assert identity, not equality).
- `debug/inspector.py` is presentation-only (rich/plain rendering).
- `debug/tracer.py` wraps `_rust.PyTracer`: record/summary/to_dict/to_file/
  load_trace run Rust-side; `task_traces` hydrates into Python `TaskTrace`;
  `started_at` is a Python monotonic float for inspection only.

### Rust side — crates are the source of truth

**Task** (`crates/llmalib-core/src/task.rs`): immutable struct, required
`name`/`prompt_template`/`output_schema`; `Task::new` applies the documented
Python defaults (model `llama3.2`, temperature 0.1, max_tokens 1024,
timeout 120s, `max_retries` 3 = **total attempts**, token_budget 2048);
`Task::builder(...)` for overrides; `guards: Vec<Arc<dyn Guard>>` (clones
share guards via `Arc`, like Python shallow copies).

**Result / Context** (`crates/llmalib-core/src/{result,context}.rs`):
envelope `{ok, value, error}` + attempt tracking; `Context{vars, results,
run_id}` with 8-char uuid run ids; only successful results merge their
fields into `vars` under plain field names (last-write-wins; prevents error
propagation into templates).

**Client** (`crates/llmalib-core/src/client.rs`): `Caller` trait
(`call(messages, config) -> Result<String, ClientError>`); reqwest-blocking
default. pyo3 exposes `_testing.set_caller(fn)` for the test seam; caller
exceptions map to `ClientError` and are captured into envelopes.

**Guards** (`crates/llmalib-reliability/src/guard.rs`): `Guard` trait in
core (`validate(&Value, &Context) -> Vec<String>` + `as_any` for binding
recovery). Native rules use the Python-parity coercion helpers
(`py_str_repr`/`py_str`/`py_repr`/`truthy_str`/`python_float`) so error
messages stay byte-identical — they feed the reflection loop.

**Pipeline** (`crates/llmalib-pipeline/src/pipeline.rs`): `run_pipeline`
(routers as `Arc<dyn Fn(&TaskResult, Vec<Task>) -> Vec<Task>>`, stores,
tracer, decomposition); errors as typed `PipelineError` with Python-identical
messages.

### Code Patterns (both sides)

1. **Functional style**: operations are functions, not methods on core types
2. **Immutable data**: frozen dataclasses (Python) / plain structs (Rust)
3. **Schema validation**: every task declares its output shape upfront;
   values validated against the compiled JSON Schema rust-side and
   re-hydrated into pydantic instances Python-side
4. **Templating**: minijinja (Rust); undefined vars render empty
5. **Error envelopes**: never raise out of run functions except
   `ClientError`-family; `attempts` trace is always present
6. **Strict key contract**: serde field names of the bridge dicts
   (`TaskResult`, `Attempt`, `TaskTrace`) are hydration contracts — change
   them only together with `_bridge.py` / `_rust.pyi`

## Dependencies and Configuration

### Python (runtime: pydantic>=2.0, rich>=13)

Dev: `pytest`/`pytest-cov`, `ruff`, `mypy` (+ `_rust.pyi` pydantic plugin),
`maturin>=1.5` as build backend. **No httpx, no jinja2** — HTTP and
templating live entirely in Rust.

### Rust (see crates/*/Cargo.toml)

- `serde`, `serde_json` — serialization / JSON handling
- `reqwest` (blocking + json) — HTTP client (llmalib-core)
- `jsonschema` — output validation (llmalib-reliability)
- `minijinja` — prompt templating (llmalib-pipeline)
- `regex` — JSON extraction / tokenization
- `chrono` — tracer timestamps (llmalib-debug)
- `pyo3 0.23` (`abi3-py39`) — FFI (llmalib-py only)
- `thiserror` — error types (per crate), `uuid` — run ids (llmalib-core)

### Configuration Files

- **pyproject.toml**: maturin backend, abi3 py39, line length 88,
  ruff `E,W,F,I,B,C4,UP`, strict-ish mypy, pytest coverage by default
- **Cargo.toml**: workspace members `crates/*`, shared
  `[workspace.package]` (version 0.1.0, edition 2021), resolver 2
- **.gitignore**: python + rust ignores, coverage, lock files, `*.so`

## Important Gotchas and Patterns

### 1. Extension rebuild caching

`uv sync` will NOT rebuild `llmalib._rust` when only crate sources change
(version-based cache). Always pass `--reinstall-package llmalib` after
editing `crates/`.

### 2. Task Immutability

Tasks are frozen dataclasses — override defaults at construction:

```python
task = Task(
    name="my_task",
    prompt_template="Process: {{ input }}",
    output_schema=MySchema,
    model="mistral",   # default "llama3.2"
    max_retries=5,     # default 3 — TOTAL attempts, not extra retries
)
```

### 3. Context Variable Merging

- Successful results merge their fields into `ctx.vars` (plain names,
  last-write-wins); failed results do not.
- `update_context()` is explicit for standalone use and delegates the merge
  to `_rust.merge_value_fields`.

### 4. Token Budget / Retry / Guards

- Every task has `token_budget` (default 2048); trimming happens pre-retry.
- Retry is reflection-based: failed attempts append parse/guard errors as a
  user turn (see `build_reflection_message`).
- Guards are lightweight heuristics (no ML); custom guards are plain
  Python callables `(value, ctx) -> list[str]` and get pydantic values even
  through the Rust bridge.

### 5. JSON Extraction & Client

- Three-strategy extraction (direct parse → markdown fence → greedy
  outermost object/array). Never raises; returns parse envelopes.
- Requires an OpenAI-compatible local server (Ollama, llama.cpp, vLLM);
  default base_url `http://localhost:11434/v1`, timeout 120s.

### 6. Test Seam

- Python tests install fakes via `llmalib._testing`:
  `sequence_caller(*responses)` (repeats the last response, like
  `mock.return_value`), `raising_caller(exc)`, `inspecting_caller(resp,
  recorder)`. `tests/conftest.py` resets the global caller automatically —
  never patch HTTP/transport modules directly.

## Development Workflow

1. **Engine changes**: implement in the relevant crate first + crate tests;
   then extend the FFI (`crates/llmalib-py/src/api.rs`), `_rust.pyi`, and
   the Python adapter.
2. **Python-side changes**: keep it to dataclasses, marshalling,
   presentation, and delegation — rule logic belongs in crates.
3. **Testing**: run both suites; parity assertions in pytest must not be
   weakened.
4. **Documentation**: update `docs/` module docs and `docs/MIGRATION.md`
   when the boundary contract changes.
5. **Quality gate**: the full command set under "Code Quality" must pass
   before commits; CI (`.github/workflows/ci.yml`) runs the same gates.
6. **Breaking changes**: alpha software — expected; note divergences in
   `.plan.md` (local, untracked).

## Research Context

The library is built on specific research findings:

- Small models excel at narrow, focused tasks (arxiv:2506.02153)
- Context rotation degrades quality beyond few thousand tokens
- Structured output constraints prevent hallucinations effectively
- Iterative reflection improves reliability more than simple retry

See `docs/research.md` for detailed references.
