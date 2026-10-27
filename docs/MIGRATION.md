# llmalib Rust Migration

This document describes the Rust-port architecture, the Python↔Rust boundary
contract, and how to extend the library post-migration.

The migration is **complete**: the Python package (`llmalib/`) is a thin
wrapper over the `llmalib._rust` pyo3 extension module, and every rule /
engine behavior lives in the Rust crates under `crates/`. The public Python
API — module paths, signatures, keyword arguments, defaults, error strings,
and result types — is unchanged from the original pure-Python library.

---

## 1. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Python (llmalib/)                                           │
│   public API: dataclasses, factories, value re-hydration     │
│   + thin adapters delegating to _rust                        │
└──────────────────────────┬──────────────────────────────────┘
                           │ llmalib._rust (pyo3, maturin)
┌──────────────────────────▼──────────────────────────────────┐
│ crates/llmalib-py/     ← FFI/marshalling cone only           │
│   api.rs       one pyfunction per Python entry point        │
│   convert.rs   python dict / JSON-Schema ↔ Rust Task/Value  │
│   bridges.rs   PyGuard / PyRouter / PyCaller / set_caller   │
│   classes.rs   PyStore / PyTracer / ContextView / NativeGuard│
└──────────┬───────────┬───────────┬───────────┬──────────────┘
           │           │           │           │
    llmalib-core  llmalib-memory  llmalib-reliability  llmalib-debug
    Task/Result/  context_window  validator/guards    tracer/replay
    Context/      (budget)        retry               (Rust-native)
    client/Guard  store (BM25)                        (+inspector data)
           │
    llmalib-pipeline  run_pipeline (routers, decomposer, replay)
```

### Crates

| Crate | Owns | Python adapter |
|---|---|---|
| `llmalib-core` | `Task`, `TaskResult`/`Attempt`, `Context`, HTTP `Caller` trait + reqwest default, `Guard` trait | `core/task.py`, `core/result.py`, `core/context.py`, `core/client.py` (dataclasses + factories; presentation only) |
| `llmalib-memory` | token budgeting (`count_tokens*`, `trim_to_budget*`, `keep_first_and_last_sentence`), KV stores (`Store`, `bm25_rank`, TTL), few-shot `select_examples` | `memory/context_window.py`, `memory/store.py`, `memory/examples.py` |
| `llmalib-reliability` | JSON-Schema validation (`jsonschema` crate), `_extract_json` (3 strategies), reflection retry, native guard rules (field_in_set / float_in_range / max_length / grounding) | `reliability/validator.py`, `reliability/retry.py`, `reliability/guards.py` (factories return `NativeGuard`) |
| `llmalib-debug` | Rust-native `Tracer` (record/summary/to_dict/to_file/load_trace), attempt/trace serde | `debug/tracer.py` (`_rust.PyTracer` wrapper), `debug/replay.py` (engine driver), `debug/inspector.py` (presentation only) |
| `llmalib-pipeline` | `run_pipeline`, routing decisions, decomposer, replay engine | `pipeline/pipeline.py`, `pipeline/router.py` (decision delegation + task-list plumbing), `pipeline/decomposer.py` |
| `llmalib-py` | the only pyo3 boundary | nothing (private FFI) |

### What still lives in Python (by design)

- **Type surface**: frozen `Task`/`Attempt`/`Result` dataclasses, pydantic
  schemas, `Example` — these are the library's declared types.
- **Object-graph marshalling** (`_bridge.py` hydration of `Task`↔spec dicts,
  pydantic value re-hydration, guard/router adapters).
- **Presentation**: `debug/inspector.py` (rich/plain rendering).
- **Delegating factories**: `guards.py`, `router.py` return Python callables
  that immediately delegate decisions into `_rust` (Router factories keep
  Python task-list plumbing because the public contract passes `Task`
  *objects* through and tests assert identity, not equality).

---

## 2. Boundary contract

### Marshalling flow

```
Python Task ──spec_from_task──▶ flat spec dict
  ├─ output_schema   = schema.model_json_schema()   (pydantic → JSON Schema, Python-side)
  ├─ guards          = adapt_guard(g, schema)       (dict→pydantic re-wrap, marked
  │                                                 `_llmalib_original` so round-trips
  │                                                 never double-wrap)
  └─ examples        = example_to_dict(ex)          (pydantic output → dict)

spec dict ──spec_to_task (convert.rs)──▶ llmalib_core::Task
  guards = PyGuard(Arc<dyn Caller-free wrapper around the Python callable>)
```

- **Results flow back** as `TaskResult` serde dicts:
  `{task_name, ok, value, error, attempts: usize, trace: [Attempt…]}`
  with `Attempt` = `{attempt_number, rendered_prompt, raw_response,
  parse_error, guard_errors, duration_ms}`. The Python adapter re-hydrates
  `value` into the task's original pydantic class via `model_validate`,
  so `result.value.field` keeps working — never assume the value side is a
  raw `dict` in user code (e.g. `replay`, `tracer`, `inspector`).
- **Validation authority**: Rust compiles the JSON Schema (`jsonschema`
  crate) and validates the model's parsed output; the Python-side pydantic
  re-hydration is a second independent check. Divergent semantics are
  listed in `.plan.md` (Phase 4 notes).
- **Context**: Rust `Context{vars, results, run_id}`; only *successful*
  results merge fields into `vars` under plain field names
  (last-write-wins). Python `update_context` delegates the merge to
  `_rust.merge_value_fields` and owns `ctx.results[task_name] = result`
  (python object bookkeeping).
- **Templating**: minijinja renders prompts; undefined variables render
  empty (matching Jinja2 legacy behavior). The system prompt is generated
  rust-side from `format_schema_hint` (alphabetical key order).
- **Guards across the bridge**: the engine hands a plain dict + a
  `ContextView` snapshot to Python guards; Python adapters re-create the
  pydantic instance first, so *user* guards always see pydantic values, exactly
  like the original API. Rust-native guards (`NativeGuard`) keep their
  `Arc<dyn Guard>` on the Rust side and skip the Python hop entirely.
- **Caller seam**: `Rust Caller` trait; the default implementation is
  reqwest-blocking HTTP. `_testing.set_caller(fn)` installs a
  Python-callable caller (PyCaller) used by the entire pytest suite instead
  of patching HTTP. Caller exceptions map onto `ClientError` and are
  captured into the result envelope; Python exceptions inside guards /
  routers propagate as `pyo3_runtime.PanicException` (raise-through, as
  before).
- **ContextView snapshot**: guards see vars/results recorded *before* the
  current task (pre-existing vs newly-merged results never appear within
  the guard run).

### Public invariants (do not break)

1. Zero-arg or keyword-compatible constructors, defaults, and error strings
   must match the pre-port Python exactly.
2. `max_retries` = **total attempts** (a reflection-based retry mirrors the
   original semantics, not naive re-call).
3. Never raise out of `run_*` except `ClientError`-family — results always
   come back as envelopes; the trace (`attempts`) is always present.
4. Every new engine behavior needs a Rust-side unit/integration test first
   (the Python tests are behavioral parity tests, not the engine's
   spec-tests).

---

## 3. Extensibility

### Adding a native guard rule

1. Implement the rule in `crates/llmalib-reliability/src/guard.rs` as an
   `Arc<dyn CoreGuard>` factory, using the crate's Python-parity helpers
   (`py_str_repr`, `py_repr`, `py_str`, `truthy_str`, `python_float`,
   `parse_python_float`) so messages stay byte-identical to the Python
   surface.
2. Add a crate-level unit test (assert the exact error string).
3. Expose a `#[pyfunction] guard_<name>` in `crates/llmalib-py/src/api.rs`
   returning `NativeGuard { inner: rule }` and add it to `register`.
4. Extend `llmalib/_rust.pyi` and add a delegating factory in
   `llmalib/reliability/guards.py`.

### Adding a new pipeline behavior

Engine semantics belong in `llmalib-pipeline` as Rust functions *first*;
then expose the decision over the FFI (see `_rust.route_stop_on_failure`
/ `route_branch_key` for the pattern: decisions return primitives/keys,
Python keeps object-graph plumbing) and re-hydrate results Python-side.

### Adding a crate

Workspace layout lives in `crates/` with a shared `[workspace.package]`
(edition 2021, version 0.1.0). New crates must pass
`cargo fmt --check`, `cargo clippy --workspace -- -D warnings`, and
`cargo test --workspace`; their behavior must be exercised by the Python
pytest suite through the FFI surface in `llmalib-py`.

---

## 4. Development workflow

```bash
uv sync --extra dev                       # py env + build ext via maturin
uv sync --extra dev --reinstall-package llmalib  # force rebuild after crate edits

cargo fmt --check
cargo clippy --workspace -- -D warnings
cargo test --workspace                    # engine test suite
uv run pytest tests/ -q -o addopts=""     # parity test suite (115)
uv run ruff check llmalib/                # python lint
uv run mypy llmalib/                      # strict typecheck
scripts/cleanup.sh                        # caches + stray artifacts
```

CI runs the same gates — see `.github/workflows/ci.yml`.
