//! The `llmalib._rust` module surface: one pyfunction per Python-facing
//! entry point, marshalling through [`crate::convert`].

use llmalib_core::client::{call as http_call, ClientError as CoreClientError};
use llmalib_core::result::TaskResult;
use llmalib_debug::replay::{replay_task_with, ReplayError};
use llmalib_memory::context_window::{
    count_tokens_dyn, keep_first_and_last_sentence as rust_keep_first_and_last_sentence,
    trim_to_budget_with,
};
use llmalib_memory::examples::{
    format_examples_block as rust_format_examples_block, select_examples as rust_select_examples,
    FormattedExample,
};
use llmalib_memory::store::{bm25_rank_default, StoreEntry};
use llmalib_pipeline::decomposer::{decompose_with, DecompositionError};
use llmalib_pipeline::pipeline::{run_pipeline_with, PipelineError};
use llmalib_reliability::retry::{
    build_reflection_message as rust_build_reflection_message, run_with_retry_with,
};
use llmalib_reliability::validator::{
    extract_json as rust_extract_json, format_schema_hint as rust_format_schema_hint,
    parse_response as rust_parse_response,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;

use crate::bridges::{py_router, resolve_caller, set_global_caller};
use crate::classes::{
    load_trace, make_file_store, make_memory_store, make_tracer, PyStore, PyTracer,
};
use crate::convert::{
    config_from_py, context_from_py, py_to_value, result_to_py, spec_to_task, task_to_spec,
    value_to_py,
};
use llmalib_memory::store::Store;

// ---------------------------------------------------------------------------
// validator
// ---------------------------------------------------------------------------

/// Python `parse_response(raw, schema)` → `{"ok": bool, "value": …, "error": …}`.
#[pyfunction]
pub fn parse_response(
    py: Python<'_>,
    raw: String,
    schema: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let schema_value = py_to_value(py, schema)?;
    let parsed = rust_parse_response(&raw, &schema_value);
    let d = PyDict::new(py);
    d.set_item("ok", parsed.ok)?;
    match &parsed.value {
        Some(v) => d.set_item("value", value_to_py(py, v)?)?,
        None => d.set_item("value", py.None())?,
    }
    match &parsed.error {
        Some(e) => d.set_item("error", e.clone())?,
        None => d.set_item("error", py.None())?,
    }
    Ok(d.into_any().unbind())
}

/// Python `extract_json(raw)` → str or None.
#[pyfunction]
pub fn extract_json(raw: String) -> Option<String> {
    rust_extract_json(&raw)
}

/// Python `format_schema_hint(schema)` → str.
#[pyfunction]
pub fn format_schema_hint(py: Python<'_>, schema: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(rust_format_schema_hint(&py_to_value(py, schema)?))
}

/// Python `build_reflection_message(parse_error, guard_errors)` →
/// `{"role": "user", "content": …}` dict.
#[pyfunction]
#[pyo3(signature = (parse_error=None, guard_errors=None))]
pub fn build_reflection_message<'py>(
    py: Python<'py>,
    parse_error: Option<String>,
    guard_errors: Option<Vec<String>>,
) -> PyResult<PyObject> {
    let content =
        rust_build_reflection_message(parse_error.as_deref(), &guard_errors.unwrap_or_default());
    let d = pyo3::types::PyDict::new(py);
    d.set_item("role", "user")?;
    d.set_item("content", content)?;
    Ok(d.into_any().unbind())
}

// ---------------------------------------------------------------------------
// context window
// ---------------------------------------------------------------------------

fn py_tokenizer<'py>(
    py: Python<'py>,
    tokenizer: Option<Py<PyAny>>,
) -> PyResult<Option<impl Fn(&str) -> usize + 'py>> {
    match tokenizer {
        Some(f) => Ok(Some(move |text: &str| {
            f.bind(py)
                .call1((text,))
                .and_then(|o| o.extract::<usize>())
                .unwrap_or_else(|e| panic!("tokenizer raised: {e}"))
        })),
        None => Ok(None),
    }
}

/// Python `count_tokens(text, tokenizer=None)`.
#[pyfunction]
#[pyo3(signature = (text, tokenizer=None))]
pub fn count_tokens(py: Python<'_>, text: String, tokenizer: Option<Py<PyAny>>) -> PyResult<usize> {
    let t = py_tokenizer(py, tokenizer)?;
    Ok(count_tokens_dyn(
        &text,
        t.as_ref().map(|f| f as &dyn Fn(&str) -> usize),
    ))
}

/// Python `trim_to_budget(messages, budget, tokenizer=None)` — raises
/// `BudgetExceededError` when the protected messages alone exceed budget.
#[pyfunction]
#[pyo3(signature = (messages, budget, tokenizer=None))]
pub fn trim_to_budget(
    py: Python<'_>,
    messages: &Bound<'_, PyAny>,
    budget: usize,
    tokenizer: Option<Py<PyAny>>,
) -> PyResult<PyObject> {
    let msgs = py_to_value(py, messages)?;
    let t = py_tokenizer(py, tokenizer)?;
    let trimmed = trim_to_budget_with(
        msgs.as_array().cloned().unwrap_or_default(),
        budget,
        t.as_ref().map(|f| f as &dyn Fn(&str) -> usize),
    )
    .map_err(|e| crate::BudgetExceededError::new_err(e.to_string()))?;
    value_to_py(py, &Value::Array(trimmed))
}

/// Python `_keep_first_and_last_sentence(text)`.
#[pyfunction]
pub fn keep_first_and_last_sentence(text: String) -> String {
    rust_keep_first_and_last_sentence(&text)
}

// ---------------------------------------------------------------------------
// store helpers
// ---------------------------------------------------------------------------

/// Python `_tokenize(text)`.
#[pyfunction]
pub fn tokenize(text: String) -> Vec<String> {
    llmalib_memory::store::tokenize(&text)
}

/// Python `_bm25_rank(query, entries, top_k)` over entry dicts.
#[pyfunction]
pub fn bm25_rank(
    py: Python<'_>,
    query: String,
    entries: &Bound<'_, PyAny>,
    top_k: usize,
) -> PyResult<PyObject> {
    let list: Vec<PyObject> = entries.extract()?;
    let mut parsed = Vec::with_capacity(list.len());
    for e in &list {
        let v = py_to_value(py, e.bind(py))?;
        let entry: StoreEntry = serde_json::from_value(v)
            .map_err(|e| PyTypeError::new_err(format!("bad store entry: {e}")))?;
        parsed.push(entry);
    }
    let ranked = bm25_rank_default(&query, parsed, top_k);
    let out: Vec<PyObject> = ranked
        .iter()
        .map(|e| value_to_py(py, &serde_json::to_value(e).unwrap()))
        .collect::<PyResult<_>>()?;
    Ok(pyo3::types::PyList::new(py, out)?.into_any().unbind())
}

// ---------------------------------------------------------------------------
// examples
// ---------------------------------------------------------------------------

/// Python `select_examples(task, ctx_vars, store=None, max_examples=4)` →
/// list of `{"text": …, "source": …}` dicts.
#[pyfunction]
#[pyo3(signature = (task_spec, vars, store=None, max_examples=4))]
pub fn select_examples(
    py: Python<'_>,
    task_spec: &Bound<'_, PyAny>,
    vars: &Bound<'_, PyAny>,
    store: Option<Py<PyStore>>,
    max_examples: usize,
) -> PyResult<PyObject> {
    let task = spec_to_task(py, task_spec)?;
    let vars_value = py_to_value(py, vars)?;
    let vars_map = match vars_value {
        Value::Object(map) => map,
        _ => return Err(PyTypeError::new_err("vars must be a dict")),
    };
    let selected = match &store {
        Some(s) => {
            let mut store_ref = s.borrow_mut(py);
            rust_select_examples(&task, &vars_map, Some(store_ref.inner_mut()), max_examples)
        }
        None => rust_select_examples(&task, &vars_map, None, max_examples),
    };
    let list: Vec<PyObject> = selected
        .iter()
        .map(|ex| {
            value_to_py(
                py,
                &serde_json::json!({"text": &ex.text, "source": &ex.source}),
            )
        })
        .collect::<PyResult<_>>()?;
    Ok(pyo3::types::PyList::new(py, list)?.into_any().unbind())
}

/// Python `format_examples_block(examples)` over `{"text", "source"}` dicts.
#[pyfunction]
pub fn format_examples_block(py: Python<'_>, examples: &Bound<'_, PyAny>) -> PyResult<String> {
    let list: Vec<PyObject> = examples.extract()?;
    let mut formatted = Vec::with_capacity(list.len());
    for e in &list {
        let v = py_to_value(py, e.bind(py))?;
        formatted.push(FormattedExample {
            text: v
                .get("text")
                .and_then(|t| t.as_str())
                .unwrap_or_default()
                .to_string(),
            source: v
                .get("source")
                .and_then(|s| s.as_str())
                .unwrap_or_default()
                .to_string(),
        });
    }
    Ok(rust_format_examples_block(&formatted))
}

// ---------------------------------------------------------------------------
// client
// ---------------------------------------------------------------------------

/// Python `call(messages, config)` — real HTTP POST (usually bypassed via
/// `set_caller` in tests).
#[pyfunction]
pub fn call(
    py: Python<'_>,
    messages: &Bound<'_, PyAny>,
    config: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let msgs = py_to_value(py, messages)?;
    let d = config
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("config must be a dict"))?;
    let cfg = config_from_py(d)?;
    let msgs_vec = msgs.as_array().cloned().unwrap_or_default();
    // Release the GIL for the (potentially slow) HTTP round-trip. Without
    // this, reqwest's background I/O thread never progresses under the
    // interpreter (request connects but is never written).
    let out = py.allow_threads(|| http_call(msgs_vec, &cfg));
    out.map_err(|e| crate::ClientError::new_err(client_error_message(&e)))
}

pub fn client_error_message(e: &CoreClientError) -> String {
    match e.status_code {
        Some(code) => format!("{} [status {}]", e.message, code),
        None => e.message.clone(),
    }
}

// ---------------------------------------------------------------------------
// retry
// ---------------------------------------------------------------------------

/// Python `run_with_retry(task, messages, config, ctx=None)` → result dict.
/// Raises `ClientError` when the caller fails.
#[pyfunction]
#[pyo3(signature = (task_spec, messages, config, ctx=None, caller=None))]
pub fn run_with_retry(
    py: Python<'_>,
    task_spec: &Bound<'_, PyAny>,
    messages: &Bound<'_, PyAny>,
    config: &Bound<'_, PyAny>,
    ctx: Option<&Bound<'_, PyAny>>,
    caller: Option<Py<PyAny>>,
) -> PyResult<PyObject> {
    let task = spec_to_task(py, task_spec)?;
    let msgs = py_to_value(py, messages)?
        .as_array()
        .cloned()
        .unwrap_or_default();
    let d = config
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("config must be a dict"))?;
    let cfg = config_from_py(d)?;
    let ctx = context_from_py(py, ctx)?;
    let caller = resolve_caller(py, caller);

    let result: TaskResult = py
        .allow_threads(|| run_with_retry_with(&task, msgs, &cfg, &ctx, caller.as_ref()))
        .map_err(|e| crate::ClientError::new_err(client_error_message(&e)))?;
    result_to_py(py, &result)
}

// ---------------------------------------------------------------------------
// pipeline
// ---------------------------------------------------------------------------

/// Python `run_pipeline(tasks, ctx, router=None, store=None, tracer=None,
/// debug=False)` → `{"run_id", "results", "vars", "context_results"}`.
#[pyfunction]
#[pyo3(signature = (task_specs, vars=None, run_id=None, router=None, store=None, tracer=None, caller=None, debug=false))]
#[allow(clippy::too_many_arguments)]
pub fn run_pipeline(
    py: Python<'_>,
    task_specs: &Bound<'_, PyAny>,
    vars: Option<&Bound<'_, PyAny>>,
    run_id: Option<String>,
    router: Option<Py<PyAny>>,
    store: Option<Py<PyStore>>,
    tracer: Option<Py<PyTracer>>,
    caller: Option<Py<PyAny>>,
    debug: bool,
) -> PyResult<PyObject> {
    let spec_list: Vec<PyObject> = task_specs.extract()?;
    let mut tasks = Vec::with_capacity(spec_list.len());
    for s in &spec_list {
        tasks.push(spec_to_task(py, s.bind(py))?);
    }

    let mut ctx = match run_id {
        Some(id) => llmalib_core::context::Context::with_run_id(id),
        None => llmalib_core::context::Context::new(),
    };
    if let Some(vars) = vars {
        let v = py_to_value(py, vars)?;
        if let Value::Object(map) = v {
            ctx.vars = map;
        }
    }

    let router_fn = router.map(|f| py_router(py, f));
    let caller = resolve_caller(py, caller);

    // Tracer: the Rust tracer is cloned for the engine run and copied back
    // afterwards (Tracer is a plain append-record value).
    let mut engine_tracer = tracer.as_ref().map(|t| t.borrow(py).inner.clone());

    // Store: the backend is extracted from the pyclass so it can be used
    // while the GIL is released; a Drop guard restores it even on panic.
    // Everything engine-visible is moved into the closure; nothing GIL-bound
    // (Python tokens, borrows) crosses it — bridges re-acquire the GIL
    // themselves when they call back into Python.
    let results = {
        let mut store_guard = store.map(|s| StoreLease::take(py, s));
        py.allow_threads(move || {
            let store_ref: Option<&mut dyn Store> = match &mut store_guard {
                Some(lease) => match lease.backend.as_mut() {
                    Some(b) => Some(&mut **b),
                    None => None,
                },
                None => None,
            };
            let tracer_ref = engine_tracer.as_mut();
            run_pipeline_with(
                tasks,
                &mut ctx,
                router_fn,
                store_ref,
                tracer_ref,
                caller.as_ref(),
                debug,
            )
            .map(|results| (results, ctx, engine_tracer))
            .map_err(map_pipeline_error)
        })?
    };

    let (results, ctx, engine_tracer) = results;
    if let (Some(t), Some(inner)) = (tracer, engine_tracer) {
        t.borrow_mut(py).inner = inner;
    }

    let d = PyDict::new(py);
    d.set_item("run_id", ctx.run_id.clone())?;
    let result_list: Vec<PyObject> = results
        .iter()
        .map(|r| result_to_py(py, r))
        .collect::<PyResult<_>>()?;
    d.set_item("results", result_list)?;
    d.set_item("vars", value_to_py(py, &Value::Object(ctx.vars.clone()))?)?;
    d.set_item(
        "context_results",
        value_to_py(py, &Value::Object(ctx.results.clone()))?,
    )?;
    Ok(d.into_any().unbind())
}

fn map_pipeline_error(e: PipelineError) -> PyErr {
    match e {
        PipelineError::Client(c) => crate::ClientError::new_err(client_error_message(&c)),
        PipelineError::Budget(b) => crate::BudgetExceededError::new_err(b.to_string()),
    }
}

/// Extracts the backend `Box<dyn Store>` out of a `PyStore` pyclass for the
/// duration of a `py.allow_threads` engine run, and restores it afterwards —
/// including on panic (bridges panic when Python exceptions propagate).
struct StoreLease {
    pyclass: Py<PyStore>,
    backend: Option<Box<dyn Store>>,
}

impl StoreLease {
    fn take(py: Python<'_>, store: Py<PyStore>) -> Self {
        let backend = {
            let mut borrowed = store.bind(py).borrow_mut();
            borrowed.replace_inner(Box::new(TakenStore))
        };
        Self {
            pyclass: store,
            backend: Some(backend),
        }
    }
}

impl Drop for StoreLease {
    fn drop(&mut self) {
        if let Some(backend) = self.backend.take() {
            Python::with_gil(|py| {
                let mut borrowed = self.pyclass.bind(py).borrow_mut();
                borrowed.replace_inner(backend);
            });
        }
    }
}

/// Placeholder left in place of the extracted backend while the engine runs.
/// Python-side access during a released-GIL engine call is a programming
/// error (the caller's thread holds no GIL), so fail loudly.
struct TakenStore;

impl Store for TakenStore {
    fn set(
        &mut self,
        _key: String,
        _value: serde_json::Value,
        _score: f64,
        _ttl: Option<std::time::Duration>,
    ) {
        panic!("llmalib store backend is in use by an engine run");
    }
    fn get(&self, _key: &str) -> Option<llmalib_memory::store::StoreEntry> {
        panic!("llmalib store backend is in use by an engine run");
    }
    fn delete(&mut self, _key: &str) {
        panic!("llmalib store backend is in use by an engine run");
    }
    fn list(&mut self, _prefix: &str) -> Vec<String> {
        panic!("llmalib store backend is in use by an engine run");
    }
    fn get_relevant(
        &mut self,
        _query: &str,
        _top_k: usize,
    ) -> Vec<llmalib_memory::store::StoreEntry> {
        panic!("llmalib store backend is in use by an engine run");
    }
}

// ---------------------------------------------------------------------------
// decomposer
// ---------------------------------------------------------------------------

/// Python `decompose(prompt, task_registry, config, max_tasks=8)` → list of
/// task specs. Raises `DecompositionError` on failure.
#[pyfunction]
#[pyo3(signature = (prompt, registry, config, max_tasks=8, caller=None))]
pub fn decompose(
    py: Python<'_>,
    prompt: String,
    registry: &Bound<'_, PyDict>,
    config: &Bound<'_, PyAny>,
    max_tasks: usize,
    caller: Option<Py<PyAny>>,
) -> PyResult<PyObject> {
    let d = config
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("config must be a dict"))?;
    let cfg = config_from_py(d)?;

    let mut reg: Vec<(String, llmalib_core::task::Task)> = Vec::new();
    for (name, spec) in registry.iter() {
        let name: String = name.extract()?;
        reg.push((name, spec_to_task(py, &spec)?));
    }

    let caller = resolve_caller(py, caller);

    let tasks = py
        .allow_threads(|| decompose_with(&prompt, &reg, &cfg, max_tasks, caller.as_ref()))
        .map_err(map_decomposition_error)?;
    let specs: Vec<PyObject> = tasks
        .iter()
        .map(|t| task_to_spec(py, t))
        .collect::<PyResult<_>>()?;
    Ok(pyo3::types::PyList::new(py, specs)?.into_any().unbind())
}

fn map_decomposition_error(e: DecompositionError) -> PyErr {
    match e {
        DecompositionError::Client(c) => crate::ClientError::new_err(client_error_message(&c)),
        other => crate::DecompositionError::new_err(other.to_string()),
    }
}

// ---------------------------------------------------------------------------
// replay
// ---------------------------------------------------------------------------

/// Python `replay_task(tracer, task_name, task, attempt_number=1,
/// model=None, base_url=None, prompt_override=None, schema_override=None,
/// diff=False)` → result dict.
#[pyfunction]
#[pyo3(signature = (tracer, task_name, task_spec, attempt_number=1, model=None, base_url=None, prompt_override=None, schema_override=None, diff=false, caller=None))]
#[allow(clippy::too_many_arguments)]
pub fn replay_task(
    py: Python<'_>,
    tracer: &PyTracer,
    task_name: String,
    task_spec: &Bound<'_, PyAny>,
    attempt_number: usize,
    model: Option<String>,
    base_url: Option<String>,
    prompt_override: Option<String>,
    schema_override: Option<&Bound<'_, PyAny>>,
    diff: bool,
    caller: Option<Py<PyAny>>,
) -> PyResult<PyObject> {
    let task = spec_to_task(py, task_spec)?;
    let schema = match schema_override {
        Some(s) => Some(py_to_value(py, s)?),
        None => None,
    };
    let caller = resolve_caller(py, caller);
    // replay only reads the recorded attempts — an owned clone crosses the
    // GIL release safely.
    let engine_tracer = tracer.inner.clone();

    let result = py
        .allow_threads(|| {
            replay_task_with(
                &engine_tracer,
                &task_name,
                &task,
                attempt_number,
                model.as_deref(),
                base_url.as_deref(),
                prompt_override.as_deref(),
                schema,
                diff,
                caller.as_ref(),
            )
        })
        .map_err(|e| PyRuntimeError::new_err(replay_error_message(&e)))?;
    result_to_py(py, &result)
}

fn replay_error_message(e: &ReplayError) -> String {
    e.to_string()
}

// ---------------------------------------------------------------------------
// test seam
// ---------------------------------------------------------------------------

/// `llmalib._testing.set_caller(fn)` — install the fake transport used by
/// the Python test suite instead of patching httpx.
#[pyfunction]
#[pyo3(signature = (f=None))]
pub fn set_caller(f: Option<Py<PyAny>>) {
    set_global_caller(f);
}

// ---------------------------------------------------------------------------
// registration
// ---------------------------------------------------------------------------

/// Register every function/class/exception on the `_rust` module.
pub fn register(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_response, m)?)?;
    m.add_function(wrap_pyfunction!(extract_json, m)?)?;
    m.add_function(wrap_pyfunction!(format_schema_hint, m)?)?;
    m.add_function(wrap_pyfunction!(build_reflection_message, m)?)?;
    m.add_function(wrap_pyfunction!(count_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(trim_to_budget, m)?)?;
    m.add_function(wrap_pyfunction!(keep_first_and_last_sentence, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(bm25_rank, m)?)?;
    m.add_function(wrap_pyfunction!(select_examples, m)?)?;
    m.add_function(wrap_pyfunction!(format_examples_block, m)?)?;
    m.add_function(wrap_pyfunction!(call, m)?)?;
    m.add_function(wrap_pyfunction!(run_with_retry, m)?)?;
    m.add_function(wrap_pyfunction!(run_pipeline, m)?)?;
    m.add_function(wrap_pyfunction!(decompose, m)?)?;
    m.add_function(wrap_pyfunction!(replay_task, m)?)?;
    m.add_function(wrap_pyfunction!(set_caller, m)?)?;
    m.add_function(wrap_pyfunction!(make_memory_store, m)?)?;
    m.add_function(wrap_pyfunction!(make_file_store, m)?)?;
    m.add_function(wrap_pyfunction!(make_tracer, m)?)?;
    m.add_function(wrap_pyfunction!(load_trace, m)?)?;
    m.add_class::<PyStore>()?;
    m.add_class::<PyTracer>()?;
    m.add_class::<crate::classes::ContextView>()?;
    m.add("StoreError", py.get_type::<crate::classes::StoreError_>())?;
    m.add("ClientError", py.get_type::<crate::ClientError>())?;
    m.add(
        "BudgetExceededError",
        py.get_type::<crate::BudgetExceededError>(),
    )?;
    m.add(
        "DecompositionError",
        py.get_type::<crate::DecompositionError>(),
    )?;
    Ok(())
}
