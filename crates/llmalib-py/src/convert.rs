//! Marshalling between Python objects and the Rust crate types.
//!
//! Everything crossing the `llmalib._rust` boundary goes through here:
//! JSON values, task specs, client configs, results.
//!
//! Task specs are plain dicts so the Phase 4 Python adapters can build them
//! straight from the `Task` dataclass fields:
//!
//! ```python
//! {
//!     "name": ..., "prompt_template": ..., "output_schema": {...},
//!     "model": ..., "base_url": ..., "temperature": ..., "max_tokens": ...,
//!     "timeout": ..., "max_retries": ..., "token_budget": ...,
//!     "system_prompt": ... or None,
//!     "guards": [callable, ...],
//!     "examples": [{"input_text": ..., "output": {...}, "description": ...}],
//! }
//! ```

use llmalib_core::client::ClientConfig;
use llmalib_core::context::Context;
use llmalib_core::result::TaskResult;
use llmalib_core::task::{Example, Task};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;

use crate::bridges::{guard_to_py, PyGuard, PyGuardRef};

// ---------------------------------------------------------------------------
// JSON values
// ---------------------------------------------------------------------------

/// `serde_json::Value` → Python object via the `json` module (round-trips
/// exactly: ints stay ints, floats stay floats, `null` becomes `None`).
pub fn value_to_py(py: Python<'_>, v: &Value) -> PyResult<PyObject> {
    let s = serde_json::to_string(v)
        .map_err(|e| PyErr::new::<PyTypeError, _>(format!("value not serializable: {e}")))?;
    let json = py.import("json")?;
    Ok(json.getattr("loads")?.call1((s,))?.into_any().unbind())
}

/// Python object → `serde_json::Value` via the `json` module. Callables and
/// other non-JSON types raise (extract them as `Py<PyAny>` first).
pub fn py_to_value(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    let json = py.import("json")?;
    let s: String = json.getattr("dumps")?.call1((obj,))?.extract()?;
    serde_json::from_str(&s)
        .map_err(|e| PyErr::new::<PyTypeError, _>(format!("object not JSON-compatible: {e}")))
}

// ---------------------------------------------------------------------------
// Task specs
// ---------------------------------------------------------------------------

fn get_item<'py>(
    d: &'py Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Option<Bound<'py, pyo3::types::PyAny>>> {
    d.get_item(key)
        .map_err(|e| PyErr::new::<PyTypeError, _>(format!("bad task spec key '{key}': {e}")))
}

fn opt_str(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    match get_item(d, key)? {
        Some(v) if !v.is_none() => v
            .extract()
            .map(Some)
            .map_err(|_| PyErr::new::<PyTypeError, _>(format!("spec['{key}'] must be a str"))),
        _ => Ok(None),
    }
}

fn opt_f64(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<f64>> {
    match get_item(d, key)? {
        Some(v) if !v.is_none() => v
            .extract()
            .map(Some)
            .map_err(|_| PyErr::new::<PyTypeError, _>(format!("spec['{key}'] must be a number"))),
        _ => Ok(None),
    }
}

fn opt_u32(d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<u32>> {
    match opt_f64(d, key)? {
        Some(v) => Ok(Some(v as u32)),
        None => Ok(None),
    }
}

fn opt_value(py: Python<'_>, d: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<Value>> {
    match get_item(d, key)? {
        Some(v) if !v.is_none() => Ok(Some(py_to_value(py, &v)?)),
        _ => Ok(None),
    }
}

/// Task spec dict → Rust [`Task`]. Guard entries may be Python callables
/// (wrapped in [`PyGuard`]) or [`PyGuardRef`] handles produced by a previous
/// [`task_to_spec`] round-trip.
pub fn spec_to_task(py: Python<'_>, spec: &Bound<'_, PyAny>) -> PyResult<Task> {
    let d = spec
        .downcast::<PyDict>()
        .map_err(|_| PyErr::new::<PyTypeError, _>("task spec must be a dict"))?;

    let name = opt_str(d, "name")?.ok_or_else(|| missing("name"))?;
    let prompt_template =
        opt_str(d, "prompt_template")?.ok_or_else(|| missing("prompt_template"))?;
    let output_schema =
        opt_value(py, d, "output_schema")?.ok_or_else(|| missing("output_schema"))?;

    let mut builder = Task::builder(name, prompt_template, output_schema);

    if let Some(v) = opt_str(d, "model")? {
        builder = builder.model(v);
    }
    if let Some(v) = opt_str(d, "base_url")? {
        builder = builder.base_url(v);
    }
    if let Some(v) = opt_f64(d, "temperature")? {
        builder = builder.temperature(v as f32);
    }
    if let Some(v) = opt_u32(d, "max_tokens")? {
        builder = builder.max_tokens(v);
    }
    if let Some(v) = opt_f64(d, "timeout")? {
        builder = builder.timeout(v as f32);
    }
    if let Some(v) = opt_u32(d, "max_retries")? {
        builder = builder.max_retries(v);
    }
    if let Some(v) = opt_u32(d, "token_budget")? {
        builder = builder.token_budget(v);
    }
    if let Some(v) = opt_str(d, "system_prompt")? {
        builder = builder.system_prompt(Some(v));
    }

    // Guards: callables or round-tripped PyGuardRef handles.
    if let Some(guards) = get_item(d, "guards")? {
        if !guards.is_none() {
            let list: Vec<PyObject> = guards.extract().map_err(|_| {
                PyErr::new::<PyTypeError, _>("spec['guards'] must be a list of callables")
            })?;
            for g in list {
                let bound = g.bind(py);
                if let Ok(handle) = bound.extract::<PyRef<'_, PyGuardRef>>() {
                    builder = builder.guard(Arc::clone(&handle.guard));
                } else {
                    builder = builder.guard(Arc::new(PyGuard::new(g)));
                }
            }
        }
    }

    // Examples: [{input_text, output, description}]
    if let Some(examples) = opt_value(py, d, "examples")? {
        let mut parsed = Vec::new();
        for ex in examples.as_array().cloned().unwrap_or_default() {
            parsed.push(example_from_value(&ex)?);
        }
        builder = builder.examples(parsed);
    }

    Ok(builder.build())
}

fn example_from_value(ex: &Value) -> PyResult<Example> {
    let input_text = ex
        .get("input_text")
        .and_then(|v| v.as_str())
        .ok_or_else(|| PyErr::new::<PyTypeError, _>("example missing 'input_text'"))?
        .to_string();
    let output = ex.get("output").cloned().unwrap_or(Value::Null);
    let description = ex
        .get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    Ok(Example {
        input_text,
        output,
        description,
    })
}

fn missing(field: &str) -> PyErr {
    PyErr::new::<PyTypeError, _>(format!("task spec missing required field '{field}'"))
}

/// Rust [`Task`] → spec dict. Guards that were originally Python callables
/// come back as those same callables; Rust-native guards are wrapped in
/// opaque [`PyGuardRef`] handles that [`spec_to_task`] understands.
pub fn task_to_spec(py: Python<'_>, task: &Task) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("name", task.name.clone())?;
    d.set_item("prompt_template", task.prompt_template.clone())?;
    d.set_item("output_schema", value_to_py(py, &task.output_schema)?)?;
    d.set_item("model", task.model.clone())?;
    d.set_item("base_url", task.base_url.clone())?;
    d.set_item("temperature", f64::from(task.temperature))?;
    d.set_item("max_tokens", task.max_tokens)?;
    d.set_item("timeout", f64::from(task.timeout))?;
    d.set_item("max_retries", task.max_retries)?;
    d.set_item("token_budget", task.token_budget)?;
    match &task.system_prompt {
        Some(p) => d.set_item("system_prompt", p.clone())?,
        None => d.set_item("system_prompt", py.None())?,
    }

    let guards: Vec<PyObject> = task
        .guards
        .iter()
        .map(|g| guard_to_py(py, g))
        .collect::<PyResult<_>>()?;
    d.set_item("guards", guards)?;

    let examples: Vec<Value> = task
        .examples
        .iter()
        .map(|ex| {
            serde_json::json!({
                "input_text": ex.input_text,
                "output": ex.output,
                "description": ex.description,
            })
        })
        .collect();
    d.set_item("examples", value_to_py(py, &Value::Array(examples))?)?;

    Ok(d.into_any().unbind())
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// Build a [`Context`] from the optional `{vars, run_id, results}` dict the
/// Python adapter passes.
pub fn context_from_py(py: Python<'_>, ctx: Option<&Bound<'_, PyAny>>) -> PyResult<Context> {
    match ctx {
        Some(obj) => {
            let d = obj
                .downcast::<PyDict>()
                .map_err(|_| PyErr::new::<PyTypeError, _>("ctx must be a dict"))?;
            let run_id = opt_str(d, "run_id")?;
            let mut c = match run_id {
                Some(id) => Context::with_run_id(id),
                None => Context::new(),
            };
            if let Some(Value::Object(map)) = opt_value(py, d, "vars")? {
                c.vars = map;
            }
            if let Some(Value::Object(map)) = opt_value(py, d, "results")? {
                c.results = map;
            }
            Ok(c)
        }
        None => Ok(Context::new()),
    }
}

// ---------------------------------------------------------------------------
// ClientConfig
// ---------------------------------------------------------------------------

/// ClientConfig dict → [`ClientConfig`] (missing keys fall back to defaults).
pub fn config_from_py(d: &Bound<'_, PyDict>) -> PyResult<ClientConfig> {
    let mut config = ClientConfig::default();
    if let Some(v) = opt_str(d, "base_url")? {
        config.base_url = v;
    }
    if let Some(v) = opt_str(d, "model")? {
        config.model = v;
    }
    if let Some(v) = opt_f64(d, "temperature")? {
        config.temperature = v as f32;
    }
    if let Some(v) = opt_f64(d, "timeout")? {
        config.timeout = Duration::from_secs_f64(v);
    }
    if let Some(v) = opt_u32(d, "max_tokens")? {
        config.max_tokens = v as usize;
    }
    Ok(config)
}

/// [`ClientConfig`] → dict (handed to Python caller callables).
pub fn config_to_py(py: Python<'_>, config: &ClientConfig) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    d.set_item("base_url", config.base_url.clone())?;
    d.set_item("model", config.model.clone())?;
    d.set_item("temperature", f64::from(config.temperature))?;
    d.set_item("timeout", config.timeout.as_secs_f64())?;
    d.set_item("max_tokens", config.max_tokens)?;
    Ok(d.into_any().unbind())
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// [`TaskResult`] → dict (task_name, ok, value, error, attempts, trace).
pub fn result_to_py(py: Python<'_>, result: &TaskResult) -> PyResult<PyObject> {
    value_to_py(py, &serde_json::to_value(result).map_err(map_serde)?)
}

/// dict → [`TaskResult`] (tracer record seam).
pub fn result_from_py(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<TaskResult> {
    let v = py_to_value(py, obj)?;
    serde_json::from_value(v).map_err(map_serde)
}

fn map_serde(e: serde_json::Error) -> PyErr {
    PyErr::new::<PyTypeError, _>(format!("result serialization error: {e}"))
}
