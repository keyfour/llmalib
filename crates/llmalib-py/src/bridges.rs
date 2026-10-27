//! Bridges: Python callables implementing Rust traits.
//!
//! - [`PyGuard`] — Python guard callable → core `Guard` trait
//! - [`PyGuardRef`] — opaque handle wrapping a Rust-native guard (so task
//!   specs round-trip even when a guard has no Python side)
//! - [`PyRouter`] — Python router callable → pipeline `Router`
//! - [`PyCaller`] — Python caller callable → core `Caller` trait, plus the
//!   module-global caller installed via `llmalib._testing.set_caller`
//!
//! A Python exception raised inside a bridge callable propagates by panicking
//! out of the trait method; pyo3 converts the unwinding panic into a
//! `pyo3_runtime.PanicException` at the function boundary, which matches
//! Python's raise-through behaviour.

use llmalib_core::client::{Caller, ClientConfig, ClientError as CoreClientError, HttpCaller};
use llmalib_core::context::Context;
use llmalib_core::guard::Guard as CoreGuard;
use llmalib_core::result::TaskResult;
use llmalib_core::task::Task;
use pyo3::prelude::*;
use serde_json::Value;
use std::any::Any;
use std::sync::{Arc, RwLock};

use crate::classes::context_view;
use crate::convert::{config_to_py, result_to_py, task_to_spec, value_to_py};

// ---------------------------------------------------------------------------
// Guards
// ---------------------------------------------------------------------------

/// A Python guard callable `(value: dict, ctx) -> list[str]`.
pub struct PyGuard {
    f: Py<PyAny>,
}

impl PyGuard {
    pub fn new(f: Py<PyAny>) -> Self {
        Self { f }
    }
}

impl CoreGuard for PyGuard {
    fn validate(&self, value: &Value, ctx: &Context) -> Vec<String> {
        Python::with_gil(|py| {
            let v = value_to_py(py, value).unwrap_or_else(|e| panic!("guard value marshal: {e}"));
            let view = context_view(py, ctx).unwrap_or_else(|e| panic!("guard ctx marshal: {e}"));
            match self.f.bind(py).call1((v, view)) {
                Ok(out) => out
                    .extract::<Vec<String>>()
                    .unwrap_or_else(|e| panic!("guard must return list[str], got {e}")),
                Err(e) => panic!("guard raised: {e}"),
            }
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Opaque pyclass handle around a Rust-native guard, produced when a Task
/// carrying Rust guards is marshalled back to a spec dict.
#[pyclass]
pub struct PyGuardRef {
    pub guard: Arc<dyn CoreGuard>,
}

/// Recover a guard's Python representation: the original callable for
/// [`PyGuard`]s, or a [`PyGuardRef`] handle for Rust-native guards.
pub fn guard_to_py(py: Python<'_>, guard: &Arc<dyn CoreGuard>) -> PyResult<PyObject> {
    if let Some(pg) = guard.as_ref().as_any().downcast_ref::<PyGuard>() {
        return Ok(pg.f.clone_ref(py));
    }
    Ok(Py::new(
        py,
        PyGuardRef {
            guard: Arc::clone(guard),
        },
    )?
    .into_any())
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Wrap a Python router `(result: dict, remaining: list[spec]) -> list[spec]`
/// as the pipeline `Router` type. Specs round-trip through
/// [`crate::convert::spec_to_task`].
pub fn py_router(_py: Python<'_>, f: Py<PyAny>) -> llmalib_pipeline::Router {
    Arc::new(
        move |result: &TaskResult, remaining: Vec<Task>| -> Vec<Task> {
            Python::with_gil(|py| {
                let rdict = result_to_py(py, result)
                    .unwrap_or_else(|e| panic!("router result marshal: {e}"));
                let specs = remaining
                    .iter()
                    .map(|t| task_to_spec(py, t))
                    .collect::<PyResult<Vec<_>>>()
                    .unwrap_or_else(|e| panic!("router task marshal: {e}"));
                let out = f
                    .bind(py)
                    .call1((rdict, specs))
                    .unwrap_or_else(|e| panic!("router raised: {e}"));
                let returned: Vec<PyObject> = out
                    .extract()
                    .unwrap_or_else(|e| panic!("router must return list of task specs: {e}"));
                returned
                    .iter()
                    .map(|s| crate::convert::spec_to_task(py, s.bind(py)))
                    .collect::<PyResult<Vec<_>>>()
                    .unwrap_or_else(|e| panic!("router returned a bad task spec: {e}"))
            })
        },
    )
}

// ---------------------------------------------------------------------------
// Caller
// ---------------------------------------------------------------------------

/// A Python caller callable `(messages: list[dict], config: dict) -> str`.
pub struct PyCaller {
    f: Py<PyAny>,
}

impl Caller for PyCaller {
    fn call(&self, messages: Vec<Value>, config: &ClientConfig) -> Result<String, CoreClientError> {
        Python::with_gil(|py| {
            let msgs = value_to_py(py, &Value::Array(messages)).map_err(|e| CoreClientError {
                message: format!("caller message marshal: {e}"),
                status_code: None,
            })?;
            let cfg = config_to_py(py, config).map_err(|e| CoreClientError {
                message: format!("caller config marshal: {e}"),
                status_code: None,
            })?;
            let out = self
                .f
                .bind(py)
                .call1((msgs, cfg))
                .map_err(|e| CoreClientError {
                    message: format!("caller raised: {e}"),
                    status_code: None,
                })?;
            out.extract::<String>().map_err(|e| CoreClientError {
                message: format!("caller must return str: {e}"),
                status_code: None,
            })
        })
    }
}

/// Module-global caller installed by `llmalib._testing.set_caller` (the
/// Python-side replacement for patching `httpx.post`).
static GLOBAL_CALLER: RwLock<Option<Py<PyAny>>> = RwLock::new(None);

pub fn set_global_caller(f: Option<Py<PyAny>>) {
    *GLOBAL_CALLER.write().unwrap() = f;
}

/// Caller resolution order: explicit parameter → global seam → real HTTP.
/// The returned `Arc` can cross `py.allow_threads` (bridges re-acquire the
/// GIL internally when they call back into Python).
pub fn resolve_caller(py: Python<'_>, param: Option<Py<PyAny>>) -> Arc<dyn Caller> {
    if let Some(p) = param {
        return Arc::new(PyCaller { f: p });
    }
    let global = GLOBAL_CALLER
        .read()
        .unwrap()
        .as_ref()
        .map(|g| g.clone_ref(py));
    if let Some(g) = global {
        return Arc::new(PyCaller { f: g });
    }
    Arc::new(HttpCaller)
}
