//! pyclasses exposed to Python: context views, stores, tracer.

use llmalib_core::context::Context;
use llmalib_debug::tracer::{self, Tracer};
use llmalib_memory::store::{
    make_file_store as rust_make_file_store, make_memory_store as rust_make_memory_store, Store,
    StoreEntry, StoreError,
};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyRuntimeError};
use pyo3::prelude::*;
use serde_json::Value;
use std::time::Duration;

use crate::convert::{py_to_value, result_from_py, value_to_py};

create_exception!(_rust, StoreError_, PyException);

// ---------------------------------------------------------------------------
// ContextView — what Python guards receive instead of the Rust Context
// ---------------------------------------------------------------------------

/// Read-only snapshot of a Rust [`Context`], handed to Python guard
/// callables. Attribute access mirrors the Python `Context` dataclass
/// (`ctx.vars`, `ctx.results`, `ctx.run_id`).
#[pyclass]
pub struct ContextView {
    pub vars: PyObject,
    pub results: PyObject,
    pub run_id: String,
}

/// Build a fresh [`ContextView`] snapshot for a guard call.
pub fn context_view(py: Python<'_>, ctx: &Context) -> PyResult<ContextView> {
    Ok(ContextView {
        vars: value_to_py(py, &Value::Object(ctx.vars.clone()))?,
        results: value_to_py(py, &Value::Object(ctx.results.clone()))?,
        run_id: ctx.run_id.clone(),
    })
}

#[pymethods]
impl ContextView {
    #[getter]
    fn vars(&self, py: Python<'_>) -> PyObject {
        self.vars.clone_ref(py)
    }

    #[getter]
    fn results(&self, py: Python<'_>) -> PyObject {
        self.results.clone_ref(py)
    }

    #[getter]
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    fn __repr__(&self) -> String {
        format!("ContextView(run_id='{}')", self.run_id)
    }
}

// ---------------------------------------------------------------------------
// PyStore — InMemoryStore / FileStore behind one class
// ---------------------------------------------------------------------------

/// Memory or file-backed KV store (Python's `InMemoryStore` / `FileStore`
/// unified; `make_memory_store()` / `make_file_store(path)` construct them).
///
/// `unsendable`: `Box<dyn Store>` has no `Send` bound and stores are used
/// single-threaded, like the Python originals.
#[pyclass(unsendable)]
pub struct PyStore {
    inner: Box<dyn Store>,
}

impl PyStore {
    pub fn inner_mut(&mut self) -> &mut dyn Store {
        &mut *self.inner
    }
}

fn entry_to_py(py: Python<'_>, entry: &StoreEntry) -> PyResult<PyObject> {
    value_to_py(
        py,
        &serde_json::to_value(entry).map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
}

#[pymethods]
impl PyStore {
    /// `set(key, value, score=1.0, ttl=None)` — ttl in seconds.
    #[pyo3(signature = (key, value, score=1.0, ttl=None))]
    pub fn set(
        &mut self,
        py: Python<'_>,
        key: String,
        value: &Bound<'_, PyAny>,
        score: f64,
        ttl: Option<f64>,
    ) -> PyResult<()> {
        let v = py_to_value(py, value)?;
        self.inner
            .set(key, v, score, ttl.map(Duration::from_secs_f64));
        Ok(())
    }

    /// `get(key)` → entry dict or None.
    pub fn get(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        match self.inner.get(key) {
            Some(entry) => entry_to_py(py, &entry),
            None => Ok(py.None()),
        }
    }

    /// `delete(key)`.
    pub fn delete(&mut self, key: &str) {
        self.inner.delete(key);
    }

    /// `list(prefix)` → list of non-expired keys.
    pub fn list(&mut self, prefix: &str) -> Vec<String> {
        self.inner.list(prefix)
    }

    /// `get_relevant(query, top_k)` → top entries by BM25.
    pub fn get_relevant(
        &mut self,
        py: Python<'_>,
        query: &str,
        top_k: usize,
    ) -> PyResult<PyObject> {
        let entries = self.inner.get_relevant(query, top_k);
        let list: Vec<PyObject> = entries
            .iter()
            .map(|e| entry_to_py(py, e))
            .collect::<PyResult<_>>()?;
        Ok(pyo3::types::PyList::new(py, list)?.into_any().unbind())
    }
}

/// In-memory store factory (Python `make_memory_store()`).
#[pyfunction]
pub fn make_memory_store() -> PyStore {
    PyStore {
        inner: rust_make_memory_store(),
    }
}

/// File-backed store factory (Python `make_file_store(path)`).
#[pyfunction]
pub fn make_file_store(path: String) -> PyResult<PyStore> {
    match rust_make_file_store(&path) {
        Ok(inner) => Ok(PyStore { inner }),
        Err(StoreError::Io(e)) => Err(PyRuntimeError::new_err(e.to_string())),
        Err(StoreError::Json(e)) => Err(StoreError_::new_err(e.to_string())),
        Err(other) => Err(PyRuntimeError::new_err(other.to_string())),
    }
}

// ---------------------------------------------------------------------------
// PyTracer
// ---------------------------------------------------------------------------

/// Always-on run recorder (Python `Tracer`).
#[pyclass]
pub struct PyTracer {
    pub inner: Tracer,
}

#[pymethods]
impl PyTracer {
    #[new]
    fn new(run_id: String) -> Self {
        PyTracer {
            inner: Tracer::new(run_id),
        }
    }

    /// Append a result (dict) to the trace.
    pub fn record(&mut self, py: Python<'_>, result: &Bound<'_, PyAny>) -> PyResult<()> {
        let r = result_from_py(py, result)?;
        self.inner.record(&r);
        Ok(())
    }

    /// One-line run summary.
    pub fn summary(&self) -> String {
        self.inner.summary()
    }

    /// Full trace as a dict.
    pub fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        value_to_py(py, &self.inner.to_dict())
    }

    /// Write the trace to a JSON file.
    pub fn to_file(&self, path: String) -> PyResult<()> {
        self.inner
            .to_file(&path)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[getter]
    pub fn run_id(&self) -> &str {
        &self.inner.run_id
    }

    #[getter]
    pub fn task_traces(&self, py: Python<'_>) -> PyResult<PyObject> {
        value_to_py(
            py,
            &serde_json::to_value(&self.inner.task_traces)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        )
    }
}

/// Tracer factory (Python `make_tracer(run_id)`).
#[pyfunction]
pub fn make_tracer(run_id: String) -> PyTracer {
    PyTracer {
        inner: tracer::make_tracer(run_id),
    }
}

/// Load a tracer from a trace file (Python `load_trace(path)`).
#[pyfunction]
pub fn load_trace(path: String) -> PyResult<PyTracer> {
    match tracer::load_trace(&path) {
        Ok(inner) => Ok(PyTracer { inner }),
        Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
    }
}
