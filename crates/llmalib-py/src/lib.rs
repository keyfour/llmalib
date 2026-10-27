//! `llmalib._rust` — pyo3 binding over the Rust crates.
//!
//! This module is the FFI surface for the Python package: the Phase 4 Python
//! adapters marshal their dataclass/closure arguments into the flat shapes
//! understood here (see [`convert`] for the task-spec contract).
//!
//! Exceptions raised on this boundary:
//! - `ClientError` — transport failure (retry/decompose/pipeline callers)
//! - `BudgetExceededError` — protected messages exceed the token budget
//! - `DecompositionError` — planner could not produce a valid plan

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

pub mod api;
pub mod bridges;
pub mod classes;
pub mod convert;

pub use bridges::PyGuardRef;
pub use classes::{ContextView, PyStore, PyTracer};

create_exception!(_rust, ClientError, PyException);
create_exception!(_rust, BudgetExceededError, PyException);
create_exception!(_rust, DecompositionError, PyException);

/// The `llmalib._rust` extension module entry point.
#[pymodule]
fn _rust(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    api::register(py, m)
}
