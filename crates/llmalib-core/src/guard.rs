//! Guard validators for task execution
//!
//! # Purpose
//!
//! Guards are validation functions that run before task execution to catch
//! errors early and provide context-specific validation. This mirrors Python's
//! `tuple[Guard, ...]` where `Guard = Callable[[BaseModel, Any], list[str]]`.
//!
//! # Semantics
//!
//! - Takes parsed output value (serde_json::Value) and context dict as input
//! - Returns Vec<String> of error messages if validation fails
//! - Empty Vec means validation passed (no errors)
//!
//! # Python ↔ Rust Mapping
//!
//! | Python | Rust |
//! |--------|------|
//! | `tuple[Callable, ...]` | `Vec<Guard>` |
//! | `Callable[[BaseModel, Any], list[str]]` | `fn validate(self, value: &serde_json::Value, context: &Context) -> Vec<String>` |

use crate::Context;

/// A guard validator that validates task output before it's used
///
/// Guards run before task execution and return error strings if validation fails.
/// An empty vector means the guard passed.
///
/// # Example
///
/// ```
/// use llmalib_core::guard::Guard;
/// use llmalib_core::context::Context;
/// use serde_json::json;
///
/// // A guard that checks if a field value is valid
/// fn my_guard(value: &serde_json::Value, _ctx: &Context) -> Vec<String> {
///     if let Some(data) = value.as_object() {
///         if let Some(name) = data.get("name") {
///             if name.as_str().unwrap_or("") == "" {
///                 return vec!["name cannot be empty".to_string()];
///             }
///         }
///     }
///     Vec::new()
/// }
/// ```
pub trait Guard: Send + Sync + 'static {
    /// Validates the given value and context, returning error strings or empty vec on success
    fn validate(&self, _value: &serde_json::Value, _context: &Context) -> Vec<String> {
        Vec::new()
    }
}

impl<F> Guard for F
where
    F: Fn(&serde_json::Value, &Context) -> Vec<String> + Send + Sync + 'static,
{
    fn validate(&self, value: &serde_json::Value, context: &Context) -> Vec<String> {
        self(value, context)
    }
}

// Manual Debug implementation for Box<dyn Guard>
use std::fmt;
impl fmt::Debug for Box<dyn Guard> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Box<dyn Guard>")
    }
}

// Clone is automatically implemented for Box<dyn Guard> - Box types implement Clone by default
