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

/// A guard validator that validates task output after generation.
///
/// Guards run on the parsed model output and return error strings if
/// validation fails. An empty vector means the guard passed.
///
/// Implement directly for your own structs, or wrap a closure with
/// [`llmalib_reliability::guard::ClosureGuard`]:
///
/// ```
/// use llmalib_core::guard::Guard;
/// use llmalib_core::context::Context;
/// use serde_json::json;
/// use std::sync::Arc;
///
/// // A guard that checks if a field value is valid
/// struct NonEmptyName;
///
/// impl Guard for NonEmptyName {
///     fn validate(&self, value: &serde_json::Value, _ctx: &Context) -> Vec<String> {
///         let name = value.get("name").and_then(|v| v.as_str()).unwrap_or("");
///         if name.is_empty() {
///             return vec!["name cannot be empty".to_string()];
///         }
///         Vec::new()
///     }
/// }
///
/// let guard: Arc<dyn Guard> = Arc::new(NonEmptyName);
/// assert!(guard.validate(&json!({"name": "x"}), &Context::new()).is_empty());
/// ```
pub trait Guard: Send + Sync + 'static {
    /// Validates the given value and context, returning error strings or empty vec on success
    fn validate(&self, _value: &serde_json::Value, _context: &Context) -> Vec<String>;
}
