//! Guards: post-generation validation heuristics.
//!
//! Maps to `llmalib.reliability.guards` Python module.
//!
//! # Usage
//!
//! ```ignore
//! use llmalib_reliability::{
//!     Guard, ClosureGuard, field_in_set, float_in_range, max_length,
//!     no_content_from_outside_context,
//! };
//! ```

use llmalib_core::context::Context;
use llmalib_core::guard::Guard as CoreGuard;
use serde_json::Value;
use std::sync::Arc;

/// Raw closure type stored inside [ClosureGuard].
pub type GuardFn = Arc<dyn Fn(&Value, &Context) -> Vec<String> + Send + Sync>;

/// Bridge from a plain closure / function to the core [`CoreGuard`] trait.
///
/// Rust has no blanket coercion from `Fn(&Value, &Context) -> Vec<String>`
/// onto `dyn Guard` (the compiler cannot prove the higher-ranked impl is
/// general enough), so closure-based guards must be wrapped explicitly.
/// This is the single bridge used by all guard factories below and by
/// user code with closure guards.
///
/// ```
/// use llmalib_reliability::guard::ClosureGuard;
/// use llmalib_core::guard::Guard;
/// use llmalib_core::context::Context;
/// use serde_json::json;
///
/// let guard = ClosureGuard::new(|value: &serde_json::Value, _ctx: &Context| {
///     if value.get("label").map(|v| v.as_str() == Some("x")).unwrap_or(false) {
///         vec!["label 'x' is forbidden".to_string()]
///     } else {
///         Vec::new()
///     }
/// });
/// assert!(guard.validate(&json!({"label": "y"}), &Context::new()).is_empty());
/// assert!(!guard.validate(&json!({"label": "x"}), &Context::new()).is_empty());
/// ```
pub struct ClosureGuard {
    f: GuardFn,
}

impl ClosureGuard {
    /// Wrap any closure or function with the guard signature.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&Value, &Context) -> Vec<String> + Send + Sync + 'static,
    {
        Self { f: Arc::new(f) }
    }
}

impl CoreGuard for ClosureGuard {
    fn validate(&self, value: &Value, context: &Context) -> Vec<String> {
        (self.f)(value, context)
    }
}

/// A guard that validates output values.
///
/// Guards are lightweight, rule-based heuristics that return error messages
/// (empty vector = pass). When a guard fails, error messages are fed back
/// to the model via the reflection loop.
///
/// This is the single canonical [`Guard`] type: every factory in this module
/// produces a value directly assignable to `Task::guards` (via the `guard()`
/// builder method), and custom implementations of the core trait work too.
pub type Guard = Arc<dyn CoreGuard>;

/// Check that a field value is in a set of allowed values.
///
/// Maps to Python's `field_in_set` function.
///
/// # Arguments
/// * `field` - Field name to check
/// * `allowed` - Set of allowed values
///
/// # Returns
/// A guard that checks if `value.<field>` is in `allowed`.
pub fn field_in_set(field: &str, allowed: &[String]) -> Arc<dyn CoreGuard> {
    let mut allowed_set = allowed.to_vec();
    allowed_set.sort(); // mirrored in the error message, like Python's sorted(allowed)
    let field_name = field.to_string();

    Arc::new(ClosureGuard::new(move |value, _| {
        let actual = get_nested_field_str(value, &field_name).unwrap_or_default();
        if !allowed_set.contains(&actual) {
            return vec![format!(
                "Field '{}' must be one of {:?}, got {:?}",
                field_name, allowed_set, actual
            )];
        }
        Vec::new()
    }))
}

/// Check that a float field is within a range.
///
/// Maps to Python's `float_in_range` function.
///
/// # Arguments
/// * `field` - Field name to check
/// * `min_val` - Minimum allowed value (inclusive)
/// * `max_val` - Maximum allowed value (inclusive)
///
/// # Returns
/// A guard that checks if `value.<field>` is between min and max.
pub fn float_in_range(field: &str, min_val: f64, max_val: f64) -> Arc<dyn CoreGuard> {
    let field_name = field.to_string();

    Arc::new(ClosureGuard::new(move |value, _| {
        let nested = get_nested_value(value, &field_name);
        let actual = match nested {
            // Missing key or explicit null.
            None | Some(Value::Null) => {
                return vec![format!("Field '{}' is missing or None", field_name)]
            }
            Some(v) => v,
        };
        let num = match actual.as_f64() {
            None => return vec![format!("Field '{}' is not numeric", field_name)],
            Some(f) => f,
        };
        if !(min_val..=max_val).contains(&num) {
            vec![format!(
                "Field '{}' must be between {} and {}, got {}",
                field_name, min_val, max_val, num
            )]
        } else {
            Vec::new()
        }
    }))
}

/// Check that a field string does not exceed a maximum length.
///
/// Maps to Python's `max_length` function.
///
/// # Arguments
/// * `field` - Field name to check
/// * `max_chars` - Maximum allowed character count
///
/// # Returns
/// A guard that checks string length of `value.<field>`.
pub fn max_length(field: &str, max_chars: usize) -> Arc<dyn CoreGuard> {
    let field_name = field.to_string();

    Arc::new(ClosureGuard::new(move |value, _| {
        let actual = get_nested_field_str(value, &field_name);

        // Skip None values
        if actual.is_none() {
            return Vec::new();
        }

        let value = actual.unwrap_or_default();
        let length = value.chars().count();
        if length > max_chars {
            return vec![format!(
                "Field '{}' is too long: {} chars (max {}). Provide a more concise response.",
                field_name, length, max_chars
            )];
        }

        Vec::new()
    }))
}

/// Lightweight grounding check using Jaccard similarity.
///
/// Maps to Python's `no_content_from_outside_context` function.
///
/// Verifies that a response field shares significant token overlap with
/// context from the context. Uses Jaccard similarity on lowercased word
/// tokens — no embeddings needed.
///
/// # Arguments
/// * `response_field` - Field name in the model output
/// * `context_field` - Field name in `ctx.vars` containing source text
/// * `threshold` - Minimum Jaccard similarity (0.0–1.0), default 0.3
///
/// # Returns
/// A guard that checks the response is grounded in the provided context.
pub fn no_content_from_outside_context(
    response_field: &str,
    context_field: &str,
    threshold: f64,
) -> Arc<dyn CoreGuard> {
    let response_field_name = response_field.to_string();
    let context_field_name = context_field.to_string();

    Arc::new(ClosureGuard::new(move |value, ctx| {
        // Get response text
        let response_text = get_nested_field_str(value, &response_field_name).unwrap_or_default();

        // Get context text from ctx.vars
        let context_text = ctx
            .vars
            .get(&context_field_name)
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_default();

        if context_text.trim().is_empty() {
            // No context to compare against — skip the check
            return Vec::new();
        }

        let response_tokens = tokenize(&response_text);
        let context_tokens = tokenize(&context_text);

        if response_tokens.is_empty() {
            return vec![format!(
                "Field '{}' is empty — no content to ground.",
                response_field_name
            )];
        }

        // Compute Jaccard similarity
        let intersection = response_tokens
            .iter()
            .filter(|t| context_tokens.contains(t))
            .count();
        let union = response_tokens.len() + context_tokens.len() - intersection;
        let jaccard = if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        };

        if jaccard < threshold {
            return vec![format!(
                "Field '{}' does not appear grounded in the provided context (similarity={:.2}, required>={}). Base your answer only on the provided input text.",
                response_field_name, jaccard, threshold
            )];
        }

        Vec::new()
    }))
}

/// Helper: Get nested field value as String from a JSON object.
fn get_nested_field_str(value: &Value, field: &str) -> Option<String> {
    let mut current = Some(value);
    for part in field.split('.') {
        current = current.and_then(|v| {
            if let Some(obj) = v.as_object() {
                obj.get(part)
            } else {
                None
            }
        });
    }
    current.and_then(|v| v.as_str()).map(|s| s.to_string())
}

/// Helper: Get a nested field by dotted path (does not coerce types).
fn get_nested_value<'a>(value: &'a Value, field: &str) -> Option<&'a Value> {
    let mut current = Some(value);
    for part in field.split('.') {
        current = current.and_then(|v| {
            if let Some(obj) = v.as_object() {
                obj.get(part)
            } else {
                None
            }
        });
    }
    current
}

/// Tokenize text into lowercase word tokens.
///
/// Maps to Python's `_tokenize` function.
fn tokenize(text: &str) -> Vec<String> {
    regex::Regex::new(r"[a-z0-9]+")
        .unwrap()
        .find_iter(text.to_lowercase().as_str())
        .map(|m| m.as_str().to_string())
        .collect()
}
