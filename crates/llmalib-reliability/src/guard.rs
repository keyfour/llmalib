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
use std::collections::HashSet;
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
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
        let actual = get_nested_value(value, &field_name);
        // Python membership test against a set of (stringified) allowed
        // values; missing fields read as `None` (never a member).
        let passes = match actual {
            Some(v) if !v.is_null() => {
                let s = py_str(v);
                allowed_set.contains(&s)
            }
            _ => false,
        };
        if !passes {
            let repr = actual.map(py_repr).unwrap_or_else(|| "None".into());
            let options = allowed_set
                .iter()
                .map(|s| py_str_repr(s))
                .collect::<Vec<_>>()
                .join(", ");
            return vec![format!(
                "Field '{}' must be one of [{options}], got {repr}",
                field_name
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
        let repr = nested.map(py_repr).unwrap_or_else(|| "None".into());
        let num = match nested {
            // Missing key or explicit null.
            None | Some(Value::Null) => {
                return vec![format!("Field '{}' is missing or None", field_name)]
            }
            // Python's `float()` coerces numbers, bools and numeric strings.
            Some(Value::Number(n)) => n.as_f64(),
            Some(Value::Bool(b)) => Some(if *b { 1.0 } else { 0.0 }),
            Some(Value::String(s)) => parse_python_float(s),
            _ => None,
        };
        let Some(num) = num else {
            return vec![format!("Field '{}' is not numeric: {repr}", field_name)];
        };
        if !(min_val..=max_val).contains(&num) {
            vec![format!(
                "Field '{}' must be between {} and {}, got {}",
                field_name,
                python_float(min_val),
                python_float(max_val),
                python_float(num)
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
        let actual = get_nested_value(value, &field_name);

        // Python's None-guard: missing fields read as None and skip the check.
        if matches!(actual, None | Some(Value::Null)) {
            return Vec::new();
        }
        // Python str(actual): numbers/bools stringify per str(); missing
        // stringification falls back to the JSON form for containers.
        let value = py_str(actual.unwrap_or(&Value::Null));
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
        // Get response text — mirrors Python's `str(getattr(value, f, "") or "")`
        let response_text = get_nested_value(value, &response_field_name)
            .map(truthy_str)
            .unwrap_or_default();

        // Get context text from ctx.vars — mirrors `str(ctx.vars.get(f, "") or "")`
        let context_text = ctx
            .vars
            .get(&context_field_name)
            .map(truthy_str)
            .unwrap_or_default();

        if context_text.trim().is_empty() {
            // No context to compare against — skip the check
            return Vec::new();
        }

        // Python builds token sets — duplicates never affect Jaccard.
        let response_tokens: HashSet<String> = tokenize(&response_text).into_iter().collect();
        let context_tokens: HashSet<String> = tokenize(&context_text).into_iter().collect();

        if response_tokens.is_empty() {
            return vec![format!(
                "Field '{}' is empty — no content to ground.",
                response_field_name
            )];
        }

        // Compute Jaccard similarity
        let intersection = response_tokens.intersection(&context_tokens).count();
        let union = response_tokens.union(&context_tokens).count();
        let jaccard = if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        };

        if jaccard < threshold {
            return vec![format!(
                "Field '{}' does not appear grounded in the provided '{}' (similarity={:.2}, required>={}). Base your answer only on the provided input text.",
                response_field_name, context_field_name, jaccard, threshold
            )];
        }

        Vec::new()
    }))
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

// ---------------------------------------------------------------------------
// Python-style coercion helpers (parity with str()/repr()/float())
// ---------------------------------------------------------------------------

/// Python `repr()` of a string: single-quoted with backslash escapes.
pub fn py_str_repr(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        match c {
            '\'' => out.push_str("\\'"),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('\'');
    out
}

/// Python `str()`/f-string formatting for a float (`0.0`, `1.5`).
pub fn python_float(f: f64) -> String {
    if f.fract() == 0.0 {
        format!("{f:.1}")
    } else {
        format!("{f}")
    }
}

/// Python `repr()` for a JSON scalar: booleans, `None`, numbers and quoted
/// strings; containers fall back to the JSON text (rare in guard messages).
pub fn py_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                n.as_f64()
                    .map(python_float)
                    .unwrap_or_else(|| n.to_string())
            }
        }
        Value::String(s) => py_str_repr(s),
        other => serde_json::to_string(other)
            .map(|s| py_str_repr(&s))
            .unwrap_or_else(|_| "None".to_string()),
    }
}

/// Python `str()` for a JSON value: strings verbatim, scalars via their
/// str/repr, containers fall back to the JSON text.
pub fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => py_repr_without_quotes(other),
    }
}

/// Python `str(value or "")`: falsy values become `""`, everything else
/// stringifies via [`py_str`].
pub fn truthy_str(v: &Value) -> String {
    let falsy = match v {
        Value::Null => true,
        Value::Bool(b) => !b,
        Value::Number(n) => n.as_f64().map(|f| f == 0.0).unwrap_or(false),
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
    };
    if falsy {
        String::new()
    } else {
        py_str(v)
    }
}

/// Python `float(string)`: trims whitespace, understands inf/nan spellings.
fn parse_python_float(s: &str) -> Option<f64> {
    let t = s.trim();
    let lowered = t.to_ascii_lowercase();
    match lowered.as_str() {
        "inf" | "+inf" | "infinity" | "+infinity" => Some(f64::INFINITY),
        "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
        "nan" | "+nan" | "-nan" => Some(f64::NAN),
        _ => lowered.parse::<f64>().ok(),
    }
}

/// Repr-style string for non-JSON-string values (numbers/bools render
/// unquoted like Python's str(); containers keep a JSON-ish form).
fn py_repr_without_quotes(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(_) => py_repr(v),
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_else(|_| "None".to_string()),
    }
}
