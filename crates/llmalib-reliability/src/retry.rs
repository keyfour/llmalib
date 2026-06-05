//! Retry: Reflection-based retry loop.
//!
//! Maps to `llmalib.reliability.retry` Python module.
//!
//! # Usage
//!
//! ```ignore
//! use llmalib_reliability::{run_with_retry, build_reflection_message};
//! ```

use llmalib_core::client::{call, ClientConfig, ClientError};
use llmalib_core::context::Context;
use llmalib_core::result::{make_error_result, make_ok_result, Attempt, TaskResult};
use llmalib_core::task::Task;
use serde_json::Value;
use std::time::Instant;

/// Build the reflection message that feeds failure details back to the model.
///
/// Maps to Python's `build_reflection_message` function.
///
/// # Arguments
/// * `parse_error` - Error from JSON extraction/validation
/// * `guard_errors` - Errors from guards
///
/// # Returns
/// A user-role message string with error details and directive to retry.
pub fn build_reflection_message(parse_error: Option<&str>, guard_errors: &[String]) -> String {
    let mut lines = vec![
        "Your previous response was invalid. Fix the following errors:",
        "",
    ];

    // Add parse error if present
    if let Some(error) = parse_error {
        for line in error.split('\n') {
            lines.push(line.trim());
        }
    }

    // Add guard errors
    for err in guard_errors {
        lines.push(err.trim());
    }

    // Add directive for the model
    lines.push("");
    lines.push(
        "Respond ONLY with valid JSON matching the required schema. No prose, no code fences, no explanation.",
    );

    lines.join("\n")
}

/// Execute task with reflection-based retry on validation or guard failure.
///
/// Maps to Python's `run_with_retry` function.
///
/// # Arguments
/// * `task` - Task configuration
/// * `messages` - Initial conversation history as JSON values
/// * `config` - Client configuration
/// * `ctx` - Context for guards
///
/// # Returns
/// TaskResult containing all attempts and final success/failure state.
pub fn run_with_retry(
    task: &Task,
    mut messages: Vec<Value>,
    config: &ClientConfig,
    ctx: &Context,
) -> Result<TaskResult, ClientError> {
    let mut trace: Vec<Attempt> = Vec::new();
    let max_retries = task.max_retries;
    let output_schema = &task.output_schema;

    for attempt_num in 1..=max_retries {
        let started_at = Instant::now();

        // Render messages for trace storage - serialize to JSON for display
        let rendered =
            serde_json::to_string(&messages).unwrap_or_else(|_| "<serialize error>".to_string());

        let raw = call(messages.clone(), config)?;

        // Parse response using this package's validator
        let parse_result = crate::validator::parse_response(&raw, output_schema);

        // Run guards on parsed value
        let guard_value = if parse_result.ok {
            parse_result.value.clone().unwrap_or(Value::Null)
        } else {
            Value::Null
        };

        let mut guard_errors: Vec<String> = Vec::new();
        if parse_result.ok {
            for guard in &task.guards {
                let result = guard.validate(&guard_value, ctx);
                guard_errors.extend(result);
            }
        }

        // Record attempt
        let duration_ms = started_at.elapsed().as_secs_f64() * 1000.0;

        let attempt = Attempt {
            attempt_number: attempt_num as usize,
            rendered_prompt: rendered,
            raw_response: raw.clone(),
            parse_error: parse_result.error.clone(),
            guard_errors: guard_errors.clone(),
            duration_ms,
        };

        trace.push(attempt);

        // Success case
        if parse_result.ok && guard_errors.is_empty() {
            return Ok(make_ok_result(
                task.name.clone(),
                parse_result.value.unwrap(),
                trace,
            ));
        }

        // Check if we can retry
        if attempt_num < max_retries {
            // Build reflection message for next attempt
            let reflection = build_reflection_message(parse_result.error.as_deref(), &guard_errors);
            messages.push(Value::Object(serde_json::Map::from_iter([
                ("role".to_string(), Value::String("assistant".to_string())),
                ("content".to_string(), Value::String(raw)),
            ])));
            messages.push(Value::Object(serde_json::Map::from_iter([
                ("role".to_string(), Value::String("user".to_string())),
                ("content".to_string(), Value::String(reflection)),
            ])));
        }
    }

    // Exhausted retries — report the last failure
    let last = trace.last().unwrap();
    let mut all_errors = Vec::new();
    if let Some(ref err) = last.parse_error {
        all_errors.push(err.clone());
    }
    all_errors.extend(last.guard_errors.clone());
    let final_error = format!(
        "Failed after {} attempts. {}",
        max_retries,
        all_errors.join(" | ")
    );

    Ok(make_error_result(task.name.clone(), final_error, trace))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validator::parse_response;
    use serde_json::json;

    #[test]
    fn test_build_reflection_message_with_parse_error() {
        let msg = build_reflection_message(Some("Missing field 'confidence'"), &[]);
        assert!(msg.contains("confidence"));
    }

    #[test]
    fn test_build_reflection_message_with_guard_errors() {
        let msg =
            build_reflection_message(None, &["Score out of range".into(), "Invalid label".into()]);
        assert!(msg.contains("Score out of range"));
    }

    #[test]
    fn test_parse_response_valid() {
        let raw = r#"{"label": "positive", "confidence": 0.9, "reason": "good"}"#;
        let result = parse_response(raw, &json!({}));
        assert!(result.ok);
    }

    #[test]
    fn test_parse_response_no_json() {
        let result = parse_response("plain text", &json!({}));
        assert!(!result.ok);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_parse_response_markdown_fence() {
        let raw = r#"```json
{"label": "positive", "confidence": 0.9}
```"#;
        let result = parse_response(raw, &json!({}));
        assert!(result.ok);
    }
}
